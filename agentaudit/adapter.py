"""Audit an agent this library did not write.

The built-in :class:`~agentaudit.agent.SelfCorrectingAgent` exists to make
the harness self-testing, not because anyone's production pipeline looks
like it. This module points the same fault taxonomy at *your* agent.

The seam is the tool boundary, which is where the injector already sits.
You hand over your tool functions; you get back functions with identical
signatures that occasionally lie. Your agent calls them without knowing,
which is the whole requirement — an agent that can tell it is being
tested produces a detection rate that means nothing.

    from agentaudit import audit_agent, Schema

    def my_agent(document, tools):
        header = tools["fetch_header"](document["id"])
        totals = tools["fetch_totals"](document["id"])
        return {**header, **totals}

    result = audit_agent(
        agent=my_agent,
        tools={"fetch_header": fetch_header, "fetch_totals": fetch_totals},
        documents=[{"document_id": "inv-1", "id": 1, "truth": {...}}, ...],
        schema=Schema.from_spec({"total": {"type": "money", "criticality": "critical"}}),
    )
    print(result.summary())

Nothing about your control flow has to change. Framework agents work the
same way: wrap the callables the framework dispatches to, then let the
framework run.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .dataset import corpus_for_injector
from .faults.injector import FaultInjector, InjectionPlan
from .metrics import Report, RunScore, score_run
from .runner import Condition, ExperimentResult, standard_conditions
from .schema import ConsistencyCheck, Schema
from .tools.base import Tool, ToolRegistry
from .trace import SpanKind, SpanStatus, Trace

# The wrapped tools have to find the run they belong to without the agent
# passing anything around. A ContextVar rather than a module global so
# concurrent audits — and async agents — stay separate.
_CURRENT: ContextVar[_RunContext | None] = ContextVar("agentaudit_run", default=None)

# Positional arguments are smuggled through the registry, which is
# keyword-oriented because its own tools are.
_ARGS = "__positional__"


class FunctionTool(Tool):
    """Adapts a plain callable to the Tool interface.

    The registry passes the document to every tool; an external function
    knows nothing about documents, so it is dropped here. Whatever the
    agent actually passes travels in the call arguments.
    """

    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name = name
        self.fn = fn
        self.description = (fn.__doc__ or "").strip().split("\n")[0]

    def run(self, document: dict[str, Any], **kwargs: Any) -> Any:
        args = kwargs.pop(_ARGS, ())
        return self.fn(*args, **kwargs)


@dataclass
class _RunContext:
    """One document's worth of audit state."""

    registry: ToolRegistry
    trace: Trace
    document: dict[str, Any]
    max_retries: int = 0
    non_dict_returns: set[str] = field(default_factory=set)

    def call(self, name: str, args: tuple, kwargs: dict) -> Any:
        payload = self.registry.call(
            name, self.document, self.trace,
            max_retries=self.max_retries,
            **{_ARGS: args, **kwargs},
        )
        if payload is not None and not isinstance(payload, dict):
            # Value-corrupting faults rewrite fields, so they need a mapping
            # to work on. Anything else can still be failed at the transport
            # layer, but it cannot be quietly corrupted — worth saying out
            # loud rather than reporting a suspiciously clean result.
            self.non_dict_returns.add(name)
        return payload


def wrap_tools(tools: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    """Return tools that behave identically until an audit is running.

    Outside an audit the wrappers are pass-throughs, so the same wired-up
    agent can serve production and the harness.
    """
    return {name: _wrap(name, fn) for name, fn in tools.items()}


def _wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        context = _CURRENT.get()
        if context is None:
            return fn(*args, **kwargs)
        return context.call(name, args, kwargs)

    return wrapper


@dataclass
class AuditResult:
    """Reports plus the caveats a reader needs to interpret them."""

    reports: list[Report] = field(default_factory=list)
    traces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    agent_errors: dict[str, list[str]] = field(default_factory=dict)
    non_dict_tools: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reports": [r.to_dict() for r in self.reports],
            "traces": self.traces,
            "agent_errors": self.agent_errors,
            "tools_not_corruptible": sorted(self.non_dict_tools),
        }

    def save(self, path: str) -> Any:
        return ExperimentResult(self.reports, self.traces).save(path)

    def summary(self) -> str:
        from .metrics import format_table

        lines = [format_table(self.reports)]
        if self.non_dict_tools:
            lines += [
                "",
                "note: these tools returned something other than a dict, so only "
                "transport faults could be applied to them — their silent-failure "
                "numbers are not meaningful:",
                "  " + ", ".join(sorted(self.non_dict_tools)),
            ]
        failed = {k: v for k, v in self.agent_errors.items() if v}
        if failed:
            lines += ["", "agent raised on some documents:"]
            for condition, errors in failed.items():
                lines.append(f"  {condition}: {len(errors)} run(s), e.g. {errors[0]}")
        return "\n".join(lines)


def audit_agent(
    agent: Callable[[dict[str, Any], dict[str, Callable[..., Any]]], dict[str, Any]],
    tools: dict[str, Callable[..., Any]],
    documents: Iterable[dict[str, Any]],
    schema: Schema,
    *,
    conditions: list[Condition] | None = None,
    checks: tuple[ConsistencyCheck, ...] = (),
    keep_traces: int = 2,
    truth_key: str = "truth",
    retry_transport_faults: bool = False,
) -> AuditResult:
    """Run ``agent`` against every condition and score what it produced.

    ``agent`` is called once per document as ``agent(document, tools)`` and
    must return the record it extracted. ``tools`` are your real functions;
    the ones handed to the agent are wrapped.

    Each document needs a ``document_id`` and a ``truth`` mapping of the
    field values it should have produced. Everything else in the document
    is yours — the harness only passes it through.

    An exception from the agent is recorded, not raised: a pipeline that
    crashes under a fault has told you something, and it should not stop
    the rest of the run.

    Transport faults are raised into your code and **not** retried, because
    retrying them here would measure this library's error handling instead
    of yours. If your pipeline has no retry logic, that is a finding, and
    it should show up as one. Set ``retry_transport_faults=True`` only when
    something outside the audited code — a gateway, a client library —
    genuinely does the retrying in production.
    """
    documents = list(documents)
    _validate(documents, truth_key)

    wrapped = wrap_tools(tools)
    conditions = conditions or standard_conditions()
    donor_corpus = _donor_corpus(documents, truth_key)

    outcome = AuditResult()
    for condition in conditions:
        injector = FaultInjector(condition.plan, corpus=donor_corpus)
        registry = ToolRegistry(
            [FunctionTool(name, fn) for name, fn in tools.items()], injector=injector
        )
        report = Report(condition=condition.name)
        kept: list[dict[str, Any]] = []
        errors: list[str] = []

        for document in documents:
            doc_id = str(document.get("document_id", ""))
            trace = Trace(document_id=doc_id)
            trace.metadata["agent"] = getattr(agent, "__name__", "agent")
            trace.metadata["condition"] = condition.name

            registry.reset()
            injector.begin_run(doc_id)

            root = trace.span("agent", SpanKind.AGENT)
            trace.push(root)
            context = _RunContext(
                registry, trace, document,
                max_retries=2 if retry_transport_faults else 0,
            )
            token = _CURRENT.set(context)
            try:
                record = agent(document, wrapped)
                root.finish(SpanStatus.OK)
                succeeded = True
                error = None
            except Exception as exc:  # the agent's failure is a result, not a crash
                record, succeeded, error = {}, False, repr(exc)
                root.finish(SpanStatus.ERROR, error=error)
                errors.append(f"{doc_id}: {error}")
            finally:
                _CURRENT.reset(token)
                trace.pop()
                outcome.non_dict_tools |= context.non_dict_returns

            report.runs.append(
                _score(doc_id, record, trace, succeeded, error,
                       document[truth_key], schema, checks)
            )
            if len(kept) < keep_traces and trace.faults_injected:
                kept.append(trace.to_dict())

        outcome.reports.append(report)
        outcome.traces[condition.name] = kept
        outcome.agent_errors[condition.name] = errors

    return outcome


def _score(
    doc_id: str,
    record: Any,
    trace: Trace,
    succeeded: bool,
    error: str | None,
    truth: dict[str, Any],
    schema: Schema,
    checks: tuple[ConsistencyCheck, ...],
) -> RunScore:
    from .agent import ExtractionResult

    if not isinstance(record, dict):
        # A non-mapping return scores as a total miss rather than raising:
        # "the agent returned something unusable" is a legitimate outcome
        # to measure, and one that faults do provoke.
        error = error or f"agent returned {type(record).__name__}, expected a mapping"
        record, succeeded = {}, False

    result = ExtractionResult(doc_id, record, trace, succeeded=succeeded, error=error)
    if checks:
        # Consistency checks are the agent's own defence, so a failure here
        # is only interesting on the trace, not in the score.
        problems = [message for check in checks if (message := check(record))]
        if problems:
            trace.metadata.setdefault("consistency_problems", problems)
    return score_run(result, truth, schema)


def _validate(documents: list[dict[str, Any]], truth_key: str) -> None:
    if not documents:
        raise ValueError("no documents to audit")
    seen: set[str] = set()
    for i, document in enumerate(documents):
        if truth_key not in document:
            raise ValueError(
                f"document {i} has no {truth_key!r} — every document needs the "
                "field values it should have produced, or nothing can be scored"
            )
        doc_id = str(document.get("document_id", ""))
        if not doc_id:
            raise ValueError(f"document {i} has no 'document_id'")
        if doc_id in seen:
            raise ValueError(f"duplicate document_id {doc_id!r}")
        seen.add(doc_id)


def _donor_corpus(
    documents: list[dict[str, Any]], truth_key: str
) -> list[dict[str, Any]]:
    """Real values from other documents, for the substitution faults.

    Ground truth is the right source here: those faults need a value that
    is genuinely valid somewhere else, which is exactly what makes them
    survive every internal consistency check.
    """
    return corpus_for_injector(
        [{"document_id": d["document_id"], "fields": d[truth_key]} for d in documents]
    )


def quick_audit(
    agent: Callable[..., dict[str, Any]],
    tools: dict[str, Callable[..., Any]],
    documents: Iterable[dict[str, Any]],
    schema: Schema,
    *,
    faults: tuple[str, ...] = ("plausible_substitution",),
    rate: float = 1.0,
    seed: int = 0,
) -> AuditResult:
    """One clean arm and one faulted arm — the smallest useful audit.

    Defaults to `plausible_substitution` because it is the fault most
    pipelines have never been tested against: the value is well-formed,
    internally consistent, and simply belongs to a different document.
    """
    return audit_agent(
        agent, tools, documents, schema,
        conditions=[
            Condition("clean", InjectionPlan(rate=0.0, seed=seed)),
            Condition(
                "+".join(faults),
                InjectionPlan(rate=rate, faults=faults, seed=seed, max_per_run=1),
            ),
        ],
    )
