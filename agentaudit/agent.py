"""Self-correcting extraction agent.

The loop is deliberately explicit rather than framework-driven, because
the benchmark needs to attribute every correction to a trigger. A
framework that hides its retry logic makes detection rate unmeasurable.

    gather -> assemble -> validate -> [corroborate] -> correct -> repeat

Each stage can fail independently, and the trace records which stage
caught what. That per-stage attribution is the whole point: knowing an
agent recovered is much less useful than knowing which check saved it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .faults.injector import ToolFailure
from .providers.base import Provider
from .schema import ConsistencyCheck, Field, Schema
from .tools.base import ToolRegistry
from .trace import SpanKind, SpanStatus, Trace

EXTRACT_SYSTEM = (
    "You extract structured records from documents. Return only a JSON "
    "object matching the requested schema. No prose, no code fences."
)

REVIEW_SYSTEM = (
    "REVIEW. You audit a candidate extraction for errors. Return only JSON: "
    '{"verdict": "accept"|"reject", "reasons": [string]}. '
    "Reject if any value is implausible, internally inconsistent, or "
    "unsupported by the evidence."
)


@dataclass
class ExtractionResult:
    document_id: str
    record: dict[str, Any]
    trace: Trace
    succeeded: bool = True
    error: str | None = None
    corroborated_fields: list[str] = field(default_factory=list)


class SelfCorrectingAgent:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        schema: Schema,
        checks: tuple[ConsistencyCheck, ...] = (),
        max_corrections: int = 2,
        corroborate: bool = True,
    ):
        self.provider = provider
        self.registry = registry
        self.schema = schema
        self.checks = checks
        self.max_corrections = max_corrections
        self.corroborate = corroborate

    # -- public ----------------------------------------------------------

    def run(self, document: dict[str, Any]) -> ExtractionResult:
        doc_id = document.get("document_id", "")
        trace = Trace(document_id=doc_id)
        trace.metadata["provider"] = self.provider.name
        trace.metadata["schema"] = self.schema.name

        self.registry.reset()
        if self.registry.injector is not None:
            self.registry.injector.begin_run(doc_id)

        root = trace.span("extract", SpanKind.AGENT)
        trace.push(root)

        try:
            record, corroborated = self._extract(document, trace)
            root.finish(SpanStatus.OK, fields=len(record))
            trace.pop()
            return ExtractionResult(doc_id, record, trace,
                                    corroborated_fields=corroborated)
        except Exception as exc:
            root.finish(SpanStatus.ERROR, error=repr(exc))
            trace.pop()
            return ExtractionResult(doc_id, {}, trace, succeeded=False,
                                    error=repr(exc))

    # -- stages ----------------------------------------------------------

    def _extract(
        self, document: dict[str, Any], trace: Trace
    ) -> tuple[dict[str, Any], list[str]]:
        evidence, failed_tools = self._gather(document, trace)
        record = self._assemble(evidence, trace)
        # Ordered set: a field re-checked on two correction rounds was still
        # only corroborated once, and counting it twice overstates the cost
        # of the defence the ablation is pricing.
        corroborated: dict[str, None] = {}

        for attempt in range(self.max_corrections + 1):
            problems, culprit_spans = self._diagnose(record, trace)

            if self.corroborate and not problems:
                extra, checked = self._corroborate(record, document, trace)
                corroborated.update(dict.fromkeys(checked))
                if extra:
                    problems.extend(extra[0])
                    culprit_spans.extend(extra[1])

            if not problems:
                break
            if attempt == self.max_corrections:
                break

            record = self._correct(
                record, problems, culprit_spans, evidence, document, trace
            )

        if failed_tools:
            trace.metadata["failed_tools"] = failed_tools
        return record, list(corroborated)

    def _gather(
        self, document: dict[str, Any], trace: Trace
    ) -> tuple[dict[str, Any], list[str]]:
        evidence: dict[str, Any] = {}
        failed: list[str] = []
        for name in ("extract_header", "extract_totals", "count_line_items"):
            try:
                evidence.update(self.registry.call(name, document, trace))
            except ToolFailure as exc:
                failed.append(f"{name}:{exc.fault_key}")
            except Exception as exc:  # pragma: no cover
                failed.append(f"{name}:{type(exc).__name__}")
        return evidence, failed

    def _assemble(self, evidence: dict[str, Any], trace: Trace) -> dict[str, Any]:
        span = trace.span("assemble", SpanKind.MODEL)
        trace.push(span)
        prompt = (
            f"{self.schema.describe()}\n\n"
            f"EVIDENCE: {json.dumps(evidence, default=str)}\n\n"
            "Return the JSON record."
        )
        completion = self.provider.complete(prompt, system=EXTRACT_SYSTEM)
        try:
            record = completion.as_json()
        except json.JSONDecodeError:
            record = {}
            span.finish(
                SpanStatus.ERROR,
                error="model returned unparseable JSON",
                stop_reason=completion.stop_reason,
            )
            trace.pop()
            return record
        span.finish(
            SpanStatus.OK,
            output_tokens=completion.output_tokens,
            stop_reason=completion.stop_reason,
            fields=sorted(record),
        )
        trace.pop()
        return record

    def _diagnose(
        self, record: dict[str, Any], trace: Trace
    ) -> tuple[list[str], list[str]]:
        """Run schema validation and consistency checks.

        Returns (problems, span_ids_of_faulted_calls_this_implicates).
        """
        span = trace.span("validate", SpanKind.VALIDATION)
        trace.push(span)

        problems = self.schema.validate(record)
        for check in self.checks:
            result = check(record)
            if result:
                problems.append(result)

        culprits = self._implicated_spans(trace) if problems else []
        span.finish(
            SpanStatus.ERROR if problems else SpanStatus.OK,
            problem_count=len(problems),
            problems=problems[:5],
        )
        trace.pop()
        return problems, culprits

    def _corroborate(
        self, record: dict[str, Any], document: dict[str, Any], trace: Trace
    ) -> tuple[tuple[list[str], list[str]] | None, list[str]]:
        """Re-derive critical fields from the raw source.

        This is the only defence against plausible substitution. It costs an
        extra tool call per critical field, which is exactly the trade-off
        the benchmark is meant to quantify.
        """
        checked: list[str] = []
        problems: list[str] = []

        critical = [f for f in self.schema.fields
                    if f.criticality.value == "critical" and f.name in record]

        for f in critical:
            try:
                result = self.registry.call(
                    "grep_source", document, trace, field=f.name
                )
            except Exception:
                continue
            checked.append(f.name)
            matches = result.get("__matches__", [])
            if not matches:
                continue
            if not any(f.matches(m, record[f.name]) for m in matches):
                problems.append(
                    f"{f.name}: extracted {record[f.name]!r} not found in source "
                    f"(source shows {matches[0]!r})"
                )

        if not problems:
            return None, checked
        return (problems, self._implicated_spans(trace)), checked

    def _correct(
        self,
        record: dict[str, Any],
        problems: list[str],
        culprit_spans: list[str],
        evidence: dict[str, Any],
        document: dict[str, Any],
        trace: Trace,
    ) -> dict[str, Any]:
        span = trace.span(
            "self_correct", SpanKind.CORRECTION,
            triggered_by=culprit_spans,
            reason=problems[0] if problems else "",
            problem_count=len(problems),
        )
        trace.push(span)

        bullet_list = "\n".join(f"  - {p}" for p in problems)
        prompt = (
            f"{self.schema.describe()}\n\n"
            f"CANDIDATE: {json.dumps(record, default=str)}\n\n"
            f"PROBLEMS FOUND:\n{bullet_list}\n\n"
            f"EVIDENCE: {json.dumps(evidence, default=str)}\n\n"
            "Assess the candidate."
        )
        verdict = self.provider.complete(prompt, system=REVIEW_SYSTEM)
        try:
            parsed = verdict.as_json()
        except json.JSONDecodeError:
            parsed = {"verdict": "reject", "reasons": problems}

        repaired = self._repair(record, problems, evidence, document, trace)
        span.finish(
            SpanStatus.OK,
            verdict=parsed.get("verdict"),
            repaired_fields=sorted(
                k for k in repaired if repaired.get(k) != record.get(k)
            ),
        )
        trace.pop()
        return repaired

    def _repair(
        self,
        record: dict[str, Any],
        problems: list[str],
        evidence: dict[str, Any],
        document: dict[str, Any],
        trace: Trace,
    ) -> dict[str, Any]:
        """Re-fetch implicated fields from source where possible.

        Deliberately conservative: it only repairs what it can re-derive.
        An agent that hallucinates a plausible replacement would score
        better on validation and worse on accuracy, and conflating those is
        the failure mode this whole harness exists to expose.
        
        Note: This is why grep_source is not a hallucination-prone LLM call.
        Simple text matching against the source document guarantees we either
        find the right value or find nothing. No middle ground.
        """
        repaired = dict(record)

        for problem in problems:
            f = self._field_from_problem(problem)
            if f is None:
                continue
            field_name = f.name
            try:
                result = self.registry.call(
                    "grep_source", document, trace, field=field_name
                )
            except Exception:
                continue
            matches = result.get("__matches__", [])
            for candidate in matches:
                try:
                    repaired[field_name] = f.normalize(candidate)
                    break
                except (ValueError, TypeError):
                    continue

        # Restore anything the model dropped that the tools did supply.
        # Scoped to all schema fields, not just required ones: an optional
        # field silently vanishing is exactly the kind of quiet degradation
        # the harness is supposed to surface, not paper over.
        for f in self.schema.fields:
            if repaired.get(f.name) is None and f.name in evidence:
                repaired[f.name] = evidence[f.name]

        return {k: v for k, v in repaired.items() if self.schema.get(k) is not None}

    def _field_from_problem(self, problem: str) -> Field | None:
        """Resolve which field a validation message refers to.

        Messages come in two shapes — "field: reason" and "missing required
        field: name" — so both sides of the colon have to be considered.
        """
        head, _, tail = problem.partition(":")
        for candidate in (head.strip(), tail.strip()):
            f = self.schema.get(candidate)
            if f is not None:
                return f
        # Consistency-check messages name their fields inline.
        for f in self.schema.fields:
            if f.name in problem:
                return f
        return None

    @staticmethod
    def _implicated_spans(trace: Trace) -> list[str]:
        """Tool spans that could plausibly have caused the current problems.

        Attribution is coarse — every prior tool call is a suspect. Precise
        attribution would require ground truth, which the agent must not have.
        
        This conservative approach (blaming all preceding calls) is intentional.
        False negatives in detection are acceptable; false positives would skew
        results by making seemingly-broken calls look recovered when they weren't.
        """
        return [s.span_id for s in trace.tool_spans() if s.status is SpanStatus.OK]
