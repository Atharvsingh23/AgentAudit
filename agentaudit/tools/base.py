"""Tool abstraction and the injection seam.

The injector sits between the agent and the tool, not inside the tool.
That separation is what lets the same tool implementation be used for
clean and faulted runs, so the only difference between arms of an
experiment is the fault plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..faults.injector import FaultContext, FaultInjector, ToolFailure
from ..trace import SpanKind, SpanStatus, Trace


class Tool(ABC):
    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, document: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        ...


class ToolRegistry:
    """Dispatches tool calls, records spans, and applies fault injection."""

    def __init__(self, tools: list[Tool], injector: FaultInjector | None = None):
        self._tools = {t.name: t for t in tools}
        self.injector = injector
        self._call_index = 0

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        return "\n".join(
            f"  - {t.name}: {t.description}" for t in self._tools.values()
        )

    def reset(self) -> None:
        self._call_index = 0

    def call(
        self,
        name: str,
        document: dict[str, Any],
        trace: Trace,
        *,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke a tool, possibly faulted, with bounded retry.

        Retries are the agent's own recovery behaviour for transport
        faults, so they are traced as correction spans rather than hidden.
        """
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}")

        tool = self._tools[name]
        document_id = document.get("document_id", "")
        attempt = 0

        while True:
            self._call_index += 1
            span = trace.span(
                f"{name}", SpanKind.TOOL,
                attempt=attempt, call_index=self._call_index,
            )
            trace.push(span)
            finished = False
            try:
                payload = tool.run(document, **kwargs)
                # What the tool really returned. `apply` corrupts a copy, so
                # this reference stays clean even when a fault lands.
                uncorrupted = payload

                fault = self.injector.decide(name) if self.injector else None
                if fault is not None:
                    try:
                        payload, detail, applied = self.injector.apply(
                            fault, payload, FaultContext(document_id, name),
                        )
                        # A handler that found nothing to corrupt leaves the
                        # span unmarked — see FaultInjector.apply.
                        if applied:
                            span.fault_injected = fault.key
                            span.fault_detail = detail
                    except ToolFailure as failure:
                        span.fault_injected = fault.key
                        span.fault_detail = {
                            "message": str(failure),
                            "retryable": failure.retryable,
                        }
                        span.finish(SpanStatus.ERROR, error=str(failure))
                        finished = True
                        trace.pop()

                        if attempt < max_retries and failure.retryable:
                            recovery = trace.span(
                                f"retry:{name}", SpanKind.CORRECTION,
                                triggered_by=[span.span_id],
                                reason=f"transport fault: {failure.fault_key}",
                                # Marked so the metrics layer can tell a
                                # transport retry apart from noticing a
                                # corrupted value — see Trace.repair_spans.
                                retry=True,
                            )
                            recovery.finish()
                            attempt += 1
                            continue
                        raise

                if self.injector is not None:
                    self.injector.observe(name, document_id, uncorrupted)

                # An external tool can return anything; only a mapping has
                # keys worth recording.
                span.finish(
                    SpanStatus.OK,
                    keys=sorted(payload) if isinstance(payload, dict)
                    else type(payload).__name__,
                )
                finished = True
                trace.pop()
                return payload

            except Exception as exc:
                # ToolFailure raised out of the block above has already been
                # recorded; anything else (including a tool raising one
                # itself) still needs its span closed, or every later span
                # is parented to a call that never ended.
                if not finished:
                    span.finish(SpanStatus.ERROR, error=repr(exc))
                    trace.pop()
                raise
