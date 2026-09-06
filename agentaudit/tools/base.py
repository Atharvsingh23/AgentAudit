"""Tool abstraction and the injection seam.

The injector sits between the agent and the tool, not inside the tool.
That separation is what lets the same tool implementation be used for
clean and faulted runs, so the only difference between arms of an
experiment is the fault plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..faults.injector import FaultInjector, ToolFailure
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
        attempt = 0

        while True:
            self._call_index += 1
            span = trace.span(
                f"{name}", SpanKind.TOOL,
                attempt=attempt, call_index=self._call_index,
            )
            trace.push(span)
            try:
                payload = tool.run(document, **kwargs)

                fault = (
                    self.injector.decide(name, self._call_index)
                    if self.injector else None
                )
                if fault is not None:
                    span.fault_injected = fault.key
                    try:
                        payload, detail = self.injector.apply(
                            fault, payload, document.get("document_id", "")
                        )
                        span.fault_detail = detail
                    except ToolFailure as failure:
                        span.fault_detail = {
                            "message": str(failure),
                            "retryable": failure.retryable,
                        }
                        span.finish(SpanStatus.ERROR, error=str(failure))
                        trace.pop()

                        if attempt < max_retries and failure.retryable:
                            recovery = trace.span(
                                f"retry:{name}", SpanKind.CORRECTION,
                                triggered_by=[span.span_id],
                                reason=f"transport fault: {failure.fault_key}",
                            )
                            recovery.finish()
                            attempt += 1
                            continue
                        raise

                span.finish(SpanStatus.OK, keys=sorted(payload))
                trace.pop()
                return payload

            except ToolFailure:
                raise
            except Exception as exc:
                span.finish(SpanStatus.ERROR, error=repr(exc))
                trace.pop()
                raise
