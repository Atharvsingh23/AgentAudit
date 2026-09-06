"""Execution tracing.

Every tool call, model call, validation pass and correction attempt is
recorded as a span. Traces are the raw material for every metric in the
harness, and they are what make a failed run diagnosable rather than just
a number in a table.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class SpanKind(str, Enum):
    AGENT = "agent"
    MODEL = "model"
    TOOL = "tool"
    VALIDATION = "validation"
    CORRECTION = "correction"
    JUDGE = "judge"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    name: str
    kind: SpanKind
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = field(default_factory=dict)

    # Fault bookkeeping. Set by the injector, never by the agent — the
    # agent must not be able to see these, or the benchmark is invalid.
    # This isolation is what makes silent failure detection meaningful.
    fault_injected: str | None = None
    fault_detail: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            return 0.0
        return (self.end - self.start) * 1000

    def finish(self, status: SpanStatus = SpanStatus.OK, **attrs: Any) -> None:
        self.end = time.perf_counter()
        self.status = status
        self.attributes.update(attrs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        d["duration_ms"] = round(self.duration_ms, 3)
        d.pop("start", None)
        d.pop("end", None)
        return d


@dataclass
class Trace:
    """A single agent run over a single document."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    document_id: str = ""
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _stack: list[str] = field(default_factory=list, repr=False)

    def span(self, name: str, kind: SpanKind, **attrs: Any) -> Span:
        s = Span(
            name=name,
            kind=kind,
            parent_id=self._stack[-1] if self._stack else None,
            attributes=dict(attrs),
        )
        self.spans.append(s)
        return s

    def push(self, span: Span) -> None:
        self._stack.append(span.span_id)

    def pop(self) -> None:
        if self._stack:
            self._stack.pop()

    # -- derived views used by the metrics layer -------------------------

    def tool_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind is SpanKind.TOOL]

    def faulted_spans(self) -> list[Span]:
        return [s for s in self.spans if s.fault_injected is not None]

    def correction_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind is SpanKind.CORRECTION]

    @property
    def correction_attempts(self) -> int:
        return len(self.correction_spans())

    @property
    def faults_injected(self) -> int:
        return len(self.faulted_spans())

    def detected_fault_ids(self) -> set[str]:
        """Span ids of faulted tool calls the agent explicitly reacted to.

        A fault counts as detected if a later correction span names the
        faulted span. This is deliberately strict: an agent that produces a
        correct answer without ever noticing the fault gets credit for
        recovery but not for detection, and that distinction matters.
        """
        detected: set[str] = set()
        for span in self.correction_spans():
            for sid in span.attributes.get("triggered_by", []):
                detected.add(sid)
        return detected

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans if s.kind is SpanKind.AGENT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "document_id": self.document_id,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render(self) -> str:
        """Compact human-readable tree, for debugging a single run."""
        by_parent: dict[str | None, list[Span]] = {}
        for s in self.spans:
            by_parent.setdefault(s.parent_id, []).append(s)

        lines: list[str] = [f"trace {self.run_id}  doc={self.document_id}"]

        def walk(parent: str | None, depth: int) -> None:
            for s in by_parent.get(parent, []):
                mark = "!" if s.fault_injected else " "
                flag = "ERR" if s.status is SpanStatus.ERROR else "ok "
                lines.append(
                    f"{mark} {'  ' * depth}{s.kind.value:<10} {s.name:<28} "
                    f"{flag} {s.duration_ms:7.1f}ms"
                    + (f"  <fault:{s.fault_injected}>" if s.fault_injected else "")
                )
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)
