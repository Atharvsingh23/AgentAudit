"""Metrics.

The headline number in most agent evaluations is end-to-end accuracy. It
is the least informative number available, because it collapses four
distinct outcomes:

    fault injected, caught, recovered      -> resilient
    fault injected, caught, not recovered  -> degraded but honest
    fault injected, not caught, correct    -> lucky
    fault injected, not caught, wrong      -> SILENT FAILURE

Only the fourth is dangerous in production, and it is invisible to
accuracy-only reporting whenever the fault rate is low. Silent failure
rate is the metric this harness exists to produce.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .agent import ExtractionResult
from .faults.taxonomy import TAXONOMY
from .schema import Schema


@dataclass
class FieldScore:
    name: str
    correct: bool
    predicted: Any
    expected: Any
    weight: float


@dataclass
class RunScore:
    document_id: str
    fields: list[FieldScore]
    faults_injected: int
    faults_detected: int
    corrections: int
    succeeded: bool
    corroborated: int = 0

    @property
    def exact_match(self) -> bool:
        return bool(self.fields) and all(f.correct for f in self.fields)

    @property
    def field_accuracy(self) -> float:
        if not self.fields:
            return 0.0
        return sum(1 for f in self.fields if f.correct) / len(self.fields)

    @property
    def weighted_accuracy(self) -> float:
        total = sum(f.weight for f in self.fields)
        if total == 0:
            return 0.0
        return sum(f.weight for f in self.fields if f.correct) / total

    @property
    def silent_failure(self) -> bool:
        """Fault landed, agent never flagged it, output is wrong.
        
        This is the central metric the harness exists to measure. In production,
        this is the only outcome that matters — accuracy metrics hide this case,
        especially when fault rates are low. A system with 99% accuracy and 20%
        silent failure rate is worse than one with 95% accuracy and 5% silent
        failure. This property isolates that fourth outcome precisely.
        """
        return (
            self.faults_injected > 0
            and self.faults_detected == 0
            and not self.exact_match
        )

    @property
    def lucky(self) -> bool:
        return (
            self.faults_injected > 0
            and self.faults_detected == 0
            and self.exact_match
        )


def score_run(
    result: ExtractionResult,
    truth: dict[str, Any],
    schema: Schema,
    rescued: set[str] | None = None,
) -> RunScore:
    """Score one run against ground truth.

    ``rescued`` names fields an LLM judge accepted as equivalent despite
    exact match failing. They are credited here rather than by writing the
    reference value into ``result.record`` — ground truth must never end up
    inside the agent's own output, where the next stage would read it back.
    """
    rescued = rescued or set()
    scores: list[FieldScore] = []
    for f in schema.fields:
        if f.name not in truth:
            continue
        predicted = result.record.get(f.name)
        scores.append(
            FieldScore(
                name=f.name,
                correct=f.name in rescued or f.matches(predicted, truth[f.name]),
                predicted=predicted,
                expected=truth[f.name],
                weight=f.criticality.weight,
            )
        )

    trace = result.trace
    faulted = trace.faulted_spans()
    # Only value-corrupting faults count toward detection accounting;
    # a retried timeout that the agent transparently recovered from is not
    # an unnoticed corruption.
    corrupting = [
        s for s in faulted
        if s.fault_injected in TAXONOMY and TAXONOMY[s.fault_injected].corrupts_value
    ]
    detected_ids = trace.detected_fault_ids()
    detected = sum(1 for s in corrupting if s.span_id in detected_ids)

    # A correction triggered while a corrupting fault is live counts as
    # detection even if span attribution is coarse. Retries are excluded:
    # a retried timeout says nothing about whether the agent noticed a
    # transposed digit somewhere else in the same run, and counting it as
    # detection credits the agent for a fault it never saw.
    if corrupting and not detected and trace.repair_attempts > 0:
        detected = min(len(corrupting), trace.repair_attempts)

    return RunScore(
        document_id=result.document_id,
        fields=scores,
        faults_injected=len(corrupting),
        faults_detected=detected,
        corrections=trace.correction_attempts,
        succeeded=result.succeeded,
        corroborated=len(result.corroborated_fields),
    )


@dataclass
class Report:
    condition: str
    runs: list[RunScore] = field(default_factory=list)

    # -- aggregate --------------------------------------------------------

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def exact_match_rate(self) -> float:
        return self._mean([1.0 if r.exact_match else 0.0 for r in self.runs])

    @property
    def field_accuracy(self) -> float:
        return self._mean([r.field_accuracy for r in self.runs])

    @property
    def weighted_accuracy(self) -> float:
        return self._mean([r.weighted_accuracy for r in self.runs])

    @property
    def completion_rate(self) -> float:
        return self._mean([1.0 if r.succeeded else 0.0 for r in self.runs])

    @property
    def faulted_runs(self) -> list[RunScore]:
        return [r for r in self.runs if r.faults_injected > 0]

    @property
    def detection_rate(self) -> float:
        faulted = self.faulted_runs
        if not faulted:
            return float("nan")
        total = sum(r.faults_injected for r in faulted)
        found = sum(r.faults_detected for r in faulted)
        return found / total if total else float("nan")

    @property
    def recovery_rate(self) -> float:
        """Of faulted runs, the share that still produced a correct record."""
        faulted = self.faulted_runs
        if not faulted:
            return float("nan")
        return self._mean([1.0 if r.exact_match else 0.0 for r in faulted])

    @property
    def silent_failure_rate(self) -> float:
        faulted = self.faulted_runs
        if not faulted:
            return float("nan")
        return self._mean([1.0 if r.silent_failure else 0.0 for r in faulted])

    @property
    def mean_corrections(self) -> float:
        return self._mean([float(r.corrections) for r in self.runs])

    @property
    def correction_overhead(self) -> float:
        """Corrections spent per faulted run — the cost side of resilience."""
        faulted = self.faulted_runs
        if not faulted:
            return 0.0
        return self._mean([float(r.corrections) for r in faulted])

    def per_field(self) -> dict[str, float]:
        totals: dict[str, list[float]] = defaultdict(list)
        for run in self.runs:
            for f in run.fields:
                totals[f.name].append(1.0 if f.correct else 0.0)
        return {k: sum(v) / len(v) for k, v in sorted(totals.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "n": self.n,
            "completion_rate": round(self.completion_rate, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "field_accuracy": round(self.field_accuracy, 4),
            "weighted_accuracy": round(self.weighted_accuracy, 4),
            "detection_rate": _r(self.detection_rate),
            "recovery_rate": _r(self.recovery_rate),
            "silent_failure_rate": _r(self.silent_failure_rate),
            "correction_overhead": round(self.correction_overhead, 3),
            "per_field_accuracy": {k: round(v, 4) for k, v in self.per_field().items()},
        }


def _r(value: float) -> float | None:
    return None if value != value else round(value, 4)


def format_table(reports: list[Report]) -> str:
    """Fixed-width comparison table across conditions."""
    headers = ["condition", "n", "exact", "field", "detect", "recover",
               "silent", "corr"]
    widths = [26, 5, 7, 7, 7, 8, 7, 6]

    def cell(value: Any, width: int) -> str:
        if value is None or (isinstance(value, float) and value != value):
            text = "—"
        elif isinstance(value, float):
            text = f"{value:.3f}"
        else:
            text = str(value)
        if len(text) >= width:
            # Truncate rather than overflow: a long condition name pushing
            # a row out of alignment makes the whole table unreadable.
            text = text[: width - 2] + "…"
        return text.ljust(width)

    lines = ["".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))]
    lines.append("-" * sum(widths))
    for rep in reports:
        row = [
            rep.condition, rep.n, rep.exact_match_rate, rep.field_accuracy,
            _r(rep.detection_rate), _r(rep.recovery_rate),
            _r(rep.silent_failure_rate), round(rep.correction_overhead, 2),
        ]
        lines.append("".join(cell(v, w) for v, w in zip(row, widths, strict=True)))
    return "\n".join(lines)
