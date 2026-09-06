"""LLM-as-judge layer.

Used only for STRING fields, and only where exact match already failed.
Judges are expensive and noisy; applying one to a money field that a
Decimal comparison settles exactly would add cost and variance for no
information. The judge exists to stop "Acme Corp." vs "Acme Corporation"
from being scored as an extraction failure.

Every judgement is traced. An unauditable judge is just a second
unverified model in the loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .providers.base import Provider
from .schema import FieldType, Schema
from .trace import SpanKind, Trace

JUDGE_SYSTEM = (
    "JUDGE. Decide whether two extracted values refer to the same real-world "
    'entity. Return only JSON: {"equivalent": true|false}. '
    "Formatting, abbreviation and legal-suffix differences are equivalent. "
    "Different entities, amounts or dates are not."
)


@dataclass
class Judgement:
    field: str
    predicted: Any
    expected: Any
    equivalent: bool


class SemanticJudge:
    def __init__(self, provider: Provider, schema: Schema):
        self.provider = provider
        self.schema = schema
        self.judgements: list[Judgement] = []

    def adjudicate(
        self,
        record: dict[str, Any],
        truth: dict[str, Any],
        trace: Trace,
    ) -> set[str]:
        """Return field names that exact-match rejected but the judge accepts."""
        rescued: set[str] = set()

        for f in self.schema.fields:
            if f.type is not FieldType.STRING:
                continue
            if f.name not in truth or f.name not in record:
                continue
            if f.matches(record[f.name], truth[f.name]):
                continue

            span = trace.span(f"judge:{f.name}", SpanKind.JUDGE)
            trace.push(span)
            prompt = (
                f"FIELD: {f.name}\n"
                f"PREDICTED: {record[f.name]}\n"
                f"REFERENCE: {truth[f.name]}\n"
            )
            completion = self.provider.complete(prompt, system=JUDGE_SYSTEM)
            try:
                equivalent = bool(completion.as_json().get("equivalent", False))
            except json.JSONDecodeError:
                equivalent = False

            self.judgements.append(
                Judgement(f.name, record[f.name], truth[f.name], equivalent)
            )
            if equivalent:
                rescued.add(f.name)
            span.finish(equivalent=equivalent)
            trace.pop()

        return rescued
