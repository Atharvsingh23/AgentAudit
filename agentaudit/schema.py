"""Extraction schema primitives.

A schema describes the fields an extraction agent is expected to produce,
along with per-field validation and comparison semantics. Field-level
comparison is what makes metrics meaningful: a pipeline that gets 9 of 10
fields right is not the same as one that fails outright, and averaging over
documents hides that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Callable

# TODO: Consider adding TypedDict for public API so type checkers catch dict unpacking errors.
#       Currently using dataclass for simplicity, but TypedDict would catch deserialization bugs earlier.
#       Decision: stick with dataclass for now because the overhead of TypedDict maintenance
#       isn't justified until we see actual schema deserialization bugs in CI.


class FieldType(str, Enum):
    STRING = "string"
    MONEY = "money"
    DATE = "date"
    INTEGER = "integer"
    ENUM = "enum"


class Criticality(str, Enum):
    """How much a wrong value for this field matters.

    Used to weight metrics. A wrong vendor name is annoying; a wrong total
    is a financial error. Reporting a single unweighted accuracy number
    treats those as equivalent, which is misleading.
    """

    CRITICAL = "critical"
    STANDARD = "standard"
    COSMETIC = "cosmetic"

    @property
    def weight(self) -> float:
        return {"critical": 3.0, "standard": 1.0, "cosmetic": 0.25}[self.value]


@dataclass(frozen=True)
class Field:
    name: str
    type: FieldType
    criticality: Criticality = Criticality.STANDARD
    required: bool = True
    choices: tuple[str, ...] = ()
    description: str = ""

    def normalize(self, value: Any) -> Any:
        """Coerce a raw value into canonical form, or raise ValueError."""
        if value is None:
            raise ValueError("value is None")

        if self.type is FieldType.MONEY:
            cleaned = re.sub(r"[^\d.\-]", "", str(value))
            if not cleaned or cleaned in {"-", ".", "-."}:
                raise ValueError(f"cannot parse money from {value!r}")
            try:
                return Decimal(cleaned).quantize(Decimal("0.01"))
            except InvalidOperation as exc:
                raise ValueError(f"cannot parse money from {value!r}") from exc

        if self.type is FieldType.INTEGER:
            cleaned = re.sub(r"[^\d\-]", "", str(value))
            if not cleaned or cleaned == "-":
                raise ValueError(f"cannot parse integer from {value!r}")
            return int(cleaned)

        if self.type is FieldType.DATE:
            if isinstance(value, datetime):
                return value.date()
            if isinstance(value, date):
                return value
            text = str(value).strip()
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y"):
                try:
                    return datetime.strptime(text, fmt).date()
                except ValueError:
                    continue
            raise ValueError(f"cannot parse date from {value!r}")

        if self.type is FieldType.ENUM:
            text = str(value).strip().lower()
            for choice in self.choices:
                if choice.lower() == text:
                    return choice
            raise ValueError(f"{value!r} not in {self.choices}")

        text = str(value).strip()
        if not text:
            raise ValueError("empty string")
        return " ".join(text.split())

    def matches(self, predicted: Any, truth: Any) -> bool:
        """Exact-match comparison after normalization.

        Returns False rather than raising on unparseable input: an
        unparseable prediction is a wrong prediction, not a crash.
        """
        try:
            norm_pred = self.normalize(predicted)
        except (ValueError, TypeError):
            return False
        try:
            norm_truth = self.normalize(truth)
        except (ValueError, TypeError):
            return False

        if self.type is FieldType.STRING:
            return norm_pred.casefold() == norm_truth.casefold()
        return norm_pred == norm_truth


@dataclass(frozen=True)
class Schema:
    name: str
    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("duplicate field names in schema")

    def get(self, name: str) -> Field | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    @property
    def required_fields(self) -> tuple[Field, ...]:
        return tuple(f for f in self.fields if f.required)

    def describe(self) -> str:
        """Human/LLM-readable schema description for prompting."""
        lines = [f"Schema: {self.name}"]
        for f in self.fields:
            req = "required" if f.required else "optional"
            extra = f" one of {list(f.choices)}" if f.choices else ""
            desc = f" — {f.description}" if f.description else ""
            lines.append(f"  - {f.name} ({f.type.value}, {req}){extra}{desc}")
        return "\n".join(lines)

    def validate(self, record: dict[str, Any]) -> list[str]:
        """Return a list of validation error strings. Empty means valid.

        This is the agent's *own* check — the thing it can run without
        access to ground truth. Whether faults are caught here is precisely
        what the benchmark measures.
        """
        errors: list[str] = []
        for f in self.fields:
            if f.name not in record or record[f.name] is None:
                if f.required:
                    errors.append(f"missing required field: {f.name}")
                continue
            try:
                f.normalize(record[f.name])
            except (ValueError, TypeError) as exc:
                errors.append(f"{f.name}: {exc}")
        unknown = set(record) - {f.name for f in self.fields}
        for key in sorted(unknown):
            errors.append(f"unexpected field: {key}")
        return errors


# Cross-field consistency checks. These catch semantic faults that pass
# type validation — the interesting case, because structural validation is
# easy and semantic validation is where real pipelines fail silently.
ConsistencyCheck = Callable[[dict[str, Any]], str | None]


def totals_add_up(record: dict[str, Any]) -> str | None:
    try:
        subtotal = Decimal(re.sub(r"[^\d.\-]", "", str(record["subtotal"])))
        tax = Decimal(re.sub(r"[^\d.\-]", "", str(record["tax"])))
        total = Decimal(re.sub(r"[^\d.\-]", "", str(record["total"])))
    except (KeyError, TypeError, InvalidOperation):
        return None
    if abs((subtotal + tax) - total) > Decimal("0.02"):
        return f"subtotal ({subtotal}) + tax ({tax}) != total ({total})"
    return None


def due_after_issue(record: dict[str, Any]) -> str | None:
    issue = record.get("issue_date")
    due = record.get("due_date")
    if not issue or not due:
        return None
    field_ = Field("d", FieldType.DATE)
    try:
        if field_.normalize(due) < field_.normalize(issue):
            return f"due_date ({due}) precedes issue_date ({issue})"
    except (ValueError, TypeError):
        return None
    return None


INVOICE_SCHEMA = Schema(
    name="invoice",
    fields=(
        Field("invoice_number", FieldType.STRING, Criticality.CRITICAL,
              description="vendor's invoice identifier"),
        Field("vendor_name", FieldType.STRING, Criticality.STANDARD),
        Field("issue_date", FieldType.DATE, Criticality.STANDARD),
        Field("due_date", FieldType.DATE, Criticality.STANDARD, required=False),
        Field("subtotal", FieldType.MONEY, Criticality.CRITICAL),
        Field("tax", FieldType.MONEY, Criticality.CRITICAL),
        Field("total", FieldType.MONEY, Criticality.CRITICAL),
        Field("currency", FieldType.ENUM, Criticality.CRITICAL,
              choices=("USD", "EUR", "GBP", "INR")),
        Field("line_item_count", FieldType.INTEGER, Criticality.COSMETIC,
              required=False),
    ),
)

INVOICE_CHECKS: tuple[ConsistencyCheck, ...] = (totals_add_up, due_after_issue)
