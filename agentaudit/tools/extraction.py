"""Concrete tools for the invoice extraction task.

These are intentionally simple and deterministic. The benchmark measures
how an agent behaves when its tools misbehave; if the tools were also
noisy on their own, fault attribution would be impossible.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Tool


class HeaderExtractor(Tool):
    name = "extract_header"
    description = "Pull invoice number, vendor and dates from the header block."

    FIELDS = ("invoice_number", "vendor_name", "issue_date", "due_date")

    def run(self, document: dict[str, Any], **_: Any) -> dict[str, Any]:
        source = document.get("fields", {})
        return {k: source[k] for k in self.FIELDS if k in source}


class TotalsExtractor(Tool):
    name = "extract_totals"
    description = "Pull subtotal, tax, total and currency from the totals block."

    FIELDS = ("subtotal", "tax", "total", "currency")

    def run(self, document: dict[str, Any], **_: Any) -> dict[str, Any]:
        source = document.get("fields", {})
        return {k: source[k] for k in self.FIELDS if k in source}


class LineItemCounter(Tool):
    name = "count_line_items"
    description = "Count the rows in the line-item table."

    def run(self, document: dict[str, Any], **_: Any) -> dict[str, Any]:
        source = document.get("fields", {})
        if "line_item_count" in source:
            return {"line_item_count": source["line_item_count"]}
        return {}


class SourceGrep(Tool):
    """Re-read the raw document text for a specific field.

    This is the only tool that can catch a corroborative fault, because it
    is the only one that goes back to the source rather than to another
    derived view. An agent that never calls it cannot detect plausible
    substitution, no matter how carefully it reasons.
    """

    name = "grep_source"
    description = (
        "Search the raw document text for a field's value. Use to "
        "independently confirm a suspicious extraction."
    )

    def run(self, document: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        field_name = kwargs.get("field")
        text = document.get("raw_text", "")
        if not field_name:
            return {"__matches__": []}
        pattern = field_name.replace("_", r"[\s_]*")
        hits = re.findall(
            rf"{pattern}\s*[:\-]?\s*([^\n]{{1,60}})", text, re.IGNORECASE
        )
        return {"__matches__": [h.strip() for h in hits[:3]], "__field__": field_name}


DEFAULT_TOOLS = [
    HeaderExtractor(),
    TotalsExtractor(),
    LineItemCounter(),
    SourceGrep(),
]
