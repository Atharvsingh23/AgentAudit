"""Synthetic invoice corpus.

Synthetic rather than scraped, for three reasons: it ships with the repo
so results are reproducible by anyone, it carries no licensing or PII
risk, and ground truth is exact by construction rather than by
annotation. The cost is realism — these documents are cleaner than real
scans, which means absolute accuracy numbers here are optimistic. The
comparisons between fault conditions, which is what the benchmark is for,
remain valid.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

VENDORS = [
    "Northwind Traders", "Acme Industrial Supply", "Blue Harbor Logistics",
    "Cedar Point Systems", "Vertex Materials", "Lakeside Print Works",
    "Ironwood Consulting", "Meridian Freight", "Copperline Electric",
    "Sunfield Agritech",
]
CURRENCIES = ["USD", "EUR", "GBP", "INR"]

TEMPLATE = """\
{vendor}
INVOICE

Invoice Number: {invoice_number}
Issue Date: {issue_date}
Due Date: {due_date}

Line Item Count: {line_item_count}

Subtotal: {currency} {subtotal}
Tax: {currency} {tax}
Total: {currency} {total}
Currency: {currency}
Vendor Name: {vendor}
"""


def build_corpus(n: int = 40, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    documents: list[dict[str, Any]] = []

    for i in range(n):
        vendor = rng.choice(VENDORS)
        issue = date(2025, 1, 1) + timedelta(days=rng.randint(0, 540))
        due = issue + timedelta(days=rng.choice([15, 30, 45, 60]))
        subtotal = Decimal(rng.randint(15_000, 900_000)) / 100
        tax_rate = Decimal(rng.choice(["0.05", "0.08", "0.12", "0.18", "0.20"]))
        tax = (subtotal * tax_rate).quantize(Decimal("0.01"))
        total = (subtotal + tax).quantize(Decimal("0.01"))

        fields = {
            "invoice_number": f"INV-{2025 + i % 2}-{rng.randint(10000, 99999)}",
            "vendor_name": vendor,
            "issue_date": issue.isoformat(),
            "due_date": due.isoformat(),
            "subtotal": str(subtotal),
            "tax": str(tax),
            "total": str(total),
            "currency": rng.choice(CURRENCIES),
            "line_item_count": rng.randint(1, 24),
        }

        raw_text = TEMPLATE.format(vendor=vendor, **fields)

        documents.append(
            {
                "document_id": f"doc-{i:03d}",
                "raw_text": raw_text,
                "fields": fields,
                # Ground truth is a separate copy: the agent reads `fields`
                # through tools that the injector may corrupt, and is scored
                # against this, which nothing touches.
                "truth": dict(fields),
            }
        )

    return documents


def corpus_for_injector(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten to the shape the injector needs for donor-based faults."""
    return [{"document_id": d["document_id"], **d["fields"]} for d in documents]
