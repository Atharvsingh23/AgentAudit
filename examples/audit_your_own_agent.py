"""Audit a pipeline AgentAudit did not write.

Run:  python examples/audit_your_own_agent.py

The agent below is deliberately ordinary: it calls two tools and trusts
what they return, which is what most production extraction pipelines do.
Nothing in it knows the harness exists.

Swap `fetch_header` / `fetch_totals` for your real functions, `my_agent`
for your real control flow, and the documents for a sample of your own
work with the answers you expect. That is the whole integration.
"""

from __future__ import annotations

from agentaudit import Schema, audit_agent, quick_audit

# --- stand-in for your data source ---------------------------------------

INVOICES = {
    f"INV-{1000 + i}": {
        "vendor": f"Vendor {i % 7}",
        "subtotal": f"{1000 + i * 13}.00",
        "tax": f"{100 + i}.00",
        "total": f"{1100 + i * 13 + i}.00",
    }
    for i in range(30)
}


# --- your tools ----------------------------------------------------------

def fetch_header(invoice_id: str) -> dict:
    """Whatever you already have: an OCR call, a vendor lookup, an API."""
    return {"vendor": INVOICES[invoice_id]["vendor"]}


def fetch_totals(invoice_id: str) -> dict:
    """Same — a real call in your pipeline, unchanged."""
    row = INVOICES[invoice_id]
    return {"subtotal": row["subtotal"], "tax": row["tax"], "total": row["total"]}


# --- your agent ----------------------------------------------------------

def my_agent(document: dict, tools: dict) -> dict:
    """Your control flow. The harness never touches it.

    `tools` holds the same callables you passed in, wrapped so the audit
    can corrupt what they return. Outside an audit they are pass-throughs,
    so this function is production code, not test code.
    """
    header = tools["fetch_header"](document["invoice_id"])
    totals = tools["fetch_totals"](document["invoice_id"])
    return {**header, **totals}


def main() -> None:
    documents = [
        {"document_id": key, "invoice_id": key, "truth": dict(fields)}
        for key, fields in INVOICES.items()
    ]
    schema = Schema.from_spec({
        "vendor": "string",
        "subtotal": {"type": "money", "criticality": "critical"},
        "tax": {"type": "money", "criticality": "critical"},
        "total": {"type": "money", "criticality": "critical"},
    })
    tools = {"fetch_header": fetch_header, "fetch_totals": fetch_totals}

    print("=" * 74)
    print("THE ONE-MINUTE VERSION: does a swapped-in value get noticed?")
    print("=" * 74)
    print(quick_audit(my_agent, tools, documents, schema).summary())
    print(
        "\n`plausible_substitution` replaces a value with a real value from a\n"
        "different invoice. Nothing is malformed, nothing is internally\n"
        "inconsistent — the totals still add up. Only re-reading the source\n"
        "would catch it, and this pipeline never does.\n"
    )

    print("=" * 74)
    print("THE FULL AUDIT: every fault layer")
    print("=" * 74)
    result = audit_agent(my_agent, tools, documents, schema)
    print(result.summary())
    print(
        "\nRead the `silent` column: the share of faulted runs that emitted a\n"
        "wrong record with nothing flagged. That is the number that does not\n"
        "show up in an accuracy report, and the reason this pipeline would\n"
        "pay a wrong invoice without anyone noticing."
    )


if __name__ == "__main__":
    main()
