"""Tests for auditing an agent this library did not write.

The subject here is a stand-in for a customer pipeline: it has its own
tools, its own control flow, and no knowledge of agentaudit. Everything
these tests assert is about not flattering it — not retrying on its
behalf, not hiding faults that could not be applied, not letting it see
what was done to it.
"""

from __future__ import annotations

import pytest

from agentaudit import (
    Condition, InjectionPlan, Schema, TAXONOMY, audit_agent, quick_audit,
    wrap_tools,
)
from agentaudit.schema import FieldType

DB = {
    f"inv-{i}": {
        "vendor": f"Vendor {i}",
        "subtotal": f"{100 + i}.00",
        "tax": f"{10 + i}.00",
        "total": f"{110 + 2 * i}.00",
    }
    for i in range(12)
}

SCHEMA = Schema.from_spec({
    "vendor": "string",
    "subtotal": {"type": "money", "criticality": "critical"},
    "tax": {"type": "money", "criticality": "critical"},
    "total": {"type": "money", "criticality": "critical"},
})


def fetch_header(invoice_id):
    """Return the vendor block."""
    return {"vendor": DB[invoice_id]["vendor"]}


def fetch_totals(invoice_id):
    """Return the money block."""
    row = DB[invoice_id]
    return {"subtotal": row["subtotal"], "tax": row["tax"], "total": row["total"]}


TOOLS = {"fetch_header": fetch_header, "fetch_totals": fetch_totals}
DOCUMENTS = [
    {"document_id": k, "invoice_id": k, "truth": dict(v)} for k, v in DB.items()
]


def trusting_agent(document, tools):
    """What most production pipelines actually do: believe the tools."""
    return {
        **tools["fetch_header"](document["invoice_id"]),
        **tools["fetch_totals"](document["invoice_id"]),
    }


# -- schemas from plain data ----------------------------------------------

def test_schema_from_spec_accepts_shorthand_and_full_form():
    schema = Schema.from_spec({
        "a": "string",
        "b": {"type": "money", "criticality": "critical", "required": False},
    })
    assert schema.get("a").type is FieldType.STRING
    assert schema.get("b").criticality.weight > schema.get("a").criticality.weight
    assert not schema.get("b").required


def test_schema_from_spec_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="type must be one of"):
        Schema.from_spec({"a": "flibble"})


# -- the audit itself -----------------------------------------------------

def test_clean_arm_is_perfect_and_faults_make_it_worse():
    result = quick_audit(trusting_agent, TOOLS, DOCUMENTS, SCHEMA)
    clean, faulted = result.reports
    assert clean.exact_match_rate == 1.0, "a clean arm that isn't perfect is a bug"
    assert faulted.exact_match_rate < clean.exact_match_rate


def test_an_agent_that_trusts_its_tools_fails_silently():
    """The finding the harness exists to produce, on a foreign agent."""
    result = quick_audit(trusting_agent, TOOLS, DOCUMENTS, SCHEMA)
    faulted = result.reports[1]
    assert faulted.silent_failure_rate == 1.0
    assert faulted.detection_rate == 0.0


def test_wrapped_tools_are_transparent_outside_an_audit():
    """The same wiring has to serve production, or nobody will adopt it."""
    wrapped = wrap_tools(TOOLS)
    assert wrapped["fetch_totals"]("inv-3") == fetch_totals("inv-3")
    assert wrapped["fetch_header"].__name__ == "fetch_header"


def test_transport_faults_are_raised_into_the_agent_not_retried_for_it():
    """Retrying on the agent's behalf would measure our error handling.

    A pipeline with no retry logic must show up as a pipeline with no
    retry logic.
    """
    condition = [Condition("transport", InjectionPlan(
        rate=1.0, faults=("timeout", "rate_limit", "transient_error"),
        seed=0, max_per_run=1))]
    result = audit_agent(trusting_agent, TOOLS, DOCUMENTS, SCHEMA,
                         conditions=condition)
    assert result.agent_errors["transport"], "the agent should have seen the failure"
    assert result.reports[0].completion_rate < 1.0

    forgiving = audit_agent(trusting_agent, TOOLS, DOCUMENTS, SCHEMA,
                            conditions=condition, retry_transport_faults=True)
    assert not forgiving.agent_errors["transport"]
    assert forgiving.reports[0].completion_rate == 1.0


def test_an_agent_that_raises_is_recorded_rather_than_stopping_the_run():
    def brittle(document, tools):
        tools["fetch_header"](document["invoice_id"])
        raise RuntimeError("kaboom")

    result = quick_audit(brittle, TOOLS, DOCUMENTS, SCHEMA)
    assert len(result.reports) == 2
    assert all(len(errors) == len(DOCUMENTS)
               for errors in result.agent_errors.values())
    assert result.reports[0].completion_rate == 0.0


def test_a_tool_that_returns_a_non_mapping_is_flagged_not_quietly_skipped():
    """Otherwise the report reads as a clean bill of health.

    Value-corrupting faults need named fields to rewrite. A tool that
    returns a string can only be failed at the transport layer, and a
    reader has to be told that before trusting its silent-failure number.
    """
    def fetch_total_as_text(invoice_id):
        return DB[invoice_id]["total"]

    def agent(document, tools):
        return {"total": tools["fetch_total_as_text"](document["invoice_id"])}

    result = quick_audit(
        agent, {"fetch_total_as_text": fetch_total_as_text}, DOCUMENTS,
        Schema.from_spec({"total": {"type": "money", "criticality": "critical"}}),
    )
    assert result.non_dict_tools == {"fetch_total_as_text"}
    assert "not meaningful" in result.summary()


def test_the_audited_agent_never_sees_what_was_done_to_it():
    """Same guarantee as the built-in agent, and for the same reason."""
    seen: list[str] = []

    def nosy(document, tools):
        seen.append(repr(document))
        payload = {**tools["fetch_header"](document["invoice_id"]),
                   **tools["fetch_totals"](document["invoice_id"])}
        seen.append(repr(payload))
        return payload

    audit_agent(nosy, TOOLS, DOCUMENTS, SCHEMA)
    blob = "\n".join(seen)
    for key in TAXONOMY:
        assert key not in blob, f"fault label {key!r} leaked to the agent"
    assert "fault_injected" not in blob


# -- input validation -----------------------------------------------------

def test_documents_must_carry_ground_truth():
    with pytest.raises(ValueError, match="truth"):
        quick_audit(trusting_agent, TOOLS, [{"document_id": "a"}], SCHEMA)


def test_document_ids_must_be_present_and_unique():
    with pytest.raises(ValueError, match="document_id"):
        quick_audit(trusting_agent, TOOLS, [{"truth": {}}], SCHEMA)
    with pytest.raises(ValueError, match="duplicate"):
        quick_audit(trusting_agent, TOOLS,
                    [{"document_id": "a", "truth": {}},
                     {"document_id": "a", "truth": {}}], SCHEMA)


def test_empty_document_set_is_rejected():
    with pytest.raises(ValueError, match="no documents"):
        quick_audit(trusting_agent, TOOLS, [], SCHEMA)
