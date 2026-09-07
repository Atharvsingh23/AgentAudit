"""Test suite.

The determinism tests are the load-bearing ones. Every claim the harness
makes rests on the assumption that a seed pins the experiment; if that
breaks, nothing else is meaningful.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentaudit import (
    INVOICE_CHECKS, INVOICE_SCHEMA, Condition, Criticality, DEFAULT_TOOLS,
    Experiment, FaultInjector, InjectionPlan, MockProvider, Report,
    SelfCorrectingAgent, ToolRegistry, build_corpus, corpus_for_injector,
    score_run,
)
from agentaudit.faults.taxonomy import TAXONOMY, Detectability, FaultLayer
from agentaudit.cli import DONOR_POOL, _documents, build_parser
from agentaudit.schema import Field, FieldType, totals_add_up


# -- schema ---------------------------------------------------------------

def test_money_normalization_handles_symbols_and_separators():
    f = Field("total", FieldType.MONEY)
    assert f.normalize("$1,234.50") == Decimal("1234.50")
    assert f.normalize("1234.5") == Decimal("1234.50")


def test_money_rejects_unparseable():
    f = Field("total", FieldType.MONEY)
    with pytest.raises(ValueError):
        f.normalize("approximately a lot")


def test_string_match_is_case_and_space_insensitive():
    f = Field("vendor_name", FieldType.STRING)
    assert f.matches("  ACME   Corp ", "acme corp")


def test_matches_returns_false_rather_than_raising():
    f = Field("total", FieldType.MONEY)
    assert f.matches(None, "10.00") is False
    assert f.matches("garbage", "10.00") is False


def test_date_parses_multiple_formats():
    f = Field("d", FieldType.DATE)
    assert f.normalize("2025-03-04") == f.normalize("04/03/2025")


def test_validate_flags_missing_required_and_unknown_fields():
    errors = INVOICE_SCHEMA.validate({"invoice_number": "X", "surprise": 1})
    assert any("missing required field: total" in e for e in errors)
    assert any("unexpected field: surprise" in e for e in errors)


def test_consistency_check_catches_broken_totals():
    assert totals_add_up({"subtotal": "100.00", "tax": "10.00", "total": "110.00"}) is None
    assert totals_add_up({"subtotal": "100.00", "tax": "10.00", "total": "999.00"})


def test_criticality_weights_are_ordered():
    assert (Criticality.CRITICAL.weight > Criticality.STANDARD.weight
            > Criticality.COSMETIC.weight)


# -- taxonomy -------------------------------------------------------------

def test_every_fault_has_a_handler():
    injector = FaultInjector(InjectionPlan())
    for key, spec in TAXONOMY.items():
        if spec.layer is FaultLayer.TRANSPORT:
            continue
        assert hasattr(injector, f"_apply_{key}"), f"no handler for {key}"


def test_transport_faults_are_non_corrupting():
    for spec in TAXONOMY.values():
        if spec.layer is FaultLayer.TRANSPORT:
            assert not spec.corrupts_value


def test_plausible_substitution_is_not_mechanically_detectable():
    """If this ever becomes MECHANICAL the benchmark has lost its point."""
    assert TAXONOMY["plausible_substitution"].detectability is Detectability.CORROBORATIVE


# -- determinism ----------------------------------------------------------

def _run_once(seed: int, n: int = 12):
    docs = build_corpus(n=n, seed=7)
    plan = InjectionPlan(rate=0.8, faults=tuple(TAXONOMY), seed=seed, max_per_run=1)
    injector = FaultInjector(plan, corpus=corpus_for_injector(docs))
    registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    out = []
    for doc in docs:
        result = agent.run(doc)
        out.append((
            result.document_id,
            sorted(s.fault_injected or "" for s in result.trace.faulted_spans()),
            score_run(result, doc["truth"], INVOICE_SCHEMA).exact_match,
        ))
    return out


def test_same_seed_produces_identical_faults_and_outcomes():
    assert _run_once(seed=1) == _run_once(seed=1)


def test_different_seeds_produce_different_faults():
    assert _run_once(seed=1) != _run_once(seed=2)


def test_injection_plan_fingerprint_is_order_stable():
    a = InjectionPlan(rate=0.5, faults=("timeout", "schema_drift"), seed=3)
    b = InjectionPlan(rate=0.5, faults=("schema_drift", "timeout"), seed=3)
    assert a.fingerprint() == b.fingerprint()


def test_max_per_run_is_respected():
    docs = build_corpus(n=10, seed=7)
    plan = InjectionPlan(rate=1.0, faults=("schema_drift",), seed=0, max_per_run=1)
    injector = FaultInjector(plan, corpus=corpus_for_injector(docs))
    registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    for doc in docs:
        assert agent.run(doc).trace.faults_injected <= 1


# -- agent behaviour ------------------------------------------------------

def test_clean_runs_are_perfect():
    """The baseline must be exact. Anything less is a harness bug."""
    docs = build_corpus(n=25, seed=7)
    registry = ToolRegistry(DEFAULT_TOOLS, injector=FaultInjector(InjectionPlan()))
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    for doc in docs:
        score = score_run(agent.run(doc), doc["truth"], INVOICE_SCHEMA)
        assert score.exact_match, f"{doc['document_id']}: clean run was not exact"


def test_transport_faults_are_fully_absorbed_by_retry():
    docs = build_corpus(n=20, seed=7)
    plan = InjectionPlan(
        rate=1.0, faults=("timeout", "rate_limit", "transient_error"),
        seed=0, max_per_run=1,
    )
    injector = FaultInjector(plan, corpus=corpus_for_injector(docs))
    registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    for doc in docs:
        assert score_run(agent.run(doc), doc["truth"], INVOICE_SCHEMA).exact_match


def test_agent_never_reads_injected_fault_labels():
    """Guards benchmark validity: fault metadata must not leak to the agent."""
    docs = build_corpus(n=5, seed=7)
    plan = InjectionPlan(rate=1.0, faults=("magnitude_shift",), seed=0, max_per_run=1)
    injector = FaultInjector(plan, corpus=corpus_for_injector(docs))

    seen: list[str] = []

    class SpyProvider(MockProvider):
        def complete(self, prompt, **kwargs):
            seen.append(prompt + kwargs.get("system", ""))
            return super().complete(prompt, **kwargs)

    registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
    agent = SelfCorrectingAgent(SpyProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    for doc in docs:
        agent.run(doc)

    blob = "\n".join(seen)
    for key in TAXONOMY:
        assert key not in blob, f"fault label {key!r} leaked into a prompt"
    assert "fault_injected" not in blob


# -- metrics --------------------------------------------------------------

def test_silent_failure_requires_undetected_and_wrong():
    docs = build_corpus(n=1, seed=7)
    doc = docs[0]
    registry = ToolRegistry(DEFAULT_TOOLS, injector=FaultInjector(InjectionPlan()))
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    score = score_run(agent.run(doc), doc["truth"], INVOICE_SCHEMA)
    assert score.faults_injected == 0
    assert not score.silent_failure


def test_report_handles_empty_faulted_set_without_dividing_by_zero():
    report = Report(condition="clean")
    assert report.recovery_rate != report.recovery_rate      # NaN
    assert report.silent_failure_rate != report.silent_failure_rate
    assert report.correction_overhead == 0.0


def test_weighted_accuracy_penalises_critical_fields_more():
    docs = build_corpus(n=1, seed=7)
    doc = docs[0]
    registry = ToolRegistry(DEFAULT_TOOLS, injector=FaultInjector(InjectionPlan()))
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    result = agent.run(doc)

    cosmetic = dict(result.record, line_item_count=999)
    critical = dict(result.record, total="999999.99")
    result.record = cosmetic
    lo = score_run(result, doc["truth"], INVOICE_SCHEMA).weighted_accuracy
    result.record = critical
    hi = score_run(result, doc["truth"], INVOICE_SCHEMA).weighted_accuracy
    assert hi < lo


# -- end to end -----------------------------------------------------------

def test_experiment_runs_all_conditions():
    docs = build_corpus(n=8, seed=7)
    experiment = Experiment(
        provider_factory=lambda: MockProvider(),
        schema=INVOICE_SCHEMA, documents=docs, checks=INVOICE_CHECKS,
    )
    conditions = [
        Condition("clean", InjectionPlan(rate=0.0)),
        Condition("semantic", InjectionPlan(
            rate=1.0, faults=("magnitude_shift",), seed=0, max_per_run=1)),
    ]
    result = experiment.run(conditions)
    assert len(result.reports) == 2
    clean, faulted = result.reports
    assert clean.exact_match_rate == 1.0
    assert faulted.exact_match_rate <= clean.exact_match_rate


def test_corroboration_reduces_silent_failures():
    """The paper's central claim, as an executable assertion."""
    docs = build_corpus(n=40, seed=7)
    faults = ("plausible_substitution", "magnitude_shift", "cross_document_bleed")
    condition = [Condition("semantic", InjectionPlan(
        rate=1.0, faults=faults, seed=0, max_per_run=1))]

    def silent_rate(corroborate: bool) -> float:
        experiment = Experiment(
            provider_factory=lambda: MockProvider(),
            schema=INVOICE_SCHEMA, documents=docs, checks=INVOICE_CHECKS,
            corroborate=corroborate,
        )
        return experiment.run(condition).reports[0].silent_failure_rate

    assert silent_rate(True) < silent_rate(False)


# -- cli and packaging ----------------------------------------------------

def test_unknown_fault_name_is_rejected_at_the_command_line():
    """It used to reach the injector and die on a KeyError deep inside."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["trace", "--faults", "not_a_fault"])
    args = parser.parse_args(["trace", "--faults", "plausible_substitution"])
    assert args.faults == ["plausible_substitution"]


def test_donor_pool_is_wider_than_the_run_being_traced():
    """`trace --n 1` built its donor pool from the single document it ran,
    so every fault that needs another document to steal from did nothing."""
    args = build_parser().parse_args(["trace", "--n", "1"])
    assert len(_documents(args)) == 1
    assert len(_documents(args, limit=DONOR_POOL)) == DONOR_POOL


def test_version_is_single_sourced():
    from agentaudit import __version__
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    assert __version__


def test_corpus_load_reports_a_missing_file_usefully(tmp_path):
    from agentaudit import load_corpus
    with pytest.raises(FileNotFoundError, match="corpus export"):
        load_corpus(tmp_path)


# -- shipped corpus -------------------------------------------------------

def test_shipped_corpus_matches_generator():
    """Guards published results against silent generator drift."""
    from agentaudit import verify_corpus
    assert verify_corpus(), "benchmarks/invoices is stale — re-export it"


def test_loaded_corpus_carries_ground_truth():
    from agentaudit import load_corpus
    documents = load_corpus()
    assert len(documents) == 100
    for doc in documents:
        assert doc["truth"], f"{doc['document_id']} has no ground truth"
        assert set(doc["truth"]) == set(doc["fields"])


def test_loaded_corpus_runs_clean():
    from agentaudit import load_corpus
    documents = load_corpus()[:20]
    registry = ToolRegistry(DEFAULT_TOOLS, injector=FaultInjector(InjectionPlan()))
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    for doc in documents:
        assert score_run(agent.run(doc), doc["truth"], INVOICE_SCHEMA).exact_match


def test_raw_text_is_consistent_with_fields():
    """Corroboration is only meaningful if grep_source can actually find values."""
    from agentaudit import load_corpus
    for doc in load_corpus()[:25]:
        for name, value in doc["fields"].items():
            assert str(value) in doc["raw_text"], f"{name} missing from raw_text"
