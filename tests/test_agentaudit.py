"""Test suite.

The determinism tests are the load-bearing ones. Every claim the harness
makes rests on the assumption that a seed pins the experiment; if that
breaks, nothing else is meaningful.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest

from agentaudit import (
    INVOICE_CHECKS, INVOICE_SCHEMA, Condition, Criticality, DEFAULT_TOOLS,
    Experiment, ExtractionResult, FaultContext, FaultInjector, InjectionPlan,
    MockProvider, Report, SelfCorrectingAgent, SpanKind, SpanStatus, Tool,
    ToolFailure, ToolRegistry, Trace, build_corpus, corpus_for_injector,
    format_table, score_run,
)
from agentaudit.faults.taxonomy import TAXONOMY, Detectability, FaultLayer
from agentaudit.cli import DONOR_POOL, _documents, build_parser
from agentaudit.schema import Field, FieldType, totals_add_up
from agentaudit.tools.extraction import SourceGrep


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
    ok = {"subtotal": "100.00", "tax": "10.00", "total": "110.00"}
    assert totals_add_up(ok) is None
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


# -- providers ------------------------------------------------------------

class _StubResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [SimpleNamespace(type="thinking", thinking="..."),
                        SimpleNamespace(type="text", text=text)]
        self.usage = SimpleNamespace(input_tokens=11, output_tokens=22)
        self.stop_reason = stop_reason


def _stub_anthropic(monkeypatch, response):
    """Install a fake `anthropic` module and capture the request kwargs."""
    seen: dict = {}

    class _Messages:
        def create(self, **kwargs):
            seen.update(kwargs)
            return response

    class _Anthropic:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=_Anthropic))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return seen


def test_anthropic_request_omits_temperature_by_default(monkeypatch):
    """Current models reject `temperature` with a 400.

    The harness never sets one — its determinism comes from the seeded
    injector and the mock provider — so sending 0.0 on every call only
    bought a failed request.
    """
    from agentaudit.providers.anthropic import DEFAULT_MODEL, AnthropicProvider

    seen = _stub_anthropic(monkeypatch, _StubResponse('{"ok": true}'))
    completion = AnthropicProvider().complete("prompt", system="sys")

    assert "temperature" not in seen
    assert seen["model"] == DEFAULT_MODEL
    # thinking blocks must not leak into the extracted text
    assert completion.text == '{"ok": true}'
    assert completion.output_tokens == 22


def test_anthropic_forwards_temperature_only_when_asked(monkeypatch):
    from agentaudit.providers.anthropic import AnthropicProvider

    seen = _stub_anthropic(monkeypatch, _StubResponse("{}"))
    AnthropicProvider(model="claude-3-5-sonnet-20240620").complete("p", temperature=0.0)
    assert seen["temperature"] == 0.0


def test_truncated_completion_is_visible_on_the_trace():
    """A record cut off at max_tokens is a fact about the run.

    Without it on the span, truncation is indistinguishable from a model
    that simply extracted badly.
    """
    class Truncating(MockProvider):
        def complete(self, prompt, **kwargs):
            completion = super().complete(prompt, **kwargs)
            completion.stop_reason = "max_tokens"
            return completion

    doc = build_corpus(n=1, seed=7)[0]
    registry = ToolRegistry(DEFAULT_TOOLS, injector=FaultInjector(InjectionPlan()))
    agent = SelfCorrectingAgent(Truncating(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    trace = agent.run(doc).trace
    assemble = [s for s in trace.spans if s.name == "assemble"]
    assert assemble and assemble[0].attributes["stop_reason"] == "max_tokens"


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


# -- measurement integrity ------------------------------------------------
#
# Each test below pins a bug that made the harness report something other
# than what happened. They are grouped because they share a failure mode:
# the numbers still looked plausible while they were wrong.

def test_money_rejects_prose_wrapped_numbers():
    """A lenient parser makes `type_violation` unmeasurable.

    Stripping non-numeric characters turned the corrupted value straight
    back into the correct one, so a MECHANICAL fault landed, passed
    validation, and scored exact.
    """
    f = Field("total", FieldType.MONEY)
    with pytest.raises(ValueError):
        f.normalize("approximately 1234.50 (see attached schedule)")
    # Currency-marked values from grep_source must still parse.
    assert f.normalize("USD 1,234.50") == Decimal("1234.50")


def test_integer_rejects_prose_but_accepts_integral_floats():
    f = Field("line_item_count", FieldType.INTEGER)
    with pytest.raises(ValueError):
        f.normalize("approximately 7 (see attached schedule)")
    with pytest.raises(ValueError):
        f.normalize("12.7")
    assert f.normalize("12.0") == 12


def test_type_violation_is_detected_rather_than_absorbed():
    docs = build_corpus(n=20, seed=7)
    plan = InjectionPlan(rate=1.0, faults=("type_violation",), seed=0, max_per_run=1)
    injector = FaultInjector(plan, corpus=corpus_for_injector(docs))
    registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    for doc in docs:
        score = score_run(agent.run(doc), doc["truth"], INVOICE_SCHEMA)
        if score.faults_injected:
            assert score.faults_detected, f"{doc['document_id']}: type violation missed"
            assert not score.lucky


def test_grep_source_does_not_match_a_field_name_inside_another():
    """`total` must not match inside `Subtotal`.

    It did, and since repair takes the first hit, a run that flagged the
    total wrote the *subtotal* into it — a wrong value produced by the
    repair path itself.
    """
    doc = build_corpus(n=1, seed=7)[0]
    grep = SourceGrep()
    total = grep.run(doc, field="total")["__matches__"]
    subtotal = grep.run(doc, field="subtotal")["__matches__"]
    money = Field("m", FieldType.MONEY)
    assert money.normalize(total[0]) == money.normalize(doc["fields"]["total"])
    assert money.normalize(subtotal[0]) == money.normalize(doc["fields"]["subtotal"])
    assert money.normalize(total[0]) != money.normalize(subtotal[0])


def test_a_fault_that_corrupts_nothing_is_not_counted_as_injected():
    """No donor documents means plausible_substitution has nothing to do.

    Recording it anyway put runs where nothing was corrupted into the
    denominator of every detection and silent-failure rate.
    """
    docs = build_corpus(n=3, seed=7)
    plan = InjectionPlan(
        rate=1.0, faults=("plausible_substitution",), seed=0, max_per_run=1)
    injector = FaultInjector(plan, corpus=[])  # no donors
    registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    for doc in docs:
        result = agent.run(doc)
        assert result.trace.faults_injected == 0
        assert score_run(result, doc["truth"], INVOICE_SCHEMA).exact_match


def test_transport_retry_is_not_scored_as_detecting_a_corrupted_value():
    """Retrying a timeout says nothing about noticing a transposed digit."""
    trace = Trace(document_id="doc-000")
    faulted = trace.span("extract_totals", SpanKind.TOOL)
    faulted.fault_injected = "digit_transposition"
    faulted.finish()
    timed_out = trace.span("extract_header", SpanKind.TOOL)
    timed_out.fault_injected = "timeout"
    timed_out.finish(SpanStatus.ERROR)
    retry = trace.span("retry:extract_header", SpanKind.CORRECTION,
                       triggered_by=[timed_out.span_id], retry=True)
    retry.finish()

    result = ExtractionResult("doc-000", {}, trace)
    score = score_run(result, {"total": "1.00"}, INVOICE_SCHEMA)
    assert score.faults_injected == 1        # the timeout does not corrupt
    assert score.faults_detected == 0
    assert trace.repair_attempts == 0 and len(trace.retry_spans()) == 1


def test_stale_cache_serves_the_same_tool_from_an_earlier_document():
    """Serving another tool's payload would be a structural fault, not a
    contextual one, and would be caught for free by any schema check."""
    injector = FaultInjector(
        InjectionPlan(rate=1.0, faults=("stale_cache",), seed=0))
    injector.observe("extract_totals", "doc-000", {"total": "10.00"})
    injector.observe("extract_header", "doc-000", {"vendor_name": "Acme"})

    payload, detail, applied = injector.apply(
        TAXONOMY["stale_cache"], {"total": "99.00"},
        FaultContext("doc-001", "extract_totals"),
    )
    assert applied and payload == {"total": "10.00"}
    assert detail["served_for"] == "doc-000"
    assert set(payload) == {"total"}, "must keep the faulted tool's own shape"

    # Nothing stale for a tool that has not run yet.
    _, _, applied_again = injector.apply(
        TAXONOMY["stale_cache"], {"line_item_count": 3},
        FaultContext("doc-001", "count_line_items"),
    )
    assert not applied_again


def test_judge_rescues_a_field_without_writing_truth_into_the_record():
    docs = build_corpus(n=1, seed=7)
    doc = docs[0]
    registry = ToolRegistry(DEFAULT_TOOLS, injector=FaultInjector(InjectionPlan()))
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    result = agent.run(doc)
    result.record["vendor_name"] = "Acme Ltd"

    score = score_run(result, doc["truth"], INVOICE_SCHEMA, rescued={"vendor_name"})
    assert score.exact_match
    assert result.record["vendor_name"] == "Acme Ltd", "ground truth leaked into output"


def test_corroborated_fields_are_counted_once_per_run():
    docs = build_corpus(n=8, seed=7)
    plan = InjectionPlan(rate=1.0, faults=("magnitude_shift",), seed=0, max_per_run=1)
    injector = FaultInjector(plan, corpus=corpus_for_injector(docs))
    registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
    agent = SelfCorrectingAgent(MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS)
    for doc in docs:
        fields = agent.run(doc).corroborated_fields
        assert len(fields) == len(set(fields))


def test_a_tool_that_raises_leaves_no_span_open():
    """An unclosed span reparents everything that follows it."""
    class Exploding(Tool):
        name = "extract_header"
        description = "always fails"

        def run(self, document, **kwargs):
            raise ToolFailure("timeout", "boom", retryable=False)

    trace = Trace(document_id="doc-000")
    outer = trace.span("extract", SpanKind.AGENT)
    trace.push(outer)
    registry = ToolRegistry([Exploding()])
    with pytest.raises(ToolFailure):
        registry.call("extract_header", {"document_id": "doc-000"}, trace)

    later = trace.span("assemble", SpanKind.MODEL)
    assert later.parent_id == outer.span_id
    tool_spans = trace.tool_spans()
    assert tool_spans and all(s.end is not None for s in tool_spans)


def test_format_table_keeps_columns_aligned_for_long_condition_names():
    reports = [Report(condition="clean"),
               Report(condition="semantic+contextual+something+long")]
    for report in reports:
        report.runs.append(score_run(
            ExtractionResult("doc-000", {}, Trace()), {}, INVOICE_SCHEMA))
    rows = format_table(reports).splitlines()
    assert len({len(r) for r in rows}) == 1, "a long name pushed a row out of line"
