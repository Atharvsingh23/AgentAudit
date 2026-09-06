# AgentAudit

**Fault injection and reliability measurement for LLM agent pipelines.**

Observability tools tell you what your agent *did*. AgentAudit tells you what your agent does when its tools **lie to it**.

You point it at an agent pipeline, it deliberately corrupts tool outputs — truncated JSON, transposed digits, a value quietly swapped for a plausible one from another document — and it measures whether the agent notices, whether it recovers, and how much that recovery cost.

Runs end to end with **no API key and no spend**.

```bash
pip install -e ".[dev]"
pytest -q                 # 27 tests, no network
agentaudit bench --n 100
```

The 100-document invoice corpus ships in `benchmarks/invoices/` with ground
truth and a SHA-256 manifest. Pre-computed results are in `results/`.

---

## Why

Agent evaluations report end-to-end accuracy. It's the least informative number available, because it collapses four outcomes that mean completely different things:

| Fault landed | Agent flagged it | Output correct | Outcome |
|---|---|---|---|
| yes | yes | yes | **resilient** |
| yes | yes | no | **degraded, but honest** — it told you |
| yes | no | yes | **lucky** — no credit deserved |
| yes | no | no | **silent failure** ← the only dangerous one |

Only the last row hurts you in production, and accuracy-only reporting hides it whenever fault rates are low. A pipeline can post 97% field accuracy while silently emitting wrong financial totals on one run in seven.

**Silent failure rate is the metric this harness exists to produce.**

## Results

Mock provider, n=100 synthetic invoices, one fault per run, `seed=0`. Reproduce with `agentaudit bench --n 100`.

```
condition                 n    exact  field  detect recover silent corr
-------------------------------------------------------------------------
clean                     100  1.000  1.000  —      —       —      0.000
transport                 100  1.000  1.000  —      —       —      0.000
structural                100  0.600  0.931  0.640  0.600   0.070  0.970
semantic                  100  0.770  0.974  0.190  0.770   0.150  0.270
contextual                100  0.210  0.822  0.790  0.210   0.200  0.970
```

Three things worth reading off this table:

**Retries fully absorb transport faults.** Timeouts and rate limits cost latency, nothing else. This is the failure mode everyone builds for, and it's the one that matters least.

**Field accuracy is actively misleading.** The `semantic` row has the *highest* field accuracy of any faulted condition (0.974) and the second-highest silent failure rate (0.150). One corrupted field out of nine barely moves an average, but it's still a wrong invoice total. Any metric that averages over fields will tell you this pipeline is fine.

**Detection collapses exactly where it matters.** Structural faults are caught 64% of the time; semantic faults, 19%. A JSON parser catches the first class for free. Nothing catches the second unless you build something that specifically goes looking.

### Ablation: does corroboration help?

Re-deriving critical fields from the source document, at the cost of an extra tool call per field:

| corroboration | detection | silent failure |
|---|---|---|
| **on** | 0.360 | **0.210** |
| off | 0.130 | 0.440 |

Silent failures **more than double** when the agent trusts its tools instead of re-checking them. This is asserted as a test (`test_corroboration_reduces_silent_failures`), so it fails CI if it ever stops holding.

> **On the mock provider.** These numbers come from a deterministic rule-based stand-in, not a real model. That's deliberate — it's a control. Because the mock's behaviour is fixed, any change in the numbers between two runs is a harness bug rather than model variance. Absolute values here are *not* claims about GPT-4 or Claude; the fault taxonomy and the relative ordering between conditions are. Run `--provider anthropic` for real numbers.

## Fault taxonomy

Faults are classified by **detectability** — how much machinery an agent needs to catch them — because that's what determines whether a defence is even possible.

| Layer | Fault | Detectability |
|---|---|---|
| **Transport** | `timeout`, `rate_limit`, `transient_error` | mechanical |
| **Structural** | `malformed_json`, `schema_drift`, `type_violation`, `field_dropped` | mechanical |
| **Semantic** | `digit_transposition`, `magnitude_shift`, `unit_mismatch` | consistency |
| | `plausible_substitution` | corroborative |
| **Contextual** | `stale_cache` | corroborative |
| | `cross_document_bleed` | opaque |

- **mechanical** — a parser or type check catches it, no reasoning needed
- **consistency** — catchable only by cross-checking fields against each other
- **corroborative** — catchable only by re-deriving the value another way
- **opaque** — not catchable from the response alone at any cost

`agentaudit faults` prints the full table with descriptions.

The `plausible_substitution` fault is the one to care about: it swaps a value for a *real value from a different document*. Nothing about the resulting record is malformed or internally inconsistent. It passes every validator you'd think to write.

## Usage

```bash
agentaudit bench --n 100                    # all conditions
agentaudit bench --ablation --no-corroborate # isolate one defence
agentaudit bench --provider anthropic        # real model (needs ANTHROPIC_API_KEY)
agentaudit faults                            # print the taxonomy
agentaudit trace --n 2 --faults plausible_substitution
```

`agentaudit trace` renders a single run, marking faulted spans with `!`:

```
trace 4b1e9c  doc=doc-000
  agent      extract                      ok       4.6ms
    tool       extract_totals               ok       0.0ms
!   tool       extract_header               ok       0.0ms  <fault:plausible_substitution>
    model      assemble                     ok       0.4ms
    validation validate                     ERR      0.0ms
    correction self_correct                 ok       0.2ms
```

### Library

```python
from agentaudit import (
    Experiment, Condition, InjectionPlan, MockProvider,
    build_corpus, INVOICE_SCHEMA, INVOICE_CHECKS, format_table,
)

experiment = Experiment(
    provider_factory=lambda: MockProvider(),
    schema=INVOICE_SCHEMA,
    documents=build_corpus(n=100),
    checks=INVOICE_CHECKS,
    corroborate=True,
)

result = experiment.run([
    Condition("clean", InjectionPlan(rate=0.0)),
    Condition("substitution", InjectionPlan(
        rate=1.0, faults=("plausible_substitution",), seed=0)),
])

print(format_table(result.reports))
```

## Design guarantees

**Reproducibility.** Faults are seeded from `sha256(seed, plan_fingerprint, document_id)`. Same seed, same plan, same dataset → byte-identical faults on byte-identical calls. Asserted in `test_same_seed_produces_identical_faults_and_outcomes`.

**No leakage.** Injected fault metadata lives on the trace span, which the agent never reads. If the agent could see its own fault labels, every detection number would be worthless. `test_agent_never_reads_injected_fault_labels` greps every prompt for every fault key and fails if one appears.

**A clean baseline must be exact.** `test_clean_runs_are_perfect` asserts 1.000 exact-match with no faults. If the baseline isn't perfect, every faulted number is contaminated by unrelated bugs — which happened during development, and is how three real bugs were found.

**Transport faults don't corrupt values.** A retried timeout is a latency cost, not a data error, and conflating the two makes recovery rate uninterpretable.

## Architecture

```
agentaudit/
├── schema.py            field types, criticality weights, consistency checks
├── trace.py             spans, fault bookkeeping, run records
├── faults/
│   ├── taxonomy.py      13 faults × 4 layers × 4 detectability levels
│   └── injector.py      deterministic seeded injection
├── tools/               tool interface + injection seam
├── providers/           mock (free, deterministic) | anthropic
├── agent.py             gather → assemble → validate → corroborate → correct
├── judge.py             LLM-as-judge, string fields only
├── metrics.py           silent failure rate and friends
└── runner.py            experiment conditions
```

The injector sits **between** the agent and the tool, not inside it. That seam is what lets the identical tool implementation serve both clean and faulted runs, so the only difference between experiment arms is the fault plan.

## Adding your own agent

Implement `Provider.complete()` and hand your agent a `ToolRegistry` with an injector attached. The registry records spans and applies faults; you keep your own control flow.

```python
from agentaudit import ToolRegistry, FaultInjector, InjectionPlan, Tool

class MyTool(Tool):
    name = "my_tool"
    description = "..."
    def run(self, document, **kwargs):
        return {"field": "value"}

registry = ToolRegistry([MyTool()], injector=FaultInjector(
    InjectionPlan(rate=0.5, faults=("magnitude_shift",), seed=0)))
```

## Limitations

Worth stating plainly, since they bound what the numbers mean:

- **Synthetic corpus.** Documents are cleaner than real scans, so absolute accuracy is optimistic. Comparisons *between* conditions — the point of the harness — are unaffected.
- **Detection attribution is coarse.** Every prior successful tool call is treated as a suspect for a given problem. Precise attribution would need ground truth, which the agent must not have.
- **One task.** Invoice extraction only. The taxonomy generalizes; the tools and schema don't yet.
- **The mock is not a model.** See the note under Results.

## Roadmap

- [ ] `smolagents` and LangGraph adapters
- [ ] Multi-fault runs (interaction effects between layers)
- [ ] Fault-detectability curves as a function of correction budget
- [ ] A second task domain to test taxonomy generalization

## What's in the box

```
benchmarks/invoices/      100 documents, ground truth, SHA-256 manifest
results/                  pre-computed JSON for every table in this README
tests/                    27 tests, no network, ~0.2s
examples/                 worked silent-failure walkthrough
.github/workflows/ci.yml  tests on 3.10–3.12 + cross-run reproducibility check
```

The corpus is both generated *and* checked in. The generator lets you scale
it up; the checked-in files mean a published number is reproducible against
exact bytes. `agentaudit corpus verify` compares the two and CI fails on
drift, so results can't silently stop matching their data.

## Development

```bash
make install       # pip install -e ".[dev]"
make test          # 27 tests, ~0.2s, no network
make bench         # headline table
make ablation      # corroboration on vs off
make results       # regenerate results/*.json
make demo          # walk through one silent failure
make all
```

See `CONTRIBUTING.md` for how to add a fault, and for the invariants that
keep the benchmark valid.

## License

MIT
