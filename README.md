# AgentAudit

**Fault injection and reliability measurement for LLM agent pipelines.**

Observability tools tell you what your agent *did*. AgentAudit tells you what your agent does when its tools **lie to it**.

You point it at an agent pipeline, it deliberately corrupts tool outputs — truncated JSON, transposed digits, a value quietly swapped for a plausible one from another document — and it measures whether the agent notices, whether it recovers, and how much that recovery cost.

Runs end to end with **no API key and no spend**.

```bash
pip install -e ".[dev]"
pytest -q                 # no network, no API key
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

Mock provider, n=100 synthetic invoices, at most one fault per run, `rate=0.85`, `seed=0`. Reproduce with `agentaudit bench --n 100`.

```
condition                 n    exact  field  detect recover silent corr
-------------------------------------------------------------------------
clean                     100  1.000  1.000  —      —       —      0.000
transport                 100  1.000  1.000  —      —       —      0.000
structural                100  0.700  0.967  0.930  0.700   0.070  0.930
semantic                  100  0.430  0.937  0.812  0.406   0.188  1.220
contextual                100  0.270  0.837  0.800  0.270   0.200  0.930
```

Two things worth reading off this table:

**Retries fully absorb transport faults.** Timeouts and rate limits cost latency, nothing else. This is the failure mode everyone builds for, and it's the one that matters least.

**Field accuracy is actively misleading.** The `semantic` row keeps 0.937 field accuracy while only 43% of records are correct and 19% of faulted runs are wrong with nothing flagged. One corrupted field out of nine barely moves an average, but it's still a wrong invoice total. Any metric that averages over fields will tell you this pipeline is fine.

### Per fault, because layers pool things that behave nothing alike

`agentaudit bench --n 100 --by-fault` — same defaults as above, at most one fault per run:

```
condition                 n    exact  field  detect recover silent corr
-------------------------------------------------------------------------
mecha:timeout             100  1.000  1.000  —      —       —      0.000
mecha:rate_limit          100  1.000  1.000  —      —       —      0.000
mecha:transient_error     100  1.000  1.000  —      —       —      0.000
mecha:malformed_json      100  0.150  0.906  1.000  0.150   0.000  1.000
mecha:schema_drift        100  1.000  1.000  1.000  1.000   0.000  1.000
mecha:type_violation      100  1.000  1.000  1.000  1.000   0.000  1.000
mecha:field_dropped       100  0.710  0.968  0.710  0.710   0.290  0.710
consi:digit_transposition 100  0.260  0.918  0.865  0.229   0.135  1.500
consi:magnitude_shift     100  0.290  0.921  0.929  0.276   0.071  1.580
consi:unit_mismatch       100  1.000  1.000  1.000  1.000   0.000  1.000
corro:plausible_substitu… 100  0.320  0.924  0.515  0.313   0.485  0.720
corro:stale_cache         100  0.140  0.728  0.980  0.131   0.020  0.980
opaqu:cross_document_ble… 100  0.300  0.922  0.520  0.300   0.480  0.740
```

Silent failure tracks detectability, which is the whole argument for classifying faults that way: everything mechanical lands at or near zero, and the two faults that need evidence from outside the response — `plausible_substitution` and `cross_document_bleed` — fail silently on roughly half of the runs they touch.

The exception is instructive. `field_dropped` is mechanically detectable yet fails silently 29% of the time, because a dropped *optional* field raises no validation error, so nothing ever goes looking for it. Detectable in principle is not the same as detected.

### Ablation: does corroboration help?

Re-deriving critical fields from the source document, at the cost of an extra tool call per field:

| corroboration | detection | silent failure |
|---|---|---|
| **on** | 0.773 | **0.227** |
| off | 0.381 | 0.619 |

Silent failures **nearly triple** when the agent trusts its tools instead of re-checking them. This is asserted as a test (`test_corroboration_reduces_silent_failures`), so it fails CI if it ever stops holding.

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
agentaudit bench --n 100                     # all conditions, one per layer
agentaudit bench --n 100 --by-fault          # one condition per fault
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
- **Detection attribution is coarse.** Every prior successful tool call is treated as a suspect for a given problem, so with more than one fault in a run, detecting one credits the others. Precise attribution would need ground truth, which the agent must not have. Transport retries are excluded — recovering from a timeout is not evidence of noticing a corrupted number — but within a run the remaining attribution is deliberately generous, which makes `detect` an upper bound and `silent` a lower one.
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
                          (standard, by_fault, ablation_on, ablation_off)
tests/                    unit + regression suite, no network
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
make test          # no network, well under a second
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
