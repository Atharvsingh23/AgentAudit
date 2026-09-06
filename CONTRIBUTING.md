# Contributing

## Setup

```bash
pip install -e ".[dev]"
make all
```

## Adding a fault

1. Add a `FaultSpec` to `TAXONOMY` in `agentaudit/faults/taxonomy.py`. The
   `detectability` field is the important one — it is what determines which
   defence could catch it, and therefore what the fault is worth measuring.
2. Add an `_apply_<key>` method to `FaultInjector`. It receives a payload
   dict and returns `(corrupted_payload, detail_dict)`. The detail dict is
   recorded on the trace span for later analysis.
3. `test_every_fault_has_a_handler` will fail until step 2 is done.

Transport-layer faults raise `ToolFailure` instead of returning a payload,
and must set `corrupts_value=False`.

## Rules that exist for a reason

**Never let fault metadata reach the agent.** Injected fault keys live on
trace spans. If a fault label ever appears in a prompt, detection rates
become meaningless. `test_agent_never_reads_injected_fault_labels` enforces
this.

**The clean baseline must stay at 1.000 exact match.** If it drops, there is
a bug in the harness, and every faulted number is contaminated by it. Fix
the baseline before interpreting anything else. This is not hypothetical —
three real bugs were found this way during initial development.

**Re-export the corpus if you touch the generator.** `agentaudit corpus verify`
compares shipped files against `build_corpus`. CI fails on drift, because
otherwise published numbers silently stop corresponding to the data.

**Keep the mock deterministic.** It is a control, not a model. Any
nondeterminism in `MockProvider` destroys the ability to tell harness bugs
apart from model variance.

## Tests

```bash
pytest -q     # 27 tests, no network, ~0.2s
```

New defences should come with an ablation showing they actually reduce
silent failure rate, in the style of `test_corroboration_reduces_silent_failures`.
