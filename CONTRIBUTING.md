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
2. Add an `_apply_<key>(self, payload, ctx)` method to `FaultInjector`. It
   receives the tool's payload and a `FaultContext` (`document_id`,
   `tool_name`), and returns `(corrupted_payload, detail_dict)`. The detail
   dict is recorded on the trace span for later analysis.
3. `test_every_fault_has_a_handler` will fail until step 2 is done.

Transport-layer faults raise `ToolFailure` instead of returning a payload,
and must set `corrupts_value=False`.

**An empty detail dict means "I could not apply this fault"** — no numeric
field to shift, no donor document to steal from. The injector then leaves
the span unmarked and refunds the run's fault budget, so the plan still
lands `max_per_run` real faults. Return a non-empty detail if and only if
you actually changed the payload; a fault recorded as injected while
corrupting nothing lands in every denominator as injected-and-missed.

Faults that need to know what a tool returned earlier (`stale_cache`) read
`FaultInjector._last_served`, which the registry fills via `observe()` with
the pre-corruption payload of each call.

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

**Never repair the corruption you are trying to measure.** Parsing and
comparison in `schema.py` must be strict enough that a corrupted value
stays corrupted. A tolerant money parser once turned `type_violation` into
a no-op: the fault landed, validation passed, and the run scored a perfect
match. `test_money_rejects_prose_wrapped_numbers` guards that boundary.

**Ground truth never enters the agent's output.** Judge rescues are passed
to `score_run(..., rescued=...)`, not written back into `result.record` —
otherwise a later stage reads the answer it was supposed to derive.

## Tests

```bash
pytest -q     # no network, well under a second
```

New defences should come with an ablation showing they actually reduce
silent failure rate, in the style of `test_corroboration_reduces_silent_failures`.
