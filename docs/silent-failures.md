# We swapped one number on an invoice. Every check passed.

Take a document extraction pipeline — the kind that reads invoices and
writes rows into an accounting system. Give it a tool that returns the
totals block. Now have that tool return the right answer *for a different
invoice*: a real vendor, a real subtotal, a real tax figure, a real total.
The numbers still add up, because they came from a real invoice. Just not
this one.

Every validator you would think to write passes. The JSON parses. The
types are right. Subtotal plus tax equals total. Nothing is null, nothing
is malformed, nothing is out of range.

The pipeline writes a wrong invoice into the ledger and reports success.

I built [AgentAudit](https://github.com/Atharvsingh23/AgentAudit) to
measure how often that happens.

## Accuracy hides the only outcome that matters

Agent evaluations report end-to-end accuracy. It's the least informative
number available, because it collapses four outcomes that mean completely
different things:

| Fault landed | Agent flagged it | Output correct | What it means |
|---|---|---|---|
| yes | yes | yes | resilient |
| yes | yes | no | degraded, but honest — it told you |
| yes | no | yes | lucky — no credit deserved |
| yes | no | no | **silent failure** |

Only the last row hurts you. The others are visible: you get an error, a
flag, a retry, a log line, something. A silent failure produces a
confident, well-formed, wrong answer, and the first time anyone finds out
is when a human notices the ledger doesn't balance three weeks later.

Accuracy averages all four together. Worse, it averages over *fields*: get
eight of nine right and you post 0.89, which looks respectable, while the
one you got wrong is the invoice total.

**Silent failure rate is the number that matters:** of the runs where
something went wrong, what share emitted a wrong answer with nothing
flagged?

## Not all faults are equally catchable

The useful way to classify a fault isn't where it comes from — it's how
much machinery you need to catch it:

- **mechanical** — a parser or type check catches it, no reasoning needed
- **consistency** — catchable by cross-checking fields against each other
- **corroborative** — catchable only by re-deriving the value another way
- **opaque** — not catchable from the response alone at any cost

This matters because a pipeline that handles the first class looks robust
while remaining wide open to the rest. Truncated JSON, a renamed field, a
number arriving as prose: all trivially catchable, and all the ones
engineers actually test for. Meanwhile the value quietly swapped for
another document's value passes every one of those checks.

Reporting a single recovery rate across all four is close to meaningless.

## What the numbers look like

Running a self-correcting agent — one that validates its output, checks
that totals reconcile, and re-reads the source document for critical
fields — against 100 synthetic invoices, one fault per run:

```
fault                     silent  detectability
------------------------------------------------
malformed_json            0.000   mechanical
schema_drift              0.000   mechanical
type_violation            0.000   mechanical
field_dropped             0.290   mechanical
digit_transposition       0.135   consistency
magnitude_shift           0.071   consistency
unit_mismatch             0.000   consistency
plausible_substitution    0.485   corroborative
stale_cache               0.020   corroborative
cross_document_bleed      0.480   opaque
```

Silent failure tracks detectability, which is the whole argument for
classifying faults that way. Everything mechanical lands at or near zero.
The two faults that need evidence from *outside* the response —
`plausible_substitution` and `cross_document_bleed` — fail silently on
roughly half the runs they touch.

The exception is the interesting one. `field_dropped` is mechanically
detectable and still fails silently 29% of the time, because a dropped
*optional* field raises no validation error, so nothing ever goes looking
for it. **Detectable in principle is not the same as detected.**

## What actually helps

The defence that works is corroboration: re-deriving critical fields from
the source document instead of trusting the tool that reported them. It
costs an extra tool call per critical field. Turning it off:

| corroboration | detection | silent failure |
|---|---|---|
| **on** | 0.773 | **0.227** |
| off | 0.381 | 0.619 |

Silent failures nearly triple when the agent trusts its tools. That's the
entire finding, and it's unsurprising once stated — but almost nothing in
the eval tooling ecosystem measures it, so almost nobody knows their own
number.

## Your pipeline is probably worse

The agent above is *trying*. It validates, it cross-checks, it re-reads
the source. Most production pipelines do none of that — they call a tool
and believe it.

Here is that pipeline, audited:

```
condition                 n    exact  field  detect recover silent
--------------------------------------------------------------------
clean                     30   1.000  1.000  —      —       —
plausible_substitution    30   0.000  0.750  0.000  0.000   1.000
```

Field accuracy 0.750 — the kind of number that ends up on a slide as "75%
accurate, needs a bit of tuning." Silent failure **1.000**. Every single
run emitted a wrong invoice and flagged nothing.

## Measure your own

Your agent doesn't need to know the harness exists. Hand over your tool
functions; you get back functions with identical signatures that
occasionally lie:

```python
pip install agentaudit
```

```python
from agentaudit import Schema, quick_audit

def my_agent(document, tools):          # your control flow, unchanged
    header = tools["fetch_header"](document["invoice_id"])
    totals = tools["fetch_totals"](document["invoice_id"])
    return {**header, **totals}

result = quick_audit(
    agent=my_agent,
    tools={"fetch_header": fetch_header, "fetch_totals": fetch_totals},
    documents=[{"document_id": "INV-1", "invoice_id": "INV-1",
                "truth": {"total": "1100.00"}}, ...],
    schema=Schema.from_spec({"total": {"type": "money",
                                       "criticality": "critical"}}),
)
print(result.summary())
```

You need a sample of your own documents and the answers you expect. That's
it. Outside an audit the wrapped tools are pass-throughs, so the same
wiring serves production and the harness.

## What these numbers are and aren't

The headline results come from a **deterministic mock provider**, not a
real model. That is deliberate: the mock is a control. Because its
behaviour is fixed, any change in the numbers between two runs is a bug in
the harness rather than model variance — which is how three real
measurement bugs got found during development, including a value parser so
tolerant that it silently repaired the corruption it was supposed to be
measuring.

So: the absolute values are not claims about any particular model. The
fault taxonomy, the ordering between conditions, and the gap between
detectable-in-principle and actually-detected are.

The corpus is synthetic and cleaner than real scans, so absolute accuracy
is optimistic. Detection attribution is deliberately generous — within a
run, catching one fault credits the others — which makes the reported
detection rate an upper bound and the silent-failure rate a lower one.

Your real number is probably worse than anything printed here. That's the
point of measuring it.

---

*[AgentAudit](https://github.com/Atharvsingh23/AgentAudit) is MIT
licensed. The corpus, the ground truth and a SHA-256 manifest ship with
it, and CI fails if the generator ever drifts from the checked-in bytes,
so every number above is reproducible against exact data.*
