"""Demonstrate a silent failure end to end.

Run:  python examples/silent_failure_demo.py

Shows a single document where a `plausible_substitution` fault lands, the
agent's own validators all pass, and the emitted record is wrong anyway —
the exact case that end-to-end accuracy reporting hides.
"""

from __future__ import annotations

from agentaudit import (
    DEFAULT_TOOLS, INVOICE_CHECKS, INVOICE_SCHEMA, FaultInjector,
    InjectionPlan, MockProvider, SelfCorrectingAgent, ToolRegistry,
    build_corpus, corpus_for_injector, score_run,
)


def main() -> None:
    documents = build_corpus(n=12, seed=7)

    # Corroboration off: the agent trusts its tools. This is what most
    # production pipelines actually do.
    plan = InjectionPlan(
        rate=1.0, faults=("plausible_substitution",), seed=3, max_per_run=1
    )
    injector = FaultInjector(plan, corpus=corpus_for_injector(documents))
    registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
    agent = SelfCorrectingAgent(
        MockProvider(), registry, INVOICE_SCHEMA, INVOICE_CHECKS,
        corroborate=False,
    )

    for document in documents:
        result = agent.run(document)
        score = score_run(result, document["truth"], INVOICE_SCHEMA)
        if not score.silent_failure:
            continue

        print("=" * 68)
        print(f"SILENT FAILURE on {document['document_id']}")
        print("=" * 68)
        print(result.trace.render())

        faulted = result.trace.faulted_spans()[0]
        print(f"\ninjected: {faulted.fault_injected}")
        print(f"detail:   {faulted.fault_detail}")

        print("\nthe agent's own validators reported:")
        problems = INVOICE_SCHEMA.validate(result.record)
        for check in INVOICE_CHECKS:
            found = check(result.record)
            if found:
                problems.append(found)
        print(f"  {problems or 'no problems — record looks clean'}")

        print("\nbut against ground truth:")
        for f in score.fields:
            if not f.correct:
                print(f"  {f.name}: emitted {f.predicted!r}, actual {f.expected!r}")

        print(
            f"\nfaults={score.faults_injected}  detected={score.faults_detected}  "
            f"field_accuracy={score.field_accuracy:.3f}"
        )
        print(
            "\nField accuracy still looks respectable. The invoice is wrong.\n"
            "Nothing in the pipeline raised a flag."
        )
        return

    print("no silent failure in this sample — try another --seed")


if __name__ == "__main__":
    main()
