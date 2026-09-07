"""Command-line interface for agentaudit benchmark.

Version: 0.1.0 (research beta)
Python: 3.10+

Examples:
    agentaudit bench --n 40 --provider mock
    agentaudit bench --ablation --out results/ablation.json
    agentaudit faults
    agentaudit trace --n 1 --faults plausible_substitution
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .corpus import export_corpus, load_corpus, verify_corpus
from .dataset import build_corpus, corpus_for_injector
from .faults.injector import FaultInjector, InjectionPlan
from .faults.taxonomy import TAXONOMY
from .metrics import format_table, score_run
from .providers.mock import MockProvider
from .runner import (
    Experiment, ablation_conditions, by_fault_conditions, standard_conditions,
)
from .schema import INVOICE_CHECKS, INVOICE_SCHEMA


def _provider_factory(name: str, vigilance: float):
    if name == "mock":
        return lambda: MockProvider(vigilance=vigilance)
    if name == "anthropic":
        from .providers.anthropic import AnthropicProvider
        return lambda: AnthropicProvider()
    raise SystemExit(f"unknown provider {name!r}")


def _documents(args: argparse.Namespace) -> list:
    """Shipped corpus by default; regenerate only when asked.

    Reading the checked-in files by default means published numbers are
    reproducible against exact bytes rather than against a generator that
    might have drifted.
    """
    if getattr(args, "generate", False):
        return build_corpus(n=args.n, seed=args.data_seed)
    try:
        documents = load_corpus()
    except FileNotFoundError:
        return build_corpus(n=args.n, seed=args.data_seed)
    return documents[: args.n]


def cmd_bench(args: argparse.Namespace) -> int:
    documents = _documents(args)
    args.n = len(documents)
    experiment = Experiment(
        provider_factory=_provider_factory(args.provider, args.vigilance),
        schema=INVOICE_SCHEMA,
        documents=documents,
        checks=INVOICE_CHECKS,
        corroborate=not args.no_corroborate,
        max_corrections=args.max_corrections,
        use_judge=not args.no_judge,
    )
    if args.ablation:
        conditions = ablation_conditions(seed=args.seed, rate=args.rate)
    elif args.by_fault:
        conditions = by_fault_conditions(seed=args.seed, rate=args.rate)
    else:
        conditions = standard_conditions(seed=args.seed, rate=args.rate)

    result = experiment.run(conditions)

    print(f"\nAgentAudit  n={args.n}  provider={args.provider}  "
          f"rate={args.rate}  corroborate={not args.no_corroborate}\n")
    print(format_table(result.reports))
    print(
        "\nexact   exact-match rate over all runs\n"
        "detect  share of injected corrupting faults the agent flagged\n"
        "recover share of faulted runs that still produced a correct record\n"
        "silent  share of faulted runs that were wrong AND unflagged\n"
        "corr    mean self-correction attempts per faulted run"
    )

    if args.out:
        path = result.save(args.out)
        print(f"\nwrote {path}")
    return 0


def cmd_faults(_args: argparse.Namespace) -> int:
    by_layer: dict[str, list[Any]] = {}
    for spec in TAXONOMY.values():
        by_layer.setdefault(spec.layer.value, []).append(spec)

    for layer, specs in by_layer.items():
        print(f"\n{layer.upper()}")
        for spec in specs:
            flag = "" if spec.corrupts_value else "  [non-corrupting]"
            print(f"  {spec.key:<26} {spec.detectability.value:<14}{flag}")
            print(f"  {'':<26} {spec.description}")
    print()
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from .agent import SelfCorrectingAgent
    from .tools.base import ToolRegistry
    from .tools.extraction import DEFAULT_TOOLS

    documents = _documents(args)
    faults = tuple(args.faults) if args.faults else tuple(TAXONOMY)
    plan = InjectionPlan(rate=1.0, faults=faults, seed=args.seed, max_per_run=1)
    injector = FaultInjector(plan, corpus=corpus_for_injector(documents))
    registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
    agent = SelfCorrectingAgent(
        MockProvider(vigilance=args.vigilance), registry,
        INVOICE_SCHEMA, INVOICE_CHECKS,
    )

    for document in documents[: args.n]:
        result = agent.run(document)
        score = score_run(result, document["truth"], INVOICE_SCHEMA)
        print(result.trace.render())
        print(
            f"\n  exact_match={score.exact_match}  "
            f"faults={score.faults_injected}  detected={score.faults_detected}  "
            f"silent_failure={score.silent_failure}"
        )
        wrong = [f for f in score.fields if not f.correct]
        for f in wrong:
            print(f"    {f.name}: got {f.predicted!r} want {f.expected!r}")
        print()
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    if args.action == "verify":
        ok = verify_corpus()
        print("corpus matches generator" if ok else "CORPUS DRIFT DETECTED")
        return 0 if ok else 1
    path = export_corpus(n=args.n, seed=args.data_seed)
    print(f"wrote {args.n} documents to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentaudit")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="run the fault-injection benchmark")
    bench.add_argument("--n", type=int, default=40)
    bench.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    bench.add_argument("--rate", type=float, default=0.85)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--data-seed", type=int, default=7)
    bench.add_argument("--vigilance", type=float, default=0.6)
    bench.add_argument("--max-corrections", type=int, default=2)
    bench.add_argument("--no-corroborate", action="store_true")
    bench.add_argument("--no-judge", action="store_true")
    bench.add_argument("--ablation", action="store_true")
    bench.add_argument("--by-fault", action="store_true",
                       help="one condition per fault instead of per layer")
    bench.add_argument("--generate", action="store_true",
                       help="regenerate the corpus instead of loading benchmarks/")
    bench.add_argument("--out")
    bench.set_defaults(func=cmd_bench)

    faults = sub.add_parser("faults", help="list the fault taxonomy")
    faults.set_defaults(func=cmd_faults)

    corpus = sub.add_parser("corpus", help="export or verify the shipped dataset")
    corpus.add_argument("action", choices=["export", "verify"])
    corpus.add_argument("--n", type=int, default=100)
    corpus.add_argument("--data-seed", type=int, default=7)
    corpus.set_defaults(func=cmd_corpus)

    trace = sub.add_parser("trace", help="render traces for individual runs")
    trace.add_argument("--n", type=int, default=1)
    trace.add_argument("--faults", nargs="*", default=[])
    trace.add_argument("--seed", type=int, default=0)
    trace.add_argument("--data-seed", type=int, default=7)
    trace.add_argument("--vigilance", type=float, default=0.6)
    trace.add_argument("--generate", action="store_true")
    trace.set_defaults(func=cmd_trace)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
