"""Experiment runner.

A condition is a named fault plan. Every condition runs against the same
documents with the same seed, so differences between conditions are
attributable to the faults rather than to sampling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent import SelfCorrectingAgent
from .dataset import corpus_for_injector
from .faults.injector import FaultInjector, InjectionPlan
from .faults.taxonomy import FaultLayer, TAXONOMY
from .judge import SemanticJudge
from .metrics import Report, score_run
from .providers.base import Provider
from .schema import ConsistencyCheck, Schema
from .tools.base import ToolRegistry
from .tools.extraction import DEFAULT_TOOLS


@dataclass
class Condition:
    name: str
    plan: InjectionPlan


def standard_conditions(seed: int = 0, rate: float = 0.85) -> list[Condition]:
    """One clean baseline plus one condition per fault layer.

    Stratifying by layer rather than pooling all faults is deliberate: a
    pooled recovery number is dominated by whichever layer happens to be
    most represented in the pool. This emerged from pilot runs where pooled
    recovery rates were artificially high when structural faults dominated
    (they're easier to catch). By separating layers, we measure robustness
    *given the fault type*, not conflate it with fault frequency.
    """
    conditions = [Condition("clean", InjectionPlan(rate=0.0, seed=seed))]
    for layer in FaultLayer:
        keys = tuple(k for k, s in TAXONOMY.items() if s.layer is layer)
        conditions.append(
            Condition(
                layer.value,
                InjectionPlan(rate=rate, faults=keys, seed=seed, max_per_run=1),
            )
        )
    return conditions


@dataclass
class ExperimentResult:
    reports: list[Report] = field(default_factory=list)
    traces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reports": [r.to_dict() for r in self.reports],
            "traces": self.traces,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n"
        )
        return path


class Experiment:
    def __init__(
        self,
        provider_factory: Callable[[], Provider],
        schema: Schema,
        documents: list[dict[str, Any]],
        checks: tuple[ConsistencyCheck, ...] = (),
        corroborate: bool = True,
        max_corrections: int = 2,
        use_judge: bool = True,
        keep_traces: int = 2,
    ):
        self.provider_factory = provider_factory
        self.schema = schema
        self.documents = documents
        self.checks = checks
        self.corroborate = corroborate
        self.max_corrections = max_corrections
        self.use_judge = use_judge
        self.keep_traces = keep_traces

    def run(self, conditions: list[Condition]) -> ExperimentResult:
        outcome = ExperimentResult()
        donor_corpus = corpus_for_injector(self.documents)

        for condition in conditions:
            provider = self.provider_factory()
            injector = FaultInjector(condition.plan, corpus=donor_corpus)
            registry = ToolRegistry(DEFAULT_TOOLS, injector=injector)
            agent = SelfCorrectingAgent(
                provider=provider,
                registry=registry,
                schema=self.schema,
                checks=self.checks,
                max_corrections=self.max_corrections,
                corroborate=self.corroborate,
            )
            judge = SemanticJudge(provider, self.schema) if self.use_judge else None

            report = Report(condition=condition.name)
            kept: list[dict[str, Any]] = []

            for document in self.documents:
                result = agent.run(document)
                truth = document["truth"]

                rescued: set[str] = set()
                if judge is not None and result.succeeded:
                    rescued = judge.adjudicate(result.record, truth, result.trace)

                report.runs.append(
                    score_run(result, truth, self.schema, rescued=rescued)
                )

                if len(kept) < self.keep_traces and result.trace.faults_injected:
                    kept.append(result.trace.to_dict())

            outcome.reports.append(report)
            outcome.traces[condition.name] = kept

        return outcome


def by_fault_conditions(seed: int = 0, rate: float = 1.0) -> list[Condition]:
    """One condition per fault, named `<detectability>:<fault>`.

    The layer-level table pools faults that behave nothing alike — a
    dropped field and a plausible substitution are both "one fault" but
    only one of them is catchable without re-reading the source. This is
    the same argument the module makes against pooling *layers*, applied
    one level further down.
    """
    return [
        Condition(
            f"{fault.detectability.value[:5]}:{key}",
            InjectionPlan(rate=rate, faults=(key,), seed=seed, max_per_run=1),
        )
        for key, fault in TAXONOMY.items()
    ]


def ablation_conditions(seed: int = 0, rate: float = 0.85) -> list[Condition]:
    """Conditions for isolating which defence catches which fault layer."""
    semantic = tuple(
        k for k, s in TAXONOMY.items()
        if s.layer in (FaultLayer.SEMANTIC, FaultLayer.CONTEXTUAL)
    )
    return [
        Condition("clean", InjectionPlan(rate=0.0, seed=seed)),
        Condition(
            "semantic+contextual",
            InjectionPlan(rate=rate, faults=semantic, seed=seed, max_per_run=1),
        ),
    ]
