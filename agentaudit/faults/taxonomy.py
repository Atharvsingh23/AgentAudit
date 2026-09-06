"""Fault taxonomy for agentic pipelines.

Observability tools record what an agent did. This module defines what can
be done *to* an agent, so that recovery behaviour can be measured rather
than assumed.

Faults are organised along two axes:

  Layer      — where in the stack the fault originates
  Detectability — how much machinery the agent needs to notice it

The second axis is the one that matters. Structural faults are trivially
detectable with a JSON parser; a pipeline that only handles those looks
robust while remaining wide open to the semantic faults that cause real
financial errors. Reporting recovery rate without stratifying by
detectability is close to meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FaultLayer(str, Enum):
    TRANSPORT = "transport"      # the call itself fails or stalls
    STRUCTURAL = "structural"    # response shape is wrong
    SEMANTIC = "semantic"        # response is well-formed but untrue
    CONTEXTUAL = "contextual"    # response is true but not of this document


class Detectability(str, Enum):
    """How the fault is, in principle, catchable.

    MECHANICAL   — a parser or type check catches it with no reasoning.
    CONSISTENCY  — catchable only by cross-checking fields against each other.
    CORROBORATIVE — catchable only by re-deriving the value another way.
    OPAQUE       — not catchable from the response alone at any cost.
    
    This stratification emerged from real pipeline failures. A recovery strategy
    that handles MECHANICAL faults well but misses CORROBORATIVE ones looks
    great on the metric that mixes them together, and terrible when you stratify.
    This is why fault detection quality cannot be reported as a single number.
    """

    MECHANICAL = "mechanical"
    CONSISTENCY = "consistency"
    CORROBORATIVE = "corroborative"
    OPAQUE = "opaque"


@dataclass(frozen=True)
class FaultSpec:
    key: str
    layer: FaultLayer
    detectability: Detectability
    description: str
    # Whether the fault changes the value the agent would extract. Faults
    # that only slow things down (e.g. a retryable timeout) should not
    # count against extraction accuracy.
    corrupts_value: bool = True


TAXONOMY: dict[str, FaultSpec] = {
    # -- transport ------------------------------------------------------
    "timeout": FaultSpec(
        "timeout", FaultLayer.TRANSPORT, Detectability.MECHANICAL,
        "Tool call exceeds its deadline and raises.",
        corrupts_value=False,
    ),
    "rate_limit": FaultSpec(
        "rate_limit", FaultLayer.TRANSPORT, Detectability.MECHANICAL,
        "Tool returns a 429-equivalent error.",
        corrupts_value=False,
    ),
    "transient_error": FaultSpec(
        "transient_error", FaultLayer.TRANSPORT, Detectability.MECHANICAL,
        "Tool raises a 5xx-equivalent error that succeeds on retry.",
        corrupts_value=False,
    ),

    # -- structural -----------------------------------------------------
    "malformed_json": FaultSpec(
        "malformed_json", FaultLayer.STRUCTURAL, Detectability.MECHANICAL,
        "Response is truncated or syntactically invalid JSON.",
    ),
    "schema_drift": FaultSpec(
        "schema_drift", FaultLayer.STRUCTURAL, Detectability.MECHANICAL,
        "Field renamed or an unexpected field appears.",
    ),
    "type_violation": FaultSpec(
        "type_violation", FaultLayer.STRUCTURAL, Detectability.MECHANICAL,
        "Field carries a value of the wrong type, e.g. money as prose.",
    ),
    "field_dropped": FaultSpec(
        "field_dropped", FaultLayer.STRUCTURAL, Detectability.MECHANICAL,
        "A required field is silently omitted.",
    ),

    # -- semantic -------------------------------------------------------
    "digit_transposition": FaultSpec(
        "digit_transposition", FaultLayer.SEMANTIC, Detectability.CONSISTENCY,
        "Two digits swapped in a numeric field; total no longer reconciles.",
    ),
    "magnitude_shift": FaultSpec(
        "magnitude_shift", FaultLayer.SEMANTIC, Detectability.CONSISTENCY,
        "Value off by a factor of ten — decimal point misplaced.",
    ),
    "unit_mismatch": FaultSpec(
        "unit_mismatch", FaultLayer.SEMANTIC, Detectability.CONSISTENCY,
        "Correct magnitude reported under the wrong currency.",
    ),
    "plausible_substitution": FaultSpec(
        "plausible_substitution", FaultLayer.SEMANTIC, Detectability.CORROBORATIVE,
        "Value replaced with a different well-formed value that violates "
        "no internal constraint. The silent-failure case.",
    ),

    # -- contextual -----------------------------------------------------
    "stale_cache": FaultSpec(
        "stale_cache", FaultLayer.CONTEXTUAL, Detectability.CORROBORATIVE,
        "Tool returns a correct result for a previous document.",
    ),
    "cross_document_bleed": FaultSpec(
        "cross_document_bleed", FaultLayer.CONTEXTUAL, Detectability.OPAQUE,
        "A subset of fields is taken from a different document in the batch.",
    ),
}


def spec(key: str) -> FaultSpec:
    if key not in TAXONOMY:
        raise KeyError(
            f"unknown fault {key!r}; known: {sorted(TAXONOMY)}"
        )
    return TAXONOMY[key]


def by_layer(layer: FaultLayer) -> list[FaultSpec]:
    return [s for s in TAXONOMY.values() if s.layer is layer]


def by_detectability(level: Detectability) -> list[FaultSpec]:
    return [s for s in TAXONOMY.values() if s.detectability is level]


ALL_FAULTS: tuple[str, ...] = tuple(TAXONOMY)
