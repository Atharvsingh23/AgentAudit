"""Deterministic fault injection.

Design constraints, in priority order:

1. Reproducible. Given the same seed, run plan and dataset, the exact same
   faults land on the exact same calls. A benchmark whose numbers move
   between runs cannot support a claim.
2. Invisible to the agent. Injected faults are recorded on the span, which
   the agent never reads. If the agent could inspect its own trace, every
   detection number would be worthless.
3. Value-preserving where it should be. A transport fault raises; it does
   not quietly corrupt data. Conflating the two makes recovery rate
   uninterpretable.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .taxonomy import FaultLayer, FaultSpec, spec


class ToolFailure(RuntimeError):
    """Raised by the injector to simulate a transport-layer failure."""

    def __init__(self, fault_key: str, message: str, retryable: bool = True):
        super().__init__(message)
        self.fault_key = fault_key
        self.retryable = retryable


@dataclass
class InjectionPlan:
    """Which faults to inject, and how often.

    ``rate`` is the probability that any single eligible tool call is
    faulted. ``faults`` restricts the pool. ``target_tools`` restricts
    which tools are eligible; empty means all.
    """

    rate: float = 0.0
    faults: tuple[str, ...] = ()
    target_tools: tuple[str, ...] = ()
    seed: int = 0
    max_per_run: int = 2

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "rate": self.rate,
                "faults": sorted(self.faults),
                "target_tools": sorted(self.target_tools),
                "seed": self.seed,
                "max_per_run": self.max_per_run,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class FaultInjector:
    """Wraps tool outputs and decides, deterministically, what to break.
    
    The three design constraints in the module docstring are load-bearing.
    Reproducibility is enforced by seeding per-document; invisibility is
    enforced by keeping fault metadata off the agent's path; value-preservation
    is enforced by distinguishing transport failures (raise) from data
    corruptions (mutate). Break any one of these and the benchmark loses its claim."""

    def __init__(self, plan: InjectionPlan, corpus: list[dict[str, Any]] | None = None):
        self.plan = plan
        # Corpus of other documents, used by contextual faults that need a
        # plausible wrong answer rather than a random one.
        self.corpus = corpus or []
        self._injected_this_run = 0
        self._rng = random.Random()
        self._last_payloads: list[dict[str, Any]] = []

    # -- lifecycle -------------------------------------------------------

    def begin_run(self, document_id: str) -> None:
        """Reseed per document so runs are independent but reproducible."""
        key = f"{self.plan.seed}:{self.plan.fingerprint()}:{document_id}"
        digest = hashlib.sha256(key.encode()).digest()
        self._rng = random.Random(int.from_bytes(digest[:8], "big"))
        self._injected_this_run = 0

    def _eligible(self, tool_name: str) -> bool:
        if self.plan.rate <= 0 or not self.plan.faults:
            return False
        if self._injected_this_run >= self.plan.max_per_run:
            return False
        if self.plan.target_tools and tool_name not in self.plan.target_tools:
            return False
        return True

    def decide(self, tool_name: str, call_index: int) -> FaultSpec | None:
        if not self._eligible(tool_name):
            return None
        if self._rng.random() >= self.plan.rate:
            return None
        self._injected_this_run += 1
        return spec(self._rng.choice(list(self.plan.faults)))

    # -- application -----------------------------------------------------

    def apply(
        self,
        fault: FaultSpec,
        payload: dict[str, Any],
        document_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (corrupted_payload, detail).

        Raises ToolFailure for transport faults.
        """
        if fault.layer is FaultLayer.TRANSPORT:
            self._raise_transport(fault)

        handler = getattr(self, f"_apply_{fault.key}", None)
        if handler is None:
            raise NotImplementedError(f"no handler for fault {fault.key}")

        corrupted, detail = handler(dict(payload), document_id)
        self._last_payloads.append(payload)
        return corrupted, detail

    def _raise_transport(self, fault: FaultSpec) -> None:
        messages = {
            "timeout": ("deadline exceeded after 30000ms", True),
            "rate_limit": ("429 too many requests; retry after 2s", True),
            "transient_error": ("503 upstream unavailable", True),
        }
        msg, retryable = messages[fault.key]
        raise ToolFailure(fault.key, msg, retryable=retryable)

    # -- structural ------------------------------------------------------

    def _apply_malformed_json(self, payload, _doc):
        raw = json.dumps(payload)
        cut = self._rng.randint(len(raw) // 3, max(len(raw) // 3 + 1, len(raw) - 2))
        return {"__raw__": raw[:cut]}, {"truncated_at": cut, "original_len": len(raw)}

    def _apply_schema_drift(self, payload, _doc):
        keys = [k for k in payload if not k.startswith("__")]
        if not keys:
            return payload, {}
        victim = self._rng.choice(keys)
        renamed = f"{victim}_v2"
        payload[renamed] = payload.pop(victim)
        return payload, {"renamed": {victim: renamed}}

    def _apply_type_violation(self, payload, _doc):
        numeric = [
            k for k, v in payload.items()
            if not k.startswith("__") and re.fullmatch(r"-?[\d,.]+", str(v) or "")
        ]
        if not numeric:
            return payload, {}
        victim = self._rng.choice(numeric)
        original = payload[victim]
        payload[victim] = f"approximately {original} (see attached schedule)"
        return payload, {"field": victim, "original": original}

    def _apply_field_dropped(self, payload, _doc):
        keys = [k for k in payload if not k.startswith("__")]
        if not keys:
            return payload, {}
        victim = self._rng.choice(keys)
        original = payload.pop(victim)
        return payload, {"field": victim, "original": original}

    # -- semantic --------------------------------------------------------

    def _numeric_fields(self, payload) -> list[str]:
        out = []
        for k, v in payload.items():
            if k.startswith("__"):
                continue
            if re.fullmatch(r"-?\d+(\.\d+)?", str(v).replace(",", "")):
                out.append(k)
        return out

    def _apply_digit_transposition(self, payload, _doc):
        candidates = self._numeric_fields(payload)
        if not candidates:
            return payload, {}
        victim = self._rng.choice(candidates)
        original = str(payload[victim])
        digits = [i for i, c in enumerate(original) if c.isdigit()]
        if len(digits) < 2:
            return payload, {}
        i = self._rng.randrange(len(digits) - 1)
        a, b = digits[i], digits[i + 1]
        chars = list(original)
        if chars[a] == chars[b]:
            chars[b] = str((int(chars[b]) + 1) % 10)
        else:
            chars[a], chars[b] = chars[b], chars[a]
        payload[victim] = "".join(chars)
        return payload, {"field": victim, "original": original, "corrupted": payload[victim]}

    def _apply_magnitude_shift(self, payload, _doc):
        candidates = self._numeric_fields(payload)
        if not candidates:
            return payload, {}
        victim = self._rng.choice(candidates)
        original = payload[victim]
        try:
            shifted = Decimal(str(original).replace(",", "")) * self._rng.choice(
                [Decimal("10"), Decimal("0.1")]
            )
        except Exception:
            return payload, {}
        payload[victim] = str(shifted.quantize(Decimal("0.01")))
        return payload, {"field": victim, "original": original, "corrupted": payload[victim]}

    def _apply_unit_mismatch(self, payload, _doc):
        if "currency" not in payload:
            return payload, {}
        original = payload["currency"]
        alternatives = [c for c in ("USD", "EUR", "GBP", "INR") if c != original]
        payload["currency"] = self._rng.choice(alternatives)
        return payload, {"field": "currency", "original": original,
                         "corrupted": payload["currency"]}

    def _apply_plausible_substitution(self, payload, document_id):
        """The dangerous one: swap in a value from another real document.

        Nothing about the resulting record is internally inconsistent. Only
        re-reading the source can catch it.
        """
        keys = [k for k in payload if not k.startswith("__") and k != "currency"]
        others = [d for d in self.corpus if d.get("document_id") != document_id]
        if not keys or not others:
            return payload, {}
        donor = self._rng.choice(others)
        shared = [k for k in keys if k in donor and donor[k] != payload.get(k)]
        if not shared:
            return payload, {}
        victim = self._rng.choice(shared)
        original = payload[victim]
        payload[victim] = donor[victim]
        return payload, {"field": victim, "original": original,
                         "corrupted": payload[victim], "donor": donor.get("document_id")}

    # -- contextual ------------------------------------------------------

    def _apply_stale_cache(self, payload, document_id):
        if not self._last_payloads:
            return payload, {}
        stale = dict(self._last_payloads[-1])
        return stale, {"served_from": "previous_call"}

    def _apply_cross_document_bleed(self, payload, document_id):
        others = [d for d in self.corpus if d.get("document_id") != document_id]
        if not others:
            return payload, {}
        donor = self._rng.choice(others)
        keys = [k for k in payload if not k.startswith("__") and k in donor]
        if not keys:
            return payload, {}
        n = max(1, len(keys) // 3)
        victims = self._rng.sample(keys, n)
        before = {k: payload[k] for k in victims}
        for k in victims:
            payload[k] = donor[k]
        return payload, {"fields": victims, "original": before,
                         "donor": donor.get("document_id")}


NO_FAULTS = InjectionPlan(rate=0.0)
