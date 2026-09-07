"""Deterministic mock provider.

This exists so the full harness runs end to end with no API key and no
spend. It is not a stand-in for a real model's judgement — it is a
control. Because its behaviour is fixed and inspectable, any change in
benchmark numbers between two runs against the mock is a bug in the
harness, not model variance. Run the mock first; it is how you tell your
own bugs apart from the model's failures.

``vigilance`` controls how hard the simulated agent looks for problems,
which lets you sanity-check that the metrics respond in the expected
direction before spending anything on a real model.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .base import Completion, Provider


class MockProvider(Provider):
    name = "mock"

    def __init__(self, vigilance: float = 0.6, seed: int = 0):
        if not 0.0 <= vigilance <= 1.0:
            raise ValueError("vigilance must be in [0, 1]")
        self.vigilance = vigilance
        self.seed = seed
        self.calls = 0

    def _hash_unit(self, text: str) -> float:
        digest = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
        return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> Completion:
        self.calls += 1

        if "JUDGE" in system:
            return self._judge(prompt)
        if "REVIEW" in system:
            return self._review(prompt)
        return self._extract(prompt)

    # -- behaviours ------------------------------------------------------

    def _extract(self, prompt: str) -> Completion:
        """Echo the tool evidence back as the extraction.

        A real model reads the document; the mock trusts its tools
        completely. That is the point — it isolates tool-layer faults from
        model-layer reasoning.
        """
        payload = self._evidence(prompt)
        return Completion(json.dumps(payload), model="mock-extract")

    def _review(self, prompt: str) -> Completion:
        """Decide whether the candidate record looks wrong.

        Structural problems are always caught (a parser would catch them).
        Consistency problems are caught in proportion to vigilance.
        Corroborative and opaque problems are never caught here, because
        catching them requires evidence this step does not have.
        """
        errors = re.findall(r"^\s*-\s*(.+)$", prompt, re.MULTILINE)
        if errors:
            return Completion(
                json.dumps({"verdict": "reject", "reasons": errors[:5]}),
                model="mock-review",
            )
        roll = self._hash_unit(prompt)
        if roll < self.vigilance * 0.25:
            return Completion(
                json.dumps({
                    "verdict": "reject",
                    "reasons": ["values do not corroborate against source text"],
                }),
                model="mock-review",
            )
        return Completion(json.dumps({"verdict": "accept", "reasons": []}),
                          model="mock-review")

    def _judge(self, prompt: str) -> Completion:
        """Semantic equivalence check. Conservative: exact-ish or nothing."""
        pred = re.search(r"PREDICTED:\s*(.*)", prompt)
        truth = re.search(r"REFERENCE:\s*(.*)", prompt)
        if not pred or not truth:
            return Completion(json.dumps({"equivalent": False}), model="mock-judge")
        a = pred.group(1).strip().casefold().rstrip(".")
        b = truth.group(1).strip().casefold().rstrip(".")
        equivalent = a == b or a.replace(" ", "") == b.replace(" ", "")
        return Completion(json.dumps({"equivalent": equivalent}), model="mock-judge")

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _evidence(prompt: str) -> dict[str, Any]:
        marker = prompt.find("EVIDENCE:")
        if marker == -1:
            return {}
        start = prompt.find("{", marker)
        if start == -1:
            return {}
        # Brace-match rather than regex: the payload can legitimately contain
        # nested objects, and a non-greedy pattern truncates them.
        depth = 0
        for i in range(start, len(prompt)):
            if prompt[i] == "{":
                depth += 1
            elif prompt[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(prompt[start : i + 1])
                    except json.JSONDecodeError:
                        return {}
                    return {k: v for k, v in data.items()
                            if not str(k).startswith("__")}
        return {}
