"""Model provider interface.

The harness is provider-agnostic on purpose. Reliability results that only
reproduce on one vendor's endpoint are not results.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    def as_json(self) -> dict[str, Any]:
        """Parse a JSON object out of the response, tolerating code fences."""
        cleaned = self.text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        return json.loads(cleaned)


class Provider(ABC):
    name: str = "provider"

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        ...
