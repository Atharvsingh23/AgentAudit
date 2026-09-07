"""Anthropic provider.

Optional. The harness is fully exercisable with MockProvider; this is for
when you want real numbers. Kept deliberately thin — retries and backoff
belong to the caller, because silently retrying inside the provider would
contaminate transport-fault measurements.
"""

from __future__ import annotations

import os
from typing import Any

from .base import Completion, Provider

DEFAULT_MODEL = "claude-opus-5"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        effort: str | None = None,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pip install anthropic  # or use MockProvider for zero-cost runs"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("set ANTHROPIC_API_KEY or pass api_key=")
        self.model = model
        # low | medium | high | xhigh | max. Left unset (server default:
        # high) so a run's cost profile is an explicit choice, not a hidden
        # one baked into the harness.
        self.effort = effort
        self._client = Anthropic(api_key=key)

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if temperature is not None:
            # Current models reject `temperature` outright (400), so it is
            # only forwarded when a caller explicitly asks for it — which
            # only makes sense against an older model id.
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)
        # Thinking blocks are skipped: only the text is the answer.
        text = "".join(b.text for b in response.content if b.type == "text")
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
            # A truncated or refused response is a fact about the run, not
            # an exception: it has to reach the trace, or a benchmark number
            # quietly absorbs it as if the model had simply answered badly.
            stop_reason=response.stop_reason or "",
        )
