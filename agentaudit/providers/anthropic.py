"""Anthropic provider.

Optional. The harness is fully exercisable with MockProvider; this is for
when you want real numbers. Kept deliberately thin — retries and backoff
belong to the caller, because silently retrying inside the provider would
contaminate transport-fault measurements.
"""

from __future__ import annotations

import os

from .base import Completion, Provider


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pip install anthropic  # or use MockProvider for zero-cost runs"
            ) from exc
        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("set ANTHROPIC_API_KEY or pass api_key=")
        self.model = model
        self._client = Anthropic(api_key=key)

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
        )
