"""Deterministic mock provider for unit testing and CI test gates.

Emulates LLM chat completions and embeddings without network requests or API costs.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from veriflux._types import TokenUsage
from veriflux.providers.base import Provider, ProviderResponse
from veriflux.tokens.counter import count_tokens


class MockProvider(Provider):
    """Deterministic, zero-cost mock provider for testing and validation pipelines."""

    def __init__(
        self,
        responses: Optional[Sequence[str]] = None,
        default_response: str = "Mock generated response",
        latency_ms: float = 0.0,
    ) -> None:
        self.responses = list(responses) if responses else []
        self.default_response = default_response
        self.latency_ms = latency_ms
        self._index = 0
        self.call_count = 0
        self.last_messages: List[Dict[str, Any]] = []
        self._error_trigger: Optional[Exception] = None

    @property
    def name(self) -> str:
        return "mock"

    def set_error(self, error: Exception) -> None:
        """Configure mock to raise an error on next call."""
        self._error_trigger = error

    def reset(self) -> None:
        """Reset call count, index, and state."""
        self._index = 0
        self.call_count = 0
        self.last_messages.clear()
        self._error_trigger = None

    def complete(self, messages: Sequence[Dict[str, Any]], **kwargs: Any) -> ProviderResponse:
        """Return next canned response."""
        self.call_count += 1
        self.last_messages = list(messages)

        if self._error_trigger is not None:
            err = self._error_trigger
            self._error_trigger = None
            raise err

        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        if self.responses:
            content = self.responses[self._index % len(self.responses)]
            self._index += 1
        else:
            content = self.default_response

        # Estimate token usage
        in_toks = sum(count_tokens(str(m.get("content", ""))) for m in messages)
        out_toks = count_tokens(content)

        return ProviderResponse(
            content=content,
            model="mock-v1",
            tokens=TokenUsage(input_tokens=in_toks, output_tokens=out_toks, total_tokens=in_toks + out_toks),
            latency_ms=self.latency_ms,
            finish_reason="stop",
            raw_response={"mock": True},
        )

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return deterministic synthetic embeddings of dimension 64."""
        results: List[List[float]] = []
        for text in texts:
            # Deterministic hash-based mock embedding
            h = hash(text)
            vec = [float(((h >> (i % 32)) & 0xFF) / 255.0) for i in range(64)]
            results.append(vec)
        return results

