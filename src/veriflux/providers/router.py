"""Intelligent provider routing, load balancing, and failover.

Routes requests across providers using round-robin, fallback, or weighted strategies.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from veriflux.exceptions import ProviderError
from veriflux.providers.base import Provider, ProviderResponse


class ProviderRouter(Provider):
    """Routes completion requests dynamically across multiple underlying providers."""

    def __init__(self, strategy: str = "fallback") -> None:
        self.strategy = strategy  # 'fallback', 'round_robin', 'weighted'
        self.providers: List[tuple[str, Provider, float]] = []
        self._rr_index = 0

    @property
    def name(self) -> str:
        return f"router:{self.strategy}"

    def add_provider(self, name: str, provider: Provider, weight: float = 1.0) -> ProviderRouter:
        """Register a provider with an associated routing weight."""
        self.providers.append((name, provider, weight))
        return self

    def complete(self, messages: Sequence[Dict[str, Any]], **kwargs: Any) -> ProviderResponse:
        """Route completion according to configured routing strategy."""
        if not self.providers:
            raise ProviderError("No providers registered in ProviderRouter")

        if self.strategy == "fallback":
            last_err: Optional[Exception] = None
            for name, prov, _ in self.providers:
                try:
                    return prov.complete(messages, **kwargs)
                except Exception as exc:
                    last_err = exc
                    continue
            raise ProviderError(f"All providers in fallback chain failed: {last_err}")

        elif self.strategy == "round_robin":
            _, prov, _ = self.providers[self._rr_index % len(self.providers)]
            self._rr_index += 1
            return prov.complete(messages, **kwargs)

        elif self.strategy == "weighted":
            weights = [w for _, _, w in self.providers]
            chosen = random.choices(self.providers, weights=weights, k=1)[0]
            return chosen[1].complete(messages, **kwargs)

        # Default fallback to first provider
        return self.providers[0][1].complete(messages, **kwargs)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Delegate embeddings to first viable provider in registry."""
        if not self.providers:
            raise ProviderError("No providers registered in ProviderRouter")
        for _, prov, _ in self.providers:
            try:
                return prov.embed(texts)
            except Exception:
                continue
        raise ProviderError("All providers failed to embed texts")

