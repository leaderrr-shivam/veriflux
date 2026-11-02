"""Abstract base interface for LLM inference providers.

Establishes provider contract for chat completions, embeddings, and telemetry collection.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from veriflux._types import TokenUsage


@dataclass(slots=True)
class ProviderResponse:
    """Standardized completion response from any model provider.

    Attributes:
        content: The primary text output.
        model: Exact model identifier reported.
        tokens: Token consumption metrics.
        latency_ms: Round-trip request latency in milliseconds.
        finish_reason: Reason generation stopped ('stop', 'length', etc.).
        raw_response: Full unmodified provider JSON payload.
    """

    content: str
    model: str
    tokens: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw_response: Dict[str, Any] = field(default_factory=dict)


class Provider(abc.ABC):
    """Abstract base class for all LLM and embedding service providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'openai', 'mock', 'ollama')."""
        ...

    @abc.abstractmethod
    def complete(self, messages: Sequence[Dict[str, Any]], **kwargs: Any) -> ProviderResponse:
        """Execute chat completion over conversation messages."""
        ...

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate vector embeddings for input texts."""
        ...

