"""Unit tests for Context Engineering Engine."""
from __future__ import annotations

import pytest
from veriflux.context import (
    ContextCompressor,
    ContextEngine,
    Deduplicator,
    SlidingWindow,
    Source,
    TokenBudget,
)


def test_context_engine_assembly() -> None:
    engine = ContextEngine(
        budget=TokenBudget(max_tokens=500, reserve_output=100),
        strategy="task_aware",
    )

    engine.add(Source.system("You are an expert aviation analyst.", priority="critical"))
    engine.add(Source.history([{"role": "user", "content": "Tell me about Apollo."}], priority="low"))
    engine.add(Source.rag([
        "Apollo 11 landed on the Moon in 1969.",
        "Neil Armstrong took one small step for man.",
    ], priority="high"))

    result = engine.build(query="When did Apollo 11 land?")
    assert result.token_count > 0
    assert result.sources_retained >= 2
    assert "Apollo 11" in result.text
    assert "expert aviation analyst" in result.text


def test_context_deduplicator() -> None:
    dedup = Deduplicator(threshold=0.8)
    chunks = [
        "The quick brown fox jumps over the lazy dog in summer",
        "The quick brown fox jumps over the lazy dog in summer",  # Duplicate
        "An entirely unrelated document concerning deep neural networks",
    ]
    unique = dedup.deduplicate(chunks)
    assert len(unique) == 2


def test_sliding_window() -> None:
    window = SlidingWindow(max_tokens=30, overlap_tokens=5)
    window.add("First sentence added to window.")
    window.add("Second sentence added to window.")
    assert len(window.get()) > 0

