"""Performance, latency, cost, and token budget assertions.

Validates operational SLAs and cost caps for LLM pipelines.
"""
from __future__ import annotations


def latency_under(duration_ms: float, max_ms: float) -> bool:
    """Assert that execution duration in milliseconds is under maximum threshold."""
    return duration_ms <= max_ms


def cost_under(cost_usd: float, max_usd: float) -> bool:
    """Assert that invocation or pipeline dollar cost is within budget."""
    return cost_usd <= max_usd


def tokens_under(token_count: int, max_tokens: int) -> bool:
    """Assert that total token consumption is strictly within token limit."""
    return token_count <= max_tokens


def throughput_above(tokens_per_sec: float, min_tps: float) -> bool:
    """Assert that generation throughput in tokens per second meets minimum SLA."""
    return tokens_per_sec >= min_tps

