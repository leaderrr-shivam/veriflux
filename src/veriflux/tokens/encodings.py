"""Model encoding specifications, context limits, and pricing metadata.

Contains up-to-date mappings for OpenAI, Anthropic, Google, and open models.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# Mapping of model names to encoding names
ENCODING_MAP: Dict[str, str] = {
    # OpenAI
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    # Anthropic
    "claude-3-5-sonnet": "claude_v3",
    "claude-3-5-sonnet-20241022": "claude_v3",
    "claude-3-opus": "claude_v3",
    "claude-3-sonnet": "claude_v3",
    "claude-3-haiku": "claude_v3",
    # Google
    "gemini-1.5-pro": "gemini_v1",
    "gemini-1.5-flash": "gemini_v1",
    "gemini-2.0-flash": "gemini_v1",
    "gemini-pro": "gemini_v1",
    # Open weights
    "llama-3": "llama3",
    "llama-3.1": "llama3",
    "mistral": "mistral",
}

# Maximum context windows (tokens)
MODEL_TOKEN_LIMITS: Dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "o3": 200000,
    "o3-mini": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "gemini-1.5-pro": 2000000,
    "gemini-1.5-flash": 1000000,
    "gemini-2.0-flash": 1048576,
    "llama-3": 8192,
    "llama-3.1": 128000,
}

# Cost per 1,000 tokens (USD): (input_cost, output_cost)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.010, 0.030),
    "gpt-4": (0.030, 0.060),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o1": (0.015, 0.060),
    "o1-mini": (0.003, 0.012),
    "o3": (0.015, 0.060),
    "o3-mini": (0.0011, 0.0044),
    "text-embedding-3-small": (0.00002, 0.0),
    "text-embedding-3-large": (0.00013, 0.0),
    # Anthropic
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-haiku": (0.00025, 0.00125),
    # Google
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-2.0-flash": (0.0001, 0.0004),
    # Local / default
    "default": (0.001, 0.002),
}


def get_encoding_for_model(model: str) -> str:
    """Return encoding family name for a given model."""
    normalized = model.lower().strip()
    return ENCODING_MAP.get(normalized, "cl100k_base")


def get_token_limit(model: str) -> int:
    """Return maximum context token length for a model."""
    normalized = model.lower().strip()
    return MODEL_TOKEN_LIMITS.get(normalized, 8192)


def get_pricing(model: str) -> Tuple[float, float]:
    """Return (input_price_per_1k, output_price_per_1k) in USD."""
    normalized = model.lower().strip()
    return MODEL_PRICING.get(normalized, MODEL_PRICING["default"])

