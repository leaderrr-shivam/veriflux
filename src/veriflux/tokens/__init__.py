"""Token management engine for Veriflux.

Provides fast token counting, budget allocation, estimation, and prompt optimization.
"""
from __future__ import annotations

from veriflux.tokens.budget_manager import (
    BudgetManager,
    BudgetSection,
    SectionPriority,
    TokenBudget,
)
from veriflux.tokens.counter import count_tokens, count_tokens_batch, estimate_tokens
from veriflux.tokens.encodings import (
    ENCODING_MAP,
    MODEL_PRICING,
    MODEL_TOKEN_LIMITS,
    get_encoding_for_model,
    get_pricing,
    get_token_limit,
)
from veriflux.tokens.estimator import FastEstimator
from veriflux.tokens.optimizer import PromptOptimizer

__all__ = [
    # Encodings & limits
    "ENCODING_MAP",
    "MODEL_TOKEN_LIMITS",
    "MODEL_PRICING",
    "get_encoding_for_model",
    "get_token_limit",
    "get_pricing",
    # Counting
    "count_tokens",
    "count_tokens_batch",
    "estimate_tokens",
    # Estimation
    "FastEstimator",
    # Budgeting
    "TokenBudget",
    "BudgetSection",
    "SectionPriority",
    "BudgetManager",
    # Optimization
    "PromptOptimizer",
]

