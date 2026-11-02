"""Unit tests for Token Management Engine."""
from __future__ import annotations

import pytest
from veriflux.tokens import (
    BudgetManager,
    BudgetSection,
    FastEstimator,
    PromptOptimizer,
    SectionPriority,
    TokenBudget,
    count_tokens,
    get_pricing,
    get_token_limit,
)


def test_token_counting() -> None:
    text = "Hello world! This is a test of the Veriflux token counter."
    toks = count_tokens(text, model="gpt-4o")
    assert 10 <= toks <= 16

    assert get_token_limit("gpt-4o") == 128000
    price_in, price_out = get_pricing("gpt-4o")
    assert price_in > 0.0 and price_out > 0.0


def test_fast_estimator_messages() -> None:
    estimator = FastEstimator()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in one sentence."},
    ]
    toks = estimator.estimate_messages(messages, model="gpt-4o")
    assert toks > 15


def test_budget_manager() -> None:
    budget = TokenBudget(max_tokens=200, reserve_output=50)
    manager = BudgetManager(budget=budget)

    sections = {
        "system": BudgetSection("system", "System instruction.", priority=SectionPriority.CRITICAL),
        "history": BudgetSection("history", "Lots of long conversation " * 10, priority=SectionPriority.LOW),
        "rag": BudgetSection("rag", "Important retrieved reference document content.", priority=SectionPriority.HIGH),
    }

    allocations = manager.allocate(sections)
    assert allocations["system"] > 0
    assert allocations["rag"] > 0
    total_alloc = sum(allocations.values())
    assert total_alloc <= budget.available_input_tokens


def test_prompt_optimizer() -> None:
    optimizer = PromptOptimizer()
    verbose_prompt = (
        "Please make sure to note that in order to achieve high accuracy, "
        "as a matter of fact, you must strictly follow the rules."
    )
    optimized = optimizer.optimize_prompt(verbose_prompt)
    assert len(optimized) < len(verbose_prompt)
    assert optimizer.compression_ratio(verbose_prompt, optimized) >= 1.0

