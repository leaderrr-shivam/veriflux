"""Multi-provider cost attribution and calculation.

Calculates fine-grained USD expenditures based on input/output token counts
and published provider pricing schedules.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from veriflux.tokens.encodings import get_pricing


@dataclass(slots=True, frozen=True)
class CostResult:
    """Detailed cost breakdown for a model invocation.

    Attributes:
        input_cost: Dollar cost for prompt / input tokens.
        output_cost: Dollar cost for completion / output tokens.
        total_cost: Total combined cost.
        currency: Currency symbol or code ('USD').
        model: Target model name.
        input_tokens: Number of prompt tokens.
        output_tokens: Number of generated tokens.
    """

    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    model: str = "default"
    input_tokens: int = 0
    output_tokens: int = 0


class CostCalculator:
    """Calculates single-call and cumulative financial expenditures."""

    def __init__(self) -> None:
        self._cumulative_cost = 0.0
        self._lock = threading.Lock()

    def calculate(self, model: str, input_tokens: int, output_tokens: int) -> CostResult:
        """Compute cost for a model invocation."""
        price_in_1k, price_out_1k = get_pricing(model)

        in_cost = (input_tokens / 1000.0) * price_in_1k
        out_cost = (output_tokens / 1000.0) * price_out_1k
        total = in_cost + out_cost

        with self._lock:
            self._cumulative_cost += total

        return CostResult(
            input_cost=round(in_cost, 6),
            output_cost=round(out_cost, 6),
            total_cost=round(total, 6),
            currency="USD",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @property
    def cumulative_cost(self) -> float:
        """Total cumulative cost tracked across all calculations."""
        with self._lock:
            return round(self._cumulative_cost, 6)

    def reset(self) -> None:
        """Reset cumulative expenditure tracking."""
        with self._lock:
            self._cumulative_cost = 0.0

    @staticmethod
    def format_cost(cost: float) -> str:
        """Format dollar amounts nicely (e.g. '$0.0024' or '<$0.0001')."""
        if cost == 0.0:
            return "$0.00"
        if cost < 0.0001:
            return "<$0.0001"
        return f"${cost:.4f}"

