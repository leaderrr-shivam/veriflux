"""Dynamic token budget allocation across prompt sections.

Guarantees critical system constraints and proportionally distributes available
headroom to conversational history, retrieved context, and tool schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Dict, Optional

from veriflux.exceptions import TokenBudgetExceeded
from veriflux.tokens.counter import count_tokens


class SectionPriority(enum.Enum):
    """Priority tiers for prompt budget allocation."""

    CRITICAL = 4   # Never truncated (system instructions, safety rules)
    HIGH = 3       # User intent / primary query
    MEDIUM = 2     # Retrieved context (RAG)
    LOW = 1        # Conversation history / examples


@dataclass(slots=True)
class BudgetSection:
    """A segment of a prompt competing for token allocation."""

    name: str
    content: str
    priority: SectionPriority = SectionPriority.MEDIUM
    min_tokens: int = 0
    max_tokens: Optional[int] = None
    allocated_tokens: int = 0


@dataclass(slots=True, frozen=True)
class TokenBudget:
    """Global token budget constraints."""

    max_tokens: int = 8192
    reserve_output: int = 1024
    min_context: int = 512

    @property
    def available_input_tokens(self) -> int:
        """Net tokens available for prompt inputs."""
        return max(0, self.max_tokens - self.reserve_output)


class BudgetManager:
    """Manages multi-section prompt token allocation and truncation."""

    def __init__(
        self,
        budget: Optional[TokenBudget] = None,
        model: str = "gpt-4o",
        max_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
    ) -> None:
        if max_tokens is not None:
            self.budget = TokenBudget(max_tokens=max_tokens)
        else:
            self.budget = budget or TokenBudget()
        self.model = model
        self.max_cost_usd = max_cost_usd

    def fits(self, text: str, budget: int) -> bool:
        """Check if text fits within the given token budget."""
        return count_tokens(text, model=self.model) <= budget

    def truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text so its token count stays within the specified budget."""
        tokens = count_tokens(text, model=self.model)
        if tokens <= budget:
            return text

        if budget <= 0:
            return ""

        # Binary search on character truncation point
        low = 0
        high = len(text)
        best_str = ""

        while low <= high:
            mid = (low + high) // 2
            cand = text[:mid]
            c_tokens = count_tokens(cand, model=self.model)
            if c_tokens <= budget:
                best_str = cand
                low = mid + 1
            else:
                high = mid - 1

        return best_str

    def allocate(self, sections: Dict[str, BudgetSection]) -> Dict[str, int]:
        """Allocate token budgets across competing sections according to priority.

        Args:
            sections: Mapping of section name to BudgetSection definition.

        Returns:
            Mapping of section name to allocated token count.

        Raises:
            TokenBudgetExceeded: If minimum critical tokens exceed total budget.
        """
        available = self.budget.available_input_tokens
        allocations: Dict[str, int] = {k: 0 for k in sections}

        # 1. Measure raw token demands
        raw_demands: Dict[str, int] = {}
        for name, sec in sections.items():
            toks = count_tokens(sec.content, model=self.model)
            if sec.max_tokens is not None:
                toks = min(toks, sec.max_tokens)
            raw_demands[name] = toks

        # 2. Guarantee min_tokens in order of priority
        sorted_by_priority = sorted(sections.values(), key=lambda s: s.priority.value, reverse=True)
        for sec in sorted_by_priority:
            guaranteed = min(sec.min_tokens, raw_demands[sec.name])
            if available < guaranteed:
                if sec.priority == SectionPriority.CRITICAL:
                    raise TokenBudgetExceeded(
                        f"Cannot satisfy minimum budget for critical section '{sec.name}'. "
                        f"Needed {guaranteed}, only {available} available."
                    )
                guaranteed = available

            allocations[sec.name] = guaranteed
            available -= guaranteed

        # 3. Distribute remaining tokens proportionally by priority weight
        while available > 0:
            unsatisfied = [
                s for s in sorted_by_priority
                if allocations[s.name] < raw_demands[s.name]
            ]
            if not unsatisfied:
                break

            total_weight = sum(s.priority.value for s in unsatisfied)
            if total_weight == 0:
                break

            allocated_in_round = 0
            for sec in unsatisfied:
                needed = raw_demands[sec.name] - allocations[sec.name]
                share = max(1, int(available * (sec.priority.value / total_weight)))
                add = min(needed, share, available)
                allocations[sec.name] += add
                available -= add
                allocated_in_round += add
                if available == 0:
                    break

            if allocated_in_round == 0:
                break

        return allocations


# Ergonomic alias
TokenBudgetManager = BudgetManager

