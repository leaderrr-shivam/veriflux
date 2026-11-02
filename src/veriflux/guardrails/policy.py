"""Declarative policy definition and governance.

Enables compliance enforcement against structured guardrail policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True, frozen=True)
class PolicyResult:
    """Outcome of evaluating content against a governance policy."""

    compliant: bool
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class Policy:
    """Represents a set of safety, privacy, and quality standards."""

    def __init__(
        self,
        name: str = "default_policy",
        block_pii: bool = True,
        max_toxicity: float = 0.5,
        block_prompt_injection: bool = True,
        pii_categories: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.block_pii = block_pii
        self.max_toxicity = max_toxicity
        self.block_prompt_injection = block_prompt_injection
        self.pii_categories = pii_categories

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Policy:
        """Instantiate a Policy from configuration dictionary."""
        return cls(
            name=data.get("name", "custom_policy"),
            block_pii=data.get("block_pii", True),
            max_toxicity=float(data.get("max_toxicity", 0.5)),
            block_prompt_injection=data.get("block_prompt_injection", True),
            pii_categories=data.get("pii_categories"),
        )

