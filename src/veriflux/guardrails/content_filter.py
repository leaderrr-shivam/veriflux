"""Configurable content filtering rules and bias detection.

Provides enterprise rule-based filtering (ALLOW, WARN, BLOCK) and lexical bias heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
import re
from typing import Any, Dict, List, Optional, Sequence


class FilterAction(enum.Enum):
    """Enforcement action for a filter rule match."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass(slots=True, frozen=True)
class FilterRule:
    """An individual custom content filtering rule."""

    name: str
    pattern: str
    action: FilterAction = FilterAction.BLOCK
    severity: str = "HIGH"
    description: str = ""


@dataclass(slots=True, frozen=True)
class FilterResult:
    """Outcome of content filtering."""

    passed: bool
    action: FilterAction
    violations: List[FilterRule] = field(default_factory=list)


class ContentFilter:
    """Configurable rule engine for enterprise content moderation."""

    def __init__(self, rules: Optional[Sequence[FilterRule]] = None) -> None:
        self.rules: List[FilterRule] = list(rules) if rules else []
        self._compiled: List[tuple[re.Pattern[str], FilterRule]] = []
        for r in self.rules:
            self._compiled.append((re.compile(r.pattern, re.IGNORECASE), r))

    def add_rule(self, rule: FilterRule) -> None:
        """Add a custom rule to the filter."""
        self.rules.append(rule)
        self._compiled.append((re.compile(rule.pattern, re.IGNORECASE), rule))

    def check(self, text: str) -> FilterResult:
        """Evaluate text against all registered filter rules."""
        if not text:
            return FilterResult(passed=True, action=FilterAction.ALLOW)

        violations: List[FilterRule] = []
        highest_action = FilterAction.ALLOW

        for pat, rule in self._compiled:
            if pat.search(text):
                violations.append(rule)
                if rule.action == FilterAction.BLOCK:
                    highest_action = FilterAction.BLOCK
                elif rule.action == FilterAction.WARN and highest_action != FilterAction.BLOCK:
                    highest_action = FilterAction.WARN

        passed = highest_action != FilterAction.BLOCK
        return FilterResult(passed=passed, action=highest_action, violations=violations)


def bias_score(text: str) -> float:
    """Compute lexical demographic / gendered bias heuristic score in [0.0, 1.0]."""
    if not text:
        return 0.0

    male_terms = len(re.findall(r"\b(?:he|him|his|men|man|boy|father|brother)\b", text, re.I))
    female_terms = len(re.findall(r"\b(?:she|her|hers|women|woman|girl|mother|sister)\b", text, re.I))
    total_gendered = male_terms + female_terms
    if total_gendered < 4:
        return 0.0

    # Imbalance ratio: 0.0 = balanced, 1.0 = completely one-sided
    diff = abs(male_terms - female_terms)
    return round(diff / total_gendered, 3)

