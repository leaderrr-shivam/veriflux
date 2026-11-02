"""Toxicity, harassment, and harmful content detection guardrail.

Provides multi-category toxicity scoring without external network APIs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(slots=True, frozen=True)
class ToxicityResult:
    """Outcome of toxicity evaluation.

    Attributes:
        is_toxic: True if toxicity score exceeds the configured threshold.
        score: Severity score in [0.0, 1.0].
        categories: Detected categories (e.g. 'THREAT', 'HATE_SPEECH', 'PROFANITY').
        flagged_terms: Terms or snippets that triggered detection.
    """

    is_toxic: bool
    score: float
    categories: List[str] = field(default_factory=list)
    flagged_terms: List[str] = field(default_factory=list)


class ToxicityGuard:
    """Heuristic and pattern-based toxicity detector."""

    # Euphemistic regex patterns targeting hostile intent, threats, slurs, harassment
    _PATTERNS: Sequence[Tuple[re.Pattern[str], str, float]] = [
        # Explicit violent threats
        (re.compile(r"\b(?:i\s+will|going\s+to|gonna)\b.*?\b(?:kill|murder|harm|attack|destroy|hunt)\b.*?\b(?:you|them|all)\b", re.I), "THREAT", 1.0),
        (re.compile(r"\b(?:die|hope\s+you\s+die|kill\s+yourself)\b", re.I), "SELF_HARM", 1.0),
        # Severe harassment and abuse
        (re.compile(r"\b(?:you\s+are\s+(?:worthless|stupid|idiot|garbage|trash|disgusting))\b", re.I), "HARASSMENT", 0.7),
        # Degrading identity slurs and hate speech heuristics
        (re.compile(r"\b(?:subhuman|scum|exterminate\s+(?:all\s+)?(?:people|them))\b", re.I), "HATE_SPEECH", 0.95),
        # Profanity heuristics
        (re.compile(r"\b(?:fuck(?:ing|er)?|shit(?:ty)?|bitch|bastard|asshole|cunt)\b", re.I), "PROFANITY", 0.6),
    ]

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def check(self, text: str) -> ToxicityResult:
        """Scan text and return toxicity evaluation."""
        if not text:
            return ToxicityResult(is_toxic=False, score=0.0)

        flagged: List[str] = []
        categories: Set[str] = set()
        max_score = 0.0

        for pat, cat, weight in self._PATTERNS:
            for m in pat.finditer(text):
                flagged.append(m.group(0))
                categories.add(cat)
                if weight > max_score:
                    max_score = weight

        is_toxic = max_score >= self.threshold

        return ToxicityResult(
            is_toxic=is_toxic,
            score=round(max_score, 3),
            categories=sorted(list(categories)),
            flagged_terms=flagged,
        )


# Ergonomic alias
ToxicityFilter = ToxicityGuard
