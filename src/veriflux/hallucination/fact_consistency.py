"""Internal fact consistency and contradiction analysis.

Detects self-contradictions, numerical clashes, and mutually exclusive assertions
within a single generated text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from veriflux.hallucination.entailment import EntailmentChecker
from veriflux.nlp.claim_extractor import ClaimExtractor


@dataclass(slots=True, frozen=True)
class ConsistencyResult:
    """Outcome of internal consistency check.

    Attributes:
        is_consistent: True if no self-contradictions were discovered.
        score: Consistency score in [0.0, 1.0].
        contradictions: List of tuples describing contradicting claim pairs.
    """

    is_consistent: bool
    score: float
    contradictions: List[Tuple[str, str, str]] = field(default_factory=list)


class ConsistencyChecker:
    """Detects internal self-contradictions and numerical clashes within a text."""

    _CONFLICTING_PAIRS = [
        ({"cancelled", "canceled", "terminated", "abandoned"}, {"continued", "active", "operating", "ongoing"}),
        ({"increase", "increased", "grew", "rose", "growth"}, {"decrease", "decreased", "fell", "dropped", "decline"}),
        ({"alive", "survived"}, {"dead", "died", "killed", "passed away"}),
        ({"succeeded", "success", "won"}, {"failed", "failure", "lost"}),
        ({"true", "valid", "correct"}, {"false", "invalid", "incorrect"}),
    ]

    def __init__(self) -> None:
        self.claim_extractor = ClaimExtractor()
        self.entailment_checker = EntailmentChecker()

    def check(self, text: str) -> ConsistencyResult:
        """Inspect text for internal self-contradictions.

        Args:
            text: Input response text.

        Returns:
            ConsistencyResult with list of conflicting claims if found.
        """
        if not text or not text.strip():
            return ConsistencyResult(is_consistent=True, score=1.0)

        claims = self.claim_extractor.extract(text)
        if len(claims) < 2:
            return ConsistencyResult(is_consistent=True, score=1.0)

        contradictions: List[Tuple[str, str, str]] = []

        # Compare pairs of claims for mutual contradiction
        num_claims = min(len(claims), 20)  # Bound quadratic search
        for i in range(num_claims):
            for j in range(i + 1, num_claims):
                c1 = claims[i].text
                c2 = claims[j].text

                res = self.entailment_checker.check(premise=c1, hypothesis=c2)
                if res.label == "CONTRADICTION" and res.confidence > 0.6:
                    contradictions.append((c1, c2, f"Contradiction confidence: {res.confidence:.2f}"))
                    continue

                # Check semantic antonym conflicts on shared entity/subject
                tokens_1 = set(c1.lower().split())
                tokens_2 = set(c2.lower().split())
                shared = tokens_1 & tokens_2 - {"the", "a", "an", "was", "were", "in", "on", "of", "to", "at", "by", "and"}
                if shared:
                    for set_a, set_b in self._CONFLICTING_PAIRS:
                        if (tokens_1 & set_a and tokens_2 & set_b) or (tokens_1 & set_b and tokens_2 & set_a):
                            contradictions.append((c1, c2, f"Semantic conflict on shared subject: {shared}"))
                            break

        if contradictions:
            penalty = min(1.0, len(contradictions) * 0.3)
            score = max(0.0, 1.0 - penalty)
            return ConsistencyResult(is_consistent=False, score=round(score, 3), contradictions=contradictions)

        return ConsistencyResult(is_consistent=True, score=1.0, contradictions=[])
