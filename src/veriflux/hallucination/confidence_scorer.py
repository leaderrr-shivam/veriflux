"""Calibrated confidence estimation for hallucination scoring.

Aggregates individual atomic claim verification verdicts into a calibrated
probability score that takes into account claim volume, claim types, and consistency.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from veriflux._types import ClaimVerdict, Verdict
from veriflux.hallucination.fact_consistency import ConsistencyResult


class ConfidenceScorer:
    """Computes calibrated confidence scores across multiple claim evaluations."""

    def __init__(self, aggregation: str = "weighted_mean") -> None:
        self.aggregation = aggregation

    def score(
        self,
        verdicts: Sequence[ClaimVerdict],
        consistency: Optional[ConsistencyResult] = None,
    ) -> float:
        """Compute aggregate factual fidelity score in [0.0, 1.0].

        Args:
            verdicts: List of claim verification outcomes.
            consistency: Optional internal consistency check outcome.

        Returns:
            Scalar confidence score where 1.0 indicates complete adherence and zero hallucination.
        """
        if not verdicts:
            return 1.0

        scores: list[float] = []
        for v in verdicts:
            if v.verdict == Verdict.SUPPORTED:
                scores.append(v.confidence)
            elif v.verdict == Verdict.PARTIALLY_SUPPORTED:
                scores.append(v.confidence * 0.7)
            elif v.verdict == Verdict.UNVERIFIABLE:
                scores.append(0.3)
            elif v.verdict == Verdict.CONTRADICTED:
                scores.append(0.0)

        if self.aggregation == "min":
            base_score = min(scores) if scores else 0.0
        elif self.aggregation == "mean":
            base_score = sum(scores) / len(scores) if scores else 0.0
        else:  # weighted_mean (give higher penalty to lower scores)
            weights = [1.5 if s < 0.5 else 1.0 for s in scores]
            total_w = sum(weights) or 1.0
            base_score = sum(s * w for s, w in zip(scores, weights)) / total_w

        # Penalise if internal self-contradictions were found
        if consistency is not None and not consistency.is_consistent:
            base_score *= consistency.score

        return round(self.calibrate(base_score, len(verdicts)), 3)

    def calibrate(self, raw_score: float, n_claims: int) -> float:
        """Apply Bayesian shrinkage / claim count calibration.

        Responses making many claims face higher cumulative risk of ungrounded statements.
        """
        if n_claims <= 1:
            return raw_score
        # Slight dampening for long speculative answers with many unverified claims
        factor = 1.0 - (0.01 * math.log(n_claims))
        return max(0.0, min(1.0, raw_score * factor))

