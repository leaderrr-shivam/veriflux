"""Semantic drift detection across model outputs over time.

Detects silent model drift, behavioral shifts, and vocabulary alterations between model revisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from veriflux.similarity.tfidf import TfidfVectorizer


@dataclass(slots=True, frozen=True)
class DriftResult:
    """Outcome of semantic drift analysis.

    Attributes:
        has_drift: True if drift exceeds threshold.
        drift_score: Drift magnitude in [0.0, 1.0] (0.0 = identical, 1.0 = completely divergent).
        similarity: Mean cosine similarity between paired response distributions.
        details: Diagnostic dictionary.
    """

    has_drift: bool
    drift_score: float
    similarity: float
    details: Dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """Measures semantic drift between baseline model outputs and new candidate responses."""

    def __init__(self, threshold: float = 0.25) -> None:
        self.threshold = threshold

    def detect(
        self,
        baseline_responses: Sequence[str],
        current_responses: Sequence[str],
    ) -> DriftResult:
        """Measure semantic drift between paired or grouped response corpora."""
        if not baseline_responses or not current_responses:
            return DriftResult(has_drift=False, drift_score=0.0, similarity=1.0)

        n = min(len(baseline_responses), len(current_responses))
        b_slice = list(baseline_responses[:n])
        c_slice = list(current_responses[:n])

        # Compute pairwise semantic similarity using TF-IDF
        vec = TfidfVectorizer()
        all_docs = b_slice + c_slice
        vectors = vec.fit_transform(all_docs)

        sims: List[float] = []
        for i in range(n):
            v_base = vectors[i]
            v_curr = vectors[n + i]
            sim = v_base.cosine_with(v_curr)
            sims.append(sim)

        mean_sim = sum(sims) / len(sims) if sims else 1.0
        drift_score = max(0.0, min(1.0, 1.0 - mean_sim))
        has_drift = drift_score >= self.threshold

        return DriftResult(
            has_drift=has_drift,
            drift_score=round(drift_score, 4),
            similarity=round(mean_sim, 4),
            details={"sample_size": n, "min_similarity": round(min(sims), 4) if sims else 1.0},
        )

