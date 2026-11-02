"""Multi-signal rank and score fusion algorithms.

Implements Reciprocal Rank Fusion (RRF), CombMNZ, weighted linear combinations,
and score normalization strategies for hybrid retrieval pipelines.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Dict, List, Sequence, Tuple


def normalize_scores(scores: Dict[int, float], method: str = "min_max") -> Dict[int, float]:
    """Normalize score values to [0.0, 1.0].

    Args:
        scores: Mapping of item ID to numeric score.
        method: Normalisation scheme ('min_max', 'z_score', 'softmax').

    Returns:
        New dict of normalized scores.
    """
    if not scores:
        return {}

    vals = list(scores.values())

    if method == "min_max":
        min_v = min(vals)
        max_v = max(vals)
        spread = max_v - min_v
        if spread == 0.0:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / spread for k, v in scores.items()}

    elif method == "z_score":
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) or 1.0
        # Sigmoid over z-score to map to [0, 1]
        res: Dict[int, float] = {}
        for k, v in scores.items():
            z = (v - mean) / std
            res[k] = 1.0 / (1.0 + math.exp(-z))
        return res

    elif method == "softmax":
        max_v = max(vals)
        exps = {k: math.exp(v - max_v) for k, v in scores.items()}
        sum_exp = sum(exps.values()) or 1.0
        return {k: v / sum_exp for k, v in exps.items()}

    raise ValueError(f"Unknown normalisation method: {method}")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Compute Reciprocal Rank Fusion (RRF) over multiple ranked lists.

    Score(d) = Sum_r (1 / (k + rank_r(d))).

    Args:
        rankings: List of ranked lists, where each list contains (doc_id, original_score).
        k: Smoothing constant (standard default is 60).

    Returns:
        Consolidated list of (doc_id, rrf_score) tuples sorted in descending order.
    """
    rrf_scores: Dict[int, float] = defaultdict(float)

    for rank_list in rankings:
        for rank_idx, (doc_id, _) in enumerate(rank_list):
            # Rank index is 1-based in standard RRF formulation
            rrf_scores[doc_id] += 1.0 / (k + (rank_idx + 1))

    sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_results


def weighted_linear(
    scores: Sequence[Dict[int, float]],
    weights: Sequence[float],
) -> Dict[int, float]:
    """Compute weighted linear combination of multiple score dictionaries.

    Score(d) = Sum_i (w_i * S_i(d)).
    """
    if len(scores) != len(weights):
        raise ValueError(f"Length of scores ({len(scores)}) must match weights ({len(weights)})")

    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("Sum of weights must be positive")

    norm_weights = [w / total_weight for w in weights]
    combined: Dict[int, float] = defaultdict(float)

    for score_dict, w in zip(scores, norm_weights):
        for doc_id, val in score_dict.items():
            combined[doc_id] += w * val

    return dict(combined)


def combmnz(rankings: Sequence[Sequence[Tuple[int, float]]]) -> List[Tuple[int, float]]:
    """Combine rankings using CombMNZ (combines scores multiplied by nonzero system count).

    CombMNZ(d) = NonzeroCount(d) * Sum(Score_i(d)).
    """
    if not rankings:
        return []

    sum_scores: Dict[int, float] = defaultdict(float)
    non_zero_counts: Dict[int, int] = defaultdict(int)

    for r_list in rankings:
        # Normalize scores within list first
        list_dict = {doc_id: s for doc_id, s in r_list}
        normed = normalize_scores(list_dict)
        for doc_id, val in normed.items():
            if val > 0:
                sum_scores[doc_id] += val
                non_zero_counts[doc_id] += 1

    final_scores: Dict[int, float] = {}
    for doc_id, s_sum in sum_scores.items():
        final_scores[doc_id] = non_zero_counts[doc_id] * s_sum

    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

