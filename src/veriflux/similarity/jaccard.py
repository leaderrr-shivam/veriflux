"""Jaccard set and string similarity metrics.

Provides standard Jaccard index, Jaccard distance, weighted Jaccard,
token-level Jaccard, and character n-gram Jaccard.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Set


def jaccard_similarity(a: Set[Any], b: Set[Any]) -> float:
    """Compute the Jaccard similarity index between two sets |A ∩ B| / |A ∪ B|.

    Args:
        a: First set.
        b: Second set.

    Returns:
        Similarity score in [0.0, 1.0]. Returns 1.0 if both sets are empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    intersection_len = len(a & b)
    union_len = len(a | b)
    return intersection_len / union_len if union_len > 0 else 0.0


def jaccard_distance(a: Set[Any], b: Set[Any]) -> float:
    """Compute the Jaccard distance between two sets: 1 - jaccard_similarity(a, b)."""
    return 1.0 - jaccard_similarity(a, b)


def weighted_jaccard(a: Dict[Any, float], b: Dict[Any, float]) -> float:
    """Compute weighted (Ruzicka) Jaccard similarity between two weight mappings.

    Sum(min(a_i, b_i)) / Sum(max(a_i, b_i)).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    all_keys = set(a.keys()) | set(b.keys())
    sum_min = 0.0
    sum_max = 0.0

    for k in all_keys:
        val_a = a.get(k, 0.0)
        val_b = b.get(k, 0.0)
        sum_min += min(val_a, val_b)
        sum_max += max(val_a, val_b)

    return sum_min / sum_max if sum_max > 0.0 else 0.0


def token_jaccard(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard similarity between two texts."""
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    return jaccard_similarity(tokens_a, tokens_b)


def ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity between two texts."""
    clean_a = text_a.lower().strip()
    clean_b = text_b.lower().strip()

    if not clean_a and not clean_b:
        return 1.0
    if not clean_a or not clean_b:
        return 0.0

    def _extract_ngrams(s: str) -> Set[str]:
        if len(s) < n:
            return {s}
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ngrams_a = _extract_ngrams(clean_a)
    ngrams_b = _extract_ngrams(clean_b)
    return jaccard_similarity(ngrams_a, ngrams_b)

