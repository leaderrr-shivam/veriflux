"""Semantic similarity and contextual evaluation assertions.

Provides groundedness, relevance, and faithfulness assertions for LLM outputs.
"""
from __future__ import annotations

from typing import Optional

from veriflux.similarity.tfidf import TfidfVectorizer


_default_checker = None
_default_vectorizer = None


def _get_groundedness_checker():
    global _default_checker
    if _default_checker is None:
        from veriflux.hallucination.groundedness import GroundednessChecker
        _default_checker = GroundednessChecker()
    return _default_checker


def groundedness_score(response: str, context: str) -> float:
    """Calculate contextual groundedness score in [0.0, 1.0]."""
    if not response or not context:
        return 0.0

    try:
        checker = _get_groundedness_checker()
        res = checker.check(response, context)
        return res.score
    except Exception:
        # Fallback to TF-IDF cosine similarity
        global _default_vectorizer
        if _default_vectorizer is None:
            _default_vectorizer = TfidfVectorizer()
        vectors = _default_vectorizer.fit_transform([response, context])
        return max(0.0, min(1.0, vectors[0].cosine_with(vectors[1])))


def relevance_score(response: str, query: str) -> float:
    """Calculate query relevance score in [0.0, 1.0]."""
    if not response or not query:
        return 0.0

    stopwords = {"who", "what", "where", "when", "why", "how", "is", "are", "was", "were", "the", "a", "an", "on", "in", "at", "by", "for", "with", "to", "of", "do", "does", "did"}
    import re
    q_words = [w for w in re.findall(r"\w+", query.lower()) if w not in stopwords and len(w) > 1]
    r_words = set(re.findall(r"\w+", response.lower()))
    coverage = sum(1 for w in q_words if w in r_words) / max(1, len(q_words)) if q_words else 1.0

    vec = TfidfVectorizer()
    vectors = vec.fit_transform([response, query])
    tfidf_sim = max(0.0, min(1.0, vectors[0].cosine_with(vectors[1])))

    score = (coverage * 0.6) + (tfidf_sim * 0.4)
    return round(max(0.0, min(1.0, score)), 3)


def is_grounded(response: str, context: str, threshold: float = 0.7) -> bool:
    """Assert that response is grounded in reference context with score >= threshold."""
    return groundedness_score(response, context) >= threshold


def is_relevant(response: str, query: str, threshold: float = 0.6) -> bool:
    """Assert that response is relevant to query with score >= threshold."""
    return relevance_score(response, query) >= threshold


def is_faithful(response: str, context: str, threshold: float = 0.7) -> bool:
    """Assert that response is faithful to the reference context without hallucination."""
    return is_grounded(response, context, threshold=threshold)
