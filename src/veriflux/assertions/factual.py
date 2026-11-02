"""Factual correctness and hallucination verification assertions.

Verifies claim groundedness, absence of fabricated URLs, and internal consistency.
"""
from __future__ import annotations

from typing import Optional

from veriflux.hallucination.detector import HallucinationDetector
from veriflux.hallucination.fact_consistency import ConsistencyChecker
from veriflux.hallucination.url_validator import URLValidator


_default_detector = None
_default_url_validator = None
_default_consistency_checker = None


def _get_detector() -> HallucinationDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = HallucinationDetector()
    return _default_detector


def _get_url_validator() -> URLValidator:
    global _default_url_validator
    if _default_url_validator is None:
        _default_url_validator = URLValidator()
    return _default_url_validator


def _get_consistency_checker() -> ConsistencyChecker:
    global _default_consistency_checker
    if _default_consistency_checker is None:
        _default_consistency_checker = ConsistencyChecker()
    return _default_consistency_checker


def no_hallucinations(response: str, context: str, threshold: float = 0.8) -> bool:
    """Assert that response has zero significant hallucinations against context."""
    detector = _get_detector()
    res = detector.detect(response, context)
    return res.score >= threshold and res.verdict != "hallucinated"


def claims_verified(response: str, context: str, min_ratio: float = 0.9) -> bool:
    """Assert that at least min_ratio proportion of claims in response are verified supported."""
    detector = _get_detector()
    res = detector.detect(response, context)
    if not res.claims:
        return True
    supported = sum(1 for c in res.claims if c.verdict.value in ("supported", "partially_supported"))
    return (supported / len(res.claims)) >= min_ratio


def no_fabricated_urls(text: str) -> bool:
    """Assert that text contains no suspicious or malformed URLs/citations."""
    validator = _get_url_validator()
    suspicious = validator.find_suspicious_urls(text)
    return len(suspicious) == 0


def internally_consistent(text: str) -> bool:
    """Assert that text contains no self-contradicting claims."""
    checker = _get_consistency_checker()
    res = checker.check(text)
    return res.is_consistent

