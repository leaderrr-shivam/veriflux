"""Veriflux hallucination detection engine — detect fabricated content from scratch.

Provides claim decomposition, entailment checking, URL validation,
consistency analysis, and groundedness scoring, all built without
external AI/ML libraries.

Usage::

    from veriflux.hallucination import HallucinationDetector, ClaimVerifier
    from veriflux.hallucination import GroundednessChecker, ConsistencyChecker

    detector = HallucinationDetector()
    result = detector.detect(response, context)
"""
from __future__ import annotations

from veriflux.hallucination.detector import HallucinationDetector, HallucinationResult
from veriflux.hallucination.claim_verifier import ClaimVerifier, ClaimVerdict
from veriflux.hallucination.groundedness import GroundednessChecker, GroundednessResult
from veriflux.hallucination.fact_consistency import ConsistencyChecker, ConsistencyResult
from veriflux.hallucination.url_validator import URLValidator, SuspiciousURL, Citation
from veriflux.hallucination.confidence_scorer import ConfidenceScorer
from veriflux.hallucination.entailment import EntailmentChecker, EntailmentResult

__all__ = [
    "HallucinationDetector",
    "HallucinationResult",
    "ClaimVerifier",
    "ClaimVerdict",
    "GroundednessChecker",
    "GroundednessResult",
    "ConsistencyChecker",
    "ConsistencyResult",
    "URLValidator",
    "SuspiciousURL",
    "Citation",
    "ConfidenceScorer",
    "EntailmentChecker",
    "EntailmentResult",
]
