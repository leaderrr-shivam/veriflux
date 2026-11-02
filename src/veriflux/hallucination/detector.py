"""Full pipeline orchestrator for hallucination detection.

Coordinates claim decomposition, contextual entailment verification, URL/citation
validation, and internal consistency checking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from veriflux._types import ClaimVerdict, Verdict
from veriflux.hallucination.claim_verifier import ClaimVerifier
from veriflux.hallucination.confidence_scorer import ConfidenceScorer
from veriflux.hallucination.fact_consistency import ConsistencyChecker
from veriflux.hallucination.url_validator import SuspiciousURL, URLValidator
from veriflux.nlp.claim_extractor import ClaimExtractor


@dataclass(slots=True, frozen=True)
class HallucinationResult:
    """Outcome of full hallucination detection pipeline.

    Attributes:
        verdict: High-level classification: 'grounded', 'partially_grounded', or 'hallucinated'.
        score: Overall fidelity score in [0.0, 1.0] (1.0 = perfectly grounded).
        claims: All evaluated atomic claims with their verdicts.
        problematic_claims: Claims that are contradicted or unverified.
        suspicious_urls: Fabricated or suspicious links/citations found in text.
        details: Diagnostic dictionary containing breakdown metrics.
    """

    verdict: str
    score: float
    claims: List[ClaimVerdict] = field(default_factory=list)
    problematic_claims: List[ClaimVerdict] = field(default_factory=list)
    suspicious_urls: List[SuspiciousURL] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if the response is considered grounded without severe hallucinations."""
        return self.verdict in ("grounded", "partially_grounded") and self.score >= 0.7


class HallucinationDetector:
    """End-to-end hallucination detection pipeline orchestrator."""

    def __init__(
        self,
        strategy: str = "claim_decomposition",
        entailment_threshold: float = 0.65,
        url_validation: bool = True,
        consistency_check: bool = True,
    ) -> None:
        self.strategy = strategy
        self.entailment_threshold = entailment_threshold
        self.url_validation = url_validation
        self.consistency_check = consistency_check

        self.claim_extractor = ClaimExtractor()
        self.claim_verifier = ClaimVerifier(entailment_threshold=entailment_threshold)
        self.confidence_scorer = ConfidenceScorer()
        self.url_validator = URLValidator() if url_validation else None
        self.consistency_checker = ConsistencyChecker() if consistency_check else None

    def detect(self, response: str, context: str, query: Optional[str] = None) -> HallucinationResult:
        """Execute full hallucination evaluation.

        Args:
            response: Generated output to evaluate.
            context: Ground-truth reference context.
            query: Optional user question that prompted the response.

        Returns:
            HallucinationResult with verdict, score, and granular claim verdicts.
        """
        if not response or not response.strip():
            return HallucinationResult(verdict="grounded", score=1.0)

        # 1. Decompose into atomic claims
        claims = self.claim_extractor.extract(response)

        # 2. Verify claims against context
        claim_verdicts = self.claim_verifier.verify_batch(claims, context)
        problematic = [
            v for v in claim_verdicts
            if v.verdict in (Verdict.CONTRADICTED, Verdict.UNVERIFIABLE)
        ]

        # 3. URL and citation validation
        suspicious_urls: List[SuspiciousURL] = []
        if self.url_validator is not None:
            suspicious_urls = self.url_validator.find_suspicious_urls(response)

        # 4. Consistency check
        consistency_res = None
        if self.consistency_checker is not None:
            consistency_res = self.consistency_checker.check(response)

        # 5. Score aggregation
        raw_score = self.confidence_scorer.score(claim_verdicts, consistency=consistency_res)

        # Apply penalty for suspicious URLs
        if suspicious_urls:
            raw_score = max(0.0, raw_score - (0.2 * len(suspicious_urls)))

        score = round(raw_score, 3)

        # 6. Assign verdict
        contradicted_count = sum(1 for v in claim_verdicts if v.verdict == Verdict.CONTRADICTED)
        if contradicted_count > 0 or score < 0.4:
            verdict = "hallucinated"
        elif score < 0.75 or len(problematic) > len(claim_verdicts) // 2:
            verdict = "partially_grounded"
        else:
            verdict = "grounded"

        details = {
            "total_claims": len(claim_verdicts),
            "problematic_claims_count": len(problematic),
            "contradicted_count": contradicted_count,
            "suspicious_urls_count": len(suspicious_urls),
            "consistency_score": consistency_res.score if consistency_res else 1.0,
        }

        return HallucinationResult(
            verdict=verdict,
            score=score,
            claims=claim_verdicts,
            problematic_claims=problematic,
            suspicious_urls=suspicious_urls,
            details=details,
        )

