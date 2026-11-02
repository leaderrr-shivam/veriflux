"""Context groundedness analysis.

Measures the factual fidelity and coverage of an LLM response relative to
reference background context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from veriflux._types import Claim, ClaimVerdict, Verdict
from veriflux.hallucination.claim_verifier import ClaimVerifier
from veriflux.nlp.claim_extractor import ClaimExtractor
from veriflux.similarity.tfidf import TfidfVectorizer


@dataclass(slots=True, frozen=True)
class GroundednessResult:
    """Outcome of groundedness analysis.

    Attributes:
        score: Consolidated groundedness fidelity in [0.0, 1.0].
        coverage: Ratio of response covered by context-backed propositions.
        precision: Proportion of atomic claims that are verified supported.
        density: Utilization density of context information.
        grounded_claims: List of claims supported by the context.
        ungrounded_claims: List of claims lacking context support or contradicted.
    """

    score: float
    coverage: float
    precision: float
    density: float
    grounded_claims: List[ClaimVerdict] = field(default_factory=list)
    ungrounded_claims: List[ClaimVerdict] = field(default_factory=list)


class GroundednessChecker:
    """Evaluates context groundedness by verifying decomposed atomic claims."""

    def __init__(self, entailment_threshold: float = 0.65) -> None:
        self.claim_extractor = ClaimExtractor()
        self.claim_verifier = ClaimVerifier(entailment_threshold=entailment_threshold)

    def check(self, response: str, context: str) -> GroundednessResult:
        """Measure the groundedness of response against context.

        Args:
            response: Generated output string.
            context: Reference context text.

        Returns:
            GroundednessResult with coverage, precision, and claim breakdowns.
        """
        if not response or not response.strip():
            return GroundednessResult(score=1.0, coverage=1.0, precision=1.0, density=0.0)
        if not context or not context.strip():
            return GroundednessResult(score=0.0, coverage=0.0, precision=0.0, density=0.0)

        # 1. Extract claims from response
        claims = self.claim_extractor.extract(response)
        if not claims:
            # Fallback to direct text similarity if no discrete factual claims found
            vec = TfidfVectorizer()
            docs = [response, context]
            vs = vec.fit_transform(docs)
            sim = vs[0].cosine_with(vs[1])
            return GroundednessResult(
                score=round(sim, 3),
                coverage=round(sim, 3),
                precision=round(sim, 3),
                density=0.1,
            )

        # 2. Verify each claim
        verdicts = self.claim_verifier.verify_batch(claims, context)
        grounded: List[ClaimVerdict] = []
        ungrounded: List[ClaimVerdict] = []

        for v in verdicts:
            if v.verdict in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED):
                grounded.append(v)
            else:
                ungrounded.append(v)

        total_claims = len(verdicts)
        precision = len(grounded) / total_claims if total_claims > 0 else 1.0

        # Approximate coverage via character counts of grounded claims vs response
        grounded_char_count = sum(len(c.claim.text) for c in grounded)
        coverage = min(1.0, grounded_char_count / max(1, len(response)))

        # Density: fraction of context tokens referenced
        resp_words = set(response.lower().split())
        ctx_words = set(context.lower().split())
        density = len(resp_words & ctx_words) / max(1, len(ctx_words))

        # Overall composite groundedness score
        score = (precision * 0.7) + (coverage * 0.3)

        return GroundednessResult(
            score=round(max(0.0, min(1.0, score)), 3),
            coverage=round(max(0.0, min(1.0, coverage)), 3),
            precision=round(max(0.0, min(1.0, precision)), 3),
            density=round(max(0.0, min(1.0, density)), 3),
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
        )

