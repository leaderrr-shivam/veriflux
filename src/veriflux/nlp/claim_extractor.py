"""Atomic claim decomposition from natural language text.

Breaks composite paragraphs and responses into verifiable atomic propositions
using syntactic clause analysis, discourse segmentation, and rule-based heuristics.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Set, Tuple

from veriflux._types import Claim, ClaimType
from veriflux.nlp.sentence_splitter import SentenceSplitter


class ClaimExtractor:
    """Extracts and classifies atomic, verifiable claims from unstructured text."""

    # Discourse markers and coordinating conjunctions that introduce independent clauses
    _CLAUSE_SPLITTERS = re.compile(
        r"""(?:\s*,\s*(?:and|but|while|whereas|although|though|however|furthermore|moreover)\s+)"""
        r"""|(?:\s*;\s*)"""
        r"""|(?:\s+which\s+is\s+)"""
        r"""|(?:\s+that\s+was\s+)""",
        re.IGNORECASE,
    )

    # Heuristics for classifying claim types
    _NUMERICAL_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)?%?|\b(?:one|two|three|four|five|ten|hundred|thousand|million|billion)\b", re.IGNORECASE)
    _TEMPORAL_PATTERN = re.compile(r"\b(?:in|during|before|after|since|until)\s+(?:19\d\d|20\d\d|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|yesterday|today|tomorrow)\b", re.IGNORECASE)
    _CAUSAL_PATTERN = re.compile(r"\b(?:because|caused by|results in|leads to|due to|therefore|as a result)\b", re.IGNORECASE)
    _COMPARATIVE_PATTERN = re.compile(r"\b(?:more than|less than|higher than|lower than|compared to|faster than|slower than|better than|worse than)\b", re.IGNORECASE)
    _DEFINITIONAL_PATTERN = re.compile(r"\b(?:is defined as|refers to|is a type of|means that|is known as)\b", re.IGNORECASE)
    _OPINION_PATTERN = re.compile(r"\b(?:i think|i believe|probably|perhaps|arguably|in my opinion|best|worst|beautiful|ugly|amazing|terrible)\b", re.IGNORECASE)

    # Filter out conversational pleasantries and questions
    _NON_FACTUAL_PATTERNS = [
        re.compile(r"^(?:hello|hi|hey|greetings|thank you|thanks|how can i help|sure|certainly|you're welcome)", re.IGNORECASE),
        re.compile(r"\?\s*$"),
        re.compile(r"^(?:let me know|feel free|hope this helps)", re.IGNORECASE),
    ]

    def __init__(self, min_claim_words: int = 3) -> None:
        self.min_claim_words = min_claim_words
        self.sentence_splitter = SentenceSplitter()

    def classify_claim(self, claim_text: str) -> ClaimType:
        """Categorise an individual claim into its primary structural form."""
        text_lower = claim_text.lower()

        if self._OPINION_PATTERN.search(text_lower):
            return ClaimType.OPINION
        if self._DEFINITIONAL_PATTERN.search(text_lower):
            return ClaimType.DEFINITIONAL
        if self._CAUSAL_PATTERN.search(text_lower):
            return ClaimType.CAUSAL
        if self._COMPARATIVE_PATTERN.search(text_lower):
            return ClaimType.COMPARATIVE
        if self._NUMERICAL_PATTERN.search(text_lower):
            return ClaimType.NUMERICAL
        if self._TEMPORAL_PATTERN.search(text_lower):
            return ClaimType.TEMPORAL

        return ClaimType.FACTUAL

    def _is_verifiable_sentence(self, sentence: str) -> bool:
        """Check whether a sentence contains potentially verifiable propositions."""
        clean = sentence.strip()
        if len(clean.split()) < self.min_claim_words:
            return False
        for pat in self._NON_FACTUAL_PATTERNS:
            if pat.search(clean):
                return False
        return True

    def extract(self, text: str, clause_level: bool = True) -> List[Claim]:
        """Extract atomic claims from text.

        Args:
            text: Input paragraph or response text.
            clause_level: When True, splits sentences further on clausal boundaries.

        Returns:
            List of Claim instances with text spans and structural types.
        """
        if not text or not text.strip():
            return []

        claims: List[Claim] = []
        seen_texts: Set[str] = set()

        sentences = self.sentence_splitter.split(text)
        current_offset = 0

        for sent in sentences:
            sent_str = sent.strip()
            # Calculate span in original text
            start_idx = text.find(sent_str, current_offset)
            if start_idx == -1:
                start_idx = current_offset
            end_idx = start_idx + len(sent_str)
            current_offset = end_idx

            if not self._is_verifiable_sentence(sent_str):
                continue

            if clause_level and self._CLAUSE_SPLITTERS.search(sent_str):
                # Split along clause boundaries
                clauses = self._CLAUSE_SPLITTERS.split(sent_str)
                clause_offset = start_idx
                for c in clauses:
                    c_clean = c.strip(" ,;.-")
                    if len(c_clean.split()) >= self.min_claim_words:
                        c_lower = c_clean.lower()
                        if c_lower not in seen_texts:
                            seen_texts.add(c_lower)
                            c_start = text.find(c_clean, clause_offset)
                            if c_start == -1:
                                c_start = clause_offset
                            c_end = c_start + len(c_clean)
                            clause_offset = c_end

                            claims.append(
                                Claim(
                                    text=c_clean,
                                    source_span=(c_start, c_end),
                                    claim_type=self.classify_claim(c_clean),
                                )
                            )
            else:
                sent_lower = sent_str.lower()
                if sent_lower not in seen_texts:
                    seen_texts.add(sent_lower)
                    claims.append(
                        Claim(
                            text=sent_str,
                            source_span=(start_idx, end_idx),
                            claim_type=self.classify_claim(sent_str),
                        )
                    )

        return claims

