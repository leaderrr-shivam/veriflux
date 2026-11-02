"""Claim verification against reference context.

Verifies individual atomic claims by locating supporting context sentences,
evaluating textual entailment, and computing confidence verdicts.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

from veriflux._types import Claim, ClaimVerdict, Verdict
from veriflux.hallucination.entailment import EntailmentChecker
from veriflux.nlp.sentence_splitter import SentenceSplitter
from veriflux.nlp.stemmer import PorterStemmer
from veriflux.similarity.tfidf import TfidfVectorizer


class ClaimVerifier:
    """Verifies atomic claims against reference context text."""

    def __init__(self, entailment_threshold: float = 0.65) -> None:
        self.entailment_threshold = entailment_threshold
        self.sentence_splitter = SentenceSplitter()
        self.entailment_checker = EntailmentChecker()
        self.stemmer = PorterStemmer()

    def verify(self, claim: Union[str, Claim], context: str) -> ClaimVerdict:
        """Verify whether an individual claim is supported, contradicted, or unmentioned in context.

        Args:
            claim: Claim instance or raw claim string.
            context: Reference context text.

        Returns:
            ClaimVerdict containing verdict enum, confidence, and best matching evidence sentence.
        """
        claim_obj = claim if isinstance(claim, Claim) else Claim(text=claim)
        claim_text = claim_obj.text.strip()

        if not context or not context.strip() or not claim_text:
            return ClaimVerdict(
                claim=claim_obj,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.0,
                evidence="",
            )

        context_sentences = self.sentence_splitter.split(context)
        if not context_sentences:
            return ClaimVerdict(
                claim=claim_obj,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.0,
                evidence="",
            )

        # 1. Rank context sentences by relevance to the claim using TF-IDF
        vectorizer = TfidfVectorizer()
        docs = [claim_text] + context_sentences
        vectors = vectorizer.fit_transform(docs)
        claim_vec = vectors[0]

        best_sentence = ""
        best_sim = -1.0
        best_idx = 0

        for idx, s_vec in enumerate(vectors[1:]):
            sim = claim_vec.cosine_with(s_vec)
            if sim > best_sim:
                best_sim = sim
                best_sentence = context_sentences[idx].strip()
                best_idx = idx

        # If similarity to all sentences is too low, the claim is unmentioned
        if best_sim < 0.15:
            return ClaimVerdict(
                claim=claim_obj,
                verdict=Verdict.UNVERIFIABLE,
                confidence=round(1.0 - best_sim, 3),
                evidence=best_sentence,
            )

        # 2. Check entailment between best matching sentence and the claim
        ent_res = self.entailment_checker.check(premise=best_sentence, hypothesis=claim_text)

        # 3. Check overall content word coverage against the full context
        stopwords = {"a", "an", "the", "and", "or", "is", "was", "were", "be", "in", "on", "at", "by", "for", "with", "to", "of", "it", "its"}
        import re
        claim_content_words = [w for w in re.findall(r"\w+", claim_text.lower()) if w not in stopwords and len(w) > 1]
        ctx_content_words = set(re.findall(r"\w+", context.lower()))
        raw_coverage = sum(1 for w in claim_content_words if w in ctx_content_words) / max(1, len(claim_content_words))

        claim_stems = [self.stemmer.stem(w) for w in claim_content_words]
        ctx_stems = set(self.stemmer.stem(w) for w in ctx_content_words)
        stemmed_coverage = sum(1 for w in claim_stems if w in ctx_stems) / max(1, len(claim_stems))
        coverage_ratio = max(raw_coverage, stemmed_coverage)

        num_ok = ent_res.features.get("numerical_consistency", 1.0) >= 0.5

        # 4. Check for unmentioned named entities or proper nouns in the claim
        common_discourse = {
            "the", "a", "an", "this", "that", "these", "those", "it", "its", "they",
            "their", "there", "here", "what", "which", "who", "whom", "where", "when",
            "why", "how", "yes", "no", "not", "according", "however", "moreover",
            "furthermore", "overall", "first", "second", "third", "finally",
            "additionally", "indeed", "similarly", "therefore", "meanwhile",
            "currently", "historically", "specifically", "in", "on", "at", "by",
            "for", "with", "to", "of", "and", "or", "as", "is", "was", "are",
            "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "well", "both", "each", "some", "many", "more", "most", "other",
            "such", "only", "own", "same", "so", "than", "too", "very",
        }
        claim_entities = [
            w.lower() for w in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", claim_text)
            if w.lower() not in common_discourse and len(w) >= 3
        ]
        ctx_lower = context.lower()
        missing_entities = [w for w in claim_entities if w not in ctx_lower]
        has_entity_hallucination = len(missing_entities) > 0 and (len(missing_entities) / max(1, len(claim_entities))) >= 0.3

        if ent_res.label == "CONTRADICTION" or not num_ok or has_entity_hallucination:
            verdict = Verdict.CONTRADICTED
            confidence = 0.92 if has_entity_hallucination else (ent_res.confidence if ent_res.label == "CONTRADICTION" else 0.85)
        elif (ent_res.label == "ENTAILMENT" or coverage_ratio >= 0.70 or best_sim >= self.entailment_threshold) and num_ok:
            verdict = Verdict.SUPPORTED
            confidence = max(ent_res.confidence, best_sim, coverage_ratio)
        elif coverage_ratio >= 0.45 or best_sim >= 0.25:
            verdict = Verdict.PARTIALLY_SUPPORTED
            confidence = max(best_sim, coverage_ratio)
        else:
            verdict = Verdict.UNVERIFIABLE
            confidence = 1.0 - coverage_ratio

        return ClaimVerdict(
            claim=claim_obj,
            verdict=verdict,
            confidence=round(min(1.0, max(0.0, confidence)), 3),
            evidence=best_sentence,
        )

    def verify_batch(self, claims: Sequence[Union[str, Claim]], context: str) -> List[ClaimVerdict]:
        """Verify multiple claims against reference context."""
        return [self.verify(c, context) for c in claims]
