"""Lightweight textual entailment checker — no ML models required.

Uses lexical overlap, entity extraction, negation detection, and
numerical consistency to determine whether a premise entails, contradicts,
or is neutral towards a hypothesis.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Set, Tuple


# ─── Negation vocabulary ────────────────────────────────────────────────────

_NEGATION_WORDS: FrozenSet[str] = frozenset({
    "no", "not", "never", "neither", "nor", "nobody", "nothing",
    "nowhere", "none", "zero", "nil", "cannot", "can't", "couldn't", "shouldn't",
    "wouldn't", "won't", "don't", "doesn't", "didn't", "isn't",
    "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "mustn't", "needn't", "without", "hardly", "barely", "scarcely",
    "seldom", "rarely", "lack", "lacking", "lacks", "absent",
    "impossible", "unable", "fail", "failed", "fails", "deny",
    "denied", "denies", "refuse", "refused", "refuses", "reject",
    "rejected", "rejects", "false", "incorrect", "wrong", "untrue",
})

_CONTRACTED_NEGATIONS: FrozenSet[str] = frozenset({
    "cant", "wont", "dont", "isnt", "arent", "wasnt", "werent",
    "havent", "hasnt", "hadnt", "shouldnt", "wouldnt", "couldnt",
    "didnt", "doesnt", "mustnt", "neednt", "aint",
})

_NEGATION_PREFIXES: Tuple[str, ...] = (
    "un", "in", "im", "il", "ir", "non", "dis", "mis",
)

# ─── Stopwords for filtering ───────────────────────────────────────────────

_STOPWORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "and",
    "but", "or", "so", "if", "that", "this", "these", "those", "it",
    "its", "he", "she", "they", "we", "you", "i", "me", "him", "her",
    "us", "them", "my", "his", "our", "your", "their", "what", "which",
    "who", "whom", "when", "where", "how", "than", "very", "just",
    "also", "about",
})

# ─── Number extraction regex ───────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r"""
    (?<!\w)(?<!\d\.)            # not preceded by word char or digit dot
    -?                          # optional negative
    (?:\d{1,3}(?:,\d{3})*      # integer with comma grouping
    |  \d+)                    # or plain integer
    (?:\.\d+)?                 # optional decimal
    (?:%|\s*(?:percent|million|billion|trillion|thousand|hundred|dozen|
               km|mi|kg|lb|lbs|m|ft|cm|mm|mph|kph|hz|ghz|mhz|gb|mb|tb|
               kb|°[cf]|celsius|fahrenheit|usd|eur|gbp|dollars?|euros?|
               pounds?|years?|months?|days?|hours?|minutes?|seconds?))?
    (?!\w)(?!\.\d)              # not followed by word char or dot digit
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PURE_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")

# ─── Entity-like extraction (capitalized multi-word sequences) ──────────────

_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)


# ─── Dataclass ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EntailmentResult:
    """Result of a textual entailment check.

    Attributes:
        label: One of ``ENTAILMENT``, ``CONTRADICTION``, ``NEUTRAL``.
        confidence: Confidence score in ``[0.0, 1.0]``.
        features: Dictionary of individual signal scores used in the decision.
    """
    label: str
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)


# ─── Helper functions ───────────────────────────────────────────────────────


def _tokenize(text: str) -> List[str]:
    """Tokenize into lowercase words, stripping punctuation."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _content_tokens(text: str) -> List[str]:
    """Return non-stopword tokens."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


def _extract_numbers(text: str) -> List[float]:
    """Extract numeric values from text."""
    results: List[float] = []
    for match in _PURE_NUMBER_RE.finditer(text):
        raw = match.group().replace(",", "")
        try:
            results.append(float(raw))
        except ValueError:
            continue
    return results


def _extract_number_contexts(text: str) -> List[Tuple[float, str]]:
    """Extract numbers with surrounding context words."""
    results: List[Tuple[float, str]] = []
    for match in _NUMBER_RE.finditer(text):
        raw_num = _PURE_NUMBER_RE.search(match.group())
        if raw_num is None:
            continue
        try:
            val = float(raw_num.group().replace(",", ""))
        except ValueError:
            continue
        # Get surrounding context (±30 chars)
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        context = text[start:end].lower()
        results.append((val, context))
    return results


def _extract_entities(text: str) -> Set[str]:
    """Extract entity-like phrases (capitalized word sequences)."""
    entities: Set[str] = set()
    for match in _ENTITY_RE.finditer(text):
        entity = match.group().strip()
        if len(entity) > 1:  # skip single chars
            entities.add(entity.lower())
    return entities


def _has_negation(tokens: List[str]) -> bool:
    """Check if a token sequence contains negation."""
    for t in tokens:
        if t in _NEGATION_WORDS or t in _CONTRACTED_NEGATIONS or t.endswith("n't"):
            return True
    return False


def _negation_asymmetry(premise_tokens: List[str], hypothesis_tokens: List[str]) -> float:
    """Detect negation asymmetry between premise and hypothesis.

    Returns a score where higher means more likely contradiction due to
    one side being negated and the other not.
    """
    premise_neg = _has_negation(premise_tokens)
    hypothesis_neg = _has_negation(hypothesis_tokens)

    if premise_neg != hypothesis_neg:
        p_text = " ".join(premise_tokens)
        h_text = " ".join(hypothesis_tokens)
        neg_pattern = r"(?:not|no|never|cannot|neither|nor|without|cant|wont|isnt|arent|wasnt|werent)\s+(?:a\s+|an\s+|the\s+)?([a-z0-9]+)"
        p_neg_terms = set(re.findall(neg_pattern, p_text))
        h_neg_terms = set(re.findall(neg_pattern, h_text))

        p_words = set(_content_tokens(p_text))
        h_words = set(_content_tokens(h_text))

        # Direct contradiction if negated term in hypothesis was explicitly affirmed in premise
        if any(term in p_words for term in h_neg_terms):
            return 0.85
        # Direct contradiction if negated term in premise was explicitly affirmed in hypothesis
        if any(term in h_words for term in p_neg_terms):
            return 0.85

        return 0.15
    return 0.0


def _lexical_overlap(premise_tokens: List[str], hypothesis_tokens: List[str]) -> float:
    """Compute lexical overlap as fraction of hypothesis tokens found in premise."""
    if not hypothesis_tokens:
        return 1.0
    p_set = set(premise_tokens)
    matches = sum(1 for t in hypothesis_tokens if t in p_set)
    return matches / len(hypothesis_tokens)


def _content_overlap(premise: str, hypothesis: str) -> float:
    """Compute overlap of content (non-stopword) tokens."""
    p_tokens = _content_tokens(premise)
    h_tokens = _content_tokens(hypothesis)
    if not h_tokens:
        return 1.0
    if not p_tokens:
        return 0.0
    p_set = set(p_tokens)
    matches = sum(1 for t in h_tokens if t in p_set)
    return matches / len(h_tokens)


def _entity_overlap(premise: str, hypothesis: str) -> float:
    """Compute fraction of hypothesis entities found in premise."""
    p_entities = _extract_entities(premise)
    h_entities = _extract_entities(hypothesis)
    if not h_entities:
        return 1.0
    if not p_entities:
        # If hypothesis has entities but premise doesn't, that's suspicious
        return 0.0
    matches = sum(1 for e in h_entities if e in p_entities)
    return matches / len(h_entities)


def _numerical_consistency(premise: str, hypothesis: str) -> float:
    """Check if numbers in the hypothesis are consistent with the premise.

    Returns 1.0 if consistent, 0.0 if contradictory.
    """
    p_nums = _extract_number_contexts(premise)
    h_nums = _extract_number_contexts(hypothesis)

    if not h_nums:
        return 1.0  # No numbers to check
    if not p_nums:
        # Hypothesis mentions numbers not in premise
        return 0.5  # Uncertain, not necessarily contradiction

    p_values = [v for v, _ in p_nums]
    contradictions = 0
    matches = 0

    for h_val, h_ctx in h_nums:
        # Check if the number exists in premise
        found_exact = False
        found_close = False
        for p_val, p_ctx in p_nums:
            if h_val == p_val:
                found_exact = True
                break
            # Check if close (within 1% for large numbers)
            if p_val != 0 and abs(h_val - p_val) / abs(p_val) < 0.01:
                found_close = True

        if found_exact:
            matches += 1
        elif found_close:
            matches += 0.8
        else:
            # Number in hypothesis not in premise — check context overlap
            h_ctx_tokens = set(_content_tokens(h_ctx))
            best_ctx_overlap = 0.0
            for p_val, p_ctx in p_nums:
                p_ctx_tokens = set(_content_tokens(p_ctx))
                if h_ctx_tokens and p_ctx_tokens:
                    ovlp = len(h_ctx_tokens & p_ctx_tokens) / max(
                        len(h_ctx_tokens), len(p_ctx_tokens)
                    )
                    best_ctx_overlap = max(best_ctx_overlap, ovlp)

            if best_ctx_overlap > 0.3:
                # Same context but different numbers — likely contradiction
                contradictions += 1
            else:
                # Different context — could be additional info
                matches += 0.3

    total = len(h_nums)
    if contradictions > 0:
        return max(0.0, 0.2 * (1.0 - (contradictions / total)))
    return min(1.0, matches / total) if total > 0 else 1.0


def _bigram_overlap(premise_tokens: List[str], hypothesis_tokens: List[str]) -> float:
    """Compute bigram overlap ratio."""
    if len(hypothesis_tokens) < 2:
        return _lexical_overlap(premise_tokens, hypothesis_tokens)

    p_bigrams: Set[Tuple[str, str]] = set()
    for i in range(len(premise_tokens) - 1):
        p_bigrams.add((premise_tokens[i], premise_tokens[i + 1]))

    h_bigrams: List[Tuple[str, str]] = []
    for i in range(len(hypothesis_tokens) - 1):
        h_bigrams.append((hypothesis_tokens[i], hypothesis_tokens[i + 1]))

    if not h_bigrams:
        return 1.0

    matches = sum(1 for bg in h_bigrams if bg in p_bigrams)
    return matches / len(h_bigrams)


# ─── Semantic Antonym Clashing Pairs ───────────────────────────────────────

_ANTONYM_PAIRS: Tuple[Tuple[FrozenSet[str], FrozenSet[str]], ...] = (
    (
        frozenset({"increase", "increases", "increased", "increasing", "grow", "grew", "growth", "rise", "rose", "rising", "spike", "spiked", "surge", "surged", "hike", "hiked", "cause", "causes", "caused", "causing", "induce", "induced", "induces", "promotes", "promoted", "accelerates", "accelerated", "exacerbates"}),
        frozenset({"decrease", "decreased", "decreasing", "drop", "dropped", "dropping", "fall", "fell", "falling", "decline", "declined", "lower", "lowered", "slow", "slowed", "prevent", "prevents", "prevented", "preventing", "protect", "protects", "protected", "protecting", "cure", "cures", "cured", "shield", "shields", "safeguard", "safeguards"}),
    ),
    (
        frozenset({"allowed", "permitted", "authorized", "encouraged", "enabled"}),
        frozenset({"forbidden", "prohibited", "banned", "restricted", "blocked", "prevented", "disallowed"}),
    ),
    (
        frozenset({"approved", "accepted", "validated", "confirmed", "passed"}),
        frozenset({"rejected", "denied", "dismissed", "declined", "failed"}),
    ),
    (
        frozenset({"active", "alive", "operational", "running", "continued"}),
        frozenset({"inactive", "dead", "cancelled", "canceled", "terminated", "stopped", "void"}),
    ),
    (
        frozenset({"true", "correct", "valid", "accurate", "binding"}),
        frozenset({"false", "incorrect", "invalid", "inaccurate", "void"}),
    ),
)


def _antonym_conflict(premise_tokens: List[str], hypothesis_tokens: List[str]) -> bool:
    """Detect if premise and hypothesis contain direct antonym polarity clashes."""
    p_set = set(premise_tokens)
    h_set = set(hypothesis_tokens)
    for pos_set, neg_set in _ANTONYM_PAIRS:
        if (p_set & pos_set and h_set & neg_set) or (p_set & neg_set and h_set & pos_set):
            return True
    return False


def _negated_scope_inversion(premise: str, hypothesis: str) -> bool:
    """Detect when a negation target is swapped between premise and hypothesis."""
    p_neg_targets = set(re.findall(r"(?:not|no|never|without)\s+(?:a\s+|an\s+|the\s+)?([a-z0-9]+)", premise.lower()))
    h_neg_targets = set(re.findall(r"(?:not|no|never|without)\s+(?:a\s+|an\s+|the\s+)?([a-z0-9]+)", hypothesis.lower()))
    if p_neg_targets and h_neg_targets:
        p_affirmed = set(_content_tokens(premise)) - p_neg_targets
        h_affirmed = set(_content_tokens(hypothesis)) - h_neg_targets
        if (p_neg_targets & h_affirmed) and (h_neg_targets & p_affirmed):
            return True
    return False


# ─── Main class ─────────────────────────────────────────────────────────────


class EntailmentChecker:
    """Lightweight textual entailment checker using lexical heuristics.

    Combines multiple signal scores (lexical overlap, entity overlap,
    negation detection, numerical consistency, bigram overlap) to
    determine whether a premise entails, contradicts, or is neutral
    towards a hypothesis.

    Args:
        entailment_threshold: Minimum combined score for ENTAILMENT label.
        contradiction_threshold: Maximum combined score for CONTRADICTION label.
        weights: Optional dict mapping feature names to weights. Defaults to
                 balanced weights.

    Example::

        checker = EntailmentChecker()
        result = checker.check(
            premise="The capital of France is Paris.",
            hypothesis="Paris is the capital of France.",
        )
        assert result.label == "ENTAILMENT"
    """

    __slots__ = ("_entailment_threshold", "_contradiction_threshold", "_weights")

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "lexical_overlap": 0.15,
        "content_overlap": 0.25,
        "entity_overlap": 0.15,
        "bigram_overlap": 0.10,
        "numerical_consistency": 0.15,
        "negation_score": 0.20,
    }

    def __init__(
        self,
        entailment_threshold: float = 0.65,
        contradiction_threshold: float = 0.40,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self._entailment_threshold = entailment_threshold
        self._contradiction_threshold = contradiction_threshold
        self._weights = weights or dict(self._DEFAULT_WEIGHTS)

    def check(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Check if *premise* entails *hypothesis*.

        Args:
            premise: The source / reference text.
            hypothesis: The claim to be checked against the premise.

        Returns:
            An :class:`EntailmentResult` with label, confidence, and feature scores.
        """
        if not premise.strip() or not hypothesis.strip():
            return EntailmentResult(
                label="NEUTRAL",
                confidence=0.5,
                features={"empty_input": 1.0},
            )

        p_tokens = _tokenize(premise)
        h_tokens = _tokenize(hypothesis)

        # Compute individual feature scores
        lex_overlap = _lexical_overlap(p_tokens, h_tokens)
        cont_overlap = _content_overlap(premise, hypothesis)
        ent_overlap = _entity_overlap(premise, hypothesis)
        bg_overlap = _bigram_overlap(p_tokens, h_tokens)
        num_consistency = _numerical_consistency(premise, hypothesis)
        neg_asym = _negation_asymmetry(p_tokens, h_tokens)

        features: Dict[str, float] = {
            "lexical_overlap": lex_overlap,
            "content_overlap": cont_overlap,
            "entity_overlap": ent_overlap,
            "bigram_overlap": bg_overlap,
            "numerical_consistency": num_consistency,
            "negation_score": neg_asym,
        }

        # Compute weighted score (negation is inverted — high negation lowers score)
        w = self._weights
        total_weight = sum(w.get(k, 0.0) for k in features if k != "negation_score") + w.get("negation_score", 0.0)
        if total_weight == 0:
            total_weight = 1.0

        # Positive signals
        pos_score = (
            w.get("lexical_overlap", 0.0) * lex_overlap
            + w.get("content_overlap", 0.0) * cont_overlap
            + w.get("entity_overlap", 0.0) * ent_overlap
            + w.get("bigram_overlap", 0.0) * bg_overlap
            + w.get("numerical_consistency", 0.0) * num_consistency
        )

        # Negation penalizes the score
        neg_penalty = w.get("negation_score", 0.0) * neg_asym

        combined = (pos_score - neg_penalty) / total_weight
        combined = max(0.0, min(1.0, combined))

        # Determine label
        if _antonym_conflict(p_tokens, h_tokens) and cont_overlap > 0.30:
            features["antonym_clash"] = 1.0
            return EntailmentResult(label="CONTRADICTION", confidence=0.90, features=features)

        if _negated_scope_inversion(premise, hypothesis):
            features["negation_inversion"] = 1.0
            return EntailmentResult(label="CONTRADICTION", confidence=0.95, features=features)

        if neg_asym > 0.5 and num_consistency < 0.5:
            label = "CONTRADICTION"
            confidence = min(1.0, neg_asym * 0.5 + (1.0 - num_consistency) * 0.5)
        elif neg_asym > 0.6 and cont_overlap > 0.4:
            label = "CONTRADICTION"
            confidence = min(1.0, neg_asym * 0.6 + cont_overlap * 0.4)
        elif num_consistency < 0.4:
            label = "CONTRADICTION"
            confidence = 1.0 - num_consistency
        elif combined >= self._entailment_threshold:
            label = "ENTAILMENT"
            confidence = combined
        elif combined <= self._contradiction_threshold and (neg_asym > 0.3 or num_consistency < 0.6):
            label = "CONTRADICTION"
            confidence = 1.0 - combined
        else:
            label = "NEUTRAL"
            confidence = 1.0 - abs(combined - 0.5) * 2  # More confident near 0.5

        features["combined_score"] = combined

        return EntailmentResult(
            label=label,
            confidence=max(0.0, min(1.0, confidence)),
            features=features,
        )

    def check_batch(
        self, premises: List[str], hypotheses: List[str]
    ) -> List[EntailmentResult]:
        """Check entailment for multiple premise-hypothesis pairs.

        Args:
            premises: List of premise strings.
            hypotheses: List of hypothesis strings (same length as *premises*).

        Returns:
            List of :class:`EntailmentResult` in corresponding order.

        Raises:
            ValueError: If lists have different lengths.
        """
        if len(premises) != len(hypotheses):
            raise ValueError(
                f"Mismatched lengths: {len(premises)} premises vs "
                f"{len(hypotheses)} hypotheses"
            )
        return [self.check(p, h) for p, h in zip(premises, hypotheses)]
