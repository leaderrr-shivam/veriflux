"""
Sentence segmentation for Veriflux NLP.

Splits running text into individual sentences with high accuracy,
handling abbreviations, decimal numbers, ellipses, URLs, email
addresses, parenthetical sentences, and quoted speech.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ─── Span dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Span:
    """A contiguous span of text with character offsets.

    Attributes:
        start: Start character offset (inclusive).
        end: End character offset (exclusive).
        text: The text content of this span.
    """
    start: int
    end: int
    text: str


# ─── Abbreviation sets ─────────────────────────────────────────────────────

_HONORIFICS: frozenset[str] = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "rev", "gov", "sgt",
    "cpl", "pvt", "capt", "cmdr", "lt", "col", "gen", "cpt", "maj",
    "brig", "hon", "pres", "rep", "sen", "amb", "atty", "st",
})

# 100+ common abbreviations whose trailing period does NOT signal a
# sentence boundary.  All stored lowercase.
_ABBREVIATIONS: frozenset[str] = frozenset({
    # Titles
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "rev", "gov", "sgt",
    "cpl", "pvt", "capt", "cmdr", "lt", "col", "gen", "cpt", "maj",
    "brig", "hon", "pres", "msgr", "fr", "br", "supt", "det", "insp",
    "const", "rep", "sen", "amb", "treas", "sec", "atty",
    # Months
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    # Days
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat",
    "sun",
    # US states (selected common)
    "ala", "ariz", "ark", "calif", "colo", "conn", "del", "fla", "ill",
    "ind", "kan", "ky", "la", "md", "mass", "mich", "minn", "miss",
    "mo", "mont", "neb", "nev", "okla", "ore", "pa", "tenn", "tex",
    "vt", "va", "wash", "wis", "wyo",
    # Geography / address
    "st", "ave", "blvd", "rd", "hwy", "rt", "mt", "ft",
    # Academic / professional
    "ph", "dept", "univ", "assn", "bros", "inc", "ltd", "co", "corp",
    "assoc", "natl", "intl",
    # Common abbreviations
    "etc", "vs", "approx", "appt", "apt", "dept", "dpt", "est", "min",
    "max", "misc", "no", "vol", "govt", "fig", "figs", "eq", "eqs",
    "ref", "refs", "ch", "sec", "para", "pp", "pg", "ed", "eds",
    "trans", "publ", "rev", "tel", "fax", "ext",
    # Latin
    "al", "eg", "ie", "cf", "viz",
    # Metric / units
    "oz", "pt", "qt", "gal", "lb", "lbs", "ft", "yd", "mi",
    "cm", "mm", "km", "mg", "kg", "ml",
    # Multi-letter abbreviations (uppercase patterns handled separately)
    "u.s", "u.s.a", "u.k", "e.g", "i.e", "a.m", "p.m", "b.c", "a.d",
    "ph.d", "m.d", "b.a", "m.a", "d.d.s", "r.n", "b.sc", "m.sc",
})


# ─── Regex patterns ────────────────────────────────────────────────────────

# URL pattern (simplified but covers most common cases).
_URL_RE = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)

# Email pattern.
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

# Decimal number pattern (e.g. 3.14, $12.50).
_DECIMAL_RE = re.compile(
    r"\d+\.\d+",
)

# Ellipsis.
_ELLIPSIS_RE = re.compile(r"\.{2,}")

# Initials like "U.S.A.", "U.S.", "A.B.C."
_INITIALS_RE = re.compile(
    r"(?:[A-Z]\.){2,}[A-Z]?\.?"
)

# Sentence-ending punctuation.
_SENT_END_RE = re.compile(r"[.!?]")


class SentenceSplitter:
    """Rule-based sentence segmentation.

    Handles abbreviations, decimal numbers, ellipses, URLs, emails,
    parenthetical sentences, and quoted text.

    Args:
        abbreviations: Optional additional abbreviations (lowercase,
                       without trailing period).

    Example::

        splitter = SentenceSplitter()
        splitter.split("Dr. Smith went to Washington. He arrived at 3 p.m.")
        # => ['Dr. Smith went to Washington.', 'He arrived at 3 p.m.']
    """

    __slots__ = ("_abbreviations",)

    def __init__(
        self,
        abbreviations: Optional[Sequence[str]] = None,
    ) -> None:
        if abbreviations is not None:
            self._abbreviations = _ABBREVIATIONS | frozenset(
                a.lower().rstrip(".") for a in abbreviations
            )
        else:
            self._abbreviations = _ABBREVIATIONS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _protect_spans(self, text: str) -> List[Tuple[int, int]]:
        """Find spans that should never be split on (URLs, emails, etc.)."""
        protected: List[Tuple[int, int]] = []
        for pattern in (_URL_RE, _EMAIL_RE, _DECIMAL_RE, _ELLIPSIS_RE, _INITIALS_RE):
            for m in pattern.finditer(text):
                protected.append((m.start(), m.end()))
        return protected

    def _in_protected(self, pos: int, protected: List[Tuple[int, int]]) -> bool:
        """Return ``True`` if *pos* falls within a protected span."""
        for start, end in protected:
            if start <= pos < end:
                return True
        return False

    def _is_abbreviation(self, text: str, dot_pos: int) -> bool:
        """Check if the period at *dot_pos* is part of an abbreviation."""
        # Walk backwards to find the token preceding the dot.
        i = dot_pos - 1
        while i >= 0 and text[i].isalpha():
            i -= 1
        token = text[i + 1 : dot_pos].lower()
        if not token:
            return False

        # Check direct match.
        if token in self._abbreviations:
            return True

        # Check dotted abbreviations like "u.s" — collect all parts.
        # Walk further back to pick up patterns like "U.S"
        j = i
        while j >= 0 and (text[j].isalpha() or text[j] == "."):
            j -= 1
        full_token = text[j + 1 : dot_pos].lower().rstrip(".")
        if full_token in self._abbreviations:
            return True

        # Single uppercase letter followed by period (initial): "J."
        if len(token) == 1 and token.isalpha():
            return True

        return False

    def _is_sentence_boundary(
        self,
        text: str,
        pos: int,
        protected: List[Tuple[int, int]],
    ) -> bool:
        """Determine whether the punctuation at *pos* marks a sentence end."""
        if self._in_protected(pos, protected):
            return False

        ch = text[pos]
        length = len(text)

        if ch == ".":
            # Check for abbreviation.
            if self._is_abbreviation(text, pos):
                # Honorifics (Dr., Mr., etc.) are followed by capitalized names and do NOT end sentences
                i = pos - 1
                while i >= 0 and text[i].isalpha():
                    i -= 1
                prev_token = text[i + 1 : pos].lower()
                if prev_token in _HONORIFICS:
                    return False

                # Other abbreviations might end a sentence if followed by an uppercase letter
                next_pos = pos + 1
                while next_pos < length and text[next_pos] in " \t":
                    next_pos += 1
                if next_pos >= length:
                    return True
                if text[next_pos].isupper():
                    # Still could be a middle initial; check if it looks
                    # like a new sentence.  Heuristic: if the next word is
                    # longer than 1 char or followed by lowercase, treat
                    # as sentence start.
                    word_end = next_pos + 1
                    while word_end < length and text[word_end].isalpha():
                        word_end += 1
                    next_word = text[next_pos:word_end]
                    if len(next_word) > 1:
                        return True
                    # Single uppercase letter — probably not a sentence boundary.
                    return False
                return False

            # Check for decimal number: digit before and after the period.
            if pos > 0 and text[pos - 1].isdigit():
                next_pos = pos + 1
                if next_pos < length and text[next_pos].isdigit():
                    return False

            # Regular period: it's a boundary if followed by whitespace+uppercase
            # or end of string.
            next_pos = pos + 1
            while next_pos < length and text[next_pos] in " \t\n\r":
                next_pos += 1
            if next_pos >= length:
                return True
            # Must be followed by uppercase, opening quote, or opening paren.
            if text[next_pos].isupper() or text[next_pos] in "\"'([":
                return True
            return False

        if ch in "!?":
            # Generally always a sentence boundary.
            # But not if it's part of a sequence like "?!" or inside quotes.
            next_pos = pos + 1
            # Skip additional punctuation in the same terminator cluster.
            while next_pos < length and text[next_pos] in "!?":
                next_pos += 1
            # Skip closing quotes/parens that belong to this sentence.
            while next_pos < length and text[next_pos] in "\"')]:":
                next_pos += 1
            # Now expect whitespace or end of string.
            if next_pos >= length:
                return True
            while next_pos < length and text[next_pos] in " \t\n\r":
                next_pos += 1
            if next_pos >= length:
                return True
            return True

        return False

    def _find_split_end(self, text: str, pos: int) -> int:
        """Find the end position of a sentence boundary at *pos*.

        The boundary includes trailing punctuation clusters (e.g. ``?!``)
        and closing quotes/parens attached to the sentence.
        """
        length = len(text)
        end = pos + 1

        # Absorb same-type punctuation clusters (e.g. "!!!", "?!")
        while end < length and text[end] in ".!?":
            end += 1

        # Absorb closing quotes and parens.
        while end < length and text[end] in "\"')]:":
            end += 1

        return end

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text: str) -> List[str]:
        """Split *text* into sentences.

        Args:
            text: Input text (may contain multiple sentences).

        Returns:
            List of sentence strings, each stripped of leading/trailing
            whitespace.
        """
        if not text or not text.strip():
            return []

        protected = self._protect_spans(text)
        length = len(text)

        sentences: List[str] = []
        sent_start = 0
        i = 0

        while i < length:
            ch = text[i]

            if ch in ".!?":
                if self._is_sentence_boundary(text, i, protected):
                    end = self._find_split_end(text, i)
                    sentence = text[sent_start:end].strip()
                    if sentence:
                        sentences.append(sentence)
                    sent_start = end
                    i = end
                    continue

            i += 1

        # Remaining text after the last boundary.
        remainder = text[sent_start:].strip()
        if remainder:
            sentences.append(remainder)

        return sentences

    def split_spans(self, text: str) -> List[Span]:
        """Split *text* into sentences and return ``Span`` objects.

        Args:
            text: Input text.

        Returns:
            List of ``Span`` instances preserving character offsets.
        """
        if not text or not text.strip():
            return []

        protected = self._protect_spans(text)
        length = len(text)

        spans: List[Span] = []
        sent_start = 0
        i = 0

        while i < length:
            ch = text[i]

            if ch in ".!?":
                if self._is_sentence_boundary(text, i, protected):
                    end = self._find_split_end(text, i)
                    # Trim whitespace but record character offsets within
                    # the raw text.
                    s = sent_start
                    while s < end and text[s] in " \t\n\r":
                        s += 1
                    e = end
                    while e > s and text[e - 1] in " \t\n\r":
                        e -= 1
                    sentence_text = text[s:e]
                    if sentence_text:
                        spans.append(Span(start=s, end=e, text=sentence_text))
                    sent_start = end
                    i = end
                    continue

            i += 1

        # Remaining.
        s = sent_start
        while s < length and text[s] in " \t\n\r":
            s += 1
        e = length
        while e > s and text[e - 1] in " \t\n\r":
            e -= 1
        if s < e:
            spans.append(Span(start=s, end=e, text=text[s:e]))

        return spans

    def __repr__(self) -> str:
        return f"SentenceSplitter(abbreviations={len(self._abbreviations)})"


# ─── Module-level convenience ──────────────────────────────────────────────

_DEFAULT_SPLITTER = SentenceSplitter()


def split_sentences(text: str) -> List[str]:
    """Split *text* into sentences using the default splitter.

    Args:
        text: Input text.

    Returns:
        List of sentence strings.
    """
    return _DEFAULT_SPLITTER.split(text)
