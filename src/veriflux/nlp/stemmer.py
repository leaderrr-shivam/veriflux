"""
Porter Stemming Algorithm — complete from-scratch implementation.

Implements all five steps of the original Porter stemmer as described in:

    Porter, M.F. (1980). "An algorithm for suffix stripping."
    *Program*, 14(3), 130–137.

This is a faithful, production-grade implementation covering every rule
in all five steps (1a, 1b, 1c, 2, 3, 4, 5a, 5b).
"""
from __future__ import annotations

from typing import List, Sequence


def _is_consonant(word: str, i: int) -> bool:
    """Return ``True`` if ``word[i]`` is a consonant in Porter's sense.

    A letter is a consonant unless it is:
    - a vowel (a, e, i, o, u), or
    - 'y' preceded by a consonant (making 'y' a vowel).

    Args:
        word: The word being analysed.
        i: Character index.

    Returns:
        ``True`` if the character at position *i* is a consonant.
    """
    ch = word[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        if i == 0:
            return True
        return not _is_consonant(word, i - 1)
    return True


class PorterStemmer:
    """Porter stemmer with all five algorithmic steps.

    Example::

        stemmer = PorterStemmer()
        stemmer.stem("running")   # => "run"
        stemmer.stem("caresses")  # => "caress"
    """

    __slots__ = ()

    # ------------------------------------------------------------------
    # Helper (measuring) methods
    # ------------------------------------------------------------------

    @staticmethod
    def _measure(stem: str) -> int:
        """Compute the *measure* m of a stem.

        The measure counts the number of vowel-consonant (VC) sequences.
        For example::

            TR          → 0   (no vowel)
            EE          → 0   (no consonant after vowel)
            TREE        → 0
            Y           → 1
            BY          → 0
            TROUBLE     → 1
            OATS        → 1
            TREES       → 1
            IVY         → 1
            TROUBLES    → 2
            PRIVATE     → 2
            OATEN       → 2

        Args:
            stem: The stem to measure.

        Returns:
            Integer measure ≥ 0.
        """
        n = len(stem)
        if n == 0:
            return 0

        # Build a consonant/vowel pattern.
        i = 0
        # Skip leading consonants.
        while i < n and _is_consonant(stem, i):
            i += 1

        m = 0
        while i < n:
            # Skip vowels.
            while i < n and not _is_consonant(stem, i):
                i += 1
            if i >= n:
                break
            # Skip consonants — this completes one VC pair.
            while i < n and _is_consonant(stem, i):
                i += 1
            m += 1

        return m

    @staticmethod
    def _has_vowel(stem: str) -> bool:
        """Return ``True`` if *stem* contains at least one vowel.

        Args:
            stem: The stem string.

        Returns:
            Boolean.
        """
        for i in range(len(stem)):
            if not _is_consonant(stem, i):
                return True
        return False

    @staticmethod
    def _ends_double_consonant(stem: str) -> bool:
        """Return ``True`` if *stem* ends with a double consonant.

        Args:
            stem: The stem string.

        Returns:
            Boolean.
        """
        if len(stem) < 2:
            return False
        if stem[-1] != stem[-2]:
            return False
        return _is_consonant(stem, len(stem) - 1)

    @staticmethod
    def _cvc(stem: str) -> bool:
        """Return ``True`` if *stem* ends with consonant-vowel-consonant,
        where the final consonant is not ``w``, ``x``, or ``y``.

        This is the *o condition from the original algorithm.

        Args:
            stem: The stem string.

        Returns:
            Boolean.
        """
        n = len(stem)
        if n < 3:
            return False
        if (
            _is_consonant(stem, n - 1)
            and not _is_consonant(stem, n - 2)
            and _is_consonant(stem, n - 3)
        ):
            if stem[-1] not in "wxy":
                return True
        return False

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def _step1a(self, word: str) -> str:
        """Step 1a: Plurals.

        Rules::

            SSES  →  SS
            IES   →  I
            SS    →  SS
            S     →  (delete)
        """
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step1b(self, word: str) -> str:
        """Step 1b: Past tenses and progressive forms.

        Rules::

            (m > 0) EED   →  EE
            (*v*) ED      →  (delete, then step1b_extra)
            (*v*) ING     →  (delete, then step1b_extra)

        Step 1b extra rules (applied after ED/ING deletion)::

            AT  →  ATE
            BL  →  BLE
            IZ  →  IZE
            (*d and not (*L or *S or *Z))  →  single letter
            (m == 1 and *o)  →  E
        """
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return word[:-1]  # EED → EE
            return word

        changed = False
        if word.endswith("ed"):
            stem = word[:-2]
            if self._has_vowel(stem):
                word = stem
                changed = True
        elif word.endswith("ing"):
            stem = word[:-3]
            if self._has_vowel(stem):
                word = stem
                changed = True

        if changed:
            if word.endswith("at"):
                return word + "e"
            if word.endswith("bl"):
                return word + "e"
            if word.endswith("iz"):
                return word + "e"
            if self._ends_double_consonant(word) and not word.endswith(
                ("l", "s", "z")
            ):
                return word[:-1]
            if self._measure(word) == 1 and self._cvc(word):
                return word + "e"

        return word

    def _step1c(self, word: str) -> str:
        """Step 1c: Suffix Y.

        Rule::

            (*v*) Y  →  I
        """
        if word.endswith("y"):
            stem = word[:-1]
            if self._has_vowel(stem):
                return stem + "i"
        return word

    def _step2(self, word: str) -> str:
        """Step 2: Map double suffixes to single ones.

        Each rule requires ``m > 0`` for the stem portion preceding
        the suffix.
        """
        rules: list[tuple[str, str]] = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
            ("logi", "log"),
        ]
        for suffix, replacement in rules:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step3(self, word: str) -> str:
        """Step 3: Remove further suffixes.

        Each rule requires ``m > 0``.
        """
        rules: list[tuple[str, str]] = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ]
        for suffix, replacement in rules:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step4(self, word: str) -> str:
        """Step 4: Remove final suffixes.

        Each rule requires ``m > 1``.
        """
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
            "ous", "ive", "ize",
        ]
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if suffix == "ion":
                    # Special: the stem must end in 's' or 't'.
                    if self._measure(stem) > 1 and len(stem) > 0 and stem[-1] in "st":
                        return stem
                    return word
                if self._measure(stem) > 1:
                    return stem
                return word
        return word

    def _step5a(self, word: str) -> str:
        """Step 5a: Remove trailing ``e``.

        Rules::

            (m > 1) E  →  (delete)
            (m == 1 and not *o) E  →  (delete)
        """
        if word.endswith("e"):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1:
                return stem
            if m == 1 and not self._cvc(stem):
                return stem
        return word

    def _step5b(self, word: str) -> str:
        """Step 5b: Remove double ``l``.

        Rule::

            (m > 1 and *d and *L)  →  single l
        """
        if (
            self._measure(word) > 1
            and self._ends_double_consonant(word)
            and word.endswith("l")
        ):
            return word[:-1]
        return word

    # ------------------------------------------------------------------
    # Main stem method
    # ------------------------------------------------------------------

    def stem(self, word: str) -> str:
        """Stem a single word using the Porter algorithm.

        Words of length ≤ 2 are returned unchanged.

        Args:
            word: Word to stem.

        Returns:
            Stemmed word (lower-cased).
        """
        word = word.lower().strip()
        if len(word) <= 2:
            return word

        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5a(word)
        word = self._step5b(word)

        return word

    def stem_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Stem a list of tokens.

        Args:
            tokens: Sequence of token strings.

        Returns:
            List of stemmed tokens.
        """
        return [self.stem(t) for t in tokens]

    def __repr__(self) -> str:
        return "PorterStemmer()"


# ─── Module-level convenience ──────────────────────────────────────────────

_DEFAULT_STEMMER = PorterStemmer()


def stem_tokens(tokens: Sequence[str]) -> List[str]:
    """Convenience function: stem a list of tokens with the default Porter stemmer.

    Args:
        tokens: Sequence of token strings.

    Returns:
        List of stemmed tokens.
    """
    return _DEFAULT_STEMMER.stem_tokens(tokens)
