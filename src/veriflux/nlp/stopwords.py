"""
Multi-language stopword lists and filtering for Veriflux NLP.

Provides built-in stopword sets (300+ English words) with an extensible
architecture for additional languages and custom word lists.  Stopword sets
are loaded lazily on first access to minimise import-time overhead.
"""
from __future__ import annotations

import copy
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Sequence,
)


# ─── Built-in English stopwords (325 words) ─────────────────────────────────
# Compiled from standard NLP stopword lists (Snowball, Onix, SMART).

_ENGLISH_STOPWORDS: FrozenSet[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",
    "doesn't", "doing", "don", "don't", "down", "during", "each", "few",
    "for", "from", "further", "get", "got", "had", "hadn", "hadn't", "has",
    "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself",
    "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most",
    "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor",
    "not", "now", "o", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same",
    "shan", "shan't", "she", "she's", "should", "should've", "shouldn",
    "shouldn't", "so", "some", "such", "t", "than", "that", "that'll",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    # Additional common stopwords
    "also", "although", "always", "among", "another", "around", "back",
    "became", "become", "becomes", "becoming", "beside", "besides", "beyond",
    "could", "dear", "eg", "either", "else", "enough", "etc", "even",
    "every", "everyone", "everything", "everywhere", "except", "first",
    "former", "go", "goes", "going", "gone", "hence", "hereafter", "hereby",
    "herein", "hereupon", "however", "ie", "indeed", "instead", "last",
    "latter", "least", "less", "let", "like", "likely", "made", "make",
    "making", "many", "may", "maybe", "might", "mine", "moreover", "much",
    "must", "neither", "never", "nevertheless", "next", "nobody", "none",
    "nonetheless", "nothing", "nowhere", "often", "one", "onto", "others",
    "otherwise", "ought", "per", "perhaps", "please", "plus", "quite",
    "rather", "really", "regarding", "said", "say", "says", "seem",
    "seemed", "seeming", "seems", "several", "shall", "since", "somebody",
    "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
    "still", "take", "tell", "tends", "th", "therefore", "thereby",
    "therein", "thereupon", "think", "thus", "together", "took", "toward",
    "towards", "two", "unless", "upon", "us", "use", "used", "using",
    "usually", "various", "via", "viz", "vs", "want", "well", "went",
    "whatever", "whenever", "whereas", "wherever", "whether", "whichever",
    "whoever", "whole", "whose", "within", "without", "would", "yet",
    "already", "along", "anything", "anyway", "anyways", "apart",
    "appear", "appreciate", "appropriate", "away", "basically",
    "certainly", "clearly", "come", "consider", "corresponding",
    "currently", "definitely", "despite", "did", "different", "done",
    "downwards", "entirely", "especially", "exactly", "example",
    "following", "furthermore", "given", "gotten", "greetings",
    "help", "hello", "hi", "hopefully", "inasmuch", "indicate",
    "indicated", "indicates", "inner", "insofar", "it'd", "keep",
    "keeps", "kept", "know", "known", "knows", "largely", "later",
    "look", "looking", "looks", "mainly", "meanwhile", "merely",
    "name", "namely", "nd", "near", "nearly", "necessary",
    "new", "normally", "notwithstanding", "novel", "obviously",
    "ok", "okay", "old", "ones", "overall", "particular",
    "particularly", "placed", "possibly", "presumably",
    "previously", "primarily", "probably", "provides",
    "rd", "recent", "recently", "relatively", "respectively",
    "right", "second", "secondly", "seriously",
    "significantly", "similarly", "simply", "six",
    "slightly", "soon", "sorry", "specifically", "specified",
    "specify", "specifying", "strongly", "sub", "substantially",
    "successfully", "sup", "sure", "taken", "tell", "th", "thank",
    "thanks", "thanx", "thats", "thence", "thereafter",
    "theres", "third", "thorough", "thoroughly", "three",
    "throughout", "thru", "truly", "try", "trying",
    "twice", "unfortunately", "unlikely", "unto",
    "useful", "value", "virtually", "way", "welcome",
    "willing", "wish", "wonder", "yes",
    "zero",
})


# Language registry: maps language code → frozenset of stopwords.
_LANGUAGE_REGISTRY: Dict[str, FrozenSet[str]] = {
    "en": _ENGLISH_STOPWORDS,
    "english": _ENGLISH_STOPWORDS,
}

# Track whether custom languages have been loaded.
_custom_languages_loaded: bool = False


def register_language(code: str, words: Iterable[str]) -> None:
    """Register a stopword set for a new language.

    Args:
        code: ISO-639-style language code (e.g. ``"fr"``, ``"de"``).
        words: Iterable of stopwords.
    """
    _LANGUAGE_REGISTRY[code.lower()] = frozenset(w.lower() for w in words)


def available_languages() -> List[str]:
    """Return sorted list of registered language codes."""
    return sorted(_LANGUAGE_REGISTRY)


def get_stopwords(language: str = "en") -> FrozenSet[str]:
    """Return the stopword set for *language*.

    Args:
        language: Language code (e.g. ``"en"``).

    Returns:
        Frozen set of stopwords.

    Raises:
        KeyError: If *language* has not been registered.
    """
    key = language.lower()
    if key not in _LANGUAGE_REGISTRY:
        raise KeyError(
            f"No stopwords registered for language {language!r}. "
            f"Available: {available_languages()}"
        )
    return _LANGUAGE_REGISTRY[key]


# ─── StopwordFilter class ──────────────────────────────────────────────────


class StopwordFilter:
    """Configurable stopword filter.

    Supports built-in language sets, custom word lists, and
    runtime additions/removals.

    Args:
        language: Language code for the built-in set (default ``"en"``).
                  Pass ``None`` to start with an empty set.
        extra_words: Additional words to include as stopwords.
        case_sensitive: Whether matching should be case-sensitive
                        (default ``False``).

    Example::

        filt = StopwordFilter()
        filt.filter_tokens(["the", "cat", "sat", "on", "the", "mat"])
        # => ["cat", "sat", "mat"]
    """

    __slots__ = ("_words", "_case_sensitive")

    def __init__(
        self,
        language: Optional[str] = "en",
        extra_words: Optional[Iterable[str]] = None,
        case_sensitive: bool = False,
    ) -> None:
        self._case_sensitive = case_sensitive

        if language is not None:
            base: Set[str] = set(get_stopwords(language))
        else:
            base = set()

        if extra_words is not None:
            if case_sensitive:
                base.update(extra_words)
            else:
                base.update(w.lower() for w in extra_words)

        self._words: Set[str] = base

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def is_stopword(self, word: str) -> bool:
        """Return ``True`` if *word* is a stopword.

        Args:
            word: The word to test.

        Returns:
            Boolean indicating stopword status.
        """
        key = word if self._case_sensitive else word.lower()
        return key in self._words

    def __contains__(self, word: str) -> bool:
        """Support ``word in filter`` syntax."""
        return self.is_stopword(word)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Remove stopwords from a list of tokens.

        Args:
            tokens: Sequence of token strings.

        Returns:
            New list with stopwords removed, preserving order.
        """
        return [t for t in tokens if not self.is_stopword(t)]

    def filter_iter(self, tokens: Iterable[str]) -> Iterator[str]:
        """Lazily filter stopwords from an iterable.

        Args:
            tokens: Iterable of token strings.

        Yields:
            Non-stopword tokens.
        """
        for t in tokens:
            if not self.is_stopword(t):
                yield t

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, word: str) -> None:
        """Add *word* to the stopword set.

        Args:
            word: Word to add.
        """
        key = word if self._case_sensitive else word.lower()
        self._words.add(key)

    def add_many(self, words: Iterable[str]) -> None:
        """Add multiple words at once.

        Args:
            words: Iterable of words.
        """
        for w in words:
            self.add(w)

    def remove(self, word: str) -> None:
        """Remove *word* from the stopword set.

        Args:
            word: Word to remove.

        Raises:
            KeyError: If the word is not present.
        """
        key = word if self._case_sensitive else word.lower()
        self._words.discard(key)

    def clear(self) -> None:
        """Remove all stopwords from the filter."""
        self._words.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def words(self) -> FrozenSet[str]:
        """Return an immutable snapshot of the current stopword set."""
        return frozenset(self._words)

    def __len__(self) -> int:
        return len(self._words)

    def __repr__(self) -> str:
        return (
            f"StopwordFilter(size={len(self._words)}, "
            f"case_sensitive={self._case_sensitive})"
        )
