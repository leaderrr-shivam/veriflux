"""
Text normalization utilities for Veriflux NLP.

Provides composable text normalization functions and a configurable
``TextNormalizer`` pipeline that chains them in user-defined order.
All functions are pure: they accept a string and return a new string.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence


# ─── Standalone normalization functions ─────────────────────────────────────


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize Unicode representation.

    Args:
        text: Input text.
        form: One of ``"NFC"``, ``"NFD"``, ``"NFKC"``, ``"NFKD"``.

    Returns:
        Unicode-normalized string.

    Raises:
        ValueError: If *form* is not a recognized Unicode normalization form.
    """
    form_upper = form.upper()
    if form_upper not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(
            f"Unknown Unicode normalization form {form!r}; "
            f"expected one of NFC, NFD, NFKC, NFKD"
        )
    return unicodedata.normalize(form_upper, text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip edges.

    Handles spaces, tabs, newlines, carriage returns, vertical tabs,
    form-feeds, non-breaking spaces (``\\u00a0``), and other Unicode
    whitespace characters.

    Args:
        text: Input text.

    Returns:
        Whitespace-normalized string.
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_case(text: str, mode: str = "lower") -> str:
    """Normalize the case of *text*.

    Args:
        text: Input text.
        mode: ``"lower"`` for lowercase, ``"upper"`` for uppercase,
              ``"casefold"`` for aggressive Unicode case-folding,
              ``"title"`` for title case.

    Returns:
        Case-normalized string.

    Raises:
        ValueError: If *mode* is unrecognised.
    """
    mode_lower = mode.lower()
    if mode_lower == "lower":
        return text.lower()
    elif mode_lower == "upper":
        return text.upper()
    elif mode_lower == "casefold":
        return text.casefold()
    elif mode_lower == "title":
        return text.title()
    else:
        raise ValueError(
            f"Unknown case mode {mode!r}; expected lower, upper, casefold, or title"
        )


# Pre-compiled pattern for control characters.
# Matches C0 controls (U+0000–U+001F except tab/newline/CR),
# DEL (U+007F), and C1 controls (U+0080–U+009F).
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


def remove_control_chars(text: str) -> str:
    """Remove non-printable control characters.

    Preserves newline (``\\n``), carriage return (``\\r``), and tab
    (``\\t``) as they are commonly meaningful in text processing.

    Args:
        text: Input text.

    Returns:
        Cleaned string with control characters stripped.
    """
    return _CONTROL_CHAR_RE.sub("", text)


# Mapping of "smart" / typographic quote characters to plain ASCII.
_SMART_QUOTE_MAP: Dict[int, str] = {
    0x2018: "'",   # LEFT SINGLE QUOTATION MARK '
    0x2019: "'",   # RIGHT SINGLE QUOTATION MARK '
    0x201A: "'",   # SINGLE LOW-9 QUOTATION MARK ‚
    0x201B: "'",   # SINGLE HIGH-REVERSED-9 QUOTATION MARK ‛
    0x201C: '"',   # LEFT DOUBLE QUOTATION MARK "
    0x201D: '"',   # RIGHT DOUBLE QUOTATION MARK "
    0x201E: '"',   # DOUBLE LOW-9 QUOTATION MARK „
    0x201F: '"',   # DOUBLE HIGH-REVERSED-9 QUOTATION MARK ‟
    0x2039: "'",   # SINGLE LEFT-POINTING ANGLE QUOTATION ‹
    0x203A: "'",   # SINGLE RIGHT-POINTING ANGLE QUOTATION ›
    0x00AB: '"',   # LEFT-POINTING DOUBLE ANGLE QUOTATION «
    0x00BB: '"',   # RIGHT-POINTING DOUBLE ANGLE QUOTATION »
    0x0060: "'",   # GRAVE ACCENT `
    0x00B4: "'",   # ACUTE ACCENT ´
    0x275B: "'",   # HEAVY SINGLE TURNED COMMA QUOTATION MARK ❛
    0x275C: "'",   # HEAVY SINGLE COMMA QUOTATION MARK ❜
    0x275D: '"',   # HEAVY DOUBLE TURNED COMMA QUOTATION MARK ❝
    0x275E: '"',   # HEAVY DOUBLE COMMA QUOTATION MARK ❞
}

_SMART_QUOTE_TABLE = str.maketrans(
    {k: v for k, v in _SMART_QUOTE_MAP.items()}
)


def normalize_quotes(text: str) -> str:
    """Normalize typographic / smart quotes to plain ASCII quotes.

    Converts single-style quotes to ``'`` and double-style quotes to ``"``.

    Args:
        text: Input text.

    Returns:
        Text with normalized quotes.
    """
    return text.translate(_SMART_QUOTE_TABLE)


# Mapping of dash-like Unicode characters to a plain ASCII hyphen.
_DASH_MAP: Dict[int, str] = {
    0x2010: "-",  # HYPHEN
    0x2011: "-",  # NON-BREAKING HYPHEN
    0x2012: "-",  # FIGURE DASH
    0x2013: "-",  # EN DASH –
    0x2014: "-",  # EM DASH —
    0x2015: "-",  # HORIZONTAL BAR ―
    0x2212: "-",  # MINUS SIGN −
    0xFE58: "-",  # SMALL EM DASH ﹘
    0xFE63: "-",  # SMALL HYPHEN-MINUS ﹣
    0xFF0D: "-",  # FULLWIDTH HYPHEN-MINUS -
}

_DASH_TABLE = str.maketrans({k: v for k, v in _DASH_MAP.items()})


def normalize_dashes(text: str) -> str:
    """Normalize em dashes, en dashes, and other dash-like characters to ``-``.

    Args:
        text: Input text.

    Returns:
        Text with normalized dashes.
    """
    return text.translate(_DASH_TABLE)


def normalize_full(text: str) -> str:
    """Apply all normalization steps in a sensible default order.

    The pipeline order is:

    1. Unicode normalization (NFC)
    2. Remove control characters
    3. Normalize quotes
    4. Normalize dashes
    5. Normalize whitespace
    6. (Case normalization is intentionally *not* included by default
       because many downstream tasks are case-sensitive.)

    Args:
        text: Input text.

    Returns:
        Fully normalized string.
    """
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = normalize_quotes(text)
    text = normalize_dashes(text)
    text = normalize_whitespace(text)
    return text


# ─── Configurable pipeline class ───────────────────────────────────────────


# Registry of built-in step names → functions.
_BUILTIN_STEPS: Dict[str, Callable[[str], str]] = {
    "unicode": normalize_unicode,
    "whitespace": normalize_whitespace,
    "lower": lambda t: normalize_case(t, "lower"),
    "upper": lambda t: normalize_case(t, "upper"),
    "casefold": lambda t: normalize_case(t, "casefold"),
    "control_chars": remove_control_chars,
    "quotes": normalize_quotes,
    "dashes": normalize_dashes,
}


class TextNormalizer:
    """Configurable text normalization pipeline.

    Build a pipeline by specifying step names (strings that reference
    built-in normalizers) and/or arbitrary callables.

    Example::

        normalizer = TextNormalizer(steps=["unicode", "whitespace", "lower"])
        clean = normalizer("  Hello   World  ")
        # => "hello world"

    Custom callables can be intermixed::

        normalizer = TextNormalizer(steps=[
            "unicode",
            lambda t: t.replace("@", " at "),
            "whitespace",
        ])

    Args:
        steps: Ordered sequence of step names (strings) and/or callables.
               If ``None``, defaults to the ``normalize_full`` pipeline.
    """

    __slots__ = ("_steps",)

    def __init__(
        self,
        steps: Optional[Sequence[str | Callable[[str], str]]] = None,
    ) -> None:
        if steps is None:
            # Mirror normalize_full
            self._steps: List[Callable[[str], str]] = [
                normalize_unicode,
                remove_control_chars,
                normalize_quotes,
                normalize_dashes,
                normalize_whitespace,
            ]
        else:
            resolved: List[Callable[[str], str]] = []
            for step in steps:
                if isinstance(step, str):
                    func = _BUILTIN_STEPS.get(step)
                    if func is None:
                        raise ValueError(
                            f"Unknown normalization step {step!r}. "
                            f"Available: {sorted(_BUILTIN_STEPS)}"
                        )
                    resolved.append(func)
                elif callable(step):
                    resolved.append(step)
                else:
                    raise TypeError(
                        f"Step must be a string or callable, got {type(step).__name__}"
                    )
            self._steps = resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, text: str) -> str:
        """Run the full pipeline on *text* and return the result."""
        for step in self._steps:
            text = step(text)
        return text

    def normalize(self, text: str) -> str:
        """Alias for ``__call__``."""
        return self(text)

    def normalize_batch(self, texts: Sequence[str]) -> List[str]:
        """Normalize a batch of texts.

        Args:
            texts: Iterable of strings.

        Returns:
            List of normalized strings in the same order.
        """
        return [self(t) for t in texts]

    @property
    def steps(self) -> List[Callable[[str], str]]:
        """Return a copy of the internal step list."""
        return list(self._steps)

    def __repr__(self) -> str:
        step_names: List[str] = []
        for s in self._steps:
            name = getattr(s, "__name__", None) or getattr(s, "__qualname__", repr(s))
            step_names.append(name)
        return f"TextNormalizer(steps={step_names})"
