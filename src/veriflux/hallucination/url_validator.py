"""URL and citation validation for hallucination detection.

Detects fabricated URLs, suspicious DOIs, made-up arXiv IDs, and
extracts academic-style citations from text — all via regex heuristics
without any network requests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Set, Tuple


# ─── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SuspiciousURL:
    """A URL flagged as potentially fabricated.

    Attributes:
        url: The full URL string.
        reason: Human-readable explanation of why it is suspicious.
        confidence: Confidence that the URL is fabricated, in ``[0.0, 1.0]``.
    """
    url: str
    reason: str
    confidence: float


@dataclass(frozen=True, slots=True)
class Citation:
    """An academic-style citation extracted from text.

    Attributes:
        text: The extracted citation text.
        citation_type: Type of citation (e.g. ``"doi"``, ``"arxiv"``,
                       ``"author_year"``, ``"numbered"``, ``"url"``).
        raw_match: The raw regex match string.
    """
    text: str
    citation_type: str
    raw_match: str


# ─── Compiled regex patterns ───────────────────────────────────────────────

# General URL pattern
_URL_RE = re.compile(
    r"https?://"
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,63}"
    r"(?::\d{1,5})?"
    r"(?:/[^\s\]\)\},\"'<>]*)?"
    r"|"
    r"(?:www\.)"
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,63}"
    r"(?::\d{1,5})?"
    r"(?:/[^\s\]\)\},\"'<>]*)?",
    re.IGNORECASE,
)

# DOI pattern
_DOI_RE = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)"
    r"(10\.\d{4,}/[^\s,;\"'<>\]\)]+)",
    re.IGNORECASE,
)

# arXiv pattern
_ARXIV_RE = re.compile(
    r"(?:arxiv:\s*|https?://arxiv\.org/(?:abs|pdf)/)"
    r"(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)

# Valid TLDs (common subset for heuristic checking)
_VALID_TLDS: FrozenSet[str] = frozenset({
    "com", "org", "net", "edu", "gov", "mil", "int",
    "io", "co", "us", "uk", "de", "fr", "jp", "cn", "au", "ca",
    "in", "br", "ru", "it", "es", "nl", "se", "no", "fi", "dk",
    "pl", "pt", "ch", "at", "be", "cz", "ie", "nz", "za", "kr",
    "tw", "hk", "sg", "mx", "ar", "cl", "id", "th", "my", "ph",
    "vn", "info", "biz", "name", "pro", "museum", "coop", "aero",
    "app", "dev", "ai", "ml", "tech", "cloud", "online", "site",
    "store", "blog", "xyz", "top", "wang", "club", "space",
    "me", "tv", "cc", "ws", "mobi", "asia", "tel", "travel",
    "health", "science", "engineering", "academy",
})

# Suspicious domain patterns
_SUSPICIOUS_DOMAIN_PATTERNS: List[Tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"example\.com", re.IGNORECASE), "Uses example.com placeholder domain", 0.95),
    (re.compile(r"test\.com", re.IGNORECASE), "Uses test.com placeholder domain", 0.8),
    (re.compile(r"\bfake(?:site|url|domain|page|[-a-z0-9]*)", re.IGNORECASE), "Domain contains 'fake' indicator", 0.9),
    (re.compile(r"(?:lorem|ipsum|dolor|sit|amet)", re.IGNORECASE), "Domain contains Lorem Ipsum text", 0.85),
    (re.compile(r"sample(?:site|page|url|[-a-z0-9]*)", re.IGNORECASE), "Domain contains 'sample' indicator", 0.8),
    (re.compile(r"placeholder", re.IGNORECASE), "Domain contains 'placeholder'", 0.85),
]

# Suspicious URL path patterns
_SUSPICIOUS_PATH_PATTERNS: List[Tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"/(?:article|paper|study)/\d{10,}"), "Implausibly long article ID", 0.7),
    (re.compile(r"/[a-f0-9]{32,}"), "Path has very long hex string (possibly fabricated hash)", 0.6),
    (re.compile(r"(?:foo|bar|baz|qux|test|sample|example)/"), "Path uses placeholder names", 0.75),
]

# Citation patterns
_AUTHOR_YEAR_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+|&\s+[A-Z][a-z]+))?)"
    r"\s*\((\d{4})\)",
)

_NUMBERED_CITE_RE = re.compile(r"\[(\d{1,3})\]")

_PARENTHETICAL_YEAR_RE = re.compile(
    r"\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?),?\s*(\d{4})\)",
)


# ─── Helper functions ───────────────────────────────────────────────────────


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL string."""
    # Strip protocol
    domain = url
    for prefix in ("https://", "http://", "www."):
        if domain.lower().startswith(prefix):
            domain = domain[len(prefix):]

    # Strip path
    slash_idx = domain.find("/")
    if slash_idx != -1:
        domain = domain[:slash_idx]

    # Strip port
    colon_idx = domain.find(":")
    if colon_idx != -1:
        domain = domain[:colon_idx]

    return domain.lower()


def _extract_tld(domain: str) -> str:
    """Extract the top-level domain from a domain string."""
    parts = domain.rstrip(".").split(".")
    if len(parts) >= 2:
        return parts[-1].lower()
    return domain.lower()


def _is_plausible_doi(doi: str) -> bool:
    """Check if a DOI string looks plausible (basic format check)."""
    # DOIs should be 10.XXXX/something
    if not doi.startswith("10."):
        return False
    parts = doi.split("/", 1)
    if len(parts) != 2:
        return False
    prefix = parts[0]  # e.g., 10.1234
    suffix = parts[1]

    # Prefix should be 10. followed by 4-9 digits
    prefix_digits = prefix[3:]
    if not prefix_digits.replace(".", "").isdigit():
        return False
    if len(prefix_digits) < 4 or len(prefix_digits) > 9:
        return False

    # Suffix should not be empty and not too long
    if not suffix or len(suffix) > 100:
        return False

    # Check for obviously fake suffixes
    fake_patterns = ["abc", "xyz", "test", "fake", "example", "sample", "1234567890"]
    suffix_lower = suffix.lower()
    for fp in fake_patterns:
        if suffix_lower == fp:
            return False

    return True


def _is_plausible_arxiv(arxiv_id: str) -> bool:
    """Check if an arXiv ID looks plausible."""
    # New format: YYMM.NNNNN
    match = re.match(r"(\d{2})(\d{2})\.(\d{4,5})(?:v\d+)?$", arxiv_id)
    if not match:
        return False
    year = int(match.group(1))
    month = int(match.group(2))
    # arXiv IDs started in ~0704 (2007 April) for new format
    if year < 7 or year > 40:  # Up to 2040
        return False
    if month < 1 or month > 12:
        return False
    return True


# ─── Main class ─────────────────────────────────────────────────────────────


class URLValidator:
    """Detect fabricated URLs and citations in text.

    Uses pattern matching and heuristic checks to identify URLs that
    are likely hallucinated by an AI model, without making any network
    requests.

    Example::

        validator = URLValidator()
        urls = validator.extract_urls("See https://example.com/article/123")
        suspicious = validator.find_suspicious_urls("See https://example.com/article/123")
    """

    __slots__ = ("_extra_suspicious_domains",)

    def __init__(
        self,
        extra_suspicious_domains: List[str] | None = None,
    ) -> None:
        self._extra_suspicious_domains: Set[str] = set(
            d.lower() for d in (extra_suspicious_domains or [])
        )

    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from *text*.

        Args:
            text: Input text to scan.

        Returns:
            List of URL strings found in the text.
        """
        urls: List[str] = []
        for match in _URL_RE.finditer(text):
            url = match.group().rstrip(".,;:!?)")
            if url not in urls:
                urls.append(url)
        return urls

    def validate_url_format(self, url: str) -> bool:
        """Check whether *url* has a valid URL format.

        Args:
            url: URL string to validate.

        Returns:
            ``True`` if the URL format is valid.
        """
        # Must have a scheme or www prefix
        has_scheme = url.lower().startswith(("http://", "https://"))
        has_www = url.lower().startswith("www.")

        if not has_scheme and not has_www:
            return False

        domain = _extract_domain(url)
        if not domain:
            return False

        # Domain must have at least one dot
        if "." not in domain:
            return False

        # TLD must be alphabetic
        tld = _extract_tld(domain)
        if not tld.isalpha():
            return False

        # TLD length should be reasonable (2-20 chars)
        if len(tld) < 2 or len(tld) > 20:
            return False

        return True

    def find_suspicious_urls(self, text: str) -> List[SuspiciousURL]:
        """Find URLs that look potentially fabricated.

        Args:
            text: Input text to scan.

        Returns:
            List of :class:`SuspiciousURL` objects for suspicious URLs found.
        """
        urls = self.extract_urls(text)
        suspicious: List[SuspiciousURL] = []

        for url in urls:
            reasons: List[Tuple[str, float]] = []

            # Check URL format
            if not self.validate_url_format(url):
                reasons.append(("Invalid URL format", 0.9))

            domain = _extract_domain(url)
            tld = _extract_tld(domain)

            # Check for non-existent TLDs
            if tld and tld not in _VALID_TLDS and len(tld) > 3:
                reasons.append((f"Unusual TLD '.{tld}'", 0.6))

            # Check suspicious domain patterns
            for pattern, reason, conf in _SUSPICIOUS_DOMAIN_PATTERNS:
                if pattern.search(domain):
                    reasons.append((reason, conf))

            # Check suspicious path patterns
            for pattern, reason, conf in _SUSPICIOUS_PATH_PATTERNS:
                if pattern.search(url):
                    reasons.append((reason, conf))

            # Check for extra suspicious domains
            for sus_domain in self._extra_suspicious_domains:
                if sus_domain in domain:
                    reasons.append((f"Domain matches suspicious pattern '{sus_domain}'", 0.8))

            # Check DOIs
            doi_match = _DOI_RE.search(url)
            if doi_match:
                doi_str = doi_match.group(1)
                if not _is_plausible_doi(doi_str):
                    reasons.append(("DOI format appears implausible", 0.75))

            # Check arXiv IDs
            arxiv_match = _ARXIV_RE.search(url)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                if not _is_plausible_arxiv(arxiv_id):
                    reasons.append(("arXiv ID appears implausible", 0.75))

            # Check for very long random-looking paths
            path_start = url.find("/", url.find("//") + 2) if "//" in url else url.find("/")
            if path_start != -1:
                path = url[path_start:]
                # Extremely long paths are suspicious
                if len(path) > 200:
                    reasons.append(("Unusually long URL path", 0.6))
                # Check for random character sequences
                alpha_seq = re.findall(r"[a-z]{15,}", path.lower())
                if alpha_seq:
                    for seq in alpha_seq:
                        # Check if the sequence looks random (low character variety repetition)
                        unique_ratio = len(set(seq)) / len(seq)
                        if unique_ratio > 0.7 and len(seq) > 20:
                            reasons.append(("Path contains random-looking character sequence", 0.55))
                            break

            if reasons:
                # Aggregate confidence: 1 - product(1 - conf_i)
                combined_conf = 1.0
                for _, conf in reasons:
                    combined_conf *= (1.0 - conf)
                combined_conf = 1.0 - combined_conf

                combined_reason = "; ".join(r for r, _ in reasons)
                suspicious.append(SuspiciousURL(
                    url=url,
                    reason=combined_reason,
                    confidence=min(1.0, combined_conf),
                ))

        return suspicious

    def extract_citations(self, text: str) -> List[Citation]:
        """Extract academic-style citations from *text*.

        Recognizes author-year, numbered, DOI, arXiv, and URL citations.

        Args:
            text: Input text to scan.

        Returns:
            List of :class:`Citation` objects.
        """
        citations: List[Citation] = []
        seen: Set[str] = set()

        # DOI citations
        for match in _DOI_RE.finditer(text):
            raw = match.group()
            if raw not in seen:
                seen.add(raw)
                citations.append(Citation(
                    text=match.group(1),
                    citation_type="doi",
                    raw_match=raw,
                ))

        # arXiv citations
        for match in _ARXIV_RE.finditer(text):
            raw = match.group()
            if raw not in seen:
                seen.add(raw)
                citations.append(Citation(
                    text=match.group(1),
                    citation_type="arxiv",
                    raw_match=raw,
                ))

        # Author-year citations: "Smith (2020)" or "Smith et al. (2020)"
        for match in _AUTHOR_YEAR_RE.finditer(text):
            raw = match.group()
            if raw not in seen:
                seen.add(raw)
                citations.append(Citation(
                    text=f"{match.group(1)}, {match.group(2)}",
                    citation_type="author_year",
                    raw_match=raw,
                ))

        # Parenthetical citations: "(Smith, 2020)" or "(Smith et al., 2020)"
        for match in _PARENTHETICAL_YEAR_RE.finditer(text):
            raw = match.group()
            if raw not in seen:
                seen.add(raw)
                citations.append(Citation(
                    text=f"{match.group(1)}, {match.group(2)}",
                    citation_type="author_year",
                    raw_match=raw,
                ))

        # Numbered citations: [1], [23]
        for match in _NUMBERED_CITE_RE.finditer(text):
            raw = match.group()
            if raw not in seen:
                seen.add(raw)
                citations.append(Citation(
                    text=match.group(1),
                    citation_type="numbered",
                    raw_match=raw,
                ))

        # URL citations
        for url in self.extract_urls(text):
            if url not in seen:
                seen.add(url)
                citations.append(Citation(
                    text=url,
                    citation_type="url",
                    raw_match=url,
                ))

        return citations

    def has_fabricated_urls(self, text: str, threshold: float = 0.6) -> bool:
        """Check if text likely contains fabricated URLs.

        Args:
            text: Text to check.
            threshold: Confidence threshold for considering a URL fabricated.

        Returns:
            ``True`` if any URL exceeds the suspicion threshold.
        """
        suspicious = self.find_suspicious_urls(text)
        return any(s.confidence >= threshold for s in suspicious)
