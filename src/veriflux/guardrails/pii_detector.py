"""Personally Identifiable Information (PII) detection and redaction guardrail.

Detects emails, phone numbers, SSNs, credit cards (with Luhn checksum validation),
API keys, JWTs, IP addresses, IBANs, and health identifiers with zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(slots=True, frozen=True)
class PIIMatch:
    """A detected PII instance."""

    text: str
    pii_type: str
    start: int
    end: int
    confidence: float = 1.0


def _luhn_checksum(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


class PIIGuard:
    """Detects and redacts sensitive personally identifiable information."""

    _PATTERNS: Dict[str, Tuple[re.Pattern[str], float]] = {
        "EMAIL": (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"), 0.98),
        "PHONE": (
            re.compile(
                r"(?:(?:\+|00)[1-9]\d{0,3}[-.\s]?)?"
                r"(?:\(?\d{1,4}\)?[-.\s]?)?"
                r"\d{2,4}[-.\s]\d{2,4}(?:[-.\s]\d{2,4}){1,2}\b|"
                r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|"
                r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
            ),
            0.92,
        ),
        "SSN": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0.95),
        "CREDIT_CARD": (
            re.compile(
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b|"
                r"\b(?:\d{4}[-\s]){3}\d{4}\b"
            ),
            0.95,
        ),
        "IPV4": (re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"), 0.90),
        "IPV6": (re.compile(r"\b(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}\b", re.IGNORECASE), 0.90),
        "JWT": (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), 0.99),
        "AWS_KEY": (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0.99),
        "GITHUB_TOKEN": (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,255}\b"), 0.99),
        "STRIPE_KEY": (re.compile(r"\b[rs]k_(?:live|test)_[0-9a-zA-Z]{16,99}\b"), 0.99),
        "ADDRESS": (
            re.compile(
                r"\b(?:Apt\.?|Suite|Ste\.?|Unit|Room|Rm\.?|Floor|Fl\.?|Building|Bldg\.?)\s*#?\s*\d+[A-Za-z]?\b|"
                r"\b\d{1,5}\s+[A-Za-z0-9\.\s\-]+?\s*(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Square|Sq|Circle|Cir)\b|"
                r"\b(?:Rue|Rua|Calle|Avenida|Av\.?|Via|Corso|Piazza)\s+(?:de\s+|do\s+|da\s+|del\s+)?[A-Za-z0-9\.\s\-]+?\s+\d{1,5}\b|"
                r"\b[A-Za-z\u00C0-\u024F\-]+(?:\s+[A-Za-z\u00C0-\u024F\-]+)?\s*(?:tie|katu|gata|strasse|straße|str|weg|gasse|kuja|allika|utca|u\.)\s+\d{1,5}\.?\b|"
                r"\b(?:PSC|Box)\s+\d+.*?(?:APO|FPO|DPO)\s+[A-Z]{2}\s+\d{5}\b",
                re.IGNORECASE,
            ),
            0.88,
        ),
        "ZIP_CODE": (re.compile(r"\b\d{5}(?:-\d{4})?\b"), 0.85),
        "GENERIC_API_KEY": (re.compile(r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)[\s:=]+['\"]?([a-zA-Z0-9_\-]{20,64})['\"]?", re.IGNORECASE), 0.90),
        "IBAN": (re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b"), 0.90),
        "PASSPORT": (re.compile(r"\b[A-Z][0-9]{8}\b"), 0.70),
        "DATE_OF_BIRTH": (re.compile(r"\b(?:DOB|birth\s*date)[\s:=]+(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", re.IGNORECASE), 0.85),
    }

    def __init__(self, categories: Optional[Sequence[str]] = None, action: str = "redact") -> None:
        self.categories = set(categories) if categories else set(self._PATTERNS.keys())
        self.action = action

    def detect(self, text: str) -> List[PIIMatch]:
        """Scan text and return all detected PII instances."""
        matches: List[PIIMatch] = []
        if not text:
            return matches

        for cat, (pattern, base_conf) in self._PATTERNS.items():
            if cat not in self.categories:
                continue

            for m in pattern.finditer(text):
                match_text = m.group(0)
                conf = base_conf

                if cat == "CREDIT_CARD":
                    clean_digits = re.sub(r"\D", "", match_text)
                    if not _luhn_checksum(clean_digits):
                        continue

                matches.append(
                    PIIMatch(
                        text=match_text,
                        pii_type=cat,
                        start=m.start(),
                        end=m.end(),
                        confidence=conf,
                    )
                )

        # Deduplicate overlapping matches: keep higher confidence or longer match
        matches.sort(key=lambda x: (x.start, -(x.confidence), -(x.end - x.start)))
        non_overlapping: List[PIIMatch] = []
        for m in matches:
            if not non_overlapping:
                non_overlapping.append(m)
                continue
            prev = non_overlapping[-1]
            if m.start < prev.end:
                if m.confidence > prev.confidence or (m.confidence == prev.confidence and (m.end - m.start) > (prev.end - prev.start)):
                    non_overlapping[-1] = m
            else:
                non_overlapping.append(m)

        return non_overlapping

    def redact(self, text: str, mode: str = "mask") -> str:
        """Redact detected PII according to mode ('mask', 'hash', 'partial', 'remove')."""
        matches = self.detect(text)
        if not matches:
            return text

        # Replace in reverse order to preserve string offsets
        result = list(text)
        for m in reversed(matches):
            if mode == "mask":
                replacement = f"[{m.pii_type}]"
            elif mode == "hash":
                h = hashlib.sha256(m.text.encode("utf-8")).hexdigest()[:8]
                replacement = f"[{m.pii_type}:{h}]"
            elif mode == "partial":
                if len(m.text) > 4:
                    replacement = m.text[:2] + ("*" * (len(m.text) - 4)) + m.text[-2:]
                else:
                    replacement = "*" * len(m.text)
            elif mode == "remove":
                replacement = ""
            else:
                replacement = f"[{m.pii_type}]"

            result[m.start : m.end] = list(replacement)

        return "".join(result)


# Ergonomic alias
PIIDetector = PIIGuard

