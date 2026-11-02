"""Rule-based and gazetteer-driven Named Entity Recognition (NER).

Detects entities (Person, Organisation, Location, Date, Money, URL, Email, Phone)
from raw text without external machine learning dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Set, Tuple


@dataclass(slots=True, frozen=True)
class Entity:
    """An extracted named entity representation."""

    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0


class EntityExtractor:
    """Extracts entities using regex heuristics and built-in world gazetteers."""

    _EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
    _URL_RE = re.compile(r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)")
    _PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    _MONEY_RE = re.compile(r"(?:\$|€|£|¥|₹|USD|EUR|GBP|INR)\s*[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s*(?:dollars|euros|pounds|rupees|yen|cents)\b", re.IGNORECASE)
    _DATE_RE = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|"
        r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",
        re.IGNORECASE,
    )
    _ORG_SUFFIX_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\s+(?:Inc\.?|Corp\.?|Corporation|LLC|Ltd\.?|Limited|Group|Technologies|Labs|Systems|Foundation|Bank|University))\b")
    _PERSON_NAME_RE = re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b|\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b")

    _KNOWN_COUNTRIES = {
        "united states", "usa", "us", "united kingdom", "uk", "canada", "france", "germany",
        "italy", "spain", "japan", "china", "india", "australia", "brazil", "russia",
        "south korea", "mexico", "netherlands", "switzerland", "sweden", "norway", "singapore",
        "new zealand", "south africa", "argentina", "ireland", "denmark", "finland", "poland",
        "austria", "belgium", "portugal", "greece", "turkey", "israel", "uae", "saudi arabia",
    }

    _KNOWN_CITIES = {
        "new york", "london", "paris", "tokyo", "beijing", "berlin", "madrid", "rome",
        "san francisco", "los angeles", "chicago", "seattle", "boston", "toronto", "vancouver",
        "sydney", "melbourne", "singapore", "hong kong", "mumbai", "delhi", "bangalore",
        "dubai", "shanghai", "seoul", "amsterdam", "stockholm", "zurich", "geneva", "dublin",
    }

    def extract(self, text: str) -> List[Entity]:
        """Extract entities across all categories from input text."""
        entities: List[Entity] = []
        occupied_spans: List[Tuple[int, int]] = []

        def _overlaps(start: int, end: int) -> bool:
            return any(s < end and start < e for s, e in occupied_spans)

        # 1. Emails
        for m in self._EMAIL_RE.finditer(text):
            s, e = m.span()
            entities.append(Entity(text=m.group(0), entity_type="EMAIL", start=s, end=e, confidence=0.98))
            occupied_spans.append((s, e))

        # 2. URLs
        for m in self._URL_RE.finditer(text):
            s, e = m.span()
            if not _overlaps(s, e):
                entities.append(Entity(text=m.group(0), entity_type="URL", start=s, end=e, confidence=0.98))
                occupied_spans.append((s, e))

        # 3. Phone numbers
        for m in self._PHONE_RE.finditer(text):
            s, e = m.span()
            if not _overlaps(s, e):
                entities.append(Entity(text=m.group(0), entity_type="PHONE", start=s, end=e, confidence=0.95))
                occupied_spans.append((s, e))

        # 4. Monetary figures
        for m in self._MONEY_RE.finditer(text):
            s, e = m.span()
            if not _overlaps(s, e):
                entities.append(Entity(text=m.group(0), entity_type="MONEY", start=s, end=e, confidence=0.95))
                occupied_spans.append((s, e))

        # 5. Dates
        for m in self._DATE_RE.finditer(text):
            s, e = m.span()
            if not _overlaps(s, e):
                entities.append(Entity(text=m.group(0), entity_type="DATE", start=s, end=e, confidence=0.92))
                occupied_spans.append((s, e))

        # 6. Organisations
        for m in self._ORG_SUFFIX_RE.finditer(text):
            s, e = m.span()
            if not _overlaps(s, e):
                entities.append(Entity(text=m.group(0), entity_type="ORGANIZATION", start=s, end=e, confidence=0.90))
                occupied_spans.append((s, e))

        # 7. Locations from gazetteer (case-insensitive search)
        text_lower = text.lower()
        for loc in self._KNOWN_COUNTRIES:
            pattern = re.compile(r"\b" + re.escape(loc) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(text):
                s, e = m.span()
                if not _overlaps(s, e):
                    entities.append(Entity(text=text[s:e], entity_type="LOCATION", start=s, end=e, confidence=0.92))
                    occupied_spans.append((s, e))

        for city in self._KNOWN_CITIES:
            pattern = re.compile(r"\b" + re.escape(city) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(text):
                s, e = m.span()
                if not _overlaps(s, e):
                    entities.append(Entity(text=text[s:e], entity_type="LOCATION", start=s, end=e, confidence=0.92))
                    occupied_spans.append((s, e))

        # 8. Persons (proper nouns heuristic)
        for m in self._PERSON_NAME_RE.finditer(text):
            s, e = m.span()
            if not _overlaps(s, e):
                matched = m.group(1) or m.group(2) or m.group(0)
                entities.append(Entity(text=matched, entity_type="PERSON", start=s, end=e, confidence=0.75))
                occupied_spans.append((s, e))

        entities.sort(key=lambda ent: ent.start)
        return entities

