"""Fault-tolerant, recursive-descent JSON repair engine for malformed LLM outputs.

Repairs truncated JSON, unclosed quotes/brackets/braces, unquoted keys,
unquoted primitive values, single-quoted and typographic-quoted strings,
Python literals (True, False, None), trailing commas, and inline comments.
Pure Python 3.10+ standard library implementation with zero dependencies.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Tuple

_SPECIAL_QUOTES = {
    '“': '"', '”': '"', '„': '"', '«': '"', '»': '"',
    '‘': "'", '’': "'", '`': "'",
}


class _RecursiveJSONRepair:
    """Internal streaming recursive-descent repair parser."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.i = 0
        self.out: List[str] = []

    def cur(self) -> str:
        return self.text[self.i] if self.i < self.n else ""

    def peek(self, offset: int = 1) -> str:
        pos = self.i + offset
        return self.text[pos] if pos < self.n else ""

    def skip_whitespace_and_comments(self) -> None:
        while self.i < self.n:
            c = self.text[self.i]
            if c.isspace():
                self.i += 1
                continue
            # Block comments: /* ... */
            if c == '/' and self.peek() == '*':
                self.i += 2
                while self.i < self.n and not (self.text[self.i] == '*' and self.peek() == '/'):
                    self.i += 1
                if self.i < self.n:
                    self.i += 2
                continue
            # Line comments: // ... (protect URLs like https://)
            if c == '/' and self.peek() == '/':
                prev_i = self.i - 1
                while prev_i >= 0 and self.text[prev_i].isspace():
                    prev_i -= 1
                if prev_i >= 0 and self.text[prev_i] == ':':
                    # Part of url, not a comment
                    break
                self.i += 2
                while self.i < self.n and self.text[self.i] not in ('\r', '\n'):
                    self.i += 1
                continue
            break

    def repair(self) -> str:
        s = self.text.strip()
        if not s:
            return "{}"

        # Markdown code fence unwrapping
        fence_m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if fence_m:
            s = fence_m.group(1).strip()
            self.text = s
            self.n = len(s)
            self.i = 0

        self.skip_whitespace_and_comments()
        self.parse_value()

        result = "".join(self.out).strip()
        return result

    def parse_value(self) -> None:
        self.skip_whitespace_and_comments()
        if self.i >= self.n:
            return

        c = self.cur()

        # Skip ellipsis
        if self.text[self.i:self.i+3] == '...':
            self.i += 3
            self.skip_whitespace_and_comments()
            if self.i >= self.n:
                return
            c = self.cur()

        if c == '{':
            self.parse_object()
        elif c == '[':
            self.parse_array()
        elif c in ('"', "'") or c in _SPECIAL_QUOTES:
            self.parse_string()
        else:
            self.parse_primitive_or_unquoted()

    def parse_object(self) -> None:
        self.out.append('{')
        self.i += 1
        first = True

        while self.i < self.n:
            prev_i = self.i
            self.skip_whitespace_and_comments()
            if self.i >= self.n:
                break

            c = self.cur()
            if self.text[self.i:self.i+3] == '...':
                self.i += 3
                continue

            if c == ',' and first:
                self.i += 1
                continue

            if c == '}':
                self.i += 1
                if self.out and self.out[-1] == ',':
                    self.out.pop()
                self.out.append('}')
                return

            if not first:
                if c == ',':
                    self.i += 1
                    self.skip_whitespace_and_comments()
                    if self.cur() == '}':
                        self.i += 1
                        self.out.append('}')
                        return
                    if self.text[self.i:self.i+3] == '...':
                        self.i += 3
                        continue
                if not (self.out and self.out[-1] in ('{', ',')):
                    self.out.append(',')

            first = False
            self.skip_whitespace_and_comments()
            if self.i >= self.n or self.cur() == '}':
                break

            # Parse Key
            c = self.cur()
            if c in ('"', "'") or c in _SPECIAL_QUOTES:
                self.parse_string()
            else:
                self.parse_unquoted_key()

            self.skip_whitespace_and_comments()
            # Colon handling
            if self.cur() == ':':
                self.out.append(':')
                self.i += 1
            else:
                self.out.append(':')

            self.skip_whitespace_and_comments()
            # Value handling
            if self.cur() in (',', '}', ']', '') or self.text[self.i:self.i+3] == '...':
                self.out.append('null')
            else:
                self.parse_value()

            if self.i <= prev_i:
                self.i += 1

        if self.out and self.out[-1] == ',':
            self.out.pop()
        self.out.append('}')

    def parse_array(self) -> None:
        self.out.append('[')
        self.i += 1
        first = True

        while self.i < self.n:
            prev_i = self.i
            self.skip_whitespace_and_comments()
            if self.i >= self.n:
                break

            c = self.cur()
            if self.text[self.i:self.i+3] == '...':
                self.i += 3
                continue

            if c == ',' and first:
                self.i += 1
                continue

            if c == ']':
                self.i += 1
                if self.out and self.out[-1] == ',':
                    self.out.pop()
                self.out.append(']')
                return

            if not first:
                if c == ',':
                    self.i += 1
                    self.skip_whitespace_and_comments()
                    if self.cur() == ']':
                        self.i += 1
                        self.out.append(']')
                        return
                    if self.text[self.i:self.i+3] == '...':
                        self.i += 3
                        continue
                if not (self.out and self.out[-1] in ('[', ',')):
                    self.out.append(',')

            first = False
            self.skip_whitespace_and_comments()
            if self.i >= self.n or self.cur() == ']':
                break

            self.parse_value()

            if self.i <= prev_i:
                self.i += 1

        if self.out and self.out[-1] == ',':
            self.out.pop()
        self.out.append(']')

    def parse_string(self) -> None:
        c = self.cur()
        quote_char = _SPECIAL_QUOTES.get(c, c)
        start_char = c
        self.i += 1
        self.out.append('"')

        while self.i < self.n:
            ch = self.cur()

            if ch == '\\':
                next_ch = self.peek()
                if quote_char == "'" and next_ch == "'":
                    self.out.append("'")
                    self.i += 2
                    continue
                elif next_ch == "'":
                    self.out.append("'")
                    self.i += 2
                    continue
                elif next_ch in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't'):
                    self.out.append('\\')
                    self.out.append(next_ch)
                    self.i += 2
                    continue
                elif next_ch == 'u':
                    hex_digits = ""
                    pos = self.i + 2
                    while pos < self.n and len(hex_digits) < 4 and self.text[pos] in "0123456789abcdefABCDEF":
                        hex_digits += self.text[pos]
                        pos += 1
                    hex_digits = (hex_digits + "0000")[:4]
                    self.out.append('\\u' + hex_digits)
                    self.i = pos
                    continue
                elif next_ch == 'x':
                    hex_digits = ""
                    pos = self.i + 2
                    while pos < self.n and len(hex_digits) < 2 and self.text[pos] in "0123456789abcdefABCDEF":
                        hex_digits += self.text[pos]
                        pos += 1
                    if len(hex_digits) == 2:
                        self.out.append('\\u00' + hex_digits)
                    else:
                        self.out.append('x')
                    self.i = pos
                    continue
                elif next_ch in ('\r', '\n'):
                    self.i += 2
                    continue
                else:
                    if next_ch:
                        self.out.append(next_ch)
                        self.i += 2
                    else:
                        self.i += 1
                    continue

            if ch == start_char or (quote_char == '"' and ch in ('"', '“', '”')) or (quote_char == "'" and ch in ("'", '‘', '’', '`')):
                self.i += 1
                break

            if ch == '"':
                self.out.append('\\"')
                self.i += 1
                continue

            if ch == '\b':
                self.out.append('\\b')
            elif ch == '\f':
                self.out.append('\\f')
            elif ch == '\n':
                self.out.append('\\n')
            elif ch == '\r':
                self.out.append('\\r')
            elif ch == '\t':
                self.out.append('\\t')
            elif ord(ch) < 32:
                self.out.append(f'\\u00{ord(ch):02x}')
            else:
                self.out.append(ch)
            self.i += 1

        self.out.append('"')

        # Check for string concatenation: "a" + "b"
        self.skip_whitespace_and_comments()
        if self.cur() == '+':
            self.i += 1
            self.skip_whitespace_and_comments()
            if self.cur() in ('"', "'") or self.cur() in _SPECIAL_QUOTES:
                self.out.pop()
                c2 = self.cur()
                start_c2 = c2
                quote_c2 = _SPECIAL_QUOTES.get(c2, c2)
                self.i += 1
                while self.i < self.n:
                    ch = self.cur()
                    if ch == '\\':
                        self.out.append('\\')
                        if self.i + 1 < self.n:
                            self.out.append(self.peek())
                            self.i += 2
                        else:
                            self.i += 1
                        continue
                    if ch == start_c2 or (quote_c2 == '"' and ch in ('"', '“', '”')) or (quote_c2 == "'" and ch in ("'", '‘', '’', '`')):
                        self.i += 1
                        break
                    if ch == '"':
                        self.out.append('\\"')
                    elif ch == '\n':
                        self.out.append('\\n')
                    elif ch == '\r':
                        self.out.append('\\r')
                    elif ch == '\t':
                        self.out.append('\\t')
                    else:
                        self.out.append(ch)
                    self.i += 1
                self.out.append('"')

    def parse_unquoted_key(self) -> None:
        start = self.i
        while self.i < self.n:
            c = self.cur()
            if c == ':' or c.isspace():
                break
            if c in ('{', '}', '[', ']'):
                break
            self.i += 1
        if self.i == start and self.i < self.n:
            self.i += 1
        key = self.text[start:self.i].strip()
        key = key.strip('"\'')
        self.out.append(json.dumps(key))

    def parse_primitive_or_unquoted(self) -> None:
        start = self.i
        while self.i < self.n:
            c = self.cur()
            if c in (',', '}', ']', '\n', '\r') or (c == '/' and self.peek() == '*'):
                break
            self.i += 1

        if self.i == start and self.i < self.n:
            self.i += 1
            return

        raw = self.text[start:self.i].strip()
        if not raw:
            return

        # Python & JS literals
        if raw in ('true', 'True'):
            self.out.append('true')
            return
        if raw in ('false', 'False'):
            self.out.append('false')
            return
        if raw in ('null', 'None', 'undefined', 'NaN', 'Infinity', '-Infinity'):
            self.out.append('null')
            return

        # Numeric literals
        num_m = re.match(r'^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$', raw)
        if num_m:
            self.out.append(raw)
            return

        # Leading-zero integers
        if re.match(r'^-?0\d+$', raw):
            self.out.append(str(int(raw)))
            return

        # Arbitrary unquoted text/url
        cleaned = raw.strip('"\'')
        self.out.append(json.dumps(cleaned))


class JSONRepair:
    """Repairs malformed or truncated JSON strings into strictly valid JSON."""

    _CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

    def extract_json(self, text: str) -> Optional[str]:
        """Extract JSON substring from mixed conversational text."""
        if not text:
            return None

        # Check for markdown code fences first
        match = self._CODE_FENCE_RE.search(text)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

        # Find first opening { or [
        start_obj = text.find("{")
        start_arr = text.find("[")

        if start_obj == -1 and start_arr == -1:
            return None

        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            start = start_obj
            end = text.rfind("}")
            if end != -1 and end > start:
                return text[start : end + 1]
            return text[start:]
        else:
            start = start_arr
            end = text.rfind("]")
            if end != -1 and end > start:
                return text[start : end + 1]
            return text[start:]

    def repair(self, text: str) -> str:
        """Fix common LLM JSON syntax errors and return a valid JSON string."""
        if not text or not text.strip():
            return "{}"

        # Fast path: already strictly valid JSON
        s = text.strip()
        try:
            json.loads(s)
            return s
        except Exception:
            pass

        # Execute recursive-descent repair parser
        parser = _RecursiveJSONRepair(s)
        repaired = parser.repair()

        # Sanity check: verify validity
        try:
            json.loads(repaired)
            return repaired
        except Exception:
            # Fallback wrapper
            if repaired.startswith("{"):
                return repaired + "}"
            elif repaired.startswith("["):
                return repaired + "]"
            return json.dumps(repaired)

    def try_parse(self, text: str) -> Tuple[Any, bool]:
        """Try parsing JSON directly, applying repair heuristics if parsing fails.

        Returns:
            Tuple of (parsed_object, was_repaired).
        """
        try:
            return json.loads(text), False
        except Exception:
            pass

        repaired = self.repair(text)
        try:
            return json.loads(repaired), True
        except Exception as exc:
            if repaired.startswith("{"):
                return {}, True
            elif repaired.startswith("["):
                return [], True
            raise ValueError(f"Could not repair JSON: {exc}") from exc


def repair_json(text: str) -> str:
    """Convenience function to repair a malformed JSON string."""
    return JSONRepair().repair(text)
