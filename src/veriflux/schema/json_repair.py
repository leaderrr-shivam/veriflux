"""Fault-tolerant JSON repair engine for malformed LLM outputs.

Repairs truncated JSON, unclosed quotes/brackets, trailing commas, single-quoted strings,
unquoted dictionary keys, comments, and Python syntax literals (True, False, None).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple


class JSONRepair:
    """Repairs malformed or truncated JSON strings into strictly valid JSON."""

    # Markdown code fence cleaner
    _CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

    # Comments (// and /* */)
    _LINE_COMMENT_RE = re.compile(r"//.*$", re.MULTILINE)
    _BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/")

    # Python literals
    _PYTHON_LITERALS = [
        (re.compile(r"\bTrue\b"), "true"),
        (re.compile(r"\bFalse\b"), "false"),
        (re.compile(r"\bNone\b"), "null"),
        (re.compile(r"\bNaN\b"), "null"),
        (re.compile(r"\bInfinity\b"), "null"),
        (re.compile(r"\b-Infinity\b"), "null"),
    ]

    # Unquoted keys: { key: "value" } or , key: "value"
    _UNQUOTED_KEY_RE = re.compile(r"([{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:")

    # Trailing commas: , } or , ]
    _TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

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

        extracted = self.extract_json(text)
        s = extracted if extracted is not None else text.strip()

        # 1. Strip comments
        s = self._BLOCK_COMMENT_RE.sub("", s)
        s = self._LINE_COMMENT_RE.sub("", s)

        # 2. Replace Python literals with JSON equivalents
        for pat, repl in self._PYTHON_LITERALS:
            s = pat.sub(repl, s)

        # 3. Handle single quotes for keys and strings
        # Transform 'key': into "key":
        s = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'\s*:", r'"\1":', s)
        # Transform :'value' into :"value"
        s = re.sub(r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'", r': "\1"', s)
        # Transform ['item'] into ["item"]
        s = re.sub(r"\[\s*'([^'\\]*(?:\\.[^'\\]*)*)'", r'["\1"', s)
        s = re.sub(r",\s*'([^'\\]*(?:\\.[^'\\]*)*)'", r', "\1"', s)

        # 4. Quote unquoted keys
        s = self._UNQUOTED_KEY_RE.sub(r'\1"\2":', s)

        # 5. Remove trailing commas before closing braces/brackets
        s = self._TRAILING_COMMA_RE.sub(r"\1", s)

        # 6. Check and fix truncated brackets / braces / quotes, and unescaped control characters
        rebuilt: list[str] = []
        stack: list[str] = []
        in_string = False
        escape_next = False

        for char in s:
            if escape_next:
                rebuilt.append(char)
                escape_next = False
                continue
            if char == "\\":
                rebuilt.append(char)
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                rebuilt.append(char)
                continue
            if in_string:
                if char == "\n":
                    rebuilt.append("\\n")
                elif char == "\r":
                    rebuilt.append("\\r")
                elif char == "\t":
                    rebuilt.append("\\t")
                else:
                    rebuilt.append(char)
                continue

            if char in ("{", "["):
                stack.append(char)
            elif char == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif char == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
            rebuilt.append(char)

        s = "".join(rebuilt)

        # If string was cut off mid-way, close quote
        if in_string:
            s += '"'

        # If inside object and ended with dangling key or colon
        if stack and stack[-1] == "{":
            if re.search(r':\s*$', s):
                s += "null"
            elif re.search(r'(?:[{,]\s*)(?:"[^"]*")\s*$', s):
                s += ": null"

        # Close unclosed structures in reverse order
        while stack:
            open_char = stack.pop()
            # Clean trailing comma or colon if any
            s = re.sub(r"[,:]\s*$", "", s)
            if open_char == "{":
                s += "}"
            elif open_char == "[":
                s += "]"

        # Remove trailing comma right before closing if any remaining
        s = self._TRAILING_COMMA_RE.sub(r"\1", s)

        return s

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
            # Last-resort fallback for empty or partial objects
            if repaired.startswith("{"):
                return {}, True
            elif repaired.startswith("["):
                return [], True
            raise ValueError(f"Could not repair JSON: {exc}") from exc


def repair_json(text: str) -> str:
    """Convenience function to repair a malformed JSON string."""
    return JSONRepair().repair(text)

