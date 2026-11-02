"""Adversarial and synthetic test data generation.

Generates perturbations (entity swaps, negation flips, truncation) and schema edge-cases.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List

from veriflux.datasets.loader import TestCase


class SyntheticGenerator:
    """Generates synthetic adversarial test cases for robustness evaluations."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate_adversarial(self, context: str, n: int = 5) -> List[TestCase]:
        """Generate adversarial test cases from a reference context."""
        cases: List[TestCase] = []
        words = context.split()

        # 1. Negation injection
        neg_context = re.sub(r"\bis\b", "is not", context, count=1)
        cases.append(TestCase(query="Verify factual consistency", context=context, response=neg_context, metadata={"attack": "negation"}))

        # 2. Number corruption
        num_context = re.sub(r"\b\d+\b", lambda m: str(int(m.group()) * 2 + 1), context)
        if num_context != context:
            cases.append(TestCase(query="Check numerical accuracy", context=context, response=num_context, metadata={"attack": "numerical_swap"}))

        # 3. Truncation attack
        if len(words) > 10:
            half = " ".join(words[: len(words) // 2])
            cases.append(TestCase(query="Check complete answers", context=context, response=half, metadata={"attack": "truncation"}))

        # 4. Out-of-domain insertion
        alien = context + " In other news, penguins have colonized the surface of Mars in 2026."
        cases.append(TestCase(query="Hallucination injection", context=context, response=alien, metadata={"attack": "fabricated_claim"}))

        return cases[:n]

    def generate_edge_cases(self, schema: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
        """Synthesize edge-case dictionary payloads for schema validation testing."""
        cases: List[Dict[str, Any]] = []
        props = schema.get("properties", {})

        # Edge case 1: Empty / zero values
        empty_case = {}
        for k, v in props.items():
            t = v.get("type", "string")
            if t == "string":
                empty_case[k] = ""
            elif t in ("integer", "number"):
                empty_case[k] = 0
            elif t == "boolean":
                empty_case[k] = False
            elif t == "array":
                empty_case[k] = []
            elif t == "object":
                empty_case[k] = {}
        cases.append(empty_case)

        # Edge case 2: Extreme values
        extreme_case = {}
        for k, v in props.items():
            t = v.get("type", "string")
            if t == "string":
                extreme_case[k] = "A" * 1000  # Long string
            elif t == "integer":
                extreme_case[k] = 2**31 - 1
            elif t == "number":
                extreme_case[k] = 1e12
            else:
                extreme_case[k] = None
        cases.append(extreme_case)

        # Edge case 3: Special characters
        special_case = {}
        for k in props:
            special_case[k] = "Special chars: <script>alert(1)</script> \x00 '\"\\"
        cases.append(special_case)

        return cases[:n]

