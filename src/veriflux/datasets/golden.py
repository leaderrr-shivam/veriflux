"""Golden evaluation dataset management and versioning.

Maintains curated golden datasets, records ground truths, and audits dataset revisions.
"""
from __future__ import annotations

import json
from typing import List, Optional

from veriflux.datasets.loader import Dataset, TestCase


class GoldenDataset:
    """A curated, version-controlled benchmark evaluation dataset."""

    def __init__(self, name: str, cases: Optional[List[TestCase]] = None, version: str = "1.0") -> None:
        self.name = name
        self.version = version
        self.cases: List[TestCase] = list(cases) if cases else []

    def add_case(self, case: TestCase) -> GoldenDataset:
        """Add a curated test case."""
        self.cases.append(case)
        return self

    def save(self, path: str) -> None:
        """Save golden dataset to JSON file."""
        data = {
            "name": self.name,
            "version": self.version,
            "count": len(self.cases),
            "cases": [
                {
                    "query": c.query,
                    "context": c.context,
                    "response": c.response,
                    "expected": c.expected,
                    "metadata": c.metadata,
                }
                for c in self.cases
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> GoldenDataset:
        """Load golden dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = [
            TestCase(
                query=item.get("query", ""),
                context=item.get("context", ""),
                response=item.get("response", ""),
                expected=item.get("expected", ""),
                metadata=item.get("metadata", {}),
            )
            for item in data.get("cases", [])
        ]
        return cls(name=data.get("name", "golden"), cases=cases, version=data.get("version", "1.0"))

    def to_dataset(self) -> Dataset:
        """Convert to Dataset instance."""
        return Dataset(self.cases, name=self.name)

