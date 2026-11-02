"""Evaluation dataset loading and representation.

Supports JSONL, CSV, and JSON dataset ingestion with generator-based streaming for large files.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence


@dataclass(slots=True)
class TestCase:
    """An individual evaluation test item.

    Attributes:
        query: Prompt or question input.
        context: Reference background information (RAG context).
        response: Model generation to evaluate.
        expected: Ground-truth reference answer.
        metadata: Arbitrary item-level labels and tags.
    """

    query: str = ""
    context: str = ""
    response: str = ""
    expected: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class Dataset:
    """In-memory collection of evaluation test cases."""

    def __init__(self, cases: Optional[Sequence[TestCase]] = None, name: str = "dataset") -> None:
        self.cases: List[TestCase] = list(cases) if cases else []
        self.name = name

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> TestCase:
        return self.cases[idx]

    def __iter__(self) -> Iterator[TestCase]:
        return iter(self.cases)

    def filter(self, predicate: Callable[[TestCase], bool]) -> Dataset:
        """Filter dataset cases by predicate."""
        return Dataset([c for c in self.cases if predicate(c)], name=f"{self.name}_filtered")


class DatasetLoader:
    """Loads evaluation datasets from JSONL, CSV, or JSON files."""

    def load(self, path: str) -> Dataset:
        """Load dataset from disk based on file extension."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        name = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1].lower()

        if ext == ".jsonl":
            return self._load_jsonl(path, name)
        elif ext == ".csv":
            return self._load_csv(path, name)
        elif ext == ".json":
            return self._load_json(path, name)
        raise ValueError(f"Unsupported dataset format: {ext}. Expected .jsonl, .csv, or .json")

    def _load_jsonl(self, path: str, name: str) -> Dataset:
        cases: List[TestCase] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    item = json.loads(line_str)
                    cases.append(self._item_to_testcase(item))
        return Dataset(cases, name=name)

    def _load_csv(self, path: str, name: str) -> Dataset:
        cases: List[TestCase] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cases.append(self._item_to_testcase(row))
        return Dataset(cases, name=name)

    def _load_json(self, path: str, name: str) -> Dataset:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                cases = [self._item_to_testcase(item) for item in data]
                return Dataset(cases, name=name)
            elif isinstance(data, dict) and "cases" in data:
                cases = [self._item_to_testcase(item) for item in data["cases"]]
                return Dataset(cases, name=name)
        raise ValueError("Invalid JSON dataset structure: expected list of items or {'cases': [...]}")

    @staticmethod
    def _item_to_testcase(item: Dict[str, Any]) -> TestCase:
        query = str(item.get("query") or item.get("prompt") or item.get("question") or "")
        context = str(item.get("context") or item.get("passage") or item.get("reference") or "")
        response = str(item.get("response") or item.get("answer") or item.get("output") or "")
        expected = str(item.get("expected") or item.get("ground_truth") or "")
        meta = {k: v for k, v in item.items() if k not in ("query", "context", "response", "expected", "prompt", "question", "passage", "reference", "answer", "output", "ground_truth")}
        return TestCase(query=query, context=context, response=response, expected=expected, metadata=meta)

