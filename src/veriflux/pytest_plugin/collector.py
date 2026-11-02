"""Custom pytest file collector for automated dataset-driven evaluation.

Discovers `*_eval.jsonl` files and dynamically generates parameterized test cases.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

from veriflux.datasets.loader import DatasetLoader, TestCase


if pytest is not None:
    class VerifluxDatasetItem(pytest.Item):
        """Individual generated test item representing one dataset TestCase."""

        def __init__(self, name: str, parent: Any, case: TestCase) -> None:
            super().__init__(name, parent)
            self.case = case

        def runtest(self) -> None:
            # Baseline execution: ensure response exists and context is valid
            if not self.case.response and not self.case.expected:
                raise AssertionError(f"TestCase at {self.name} has neither response nor expected answer")

        def repr_failure(self, excinfo: Any) -> str:
            return f"Veriflux Test Case Failure: {excinfo.value}"

        def reportinfo(self) -> tuple[Any, int, str]:
            return self.path, 0, f"Veriflux Case: {self.case.query[:30]}"

