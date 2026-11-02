"""Pytest terminal and session report aggregator.

Collects Veriflux evaluation results during pytest execution and renders
summary metrics at terminal session conclusion.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List

from veriflux._types import EvalResult, EvalSuite


class VerifluxReporter:
    """Thread-safe collector of evaluation results during pytest runs."""

    def __init__(self) -> None:
        self.suite = EvalSuite(name="pytest_session")
        self._lock = threading.Lock()

    def record_result(self, result: EvalResult) -> None:
        with self._lock:
            self.suite.results.append(result)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return self.suite.summary()

