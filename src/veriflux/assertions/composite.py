"""Assertion combinators and comprehensive evaluation suites.

Enables aggregation of multiple assertions into structured pass/fail reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence


@dataclass(slots=True)
class EvalReport:
    """Summary report from executing an assertion suite.

    Attributes:
        passed: True if all suite expectations were satisfied.
        total: Total number of assertions evaluated.
        passed_count: Count of successful assertions.
        failed_count: Count of failed assertions.
        results: Mapping of assertion name to boolean pass/fail status.
    """

    passed: bool
    total: int
    passed_count: int
    failed_count: int
    results: Dict[str, bool] = field(default_factory=dict)


def all_pass(*assertions: bool) -> bool:
    """Verify that every assertion evaluated to True."""
    return all(assertions)


def any_pass(*assertions: bool) -> bool:
    """Verify that at least one assertion evaluated to True."""
    return any(assertions)


def at_least_n(n: int, *assertions: bool) -> bool:
    """Verify that at least N assertions evaluated to True."""
    return sum(1 for a in assertions if a) >= n


class AssertionSuite:
    """Collection of named assertions evaluated in sequence with full reporting."""

    def __init__(self, name: str = "default_suite") -> None:
        self.name = name
        self._checks: List[tuple[str, Callable[[], bool]]] = []

    def add(self, name: str, assertion_fn: Callable[[], bool]) -> AssertionSuite:
        """Register a named assertion callable."""
        self._checks.append((name, assertion_fn))
        return self

    def run(self) -> EvalReport:
        """Execute all assertions and return an EvalReport."""
        results: Dict[str, bool] = {}
        passed_count = 0

        for name, fn in self._checks:
            try:
                res = bool(fn())
            except Exception:
                res = False
            results[name] = res
            if res:
                passed_count += 1

        total = len(self._checks)
        failed_count = total - passed_count
        passed = failed_count == 0

        return EvalReport(
            passed=passed,
            total=total,
            passed_count=passed_count,
            failed_count=failed_count,
            results=results,
        )

