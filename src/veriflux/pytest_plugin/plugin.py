"""Official Pytest plugin hook implementations for Veriflux.

Registers CLI options, markers, and custom terminal summaries.
"""
from __future__ import annotations

from typing import Any, List

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

from veriflux.pytest_plugin.fixtures import *
from veriflux.pytest_plugin.markers import extract_veriflux_marker
from veriflux.pytest_plugin.reporter import VerifluxReporter

_global_reporter = VerifluxReporter()


def pytest_addoption(parser: Any) -> None:
    """Register custom CLI options with pytest."""
    group = parser.getgroup("veriflux", "Veriflux AI Evaluation")
    group.addoption(
        "--veriflux-layer",
        action="store",
        default=None,
        help="Filter tests by Veriflux layer: 'fast', 'standard', 'deep', 'nightly'.",
    )
    group.addoption(
        "--veriflux-baseline",
        action="store",
        default=None,
        help="Compare test execution metrics against a named baseline.",
    )


def pytest_configure(config: Any) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "veriflux_test(dataset=None, layer='standard', timeout=None): mark test as a Veriflux AI evaluation.",
    )


def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    """Filter test items if --veriflux-layer is specified."""
    target_layer = config.getoption("--veriflux-layer")
    if not target_layer:
        return

    selected = []
    deselected = []

    for item in items:
        marker_info = extract_veriflux_marker(item)
        if marker_info:
            item_layer = marker_info.get("layer", "standard")
            if item_layer == target_layer:
                selected.append(item)
            else:
                deselected.append(item)
        else:
            deselected.append(item)

    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    """Render Veriflux session summary banner."""
    summary = _global_reporter.summary()
    if summary["total"] > 0:
        terminalreporter.section("Veriflux AI Evaluation Summary", sep="=")
        terminalreporter.write_line(f"Evaluated Metrics: {summary['total']}")
        terminalreporter.write_line(f"Pass Rate:        {summary['pass_rate'] * 100:.1f}%")
        terminalreporter.write_line(f"Mean Score:       {summary['mean_score']:.3f}")

