"""CI/CD Integration layer for Veriflux.

Provides quality gates, multi-format reporters (JUnit XML, HTML, Markdown),
dynamic SVG badges, and GitHub Actions integrations.
"""
from __future__ import annotations

from veriflux.cicd.artifacts import ArtifactManager
from veriflux.cicd.badge import BadgeGenerator
from veriflux.cicd.gates import Gate, GateResult, QualityGate
from veriflux.cicd.github_actions import GitHubActionsReporter
from veriflux.cicd.reporters import (
    HTMLReporter,
    JSONReporter,
    JUnitReporter,
    MarkdownReporter,
)

__all__ = [
    # Gates
    "QualityGate",
    "Gate",
    "GateResult",
    # Reporters
    "JUnitReporter",
    "JSONReporter",
    "HTMLReporter",
    "MarkdownReporter",
    # Badge
    "BadgeGenerator",
    # Artifacts
    "ArtifactManager",
    # GitHub Actions
    "GitHubActionsReporter",
]

