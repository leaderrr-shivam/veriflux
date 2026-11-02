"""Native GitHub Actions integration.

Emits workflow commands, annotations (::error, ::warning), sets job outputs,
and generates ready-to-use CI workflow YAML files.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

from veriflux._types import EvalResult
from veriflux.cicd.reporters import MarkdownReporter


class GitHubActionsReporter:
    """GitHub Actions workflow runner integration."""

    def annotate(self, results: Sequence[EvalResult]) -> None:
        """Emit workflow log annotations (::warning and ::error) for failed evaluations."""
        for r in results:
            if not r.passed:
                severity = "error" if r.score < 0.5 else "warning"
                msg = f"Evaluation failed for {r.metric_name} (score: {r.score:.3f})"
                print(f"::{severity} title=Veriflux Evaluation::{msg}")

    def set_output(self, name: str, value: str) -> None:
        """Write key-value output to GITHUB_OUTPUT environment file."""
        github_output_path = os.environ.get("GITHUB_OUTPUT")
        if github_output_path:
            with open(github_output_path, "a", encoding="utf-8") as f:
                f.write(f"{name}={value}\n")
        else:
            print(f"::set-output name={name}::{value}")

    def pr_comment_body(self, results: Sequence[EvalResult]) -> str:
        """Generate markdown payload suitable for automated PR comment bots."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        status = "✅ All Checks Passed" if passed == total else f"⚠️ {total - passed} Checks Failed"

        lines = [
            f"### 🔱 Veriflux CI Report — {status}",
            "",
            MarkdownReporter().generate(results),
        ]
        return "\n".join(lines)

    @staticmethod
    def generate_workflow() -> str:
        """Generate a complete GitHub Actions workflow YAML."""
        return """name: Veriflux AI Evaluation
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install . pytest
      - name: Run Veriflux Gates
        run: pytest -m "not slow" --junitxml=reports/junit.xml
      - name: Upload Test Report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: veriflux-report
          path: reports/junit.xml
"""

