"""Report format generators for CI/CD runners and humans.

Generates JUnit XML for test runners, JSON for machine pipelines,
styled HTML for visual inspection, and Markdown tables for PR comment bots.
"""
from __future__ import annotations

import html
import json
from typing import List, Optional, Sequence
import xml.etree.ElementTree as ET

from veriflux._types import EvalResult


class JUnitReporter:
    """Generates standard JUnit XML compliant with GitHub Actions, GitLab CI, and Jenkins."""

    def generate(self, results: Sequence[EvalResult], suite_name: str = "veriflux") -> str:
        """Render results into a JUnit XML string."""
        total = len(results)
        failures = sum(1 for r in results if not r.passed)
        total_time_sec = sum(r.duration_ms for r in results) / 1000.0

        testsuite = ET.Element(
            "testsuite",
            name=suite_name,
            tests=str(total),
            failures=str(failures),
            errors="0",
            time=f"{total_time_sec:.3f}",
        )

        for r in results:
            case_time = f"{r.duration_ms / 1000.0:.3f}"
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                classname=f"veriflux.{suite_name}",
                name=r.metric_name,
                time=case_time,
            )
            if not r.passed:
                err_msg = r.details.get("error", f"Score {r.score:.3f} failed threshold")
                failure = ET.SubElement(
                    testcase,
                    "failure",
                    message=html.escape(str(err_msg)),
                    type="AssertionError",
                )
                failure.text = json.dumps(r.details, indent=2)

        return ET.tostring(testsuite, encoding="utf-8", xml_declaration=True).decode("utf-8")


class JSONReporter:
    """Generates machine-parseable JSON summary reports."""

    def generate(self, results: Sequence[EvalResult], indent: int = 2) -> str:
        data = [
            {
                "metric": r.metric_name,
                "passed": r.passed,
                "score": r.score,
                "duration_ms": r.duration_ms,
                "severity": r.severity.value,
                "details": r.details,
            }
            for r in results
        ]
        return json.dumps(data, indent=indent)


class MarkdownReporter:
    """Generates GitHub-flavored markdown tables for pull request comments."""

    def generate(self, results: Sequence[EvalResult]) -> str:
        lines = [
            "| Metric | Status | Score | Duration | Details |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ]
        for r in results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            details_str = json.dumps(r.details) if r.details else "-"
            if len(details_str) > 40:
                details_str = details_str[:37] + "..."
            lines.append(f"| `{r.metric_name}` | {status} | {r.score:.2f} | {r.duration_ms:.0f}ms | {html.escape(details_str)} |")
        return "\n".join(lines)


class HTMLReporter:
    """Generates styled standalone HTML evaluation reports."""

    def generate(self, results: Sequence[EvalResult], title: str = "Veriflux Evaluation Report") -> str:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total * 100.0) if total else 0.0

        rows = []
        for r in results:
            status_badge = '<span style="color:#10b981;font-weight:bold;">PASS</span>' if r.passed else '<span style="color:#ef4444;font-weight:bold;">FAIL</span>'
            rows.append(f"""
            <tr>
                <td style="padding:10px;border-bottom:1px solid #e5e7eb;"><code>{html.escape(r.metric_name)}</code></td>
                <td style="padding:10px;border-bottom:1px solid #e5e7eb;text-align:center;">{status_badge}</td>
                <td style="padding:10px;border-bottom:1px solid #e5e7eb;text-align:right;">{r.score:.3f}</td>
                <td style="padding:10px;border-bottom:1px solid #e5e7eb;text-align:right;">{r.duration_ms:.1f}ms</td>
            </tr>
            """)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html.escape(title)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f9fafb; color: #111827; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 24px; max-width: 900px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e5e7eb; padding-bottom: 16px; margin-bottom: 24px; }}
        .stats {{ display: flex; gap: 24px; margin-bottom: 24px; }}
        .stat-box {{ background: #f3f4f6; padding: 12px 20px; border-radius: 6px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #f9fafb; padding: 10px; border-bottom: 2px solid #e5e7eb; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1 style="margin:0;font-size:24px;">🔱 {html.escape(title)}</h1>
            <span style="font-size:14px;color:#6b7280;">Veriflux 0.1.0</span>
        </div>
        <div class="stats">
            <div class="stat-box"><strong>Pass Rate:</strong> {pass_rate:.1f}%</div>
            <div class="stat-box"><strong>Passed:</strong> {passed} / {total}</div>
            <div class="stat-box"><strong>Failed:</strong> {failed}</div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th style="text-align:center;">Status</th>
                    <th style="text-align:right;">Score</th>
                    <th style="text-align:right;">Duration</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
</body>
</html>"""

