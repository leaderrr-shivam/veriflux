"""Performance report generator.

Consolidates latency, cost, memory, and throughput metrics into Markdown tables,
JSON reports, and concise summaries.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from veriflux.profiler.cost_calculator import CostResult
from veriflux.profiler.latency_tracker import LatencyStats
from veriflux.profiler.memory_profiler import MemorySnapshot
from veriflux.profiler.throughput import ThroughputStats


class PerformanceReport:
    """Consolidated performance, cost, and throughput summary generator."""

    def __init__(self, suite_name: str = "Benchmark") -> None:
        self.suite_name = suite_name
        self.latency_metrics: Dict[str, LatencyStats] = {}
        self.cost_metrics: List[CostResult] = []
        self.throughput_stats: Optional[ThroughputStats] = None
        self.memory_snapshot: Optional[MemorySnapshot] = None

    def add_latency(self, name: str, stats: LatencyStats) -> None:
        self.latency_metrics[name] = stats

    def add_cost(self, cost: CostResult) -> None:
        self.cost_metrics.append(cost)

    def set_throughput(self, stats: ThroughputStats) -> None:
        self.throughput_stats = stats

    def set_memory(self, snapshot: MemorySnapshot) -> None:
        self.memory_snapshot = snapshot

    @property
    def total_cost(self) -> float:
        return sum(c.total_cost for c in self.cost_metrics)

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.cost_metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire report into structured dictionary."""
        return {
            "suite_name": self.suite_name,
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "latency": {
                name: {
                    "count": s.count,
                    "mean_ms": s.mean,
                    "p50_ms": s.median,
                    "p95_ms": s.p95,
                    "p99_ms": s.p99,
                }
                for name, s in self.latency_metrics.items()
            },
            "throughput": {
                "tps": self.throughput_stats.tokens_per_second if self.throughput_stats else 0.0,
                "rpm": self.throughput_stats.requests_per_minute if self.throughput_stats else 0.0,
            } if self.throughput_stats else None,
            "memory": {
                "peak_kb": self.memory_snapshot.peak_kb if self.memory_snapshot else 0.0,
            } if self.memory_snapshot else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Format report as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Render report as GitHub-flavored Markdown table."""
        lines = [
            f"### ⚡ Performance & Cost Profile: {self.suite_name}",
            "",
            f"**Total Tokens:** {self.total_tokens:,} | **Total Expenditure:** ${self.total_cost:.4f}",
            "",
        ]

        if self.latency_metrics:
            lines.extend([
                "| Metric | Calls | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ])
            for name, s in self.latency_metrics.items():
                lines.append(f"| `{name}` | {s.count} | {s.mean} | {s.median} | {s.p95} | {s.p99} |")
            lines.append("")

        if self.throughput_stats:
            lines.extend([
                "#### 🚀 Throughput",
                f"- **Average TPS:** {self.throughput_stats.tokens_per_second:.1f} tokens/sec",
                f"- **Peak TPS:** {self.throughput_stats.peak_tokens_per_second:.1f} tokens/sec",
                f"- **Request Velocity:** {self.throughput_stats.requests_per_minute:.1f} RPM",
                "",
            ])

        return "\n".join(lines)

