"""Execution graph builder and Mermaid diagram visualization.

Transforms linear agent traces into directed execution graphs (DAGs)
for structural inspection and architectural visualization.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from veriflux._types import TraceStep
from veriflux.trajectory.tracer import Trace


class ExecutionGraph:
    """Directed graph representation of agent workflow steps and tool linkages."""

    def __init__(self) -> None:
        self.nodes: Dict[int, TraceStep] = {}
        self.edges: List[tuple[int, int]] = []
        self.custom_transitions: List[tuple[str, str]] = []

    def add_transition(self, src: str, dest: str) -> None:
        """Add an explicit agent-to-agent transition edge."""
        self.custom_transitions.append((src, dest))

    @classmethod
    def from_trace(cls, trace: Trace) -> ExecutionGraph:
        """Construct DAG from a completed Trace instance."""
        graph = cls()
        for s in trace.steps:
            graph.nodes[s.step_index] = s

        # Connect consecutive steps
        for i in range(len(trace.steps) - 1):
            graph.edges.append((trace.steps[i].step_index, trace.steps[i + 1].step_index))

        return graph

    def to_dict(self) -> Dict[str, Any]:
        """Export adjacency representation."""
        return {
            "nodes": {
                idx: {"action": s.action, "duration_ms": s.duration_ms}
                for idx, s in self.nodes.items()
            },
            "edges": self.edges,
        }

    def to_mermaid(self) -> str:
        """Render trace graph as a GitHub-compatible Mermaid flowchart."""
        if self.custom_transitions and not self.nodes:
            lines = ["flowchart TD"]
            for src, dest in self.custom_transitions:
                lines.append(f"    {src} --> {dest}")
            return "\n".join(lines)

        lines = ["flowchart TD"]

        for idx, step in self.nodes.items():
            # Clean label for mermaid compatibility
            clean_action = re.sub(r"[^a-zA-Z0-9_: -]", "", step.action)
            dur = f"{step.duration_ms:.0f}ms"
            label = f"\"{idx}: {clean_action} ({dur})\""

            if "error" in step.metadata:
                lines.append(f"    node{idx}[{label}]:::errorNode")
            elif "llm_call" in step.action:
                lines.append(f"    node{idx}[{label}]:::llmNode")
            else:
                lines.append(f"    node{idx}[{label}]")

        for src, dest in self.edges:
            lines.append(f"    node{src} --> node{dest}")

        for src, dest in self.custom_transitions:
            lines.append(f"    {src} --> {dest}")

        lines.append("    classDef errorNode fill:#ffcccc,stroke:#ff0000,stroke-width:2px;")
        lines.append("    classDef llmNode fill:#e6f3ff,stroke:#0066cc,stroke-width:1px;")

        return "\n".join(lines)


# Ergonomic alias
AgentGraph = ExecutionGraph

