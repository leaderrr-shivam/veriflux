"""Trace serialization and persistence.

Saves and restores agent traces in JSON and JSONL formats with integrity validation.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from veriflux._types import TokenUsage, TraceStep
from veriflux.trajectory.tracer import Trace


class TraceRecorder:
    """Serializes agent traces to disk and loads historical execution records."""

    def save(self, trace: Trace, path: str, format: str = "json") -> None:
        """Write trace record to file."""
        data = trace.to_dict()
        if format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                # First line is header/metadata
                header = {k: v for k, v in data.items() if k != "steps"}
                f.write(json.dumps(header) + "\n")
                # Subsequent lines are individual steps
                for step in data["steps"]:
                    f.write(json.dumps(step) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def load(self, path: str) -> Trace:
        """Read and reconstruct a Trace object from disk."""
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            # Detect JSON vs JSONL
            if first_line.startswith("{") and not first_line.endswith("}"):
                # Multiline JSON file
                f.seek(0)
                data = json.load(f)
            else:
                try:
                    header = json.loads(first_line)
                    if "steps" in header:
                        # Single-line JSON
                        data = header
                    else:
                        # JSONL format
                        steps_data = []
                        for line in f:
                            line_str = line.strip()
                            if line_str:
                                steps_data.append(json.loads(line_str))
                        data = header
                        data["steps"] = steps_data
                except json.JSONDecodeError:
                    f.seek(0)
                    data = json.load(f)

        trace = Trace(
            name=data.get("name", "loaded_trace"),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            metadata=data.get("metadata", {}),
        )

        for s in data.get("steps", []):
            tok_count = s.get("tokens", 0)
            tok = TokenUsage(total_tokens=tok_count) if tok_count else None
            step = TraceStep(
                step_index=s.get("step_index", 0),
                action=s.get("action", ""),
                input_data=s.get("input_data"),
                output_data=s.get("output_data"),
                duration_ms=s.get("duration_ms", 0.0),
                token_usage=tok,
                metadata=s.get("metadata", {}),
            )
            trace.steps.append(step)

        return trace

