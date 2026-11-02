"""Evaluation artifact storage and management.

Persists test traces, failure diffs, and benchmark logs to designated CI artifact folders.
"""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional


class ArtifactManager:
    """Manages disk persistence and retrieval of test run artifacts."""

    def __init__(self, directory: str = ".veriflux_artifacts") -> None:
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def save(self, name: str, data: Any, format: str = "json") -> str:
        """Save artifact to disk and return absolute file path."""
        filename = f"{name}.{format}"
        filepath = os.path.join(self.directory, filename)

        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data))

        return os.path.abspath(filepath)

    def load(self, name: str, format: str = "json") -> Any:
        """Load artifact content from disk."""
        filename = f"{name}.{format}"
        filepath = os.path.join(self.directory, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Artifact '{filename}' not found in {self.directory}")

        if format == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

    def list_artifacts(self) -> List[str]:
        """List all artifact filenames stored in the artifact directory."""
        if not os.path.exists(self.directory):
            return []
        return sorted(os.listdir(self.directory))

