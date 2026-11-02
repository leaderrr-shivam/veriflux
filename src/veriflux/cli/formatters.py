"""Terminal output formatting and ANSI color management.

Provides clean headers, tables, colored status indicators, and progress bars without external libraries.
"""
from __future__ import annotations

import os
import sys
from typing import List, Sequence


class TerminalFormatter:
    """Formats terminal diagnostics using ANSI escape codes with automatic fallback."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    def __init__(self) -> None:
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        no_color = "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb"
        self.use_color = is_tty and not no_color

    def _c(self, text: str, color_code: str) -> str:
        if not self.use_color:
            return text
        return f"{color_code}{text}{self.RESET}"

    def header(self, text: str) -> str:
        line = "=" * max(40, len(text) + 4)
        return f"\n{self._c(text, self.BOLD + self.CYAN)}\n{self._c(line, self.GRAY)}"

    def success(self, text: str) -> str:
        return f"{self._c('[PASS]', self.BOLD + self.GREEN)} {text}"

    def warning(self, text: str) -> str:
        return f"{self._c('[WARN]', self.BOLD + self.YELLOW)} {text}"

    def error(self, text: str) -> str:
        return f"{self._c('[FAIL]', self.BOLD + self.RED)} {text}"

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
        """Render a clean text table."""
        if not headers and not rows:
            return ""

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(headers)]
        header_line = " | ".join(header_cells)
        sep_cells = ["-" * col_widths[i] for i in range(len(headers))]
        sep_line = "-+-".join(sep_cells)

        lines = [self._c(header_line, self.BOLD), self._c(sep_line, self.GRAY)]
        for row in rows:
            row_cells = [str(cell).ljust(col_widths[i]) if i < len(col_widths) else str(cell) for i, cell in enumerate(row)]
            lines.append(" | ".join(row_cells))

        return "\n".join(lines)
