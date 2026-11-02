"""Dynamic SVG badge generator for GitHub READMEs.

Generates self-contained SVG badges reflecting evaluation scores and quality gate status.
"""
from __future__ import annotations


class BadgeGenerator:
    """Generates shields.io-compatible vector SVG status badges."""

    _COLORS = {
        "green": "#10b981",
        "yellow": "#f59e0b",
        "red": "#ef4444",
        "blue": "#3b82f6",
        "purple": "#8b5cf6",
        "gray": "#6b7280",
    }

    def generate(self, label: str, value: str, color: str = "green") -> str:
        """Render standalone SVG badge."""
        hex_color = self._COLORS.get(color.lower(), color)
        label_w = max(40, len(label) * 7 + 10)
        value_w = max(30, len(value) * 7 + 10)
        total_w = label_w + value_w

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20" role="img" aria-label="{label}: {value}">
    <linearGradient id="s" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <clipPath id="r">
        <rect width="{total_w}" height="20" rx="3" fill="#fff"/>
    </clipPath>
    <g clip-path="url(#r)">
        <rect width="{label_w}" height="20" fill="#555"/>
        <rect x="{label_w}" width="{value_w}" height="20" fill="{hex_color}"/>
        <rect width="{total_w}" height="20" fill="url(#s)"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
        <text x="{label_w * 5}" y="140" transform="scale(.1)">{label}</text>
        <text x="{(label_w + value_w / 2) * 10}" y="140" transform="scale(.1)" font-weight="bold">{value}</text>
    </g>
</svg>"""

    def score_badge(self, metric: str, score: float) -> str:
        """Render auto-colored badge based on threshold value."""
        pct = f"{score * 100:.1f}%"
        color = "green" if score >= 0.85 else "yellow" if score >= 0.70 else "red"
        return self.generate(metric, pct, color=color)

