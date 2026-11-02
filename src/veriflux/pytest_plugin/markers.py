"""Pytest custom marker registrations and inspection utilities.

Registers `@pytest.mark.veriflux_test` and provides layer-based filtering.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def extract_veriflux_marker(item: Any) -> Optional[Dict[str, Any]]:
    """Extract metadata kwargs from a @pytest.mark.veriflux_test or function attributes."""
    # Check function attribute first (set by @veriflux.test decorator)
    obj = getattr(item, "obj", None)
    if obj is not None and getattr(obj, "__veriflux_test__", False):
        return {
            "dataset": getattr(obj, "__veriflux_dataset__", None),
            "layer": getattr(obj, "__veriflux_layer__", "standard"),
            "timeout": getattr(obj, "__veriflux_timeout__", None),
        }

    # Check pytest marker
    marker = item.get_closest_marker("veriflux_test")
    if marker:
        return {
            "dataset": marker.kwargs.get("dataset"),
            "layer": marker.kwargs.get("layer", "standard"),
            "timeout": marker.kwargs.get("timeout"),
        }

    return None

