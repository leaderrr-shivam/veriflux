"""Deep schema and structural diff comparison engine.

Calculates exact additions, deletions, type mismatches, and value alterations
between expected schema structures and actual LLM JSON payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
from typing import Any, Dict, List, Optional


class DiffType(enum.Enum):
    """Categorisation of a schema divergence."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    TYPE_MISMATCH = "type_mismatch"


@dataclass(slots=True, frozen=True)
class DiffEntry:
    """An individual difference between two structured entities.

    Attributes:
        path: Object path where difference occurred.
        diff_type: Nature of difference (ADDED, REMOVED, CHANGED, TYPE_MISMATCH).
        expected_value: Reference value or type.
        actual_value: Observed value or type.
    """

    path: str
    diff_type: DiffType
    expected_value: Any = None
    actual_value: Any = None


class SchemaDiff:
    """Computes recursive differences across nested dictionaries, lists, and primitives."""

    def diff(self, expected: Any, actual: Any, path: str = "$") -> List[DiffEntry]:
        """Deep comparison between expected and actual data structures."""
        diffs: List[DiffEntry] = []

        # Type mismatch check
        if type(expected) is not type(actual):
            # Allow int/float interchangeability if values match
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and expected == actual:
                return []
            diffs.append(
                DiffEntry(
                    path=path,
                    diff_type=DiffType.TYPE_MISMATCH,
                    expected_value=type(expected).__name__,
                    actual_value=type(actual).__name__,
                )
            )
            return diffs

        # Dictionaries
        if isinstance(expected, dict):
            all_keys = set(expected.keys()) | set(actual.keys())
            for key in sorted(all_keys):
                sub_path = f"{path}.{key}"
                if key in expected and key not in actual:
                    diffs.append(
                        DiffEntry(
                            path=sub_path,
                            diff_type=DiffType.REMOVED,
                            expected_value=expected[key],
                            actual_value=None,
                        )
                    )
                elif key not in expected and key in actual:
                    diffs.append(
                        DiffEntry(
                            path=sub_path,
                            diff_type=DiffType.ADDED,
                            expected_value=None,
                            actual_value=actual[key],
                        )
                    )
                else:
                    diffs.extend(self.diff(expected[key], actual[key], path=sub_path))

        # Lists
        elif isinstance(expected, list):
            len_exp = len(expected)
            len_act = len(actual)
            min_len = min(len_exp, len_act)

            for i in range(min_len):
                diffs.extend(self.diff(expected[i], actual[i], path=f"{path}[{i}]"))

            if len_exp > len_act:
                for i in range(len_act, len_exp):
                    diffs.append(
                        DiffEntry(
                            path=f"{path}[{i}]",
                            diff_type=DiffType.REMOVED,
                            expected_value=expected[i],
                            actual_value=None,
                        )
                    )
            elif len_act > len_exp:
                for i in range(len_exp, len_act):
                    diffs.append(
                        DiffEntry(
                            path=f"{path}[{i}]",
                            diff_type=DiffType.ADDED,
                            expected_value=None,
                            actual_value=actual[i],
                        )
                    )

        # Primitives
        else:
            if expected != actual:
                diffs.append(
                    DiffEntry(
                        path=path,
                        diff_type=DiffType.CHANGED,
                        expected_value=expected,
                        actual_value=actual,
                    )
                )

        return diffs

