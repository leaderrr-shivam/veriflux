"""Levenshtein and Damerau-Levenshtein edit distance algorithms.

Provides linear-space distance computation, normalised similarity scores,
and step-by-step edit operation sequence tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(slots=True, frozen=True)
class EditOp:
    """A single atomic edit operation.

    Attributes:
        op_type: One of 'match', 'insert', 'delete', 'replace', 'transpose'.
        src_pos: Index in the source string.
        dest_pos: Index in the destination string.
        src_char: Character in source (or empty for insert).
        dest_char: Character in destination (or empty for delete).
    """

    op_type: str
    src_pos: int
    dest_pos: int
    src_char: str
    dest_char: str


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the minimum edit distance with O(min(len(s1), len(s2))) space.

    Args:
        s1: Source string.
        s2: Target string.

    Returns:
        Non-negative integer edit count.
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    # s2 is the shorter string
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute normalised Levenshtein distance in [0.0, 1.0]."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    dist = levenshtein_distance(s1, s2)
    return dist / max_len


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute string similarity as 1.0 - normalized_levenshtein(s1, s2)."""
    return 1.0 - normalized_levenshtein(s1, s2)


def damerau_levenshtein(s1: str, s2: str) -> int:
    """Compute Damerau-Levenshtein distance supporting adjacent transpositions."""
    len1, len2 = len(s1), len(s2)
    if s1 == s2:
        return 0
    if not s1:
        return len2
    if not s2:
        return len1

    # Matrix for full Damerau-Levenshtein DP
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            # Transposition check
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)

    return d[len1][len2]


def edit_operations(s1: str, s2: str) -> List[EditOp]:
    """Compute the sequence of edit operations transforming s1 into s2."""
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    # Backtrack to reconstruct operations
    ops: List[EditOp] = []
    i, j = len1, len2
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(EditOp("match", i - 1, j - 1, s1[i - 1], s2[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(EditOp("replace", i - 1, j - 1, s1[i - 1], s2[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(EditOp("delete", i - 1, j, s1[i - 1], ""))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(EditOp("insert", i, j - 1, "", s2[j - 1]))
            j -= 1

    ops.reverse()
    return ops

