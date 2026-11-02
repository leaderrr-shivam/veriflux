"""Context engineering and token budget optimization engine for Veriflux.

Provides multi-strategy compression, chunk ranking, deduplication, sliding windows,
and priority-based prompt assembly.
"""
from __future__ import annotations

from veriflux.context.compressor import CompressedContext, ContextCompressor
from veriflux.context.deduplicator import Deduplicator
from veriflux.context.engine import ContextEngine, ContextResult, Source
from veriflux.context.pruner import ContextPruner
from veriflux.context.ranker import ContextRanker, RankedChunk
from veriflux.context.window import SlidingWindow
from veriflux.tokens.budget_manager import TokenBudget

__all__ = [
    # Engine
    "ContextEngine",
    "ContextResult",
    "Source",
    "TokenBudget",
    # Ranking
    "ContextRanker",
    "RankedChunk",
    # Compression
    "ContextCompressor",
    "CompressedContext",
    # Deduplication
    "Deduplicator",
    # Pruning
    "ContextPruner",
    # Window
    "SlidingWindow",
]

