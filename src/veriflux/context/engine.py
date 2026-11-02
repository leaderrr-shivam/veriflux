"""Master context assembly and optimization engine.

Dynamically orchestrates heterogeneous context sources (system, history, RAG, tools)
into an optimized, deduplicated, and budget-compliant prompt payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence, Union

from veriflux.context.compressor import ContextCompressor
from veriflux.context.deduplicator import Deduplicator
from veriflux.context.pruner import ContextPruner
from veriflux.context.ranker import ContextRanker
from veriflux.tokens.budget_manager import (
    BudgetManager,
    BudgetSection,
    SectionPriority,
    TokenBudget,
)
from veriflux.tokens.counter import count_tokens


@dataclass(slots=True)
class Source:
    """A single context source stream."""

    name: str
    content: str
    source_type: str  # 'system', 'history', 'rag', 'tools'
    priority: SectionPriority = SectionPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def system(cls, text: str, priority: str = "critical") -> Source:
        p = SectionPriority[priority.upper()]
        return cls("system", text, "system", p)

    @classmethod
    def history(cls, messages: Sequence[Dict[str, str]], priority: str = "low") -> Source:
        text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        p = SectionPriority[priority.upper()]
        return cls("history", text, "history", p)

    @classmethod
    def rag(cls, chunks: Sequence[str], priority: str = "medium") -> Source:
        text = "\n\n".join(chunks)
        p = SectionPriority[priority.upper()]
        return cls("rag", text, "rag", p, metadata={"chunks": list(chunks)})

    @classmethod
    def tools(cls, tool_results: Any, priority: str = "high") -> Source:
        text = json.dumps(tool_results, indent=2) if not isinstance(tool_results, str) else tool_results
        p = SectionPriority[priority.upper()]
        return cls("tools", text, "tools", p)


@dataclass(slots=True, frozen=True)
class ContextResult:
    """Outcome of context assembly."""

    text: str
    token_count: int
    compression_ratio: float
    quality_score: float
    sources_retained: int
    sources_total: int
    sections: List[Dict[str, Any]] = field(default_factory=list)


class ContextEngine:
    """Assembles prompt contexts dynamically under strict token budgets."""

    def __init__(
        self,
        budget: Optional[TokenBudget] = None,
        strategy: str = "task_aware",
        model: str = "gpt-4o",
    ) -> None:
        self.budget = budget or TokenBudget(max_tokens=8192, reserve_output=2048)
        self.strategy = strategy
        self.model = model
        self.sources: List[Source] = []

        self.ranker = ContextRanker()
        self.deduplicator = Deduplicator()
        self.compressor = ContextCompressor(model=model)
        self.pruner = ContextPruner(model=model)
        self.budget_manager = BudgetManager(budget=self.budget, model=model)

    def add(self, source: Source) -> ContextEngine:
        """Add a context source to the assembly pipeline."""
        self.sources.append(source)
        return self

    def build(self, query: Optional[str] = None) -> ContextResult:
        """Assemble all sources into a single optimized prompt context."""
        if not self.sources:
            return ContextResult("", 0, 1.0, 1.0, 0, 0)

        # 1. Deduplicate and rank RAG chunks if query provided
        prepared_sections: Dict[str, BudgetSection] = {}
        for src in self.sources:
            content = src.content
            if src.source_type == "rag" and "chunks" in src.metadata:
                chunks = src.metadata["chunks"]
                deduped = self.deduplicator.deduplicate(chunks)
                if query:
                    ranked = self.ranker.rank(deduped, query=query, top_k=len(deduped))
                    content = "\n\n".join(r.text for r in ranked)
                else:
                    content = "\n\n".join(deduped)

            prepared_sections[src.name] = BudgetSection(
                name=src.name,
                content=content,
                priority=src.priority,
                min_tokens=20 if src.priority != SectionPriority.CRITICAL else count_tokens(content, model=self.model),
            )

        # 2. Allocate token budget across sections
        allocations = self.budget_manager.allocate(prepared_sections)

        # 3. Assemble and truncate sections according to allocation
        assembled_parts: List[str] = []
        section_summaries: List[Dict[str, Any]] = []
        total_raw_tokens = 0
        retained_count = 0

        for src in self.sources:
            sec = prepared_sections[src.name]
            raw_tokens = count_tokens(sec.content, model=self.model)
            total_raw_tokens += raw_tokens
            allowed = allocations.get(src.name, 0)

            if allowed > 0:
                fitted_content = self.budget_manager.truncate_to_budget(sec.content, allowed)
                if fitted_content.strip():
                    assembled_parts.append(fitted_content)
                    retained_count += 1

                section_summaries.append({
                    "name": src.name,
                    "type": src.source_type,
                    "tokens": count_tokens(fitted_content, model=self.model),
                    "allowed": allowed,
                })

        final_text = "\n\n".join(assembled_parts)
        final_tokens = count_tokens(final_text, model=self.model)
        compression_ratio = round(total_raw_tokens / max(1, final_tokens), 2)
        quality_score = min(1.0, max(0.6, 1.0 - (0.1 * (compression_ratio - 1.0))))

        return ContextResult(
            text=final_text,
            token_count=final_tokens,
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            sources_retained=retained_count,
            sources_total=len(self.sources),
            sections=section_summaries,
        )

