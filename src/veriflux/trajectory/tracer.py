"""Zero-overhead agent execution tracer and telemetry collector.

Captures tool executions, LLM steps, token consumptions, and state mutations
with full contextvar propagation across async boundaries.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

from veriflux._types import TokenUsage, TraceStep

_current_tracer: contextvars.ContextVar[Optional[Tracer]] = contextvars.ContextVar(
    "veriflux_current_tracer", default=None
)


def current_tracer() -> Optional[Tracer]:
    """Return the currently active Tracer instance in the async execution context."""
    return _current_tracer.get()


@dataclass(slots=True)
class Trace:
    """Complete execution record of an agent session.

    Attributes:
        name: Identifier for the session or agent workflow.
        steps: Chronological list of execution steps.
        start_time: Unix timestamp when tracing began.
        end_time: Unix timestamp when tracing completed.
        metadata: Arbitrary run-level metadata.
    """

    name: str
    steps: List[TraceStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def total_duration_ms(self) -> float:
        if self.end_time > self.start_time:
            return (self.end_time - self.start_time) * 1000.0
        return sum(s.duration_ms for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.token_usage.total_tokens for s in self.steps if s.token_usage)

    @property
    def total_cost(self) -> float:
        return sum(s.token_usage.estimated_cost for s in self.steps if s.token_usage)

    @property
    def has_errors(self) -> bool:
        return any(s.metadata.get("error") is not None for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "total_steps": self.total_steps,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "has_errors": self.has_errors,
            "metadata": self.metadata,
            "steps": [
                {
                    "step_index": s.step_index,
                    "action": s.action,
                    "duration_ms": round(s.duration_ms, 2),
                    "input_data": s.input_data,
                    "output_data": s.output_data,
                    "tokens": s.token_usage.total_tokens if s.token_usage else 0,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
        }


class Tracer:
    """Instruments agent workflows to capture execution traces."""

    def __init__(self, name: str = "agent_trace", metadata: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.trace = Trace(name=name, metadata=metadata or {})
        self._token: Optional[contextvars.Token[Optional[Tracer]]] = None

    def __enter__(self) -> Tracer:
        self.trace.start_time = time.time()
        self._token = _current_tracer.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.trace.end_time = time.time()
        if exc_val is not None:
            self.trace.metadata["exception"] = str(exc_val)
        if self._token is not None:
            _current_tracer.reset(self._token)

    def record_step(
        self,
        action: str,
        input_data: Any = None,
        output_data: Any = None,
        duration_ms: float = 0.0,
        token_usage: Optional[TokenUsage] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
        tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> TraceStep:
        """Record an arbitrary agent step."""
        meta = dict(metadata or {})
        if agent is not None:
            meta["agent"] = agent

        dur = latency_ms if latency_ms > 0 else duration_ms
        usage = token_usage
        if usage is None and tokens > 0:
            usage = TokenUsage(total_tokens=tokens, input_tokens=tokens, output_tokens=0)

        step = TraceStep(
            step_index=len(self.trace.steps),
            action=action,
            input_data=input_data,
            output_data=output_data,
            duration_ms=dur,
            token_usage=usage,
            metadata=meta,
        )
        self.trace.steps.append(step)
        return step

    @property
    def steps(self) -> List[TraceStep]:
        """Return the captured trace steps."""
        return self.trace.steps

    def analyze(self) -> Trace:
        """Return the completed execution trace summary."""
        return self.trace

    def record_llm_call(
        self,
        model: str,
        messages: Any,
        response: str,
        tokens: Optional[TokenUsage] = None,
        duration_ms: float = 0.0,
    ) -> TraceStep:
        """Specialized recording method for LLM inferences."""
        return self.record_step(
            action=f"llm_call:{model}",
            input_data=messages,
            output_data=response,
            duration_ms=duration_ms,
            token_usage=tokens,
            metadata={"model": model, "step_type": "llm_call"},
        )

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Any,
        result: Any,
        duration_ms: float = 0.0,
        error: Optional[Exception] = None,
    ) -> TraceStep:
        """Specialized recording method for agent tool invocations."""
        meta: Dict[str, Any] = {"tool_name": tool_name, "step_type": "tool_call"}
        if error is not None:
            meta["error"] = str(error)
        return self.record_step(
            action=f"tool_call:{tool_name}",
            input_data=arguments,
            output_data=result,
            duration_ms=duration_ms,
            metadata=meta,
        )


# Ergonomic alias
TrajectoryTracer = Tracer

