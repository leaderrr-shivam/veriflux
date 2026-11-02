"""Top-level convenience API for Veriflux.

Provides simple, one-line evaluation assertions, configuration functions,
and pytest decorators.
"""
from __future__ import annotations

import functools
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from veriflux._types import VerifluxConfig


_fn_is_grounded = None
_fn_is_relevant = None
_fn_is_faithful = None
_fn_no_pii = None
_fn_no_toxicity = None
_fn_no_hallucinations = None
_fn_token_count = None
_repairer = None
_validator = None


def configure(**kwargs: Any) -> VerifluxConfig:
    """Configure global Veriflux settings."""
    from veriflux.core.config import configure as _core_configure
    return _core_configure(**kwargs)


def get_config() -> VerifluxConfig:
    """Get the active Veriflux global configuration."""
    from veriflux.core.config import get_config as _core_get_config
    return _core_get_config()


def token_count(text: str, model: str = "gpt-4o") -> int:
    """Count or estimate the number of tokens in a string for a given model."""
    global _fn_token_count
    if _fn_token_count is None:
        from veriflux.tokens.counter import count_tokens
        _fn_token_count = count_tokens
    return _fn_token_count(text, model=model)


def is_valid_json(text: str, schema: Optional[Dict[str, Any]] = None) -> bool:
    """Check if text is valid JSON and optionally conforms to a schema."""
    global _repairer, _validator
    try:
        data = json.loads(text)
    except Exception:
        # Try JSON repair if available
        try:
            if _repairer is None:
                from veriflux.schema.json_repair import JSONRepair
                _repairer = JSONRepair()
            repaired = _repairer.repair(text)
            data = json.loads(repaired)
        except Exception:
            return False

    if schema is not None:
        try:
            if _validator is None:
                from veriflux.schema.validator import SchemaValidator
                _validator = SchemaValidator()
            res = _validator.validate(data, schema)
            return res.is_valid
        except Exception:
            return False

    return True


def is_grounded(response: str, context: str, threshold: float = 0.7) -> bool:
    """Check whether an LLM response is grounded in provided reference context."""
    global _fn_is_grounded
    if _fn_is_grounded is None:
        from veriflux.assertions.semantic import is_grounded as _fn
        _fn_is_grounded = _fn
    return _fn_is_grounded(response, context, threshold=threshold)


def is_relevant(response: str, query: str, threshold: float = 0.6) -> bool:
    """Check whether a response is semantically relevant to the input query."""
    global _fn_is_relevant
    if _fn_is_relevant is None:
        from veriflux.assertions.semantic import is_relevant as _fn
        _fn_is_relevant = _fn
    return _fn_is_relevant(response, query, threshold=threshold)


def is_faithful(response: str, context: str, threshold: float = 0.7) -> bool:
    """Check whether response faithfully adheres to context without introducing foreign facts."""
    global _fn_is_faithful
    if _fn_is_faithful is None:
        from veriflux.assertions.semantic import is_faithful as _fn
        _fn_is_faithful = _fn
    return _fn_is_faithful(response, context, threshold=threshold)


def no_pii(text: str, categories: Optional[List[str]] = None) -> bool:
    """Verify that text is free of sensitive personally identifiable information."""
    global _fn_no_pii
    if _fn_no_pii is None:
        from veriflux.assertions.safety import no_pii as _fn
        _fn_no_pii = _fn
    return _fn_no_pii(text, categories=categories)


def no_toxicity(text: str, threshold: float = 0.5) -> bool:
    """Verify that text does not contain toxic, abusive, or harmful language."""
    global _fn_no_toxicity
    if _fn_no_toxicity is None:
        from veriflux.assertions.safety import no_toxicity as _fn
        _fn_no_toxicity = _fn
    return _fn_no_toxicity(text, threshold=threshold)


def no_hallucinations(response: str, context: str, threshold: float = 0.8) -> bool:
    """Run full atomic hallucination detection across claims."""
    global _fn_no_hallucinations
    if _fn_no_hallucinations is None:
        from veriflux.assertions.factual import no_hallucinations as _fn
        _fn_no_hallucinations = _fn
    return _fn_no_hallucinations(response, context, threshold=threshold)


def test(
    dataset: Optional[str] = None,
    layer: str = "standard",
    timeout: Optional[float] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator marking a test function as a Veriflux-managed evaluation case.

    Usage:
        @veriflux.test(dataset="golden.jsonl", layer="fast")
        def test_llm_output(case):
            assert veriflux.is_grounded(case.response, case.context)
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        setattr(wrapper, "__veriflux_test__", True)
        setattr(wrapper, "__veriflux_dataset__", dataset)
        setattr(wrapper, "__veriflux_layer__", layer)
        setattr(wrapper, "__veriflux_timeout__", timeout)
        return wrapper

    return decorator

