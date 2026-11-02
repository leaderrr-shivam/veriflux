"""Declarative field constraints and invariant validation.

Provides modular constraint primitives (Range, Length, Pattern, Email, etc.)
and functional combinators for data validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import re
from typing import Any, Callable, List, Optional, Protocol, Sequence, Set, Union


class Constraint(Protocol):
    """Protocol for field-level constraints."""

    def validate(self, value: Any) -> bool: ...
    @property
    def message(self) -> str: ...


@dataclass(slots=True, frozen=True)
class Range:
    """Numerical range constraint [min_val, max_val]."""

    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        if self.min_val is not None and value < self.min_val:
            return False
        if self.max_val is not None and value > self.max_val:
            return False
        return True

    @property
    def message(self) -> str:
        return f"Value must be between {self.min_val} and {self.max_val}"


@dataclass(slots=True, frozen=True)
class Length:
    """Sequence length constraint [min_len, max_len]."""

    min_len: Optional[int] = None
    max_len: Optional[int] = None

    def validate(self, value: Any) -> bool:
        try:
            length = len(value)
        except TypeError:
            return False
        if self.min_len is not None and length < self.min_len:
            return False
        if self.max_len is not None and length > self.max_len:
            return False
        return True

    @property
    def message(self) -> str:
        return f"Length must be between {self.min_len} and {self.max_len}"


@dataclass(slots=True, frozen=True)
class Pattern:
    """Regex pattern constraint."""

    regex: str

    def validate(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return bool(re.search(self.regex, value))

    @property
    def message(self) -> str:
        return f"Value must match regex pattern: {self.regex}"


@dataclass(slots=True, frozen=True)
class OneOf:
    """Set membership constraint."""

    allowed: Set[Any]

    def __init__(self, allowed: Sequence[Any]) -> None:
        object.__setattr__(self, "allowed", set(allowed))

    def validate(self, value: Any) -> bool:
        return value in self.allowed

    @property
    def message(self) -> str:
        return f"Value must be one of: {sorted(list(self.allowed))}"


@dataclass(slots=True, frozen=True)
class NotEmpty:
    """Non-empty string, collection, or non-None constraint."""

    def validate(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, list, dict, set, tuple)):
            return len(value) > 0
        return True

    @property
    def message(self) -> str:
        return "Value cannot be empty"


@dataclass(slots=True, frozen=True)
class Email:
    """Email address constraint."""

    _RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}$")

    def validate(self, value: Any) -> bool:
        return isinstance(value, str) and bool(self._RE.match(value.strip()))

    @property
    def message(self) -> str:
        return "Value must be a valid email address"


@dataclass(slots=True, frozen=True)
class URL:
    """URL constraint."""

    _RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)

    def validate(self, value: Any) -> bool:
        return isinstance(value, str) and bool(self._RE.match(value.strip()))

    @property
    def message(self) -> str:
        return "Value must be a valid HTTP/HTTPS URL"


@dataclass(slots=True, frozen=True)
class Positive:
    """Strictly positive number constraint (> 0)."""

    def validate(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and value > 0

    @property
    def message(self) -> str:
        return "Value must be strictly positive (> 0)"


@dataclass(slots=True, frozen=True)
class NonNegative:
    """Non-negative number constraint (>= 0)."""

    def validate(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and value >= 0

    @property
    def message(self) -> str:
        return "Value must be non-negative (>= 0)"


class ConstraintSet:
    """Logical combination of multiple constraints with AND/OR logic."""

    def __init__(self, *constraints: Constraint, mode: str = "ALL") -> None:
        self.constraints = list(constraints)
        self.mode = mode.upper()

    def validate(self, value: Any) -> bool:
        if self.mode == "ALL":
            return all(c.validate(value) for c in self.constraints)
        elif self.mode == "ANY":
            return any(c.validate(value) for c in self.constraints)
        raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def message(self) -> str:
        sub_msgs = "; ".join(c.message for c in self.constraints)
        return f"Must satisfy {self.mode} of: [{sub_msgs}]"


def constrained(**param_constraints: Constraint) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Function decorator enforcing parameter constraints at runtime."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check kwargs
            for param, constraint in param_constraints.items():
                if param in kwargs:
                    val = kwargs[param]
                    if not constraint.validate(val):
                        raise ValueError(f"Constraint failed for parameter '{param}': {constraint.message}. Got {val!r}")
            return fn(*args, **kwargs)
        return wrapper
    return decorator

