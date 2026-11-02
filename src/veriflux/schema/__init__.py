"""Structured output validation and repair engine for Veriflux.

Provides JSON repair, JSON Schema validation, type coercion, fluent schema building,
and structural diffing.
"""
from __future__ import annotations

from veriflux.schema.constraints import (
    Constraint,
    ConstraintSet,
    Email,
    Length,
    NonNegative,
    NotEmpty,
    OneOf,
    Pattern,
    Positive,
    Range,
    URL,
    constrained,
)
from veriflux.schema.diff_engine import DiffEntry, DiffType, SchemaDiff
from veriflux.schema.json_repair import JSONRepair, repair_json
from veriflux.schema.schema_builder import FieldBuilder, SchemaBuilder
from veriflux.schema.type_coercer import TypeCoercer
from veriflux.schema.validator import (
    SchemaValidationError,
    SchemaValidator,
    ValidationResult,
    validate_json,
)

__all__ = [
    # Validation
    "SchemaValidator",
    "SchemaValidationError",
    "ValidationResult",
    "validate_json",
    # Repair
    "JSONRepair",
    "repair_json",
    # Coercion
    "TypeCoercer",
    # Builder
    "SchemaBuilder",
    "FieldBuilder",
    # Diff
    "SchemaDiff",
    "DiffEntry",
    "DiffType",
    # Constraints
    "Constraint",
    "ConstraintSet",
    "Range",
    "Length",
    "Pattern",
    "OneOf",
    "NotEmpty",
    "Email",
    "URL",
    "Positive",
    "NonNegative",
    "constrained",
]

