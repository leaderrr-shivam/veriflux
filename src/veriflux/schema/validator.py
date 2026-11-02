"""Structured JSON and schema validation engine.

Validates hierarchical JSON data structures against JSON-schema specifications
with detailed path-indexed error reporting and auto-coercion options.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from veriflux.schema.json_repair import JSONRepair
from veriflux.schema.type_coercer import TypeCoercer


@dataclass(slots=True, frozen=True)
class SchemaValidationError:
    """An individual schema validation failure.

    Attributes:
        path: Dot-notated object path (e.g. 'user.address.zipcode').
        message: Explanatory error description.
        expected: Expected type or constraint.
        actual: Actual value observed.
    """

    path: str
    message: str
    expected: Any = None
    actual: Any = None


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a data structure against a schema."""

    is_valid: bool
    errors: List[SchemaValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    coerced_data: Any = None


class SchemaValidator:
    """Validates arbitrary data structures against JSON-schema definitions."""

    def __init__(self, coerce_types: bool = False) -> None:
        self.coerce_types = coerce_types
        self.coercer = TypeCoercer() if coerce_types else None

    def validate(self, data: Any, schema: Dict[str, Any], path: str = "$") -> ValidationResult:
        """Validate data object against schema definition.

        Args:
            data: Python dictionary, list, or primitive.
            schema: JSON Schema dictionary.
            path: Current navigation path string.

        Returns:
            ValidationResult with list of discovered SchemaValidationErrors.
        """
        errors: List[SchemaValidationError] = []

        # 1. Type validation
        expected_type = schema.get("type")
        if expected_type:
            types = [expected_type] if isinstance(expected_type, str) else expected_type
            matched_type = False
            for t in types:
                if self._check_type(data, t):
                    matched_type = True
                    break

            if not matched_type and self.coerce_types and self.coercer is not None and isinstance(expected_type, str):
                try:
                    data = self.coercer.coerce(data, expected_type)
                    matched_type = True
                except Exception:
                    pass

            if not matched_type:
                errors.append(
                    SchemaValidationError(
                        path=path,
                        message=f"Expected type {expected_type}, got {type(data).__name__}",
                        expected=expected_type,
                        actual=type(data).__name__,
                    )
                )
                return ValidationResult(is_valid=False, errors=errors)

        # 2. Enum constraints
        if "enum" in schema and data not in schema["enum"]:
            errors.append(
                SchemaValidationError(
                    path=path,
                    message=f"Value {data!r} is not in allowed enum {schema['enum']}",
                    expected=schema["enum"],
                    actual=data,
                )
            )

        # 3. String constraints
        if isinstance(data, str):
            if "minLength" in schema and len(data) < schema["minLength"]:
                errors.append(SchemaValidationError(path=path, message=f"String length {len(data)} < minLength {schema['minLength']}"))
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                errors.append(SchemaValidationError(path=path, message=f"String length {len(data)} > maxLength {schema['maxLength']}"))
            if "pattern" in schema and not re.search(schema["pattern"], data):
                errors.append(SchemaValidationError(path=path, message=f"String does not match pattern {schema['pattern']}"))

        # 4. Number constraints
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            if "minimum" in schema and data < schema["minimum"]:
                errors.append(SchemaValidationError(path=path, message=f"Value {data} < minimum {schema['minimum']}"))
            if "maximum" in schema and data > schema["maximum"]:
                errors.append(SchemaValidationError(path=path, message=f"Value {data} > maximum {schema['maximum']}"))

        # 5. Object constraints
        if isinstance(data, dict):
            # Check required fields
            required = schema.get("required", [])
            for req in required:
                if req not in data:
                    errors.append(
                        SchemaValidationError(
                            path=f"{path}.{req}",
                            message=f"Required property '{req}' is missing",
                            expected="present",
                            actual="missing",
                        )
                    )

            # Check properties
            props = schema.get("properties", {})
            for key, val in data.items():
                if key in props:
                    sub_res = self.validate(val, props[key], path=f"{path}.{key}")
                    errors.extend(sub_res.errors)

        # 6. Array constraints
        if isinstance(data, list):
            if "minItems" in schema and len(data) < schema["minItems"]:
                errors.append(SchemaValidationError(path=path, message=f"Array length {len(data)} < minItems {schema['minItems']}"))
            if "maxItems" in schema and len(data) > schema["maxItems"]:
                errors.append(SchemaValidationError(path=path, message=f"Array length {len(data)} > maxItems {schema['maxItems']}"))

            item_schema = schema.get("items")
            if item_schema and isinstance(item_schema, dict):
                for idx, item in enumerate(data):
                    sub_res = self.validate(item, item_schema, path=f"{path}[{idx}]")
                    errors.extend(sub_res.errors)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            coerced_data=data,
        )

    def _check_type(self, value: Any, target_type: str) -> bool:
        t = target_type.lower()
        if t == "null":
            return value is None
        if t in ("string", "str"):
            return isinstance(value, str)
        if t in ("boolean", "bool"):
            return isinstance(value, bool)
        if t in ("integer", "int"):
            return isinstance(value, int) and not isinstance(value, bool)
        if t in ("number", "float"):
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if t in ("object", "dict"):
            return isinstance(value, dict)
        if t in ("array", "list"):
            return isinstance(value, list)
        return True


def validate_json(text: str, schema: Optional[Dict[str, Any]] = None, repair: bool = True) -> ValidationResult:
    """Parse and validate JSON string against schema."""
    repairer = JSONRepair()
    try:
        if repair:
            data, _ = repairer.try_parse(text)
        else:
            data = json.loads(text)
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            errors=[SchemaValidationError(path="$", message=f"Invalid JSON syntax: {exc}")],
        )

    if schema is None:
        return ValidationResult(is_valid=True, coerced_data=data)

    validator = SchemaValidator()
    return validator.validate(data, schema)

