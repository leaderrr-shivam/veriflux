"""Unit tests for Structured Output and Schema Engine."""
from __future__ import annotations

from dataclasses import dataclass
import pytest
from veriflux.schema import (
    JSONRepair,
    SchemaBuilder,
    SchemaDiff,
    SchemaValidator,
    TypeCoercer,
    repair_json,
    validate_json,
)


def test_json_repair() -> None:
    repairer = JSONRepair()

    # Trailing comma and single quotes
    broken_1 = "{ 'name': 'Alice', 'age': 30, }"
    fixed_1 = repairer.repair(broken_1)
    data, _ = repairer.try_parse(broken_1)
    assert data["name"] == "Alice"
    assert data["age"] == 30

    # Truncated JSON
    broken_2 = '{"title": "Test", "items": ["a", "b"'
    data_2, _ = repairer.try_parse(broken_2)
    assert data_2["title"] == "Test"
    assert len(data_2["items"]) == 2

    # Unquoted keys and Python booleans
    broken_3 = "{ active: True, count: None }"
    data_3, _ = repairer.try_parse(broken_3)
    assert data_3["active"] is True
    assert data_3["count"] is None


def test_schema_validator() -> None:
    schema = {
        "type": "object",
        "required": ["id", "username", "score"],
        "properties": {
            "id": {"type": "integer"},
            "username": {"type": "string", "minLength": 3},
            "score": {"type": "number", "minimum": 0.0, "maximum": 100.0},
            "role": {"type": "string", "enum": ["admin", "user"]},
        },
    }

    valid_payload = {"id": 1, "username": "alice", "score": 95.5, "role": "admin"}
    invalid_payload = {"id": "not_an_int", "username": "al", "score": 150.0}

    validator = SchemaValidator()
    res_valid = validator.validate(valid_payload, schema)
    res_invalid = validator.validate(invalid_payload, schema)

    assert res_valid.is_valid
    assert not res_invalid.is_valid
    assert len(res_invalid.errors) >= 3


def test_schema_builder_and_diff() -> None:
    builder = SchemaBuilder("User")
    builder.string("name").required()
    builder.integer("age").min(0)
    schema = builder.build()

    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "name" in schema["required"]

    diff_engine = SchemaDiff()
    exp = {"name": "Alice", "settings": {"theme": "dark"}}
    act = {"name": "Bob", "settings": {"theme": "light", "font": 12}}

    diffs = diff_engine.diff(exp, act)
    assert len(diffs) >= 2

