"""Fluent schema builder and schema inference from Python dataclasses or example data.

Allows programmatic JSON-schema generation using chaining without raw dictionary boilerplate.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Type


class FieldBuilder:
    """Fluent builder for a single schema property."""

    def __init__(self, name: str, field_type: str, parent: SchemaBuilder) -> None:
        self.name = name
        self.field_type = field_type
        self.parent = parent
        self.spec: Dict[str, Any] = {"type": field_type}

    def required(self) -> FieldBuilder:
        self.parent._required.add(self.name)
        return self

    def optional(self) -> FieldBuilder:
        self.parent._required.discard(self.name)
        return self

    def min(self, val: float) -> FieldBuilder:
        if self.field_type in ("integer", "number"):
            self.spec["minimum"] = val
        elif self.field_type in ("string", "array"):
            self.spec["minLength" if self.field_type == "string" else "minItems"] = int(val)
        return self

    def max(self, val: float) -> FieldBuilder:
        if self.field_type in ("integer", "number"):
            self.spec["maximum"] = val
        elif self.field_type in ("string", "array"):
            self.spec["maxLength" if self.field_type == "string" else "maxItems"] = int(val)
        return self

    def pattern(self, regex: str) -> FieldBuilder:
        self.spec["pattern"] = regex
        return self

    def enum(self, values: Sequence[Any]) -> FieldBuilder:
        self.spec["enum"] = list(values)
        return self

    def description(self, desc: str) -> FieldBuilder:
        self.spec["description"] = desc
        return self

    def default(self, val: Any) -> FieldBuilder:
        self.spec["default"] = val
        return self


class SchemaBuilder:
    """Fluent builder for top-level JSON schema specifications."""

    def __init__(self, title: str = "Schema", description: str = "") -> None:
        self.title = title
        self.description = description
        self._properties: Dict[str, Dict[str, Any]] = {}
        self._required: set[str] = set()

    def string(self, name: str) -> FieldBuilder:
        fb = FieldBuilder(name, "string", self)
        self._properties[name] = fb.spec
        return fb

    def integer(self, name: str) -> FieldBuilder:
        fb = FieldBuilder(name, "integer", self)
        self._properties[name] = fb.spec
        return fb

    def number(self, name: str) -> FieldBuilder:
        fb = FieldBuilder(name, "number", self)
        self._properties[name] = fb.spec
        return fb

    def boolean(self, name: str) -> FieldBuilder:
        fb = FieldBuilder(name, "boolean", self)
        self._properties[name] = fb.spec
        return fb

    def array(self, name: str, item_schema: Optional[Dict[str, Any]] = None) -> FieldBuilder:
        fb = FieldBuilder(name, "array", self)
        if item_schema:
            fb.spec["items"] = item_schema
        self._properties[name] = fb.spec
        return fb

    def object(self, name: str, properties: Optional[Dict[str, Any]] = None) -> FieldBuilder:
        fb = FieldBuilder(name, "object", self)
        if properties:
            fb.spec["properties"] = properties
        self._properties[name] = fb.spec
        return fb

    def build(self) -> Dict[str, Any]:
        """Construct the complete JSON schema dictionary."""
        schema: Dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.title,
            "type": "object",
            "properties": self._properties,
        }
        if self.description:
            schema["description"] = self.description
        if self._required:
            schema["required"] = sorted(list(self._required))
        return schema

    @staticmethod
    def from_example(data: Any) -> Dict[str, Any]:
        """Infer JSON schema automatically from an example Python object."""
        if isinstance(data, dict):
            props = {k: SchemaBuilder.from_example(v) for k, v in data.items()}
            return {
                "type": "object",
                "properties": props,
                "required": sorted(list(data.keys())),
            }
        elif isinstance(data, list):
            item_schema = SchemaBuilder.from_example(data[0]) if data else {"type": "string"}
            return {"type": "array", "items": item_schema}
        elif isinstance(data, bool):
            return {"type": "boolean"}
        elif isinstance(data, int):
            return {"type": "integer"}
        elif isinstance(data, float):
            return {"type": "number"}
        elif isinstance(data, str):
            return {"type": "string"}
        elif data is None:
            return {"type": "null"}
        return {"type": "string"}

    @staticmethod
    def from_dataclass(cls: Type[Any]) -> Dict[str, Any]:
        """Infer JSON schema from a dataclass definition."""
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls} is not a dataclass")

        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        props: Dict[str, Any] = {}
        required: List[str] = []

        for f in dataclasses.fields(cls):
            t_name = type_map.get(f.type, "string")
            props[f.name] = {"type": t_name}
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                required.append(f.name)

        return {
            "type": "object",
            "title": cls.__name__,
            "properties": props,
            "required": sorted(required),
        }

