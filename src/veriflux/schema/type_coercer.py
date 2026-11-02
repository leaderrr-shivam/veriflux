"""Intelligent type coercion engine for LLM structured outputs.

Safely coerces common LLM stringified types ('true', '42', 'null') into canonical
Python primitives in both strict and permissive modes.
"""
from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any, Dict, Optional

from veriflux.exceptions import TypeCoercionError


class TypeCoercer:
    """Coerces raw and stringified values into expected target data types."""

    _BOOL_TRUE = {"true", "1", "yes", "t", "y", "on"}
    _BOOL_FALSE = {"false", "0", "no", "f", "n", "off"}
    _NULL_STRINGS = {"null", "none", "nil", "undefined", "n/a", "na", ""}

    def __init__(self, mode: str = "loose") -> None:
        self.mode = mode.lower()  # "strict" or "loose"

    def coerce(self, value: Any, target_type: str) -> Any:
        """Coerce value into the specified target type.

        Supported target types: 'string', 'str', 'integer', 'int', 'number',
        'float', 'boolean', 'bool', 'null', 'date', 'datetime'.

        Raises:
            TypeCoercionError: If coercion fails in strict mode.
        """
        target = target_type.lower()

        if value is None:
            if target in ("null", "none"):
                return None
            if self.mode == "loose":
                if target in ("string", "str"):
                    return ""
                if target in ("integer", "int"):
                    return 0
                if target in ("number", "float"):
                    return 0.0
                if target in ("boolean", "bool"):
                    return False
            raise TypeCoercionError(f"Cannot coerce None to {target_type}", source_type="NoneType", target_type=target_type)

        # 1. Strings
        if target in ("string", "str"):
            return str(value)

        # 2. Booleans
        if target in ("boolean", "bool"):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                v_clean = value.strip().lower()
                if v_clean in self._BOOL_TRUE:
                    return True
                if v_clean in self._BOOL_FALSE:
                    return False
            if self.mode == "loose":
                return bool(value)
            raise TypeCoercionError(f"Cannot coerce {value!r} to boolean", source_type=type(value).__name__, target_type=target_type)

        # 3. Integers
        if target in ("integer", "int"):
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, float):
                if value.is_integer() or self.mode == "loose":
                    return int(value)
                raise TypeCoercionError(f"Loss of precision coercing float {value} to int", source_type="float", target_type="int")
            if isinstance(value, str):
                clean = re.sub(r"[\s,]", "", value)
                try:
                    return int(clean)
                except ValueError:
                    try:
                        f = float(clean)
                        if f.is_integer() or self.mode == "loose":
                            return int(f)
                    except ValueError:
                        pass
            raise TypeCoercionError(f"Cannot coerce {value!r} to integer", source_type=type(value).__name__, target_type=target_type)

        # 4. Floats / Numbers
        if target in ("number", "float"):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                clean = re.sub(r"[\s,]", "", value)
                try:
                    return float(clean)
                except ValueError:
                    pass
            raise TypeCoercionError(f"Cannot coerce {value!r} to float", source_type=type(value).__name__, target_type=target_type)

        # 5. Null
        if target in ("null", "none"):
            if isinstance(value, str) and value.strip().lower() in self._NULL_STRINGS:
                return None
            if self.mode == "loose":
                return None
            raise TypeCoercionError(f"Cannot coerce {value!r} to null", source_type=type(value).__name__, target_type=target_type)

        # 6. Dates and datetimes
        if target == "date":
            if isinstance(value, date):
                return value
            if isinstance(value, str):
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y"):
                    try:
                        return datetime.strptime(value.strip(), fmt).date()
                    except ValueError:
                        continue
            raise TypeCoercionError(f"Cannot coerce {value!r} to date", source_type=type(value).__name__, target_type=target_type)

        if target == "datetime":
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
                except ValueError:
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                        try:
                            return datetime.strptime(value.strip(), fmt)
                        except ValueError:
                            continue
            raise TypeCoercionError(f"Cannot coerce {value!r} to datetime", source_type=type(value).__name__, target_type=target_type)

        return value

    def coerce_dict(self, data: Dict[str, Any], schema_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce fields of a dictionary matching schema property definitions."""
        result: Dict[str, Any] = {}
        for key, val in data.items():
            prop = schema_properties.get(key)
            if prop and isinstance(prop, dict) and "type" in prop:
                try:
                    result[key] = self.coerce(val, prop["type"])
                except TypeCoercionError:
                    result[key] = val
            else:
                result[key] = val
        return result

