"""Fast and secure binary and JSON serialization for Veriflux.

Provides a custom type-tagged binary serialization protocol (safe against arbitrary
code execution vulnerabilities of pickle) and high-speed JSON wrappers.
"""
from __future__ import annotations

import io
import json
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

from veriflux.exceptions import SerializationError

# Type tags for custom binary wire format
TAG_NONE = 0x00
TAG_TRUE = 0x01
TAG_FALSE = 0x02
TAG_INT_POS = 0x03   # 8-byte uint64
TAG_INT_NEG = 0x04   # 8-byte uint64 (magnitude)
TAG_FLOAT = 0x05     # 8-byte IEEE 754 float
TAG_STR = 0x06       # 4-byte len + utf-8 bytes
TAG_BYTES = 0x07     # 4-byte len + raw bytes
TAG_LIST = 0x08      # 4-byte count + elements
TAG_TUPLE = 0x09     # 4-byte count + elements
TAG_DICT = 0x0A      # 4-byte count + (key, value) pairs


def _encode_val(obj: Any, buf: io.BytesIO) -> None:
    if obj is None:
        buf.write(bytes([TAG_NONE]))
    elif isinstance(obj, bool):
        buf.write(bytes([TAG_TRUE if obj else TAG_FALSE]))
    elif isinstance(obj, int):
        if obj >= 0:
            buf.write(bytes([TAG_INT_POS]))
            buf.write(struct.pack(">Q", obj))
        else:
            buf.write(bytes([TAG_INT_NEG]))
            buf.write(struct.pack(">Q", -obj))
    elif isinstance(obj, float):
        buf.write(bytes([TAG_FLOAT]))
        buf.write(struct.pack(">d", obj))
    elif isinstance(obj, str):
        data = obj.encode("utf-8")
        buf.write(bytes([TAG_STR]))
        buf.write(struct.pack(">I", len(data)))
        buf.write(data)
    elif isinstance(obj, (bytes, bytearray)):
        data = bytes(obj)
        buf.write(bytes([TAG_BYTES]))
        buf.write(struct.pack(">I", len(data)))
        buf.write(data)
    elif isinstance(obj, list):
        buf.write(bytes([TAG_LIST]))
        buf.write(struct.pack(">I", len(obj)))
        for item in obj:
            _encode_val(item, buf)
    elif isinstance(obj, tuple):
        buf.write(bytes([TAG_TUPLE]))
        buf.write(struct.pack(">I", len(obj)))
        for item in obj:
            _encode_val(item, buf)
    elif isinstance(obj, dict):
        buf.write(bytes([TAG_DICT]))
        buf.write(struct.pack(">I", len(obj)))
        for k, v in obj.items():
            _encode_val(k, buf)
            _encode_val(v, buf)
    else:
        raise SerializationError(f"Unsupported type for binary serialization: {type(obj).__name__}")


def _decode_val(stream: io.BytesIO) -> Any:
    tag_byte = stream.read(1)
    if not tag_byte:
        raise SerializationError("Unexpected end of binary stream")
    tag = tag_byte[0]

    if tag == TAG_NONE:
        return None
    elif tag == TAG_TRUE:
        return True
    elif tag == TAG_FALSE:
        return False
    elif tag == TAG_INT_POS:
        raw = stream.read(8)
        if len(raw) < 8:
            raise SerializationError("Truncated integer in binary stream")
        return struct.unpack(">Q", raw)[0]
    elif tag == TAG_INT_NEG:
        raw = stream.read(8)
        if len(raw) < 8:
            raise SerializationError("Truncated negative integer in binary stream")
        return -struct.unpack(">Q", raw)[0]
    elif tag == TAG_FLOAT:
        raw = stream.read(8)
        if len(raw) < 8:
            raise SerializationError("Truncated float in binary stream")
        return struct.unpack(">d", raw)[0]
    elif tag == TAG_STR:
        raw_len = stream.read(4)
        if len(raw_len) < 4:
            raise SerializationError("Truncated string length in binary stream")
        length = struct.unpack(">I", raw_len)[0]
        data = stream.read(length)
        if len(data) < length:
            raise SerializationError("Truncated string data in binary stream")
        return data.decode("utf-8")
    elif tag == TAG_BYTES:
        raw_len = stream.read(4)
        if len(raw_len) < 4:
            raise SerializationError("Truncated bytes length in binary stream")
        length = struct.unpack(">I", raw_len)[0]
        data = stream.read(length)
        if len(data) < length:
            raise SerializationError("Truncated bytes data in binary stream")
        return data
    elif tag == TAG_LIST:
        raw_len = stream.read(4)
        if len(raw_len) < 4:
            raise SerializationError("Truncated list length in binary stream")
        count = struct.unpack(">I", raw_len)[0]
        return [_decode_val(stream) for _ in range(count)]
    elif tag == TAG_TUPLE:
        raw_len = stream.read(4)
        if len(raw_len) < 4:
            raise SerializationError("Truncated tuple length in binary stream")
        count = struct.unpack(">I", raw_len)[0]
        return tuple(_decode_val(stream) for _ in range(count))
    elif tag == TAG_DICT:
        raw_len = stream.read(4)
        if len(raw_len) < 4:
            raise SerializationError("Truncated dict length in binary stream")
        count = struct.unpack(">I", raw_len)[0]
        d: Dict[Any, Any] = {}
        for _ in range(count):
            k = _decode_val(stream)
            v = _decode_val(stream)
            d[k] = v
        return d
    else:
        raise SerializationError(f"Unknown binary tag: 0x{tag:02x}")


def serialize(obj: Any) -> bytes:
    """Serialize a Python object graph to compact, secure binary format."""
    try:
        buf = io.BytesIO()
        _encode_val(obj, buf)
        return buf.getvalue()
    except Exception as exc:
        if isinstance(exc, SerializationError):
            raise
        raise SerializationError(f"Serialization failed: {exc}") from exc


def deserialize(data: bytes) -> Any:
    """Deserialize binary data into Python objects."""
    try:
        stream = io.BytesIO(data)
        res = _decode_val(stream)
        return res
    except Exception as exc:
        if isinstance(exc, SerializationError):
            raise
        raise SerializationError(f"Deserialization failed: {exc}") from exc


def to_msgpack_like(obj: Any) -> bytes:
    """Alias for serialize, providing a compact message-pack-like format."""
    return serialize(obj)


def to_json(obj: Any, *, indent: Optional[int] = None, sort_keys: bool = False) -> str:
    """Fast JSON serialization with standardized defaults."""
    try:
        return json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    except Exception as exc:
        raise SerializationError(f"JSON serialization error: {exc}") from exc


def from_json(data: str) -> Any:
    """Fast JSON deserialization."""
    try:
        return json.loads(data)
    except Exception as exc:
        raise SerializationError(f"JSON deserialization error: {exc}") from exc

