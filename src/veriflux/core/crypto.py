"""Cryptographic utilities for Veriflux.

All implementations use Python's standard library only (:mod:`hashlib`,
:mod:`hmac`, :mod:`secrets`).
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import secrets
from typing import Literal


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

_SUPPORTED_ALGORITHMS = frozenset(hashlib.algorithms_available)


def hash_content(data: bytes, algorithm: str = "sha256") -> str:
    """Compute a hex-digest hash of *data*.

    Parameters:
        data: Raw bytes to hash.
        algorithm: Hash algorithm name (anything supported by :mod:`hashlib`).

    Returns:
        Lowercase hex-encoded digest.

    Raises:
        ValueError: If *algorithm* is not supported.
        TypeError: If *data* is not ``bytes``.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
    if algorithm not in _SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported hash algorithm {algorithm!r}. "
            f"Available: {sorted(_SUPPORTED_ALGORITHMS)}"
        )
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()


def hash_text(text: str, algorithm: str = "sha256", encoding: str = "utf-8") -> str:
    """Convenience wrapper: hash a string after encoding to bytes.

    Parameters:
        text: String to hash.
        algorithm: Hash algorithm name.
        encoding: Text encoding (default UTF-8).

    Returns:
        Lowercase hex-encoded digest.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return hash_content(text.encode(encoding), algorithm=algorithm)


# ---------------------------------------------------------------------------
# HMAC
# ---------------------------------------------------------------------------

def hmac_sign(key: bytes, message: bytes, algorithm: str = "sha256") -> str:
    """Create an HMAC signature.

    Parameters:
        key: Secret key.
        message: Message to sign.
        algorithm: Hash algorithm for HMAC.

    Returns:
        Lowercase hex-encoded HMAC digest.
    """
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise TypeError(f"key must be bytes-like, got {type(key).__name__}")
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError(f"message must be bytes-like, got {type(message).__name__}")
    return _hmac.new(bytes(key), bytes(message), algorithm).hexdigest()


def hmac_verify(key: bytes, message: bytes, signature: str, algorithm: str = "sha256") -> bool:
    """Verify an HMAC signature in constant time.

    Parameters:
        key: Secret key.
        message: Original message.
        signature: Hex-encoded HMAC to verify.
        algorithm: Hash algorithm for HMAC.

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    expected = hmac_sign(key, message, algorithm=algorithm)
    return constant_time_compare(expected, signature)


# ---------------------------------------------------------------------------
# Random generation
# ---------------------------------------------------------------------------

def generate_trace_id() -> str:
    """Generate a 128-bit random hex trace ID (32 hex characters).

    Returns:
        A 32-character lowercase hexadecimal string.
    """
    return secrets.token_hex(16)


def generate_nonce(length: int = 32) -> str:
    """Generate a URL-safe random nonce.

    Parameters:
        length: Desired length of the nonce string.

    Returns:
        A URL-safe random string of approximately *length* characters.

    Raises:
        ValueError: If *length* is not positive.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    # token_urlsafe returns ~4/3 * nbytes characters
    # We compute nbytes to get at least *length* chars, then truncate.
    nbytes = max(1, (length * 3 + 3) // 4)
    token = secrets.token_urlsafe(nbytes)
    return token[:length]


# ---------------------------------------------------------------------------
# Constant-time comparison
# ---------------------------------------------------------------------------

def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks.

    Parameters:
        a: First string.
        b: Second string.

    Returns:
        ``True`` if the strings are equal, ``False`` otherwise.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("Both arguments must be strings")
    return _hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
