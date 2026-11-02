"""Hierarchical configuration system for Veriflux.

Configuration is resolved in priority order:

1. Code overrides (``configure(key=value)``)
2. Environment variables (``VERIFLUX_*``)
3. Config file (``veriflux.toml`` or ``.veriflux.toml``)
4. Built-in defaults

Thread-safe singleton via a module-level lock.
"""
from __future__ import annotations

import copy
import os
import pathlib
import threading
from typing import Any, Dict, List, Optional, Union

from veriflux._types import ProviderConfig, TestLayer, VerifluxConfig
from veriflux.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# TOML reader — use stdlib tomllib (3.11+) or fall back gracefully
# ---------------------------------------------------------------------------

def _load_toml(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Read a TOML file and return a dict.

    Uses :mod:`tomllib` on Python ≥ 3.11; otherwise tries the third-party
    ``tomli`` package.  If neither is available, raises
    :class:`ConfigurationError`.
    """
    filepath = pathlib.Path(path)
    if not filepath.exists():
        raise ConfigurationError(f"Config file not found: {filepath}", key=str(filepath))

    data = filepath.read_bytes()

    # Try stdlib first (Python 3.11+)
    try:
        import tomllib  # type: ignore[import-not-found]
        return tomllib.loads(data.decode("utf-8"))
    except ImportError:
        pass

    # Fallback to tomli
    try:
        import tomli  # type: ignore[import-not-found]
        return tomli.loads(data.decode("utf-8"))
    except ImportError:
        pass

    # Last resort: minimal inline TOML parser for the subset we need
    raise ConfigurationError(
        "No TOML parser available. Install 'tomli' on Python < 3.11, "
        "or upgrade to Python 3.11+.",
        key=str(filepath),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a deep copy of *base*.

    For conflicting keys:
    * If both values are dicts, merge recursively.
    * Otherwise, the *override* value wins.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _get_nested(d: Dict[str, Any], dotted_key: str) -> Any:
    """Retrieve a value from a nested dict using a dotted key like ``"a.b.c"``."""
    parts = dotted_key.split(".")
    current: Any = d
    for part in parts:
        if not isinstance(current, dict):
            raise ConfigurationError(
                f"Cannot traverse into non-dict at key segment {part!r}",
                key=dotted_key,
            )
        if part not in current:
            raise ConfigurationError(f"Key {dotted_key!r} not found", key=dotted_key)
        current = current[part]
    return current


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key."""
    parts = dotted_key.split(".")
    current = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _env_to_dict(prefix: str = "VERIFLUX_") -> Dict[str, Any]:
    """Collect environment variables with *prefix* into a nested dict.

    ``VERIFLUX_PROVIDER__API_KEY`` → ``{"provider": {"api_key": "..."}}``.
    Double underscore ``__`` is the nesting separator; single underscores
    remain as-is (lowercased).
    """
    result: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        stripped = key[len(prefix):]
        parts = [p.lower() for p in stripped.split("__")]
        # Auto-cast booleans and numbers
        coerced = _auto_cast(value)
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = coerced
    return result


def _auto_cast(value: str) -> Any:
    """Attempt to cast a string to bool, int, or float."""
    lower = value.lower()
    if lower in ("true", "yes", "1", "on"):
        return True
    if lower in ("false", "no", "0", "off"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "provider": {
        "api_key": "",
        "base_url": "",
        "model": "",
        "timeout": 30.0,
        "max_retries": 3,
    },
    "default_thresholds": {},
    "cache_enabled": True,
    "cache_max_size": 4096,
    "cache_ttl": 3600.0,
    "cache_directory": ".veriflux_cache",
    "log_level": "INFO",
    "log_format": "text",
    "test_layer": "standard",
    "max_workers": 4,
    "token_budget": 0,
}


# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------

_SEARCH_NAMES: List[str] = ["veriflux.toml", ".veriflux.toml"]


def _discover_config_file(search_dir: Optional[Union[str, pathlib.Path]] = None) -> Optional[pathlib.Path]:
    """Search for a config file in *search_dir* (default: cwd) and up the tree."""
    start = pathlib.Path(search_dir) if search_dir else pathlib.Path.cwd()
    current = start.resolve()
    while True:
        for name in _SEARCH_NAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_LOG_FORMATS = {"text", "json"}
_VALID_TEST_LAYERS = {t.value for t in TestLayer}


def _validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Validate configuration values.  Returns a list of warning messages."""
    warnings: List[str] = []

    log_level = cfg.get("log_level", "INFO")
    if isinstance(log_level, str) and log_level.upper() not in _VALID_LOG_LEVELS:
        warnings.append(f"Invalid log_level {log_level!r}; must be one of {_VALID_LOG_LEVELS}")

    log_format = cfg.get("log_format", "text")
    if isinstance(log_format, str) and log_format.lower() not in _VALID_LOG_FORMATS:
        warnings.append(f"Invalid log_format {log_format!r}; must be one of {_VALID_LOG_FORMATS}")

    test_layer = cfg.get("test_layer", "standard")
    if isinstance(test_layer, str) and test_layer.lower() not in _VALID_TEST_LAYERS:
        warnings.append(f"Invalid test_layer {test_layer!r}; must be one of {_VALID_TEST_LAYERS}")

    cache_max_size = cfg.get("cache_max_size")
    if cache_max_size is not None and isinstance(cache_max_size, (int, float)) and cache_max_size < 0:
        warnings.append(f"cache_max_size must be non-negative, got {cache_max_size}")

    cache_ttl = cfg.get("cache_ttl")
    if cache_ttl is not None and isinstance(cache_ttl, (int, float)) and cache_ttl < 0:
        warnings.append(f"cache_ttl must be non-negative, got {cache_ttl}")

    max_workers = cfg.get("max_workers")
    if max_workers is not None and isinstance(max_workers, int) and max_workers < 1:
        warnings.append(f"max_workers must be >= 1, got {max_workers}")

    provider = cfg.get("provider", {})
    if isinstance(provider, dict):
        timeout = provider.get("timeout")
        if timeout is not None and isinstance(timeout, (int, float)) and timeout <= 0:
            warnings.append(f"provider.timeout must be positive, got {timeout}")
        max_retries = provider.get("max_retries")
        if max_retries is not None and isinstance(max_retries, int) and max_retries < 0:
            warnings.append(f"provider.max_retries must be non-negative, got {max_retries}")

    return warnings


def _dict_to_config(d: Dict[str, Any]) -> VerifluxConfig:
    """Convert a flat/nested dict into a :class:`VerifluxConfig`."""
    provider_data = d.get("provider", {})
    if isinstance(provider_data, dict):
        provider = ProviderConfig(
            api_key=str(provider_data.get("api_key", "")),
            base_url=str(provider_data.get("base_url", "")),
            model=str(provider_data.get("model", "")),
            timeout=float(provider_data.get("timeout", 30.0)),
            max_retries=int(provider_data.get("max_retries", 3)),
            extra={k: v for k, v in provider_data.items()
                   if k not in ("api_key", "base_url", "model", "timeout", "max_retries")},
        )
    else:
        provider = ProviderConfig()

    test_layer_str = str(d.get("test_layer", "standard")).lower()
    try:
        test_layer = TestLayer(test_layer_str)
    except ValueError:
        test_layer = TestLayer.STANDARD

    # Collect "extra" keys not in the known set
    known_keys = {
        "provider", "default_thresholds", "cache_enabled", "cache_max_size",
        "cache_ttl", "cache_directory", "log_level", "log_format",
        "test_layer", "max_workers", "token_budget",
    }
    extra = {k: v for k, v in d.items() if k not in known_keys}

    return VerifluxConfig(
        provider=provider,
        default_thresholds=dict(d.get("default_thresholds", {})),
        cache_enabled=bool(d.get("cache_enabled", True)),
        cache_max_size=int(d.get("cache_max_size", 4096)),
        cache_ttl=float(d.get("cache_ttl", 3600.0)),
        cache_directory=str(d.get("cache_directory", ".veriflux_cache")),
        log_level=str(d.get("log_level", "INFO")).upper(),
        log_format=str(d.get("log_format", "text")).lower(),
        test_layer=test_layer,
        max_workers=int(d.get("max_workers", 4)),
        token_budget=int(d.get("token_budget", 0)),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# VerifluxConfigManager
# ---------------------------------------------------------------------------

class VerifluxConfigManager:
    """Thread-safe hierarchical configuration manager.

    Resolution order (highest priority first):

    1. Code overrides via :meth:`set`
    2. Environment variables (``VERIFLUX_*``)
    3. Config file (``veriflux.toml`` / ``.veriflux.toml``)
    4. Built-in defaults

    This class is a singleton; use :func:`get_config` and :func:`configure`
    for convenient access.
    """

    _instance: Optional[VerifluxConfigManager] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> VerifluxConfigManager:
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False  # type: ignore[attr-defined]
                cls._instance = inst
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[has-type]
            return
        self._lock = threading.Lock()
        self._overrides: Dict[str, Any] = {}
        self._file_config: Dict[str, Any] = {}
        self._env_config: Dict[str, Any] = {}
        self._merged: Dict[str, Any] = {}
        self._config: Optional[VerifluxConfig] = None
        self._config_file: Optional[pathlib.Path] = None
        self._validation_warnings: List[str] = []
        self._initialized = True
        self._reload()

    # -- Public API ---------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Set a code-level override for *key* (supports dotted notation)."""
        with self._lock:
            _set_nested(self._overrides, key, value)
            self._rebuild()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dotted key, with optional *default*."""
        with self._lock:
            try:
                return _get_nested(self._merged, key)
            except ConfigurationError:
                return default

    def get_config(self) -> VerifluxConfig:
        """Return the fully resolved :class:`VerifluxConfig` object."""
        with self._lock:
            if self._config is None:
                self._rebuild()
            assert self._config is not None
            return self._config

    def load_file(self, path: Union[str, pathlib.Path]) -> None:
        """Explicitly load a config file.

        Raises :class:`ConfigurationError` if the file cannot be read.
        """
        with self._lock:
            self._config_file = pathlib.Path(path)
            self._file_config = _load_toml(self._config_file)
            self._rebuild()

    def reload(self) -> None:
        """Re-read environment variables and config file."""
        with self._lock:
            self._reload()

    def reset(self) -> None:
        """Clear all overrides and cached state, reverting to defaults."""
        with self._lock:
            self._overrides.clear()
            self._file_config.clear()
            self._env_config.clear()
            self._config_file = None
            self._reload()

    @property
    def validation_warnings(self) -> List[str]:
        """Warnings from the most recent config validation pass."""
        with self._lock:
            return list(self._validation_warnings)

    @property
    def config_file(self) -> Optional[pathlib.Path]:
        """Path to the config file in use, or ``None``."""
        with self._lock:
            return self._config_file

    def as_dict(self) -> Dict[str, Any]:
        """Return the merged configuration as a plain dict."""
        with self._lock:
            return copy.deepcopy(self._merged)

    # -- Internals ----------------------------------------------------------

    def _reload(self) -> None:
        """Reload env vars, discover config file if needed, rebuild."""
        self._env_config = _env_to_dict()

        if self._config_file is None:
            discovered = _discover_config_file()
            if discovered is not None:
                self._config_file = discovered
                try:
                    self._file_config = _load_toml(discovered)
                except ConfigurationError:
                    self._file_config = {}

        self._rebuild()

    def _rebuild(self) -> None:
        """Merge layers and produce a new VerifluxConfig."""
        merged = copy.deepcopy(_DEFAULTS)
        merged = _deep_merge(merged, self._file_config)
        merged = _deep_merge(merged, self._env_config)
        merged = _deep_merge(merged, self._overrides)
        self._merged = merged
        self._validation_warnings = _validate_config(merged)
        self._config = _dict_to_config(merged)

    @classmethod
    def _reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing only)."""
        with cls._instance_lock:
            cls._instance = None


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def configure(**kwargs: Any) -> VerifluxConfig:
    """Set one or more configuration overrides and return the updated config.

    Example::

        configure(log_level="DEBUG", cache_enabled=False)
    """
    mgr = VerifluxConfigManager()
    for key, value in kwargs.items():
        mgr.set(key, value)
    return mgr.get_config()


def get_config() -> VerifluxConfig:
    """Return the current :class:`VerifluxConfig`."""
    return VerifluxConfigManager().get_config()
