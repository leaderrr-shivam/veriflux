"""Structured logging for Veriflux.

Provides JSON-structured and human-readable logging with trace-correlation
IDs, a custom ``TRACE`` level, and lazy message formatting.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
import time
import uuid
from typing import Any, Dict, Optional, Union


# ---------------------------------------------------------------------------
# Custom TRACE level (below DEBUG)
# ---------------------------------------------------------------------------

TRACE: int = 5
logging.addLevelName(TRACE, "TRACE")


# ---------------------------------------------------------------------------
# Context variables for trace correlation
# ---------------------------------------------------------------------------

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "veriflux_trace_id", default=""
)
_span_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "veriflux_span_id", default=""
)


def get_trace_id() -> str:
    """Return the current trace ID (empty string if unset)."""
    return _trace_id_var.get()


def set_trace_id(trace_id: str) -> contextvars.Token[str]:
    """Set the trace ID for the current context, returning a reset token."""
    return _trace_id_var.set(trace_id)


def get_span_id() -> str:
    """Return the current span ID (empty string if unset)."""
    return _span_id_var.get()


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class _HumanFormatter(logging.Formatter):
    """Human-readable log formatter with colour hints for terminals."""

    _COLORS = {
        "TRACE": "\033[90m",     # grey
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self._use_color = use_color and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        ms = int(record.msecs)
        level = record.levelname
        name = record.name

        trace_id = _trace_id_var.get("")
        trace_part = f" [{trace_id[:12]}]" if trace_id else ""

        msg = record.getMessage()

        if self._use_color:
            color = self._COLORS.get(level, "")
            return f"{ts}.{ms:03d} {color}{level:>8s}{self._RESET} {name}{trace_part} — {msg}"
        return f"{ts}.{ms:03d} {level:>8s} {name}{trace_part} — {msg}"


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter — one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: Dict[str, Any] = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        trace_id = _trace_id_var.get("")
        if trace_id:
            payload["trace_id"] = trace_id

        span_id = _span_id_var.get("")
        if span_id:
            payload["span_id"] = span_id

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        # Merge extra structured data
        extra = getattr(record, "_veriflux_extra", None)
        if extra and isinstance(extra, dict):
            payload["data"] = extra

        return json.dumps(payload, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# TraceContext (context manager for scoped trace IDs)
# ---------------------------------------------------------------------------

class TraceContext:
    """Context manager that sets a trace ID for the duration of a block.

    Example::

        with TraceContext("abc-123"):
            logger.info("Inside trace")
        # trace_id is restored here
    """

    def __init__(self, trace_id: Optional[str] = None, span_id: Optional[str] = None) -> None:
        self._trace_id = trace_id or uuid.uuid4().hex
        self._span_id = span_id or ""
        self._trace_token: Optional[contextvars.Token[str]] = None
        self._span_token: Optional[contextvars.Token[str]] = None

    def __enter__(self) -> TraceContext:
        self._trace_token = _trace_id_var.set(self._trace_id)
        if self._span_id:
            self._span_token = _span_id_var.set(self._span_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._trace_token is not None:
            _trace_id_var.reset(self._trace_token)
        if self._span_token is not None:
            _span_id_var.reset(self._span_token)

    @property
    def trace_id(self) -> str:
        """The trace ID set by this context."""
        return self._trace_id

    @property
    def span_id(self) -> str:
        """The span ID set by this context."""
        return self._span_id


# ---------------------------------------------------------------------------
# VerifluxLogger
# ---------------------------------------------------------------------------

class VerifluxLogger:
    """Structured logger wrapping :class:`logging.Logger`.

    Features:

    * Lazy message formatting — expensive ``f-strings`` are never evaluated if
      the level is disabled.
    * ``TRACE`` level for ultra-verbose diagnostics.
    * Structured ``extra`` dict attached to each record for JSON output.
    * Automatic trace-ID propagation.

    Parameters:
        name: Logger name (typically ``__name__``).
        level: Initial log level.
        fmt: ``"text"`` or ``"json"``.
    """

    def __init__(
        self,
        name: str = "veriflux",
        level: Union[int, str] = logging.INFO,
        fmt: str = "text",
    ) -> None:
        self._logger = logging.getLogger(name)
        self._set_level(level)

        # Only add a handler if the logger has none (avoid duplicate handlers)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            if fmt == "json":
                handler.setFormatter(_JSONFormatter())
            else:
                handler.setFormatter(_HumanFormatter())
            self._logger.addHandler(handler)
            self._logger.propagate = False

    # -- level helpers ------------------------------------------------------

    def _set_level(self, level: Union[int, str]) -> None:
        if isinstance(level, str):
            level = level.upper()
            if level == "TRACE":
                self._logger.setLevel(TRACE)
            else:
                self._logger.setLevel(getattr(logging, level, logging.INFO))
        else:
            self._logger.setLevel(level)

    def set_level(self, level: Union[int, str]) -> None:
        """Change the effective log level at runtime."""
        self._set_level(level)

    # -- core logging methods -----------------------------------------------

    def _log(self, level: int, msg: str, *args: Any, extra: Optional[Dict[str, Any]] = None, exc_info: Any = None) -> None:
        """Internal log dispatch with lazy formatting."""
        if not self._logger.isEnabledFor(level):
            return
        # Build the record manually to inject structured extra
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=msg,
            args=args,
            exc_info=exc_info,
        )
        if extra:
            record._veriflux_extra = extra  # type: ignore[attr-defined]
        self._logger.handle(record)

    def trace(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ``TRACE`` level (5)."""
        self._log(TRACE, msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ``DEBUG`` level."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ``INFO`` level."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ``WARNING`` level."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ``ERROR`` level."""
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ``CRITICAL`` level."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ``ERROR`` with exception info."""
        kwargs["exc_info"] = kwargs.get("exc_info", sys.exc_info())
        self._log(logging.ERROR, msg, *args, **kwargs)

    # -- convenience --------------------------------------------------------

    def is_enabled_for(self, level: Union[int, str]) -> bool:
        """Check whether a given level is enabled."""
        if isinstance(level, str):
            level_num = TRACE if level.upper() == "TRACE" else getattr(logging, level.upper(), 0)
        else:
            level_num = level
        return self._logger.isEnabledFor(level_num)

    @property
    def name(self) -> str:
        """Logger name."""
        return self._logger.name

    @property
    def effective_level(self) -> int:
        """Effective numeric level."""
        return self._logger.getEffectiveLevel()


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

_loggers: Dict[str, VerifluxLogger] = {}
_loggers_lock = threading.Lock()


def get_logger(
    name: str = "veriflux",
    level: Union[int, str, None] = None,
    fmt: Optional[str] = None,
) -> VerifluxLogger:
    """Return a :class:`VerifluxLogger` for *name*, creating one if needed.

    Subsequent calls with the same *name* return the same logger instance.
    """
    with _loggers_lock:
        if name in _loggers:
            logger = _loggers[name]
            if level is not None:
                logger.set_level(level)
            return logger

        # Resolve defaults from config if available
        effective_level: Union[int, str] = level or logging.INFO
        effective_fmt: str = fmt or "text"

        try:
            from veriflux.core.config import get_config as _gc
            cfg = _gc()
            if level is None:
                effective_level = cfg.log_level
            if fmt is None:
                effective_fmt = cfg.log_format
        except Exception:
            pass

        logger = VerifluxLogger(name=name, level=effective_level, fmt=effective_fmt)
        _loggers[name] = logger
        return logger
