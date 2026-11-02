"""Extension hook lifecycle registry.

Enables community plugins to intercept evaluation events, guardrail triggers, and regressions.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
import inspect
import threading
from typing import Any, Callable, Dict, List


class HookRegistry:
    """Thread-safe event hook registry for lifecycle extensions."""

    PREDEFINED_HOOKS = {
        "before_eval",
        "after_eval",
        "before_guard",
        "after_guard",
        "on_hallucination",
        "on_regression",
        "before_report",
        "after_report",
    }

    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable[..., Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def register(self, hook_name: str, callback: Callable[..., Any]) -> None:
        """Register a callback for an event hook."""
        with self._lock:
            self._hooks[hook_name].append(callback)

    def trigger(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Trigger all registered callbacks for an event synchronously."""
        with self._lock:
            callbacks = list(self._hooks.get(hook_name, []))

        results = []
        for cb in callbacks:
            try:
                res = cb(**kwargs)
                results.append(res)
            except Exception as exc:
                results.append(exc)
        return results

    async def trigger_async(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Trigger callbacks asynchronously supporting coroutines."""
        with self._lock:
            callbacks = list(self._hooks.get(hook_name, []))

        results = []
        for cb in callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    res = await cb(**kwargs)
                else:
                    res = cb(**kwargs)
                results.append(res)
            except Exception as exc:
                results.append(exc)
        return results

