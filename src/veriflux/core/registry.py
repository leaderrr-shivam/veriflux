"""Component registry and plugin discovery for Veriflux.

Provides a thread-safe, namespaced registry supporting decorator registration,
entry point discovery, and dynamic lookup.
"""
from __future__ import annotations

import importlib.metadata
import threading
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Thread-safe registry for managing named components and extensions."""

    def __init__(self, name: str = "default", entry_point_group: Optional[str] = None) -> None:
        self.name = name
        self.entry_point_group = entry_point_group
        self._entries: Dict[str, Dict[str, T]] = {}
        self._lock = threading.RLock()
        self._loaded_entry_points = False

    def register(
        self,
        name: str,
        target: Optional[T] = None,
        *,
        namespace: str = "default",
        override: bool = False,
    ) -> Any:
        """Register a component under a given name and namespace.

        Can be used as a decorator or direct method call:
            @registry.register("my_plugin")
            def my_func(): ...
        """
        def decorator(obj: T) -> T:
            with self._lock:
                ns_dict = self._entries.setdefault(namespace, {})
                if name in ns_dict and not override:
                    raise KeyError(f"Component '{name}' already registered in namespace '{namespace}' of registry '{self.name}'")
                ns_dict[name] = obj
            return obj

        if target is not None:
            return decorator(target)
        return decorator

    def get(self, name: str, *, namespace: str = "default", default: Optional[T] = None) -> Optional[T]:
        """Retrieve a registered component by name and namespace."""
        with self._lock:
            self._ensure_entry_points()
            ns_dict = self._entries.get(namespace, {})
            return ns_dict.get(name, default)

    def get_or_raise(self, name: str, *, namespace: str = "default") -> T:
        """Retrieve a registered component or raise KeyError."""
        item = self.get(name, namespace=namespace)
        if item is None:
            raise KeyError(f"No component named '{name}' found in namespace '{namespace}' of registry '{self.name}'")
        return item

    def list_names(self, *, namespace: str = "default") -> List[str]:
        """List all component names in a given namespace."""
        with self._lock:
            self._ensure_entry_points()
            return sorted(self._entries.get(namespace, {}).keys())

    def list_namespaces(self) -> List[str]:
        """List all registered namespaces."""
        with self._lock:
            self._ensure_entry_points()
            return sorted(self._entries.keys())

    def unregister(self, name: str, *, namespace: str = "default") -> bool:
        """Unregister a component. Returns True if removed, False if not found."""
        with self._lock:
            ns_dict = self._entries.get(namespace, {})
            if name in ns_dict:
                del ns_dict[name]
                return True
            return False

    def clear(self) -> None:
        """Clear all registered entries."""
        with self._lock:
            self._entries.clear()
            self._loaded_entry_points = False

    def _ensure_entry_points(self) -> None:
        if self._loaded_entry_points or not self.entry_point_group:
            return
        self._loaded_entry_points = True
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                group_eps = eps.select(group=self.entry_point_group)
            else:
                group_eps = eps.get(self.entry_point_group, [])  # type: ignore[assignment]
            for ep in group_eps:
                loaded = ep.load()
                self.register(ep.name, loaded, namespace="entry_points", override=True)
        except Exception:
            pass

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self._ensure_entry_points()
            yield from self.list_names()

    def __contains__(self, name: str) -> bool:
        with self._lock:
            self._ensure_entry_points()
            return any(name in ns for ns in self._entries.values())

