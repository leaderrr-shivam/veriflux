"""Multi-tier caching subsystem for Veriflux.

Provides LRU, LFU, content-addressable, disk-backed, and tiered caches,
all conforming to a common ``Cache`` protocol.
"""
from __future__ import annotations

import collections
import hashlib
import os
import pathlib
import shelve
import threading
import time
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from veriflux.exceptions import CacheError

K = TypeVar("K")
V = TypeVar("V")


# ---------------------------------------------------------------------------
# Cache Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Cache(Protocol[K, V]):
    """Minimal cache interface implemented by all Veriflux caches."""

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]: ...
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None: ...
    def delete(self, key: K) -> bool: ...
    def clear(self) -> None: ...
    def stats(self) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

class LRUCache(Generic[K, V]):
    """Thread-safe Least-Recently-Used cache backed by :class:`OrderedDict`.

    Parameters:
        maxsize: Maximum number of entries.
        default_ttl: Default time-to-live in seconds (0 = no expiry).
    """

    def __init__(self, maxsize: int = 256, default_ttl: float = 0.0) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        # value, expiry (0 = never)
        self._data: collections.OrderedDict[K, Tuple[V, float]] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Retrieve *key*, returning *default* if missing or expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return default
            value, expiry = entry
            if expiry and time.monotonic() > expiry:
                del self._data[key]
                self._misses += 1
                return default
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Insert or update *key*.  Evicts the LRU entry if at capacity."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = (time.monotonic() + effective_ttl) if effective_ttl > 0 else 0.0
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, expiry)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)
                self._evictions += 1

    def delete(self, key: K) -> bool:
        """Remove *key*.  Returns ``True`` if found."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            entry = self._data.get(key)  # type: ignore[arg-type]
            if entry is None:
                return False
            _, expiry = entry
            if expiry and time.monotonic() > expiry:
                del self._data[key]  # type: ignore[arg-type]
                return False
            return True

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "type": "LRUCache",
                "size": len(self._data),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total else 0.0,
                "evictions": self._evictions,
            }


# ---------------------------------------------------------------------------
# LFUCache — O(1) eviction using frequency buckets
# ---------------------------------------------------------------------------

class _LFUNode:
    """Doubly-linked list node for the LFU cache."""
    __slots__ = ("key", "value", "freq", "expiry", "prev", "next")

    def __init__(self, key: Any, value: Any, freq: int = 1, expiry: float = 0.0) -> None:
        self.key = key
        self.value = value
        self.freq = freq
        self.expiry = expiry
        self.prev: Optional[_LFUNode] = None
        self.next: Optional[_LFUNode] = None


class _FreqList:
    """Doubly-linked list for a single frequency bucket."""
    __slots__ = ("head", "tail", "size")

    def __init__(self) -> None:
        self.head = _LFUNode(None, None, 0)
        self.tail = _LFUNode(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def append(self, node: _LFUNode) -> None:
        """Append *node* at the tail (most recent in this freq)."""
        prev = self.tail.prev
        assert prev is not None
        prev.next = node
        node.prev = prev
        node.next = self.tail
        self.tail.prev = node
        self.size += 1

    def pop_front(self) -> Optional[_LFUNode]:
        """Remove and return the least-recently-used node in this bucket."""
        if self.size == 0:
            return None
        node = self.head.next
        assert node is not None and node is not self.tail
        self._remove(node)
        return node

    def remove(self, node: _LFUNode) -> None:
        self._remove(node)

    def _remove(self, node: _LFUNode) -> None:
        p = node.prev
        n = node.next
        assert p is not None and n is not None
        p.next = n
        n.prev = p
        node.prev = None
        node.next = None
        self.size -= 1

    def is_empty(self) -> bool:
        return self.size == 0


class LFUCache(Generic[K, V]):
    """Thread-safe Least-Frequently-Used cache with O(1) eviction.

    Parameters:
        maxsize: Maximum number of entries.
        default_ttl: Default TTL in seconds (0 = no expiry).
    """

    def __init__(self, maxsize: int = 256, default_ttl: float = 0.0) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._nodes: Dict[K, _LFUNode] = {}
        self._freq_map: Dict[int, _FreqList] = {}
        self._min_freq: int = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _touch(self, node: _LFUNode) -> None:
        """Increment the frequency of *node* and move it."""
        old_freq = node.freq
        freq_list = self._freq_map.get(old_freq)
        if freq_list is not None:
            freq_list.remove(node)
            if freq_list.is_empty():
                del self._freq_map[old_freq]
                if self._min_freq == old_freq:
                    self._min_freq = old_freq + 1

        node.freq += 1
        new_list = self._freq_map.setdefault(node.freq, _FreqList())
        new_list.append(node)

    def _evict(self) -> None:
        """Evict the least-frequently (then least-recently) used entry."""
        freq_list = self._freq_map.get(self._min_freq)
        if freq_list is None:
            return
        victim = freq_list.pop_front()
        if victim is None:
            return
        if freq_list.is_empty():
            del self._freq_map[self._min_freq]
        del self._nodes[victim.key]
        self._evictions += 1

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            node = self._nodes.get(key)
            if node is None:
                self._misses += 1
                return default
            if node.expiry and time.monotonic() > node.expiry:
                self._remove_node(node)
                self._misses += 1
                return default
            self._touch(node)
            self._hits += 1
            return node.value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = (time.monotonic() + effective_ttl) if effective_ttl > 0 else 0.0
        with self._lock:
            node = self._nodes.get(key)
            if node is not None:
                node.value = value
                node.expiry = expiry
                self._touch(node)
                return

            if len(self._nodes) >= self._maxsize:
                self._evict()

            new_node = _LFUNode(key, value, freq=1, expiry=expiry)
            self._nodes[key] = new_node
            freq_list = self._freq_map.setdefault(1, _FreqList())
            freq_list.append(new_node)
            self._min_freq = 1

    def delete(self, key: K) -> bool:
        with self._lock:
            node = self._nodes.get(key)
            if node is None:
                return False
            self._remove_node(node)
            return True

    def _remove_node(self, node: _LFUNode) -> None:
        freq_list = self._freq_map.get(node.freq)
        if freq_list is not None:
            freq_list.remove(node)
            if freq_list.is_empty():
                del self._freq_map[node.freq]
                if self._min_freq == node.freq:
                    self._min_freq = node.freq + 1
        del self._nodes[node.key]

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._freq_map.clear()
            self._min_freq = 0

    def __len__(self) -> int:
        return len(self._nodes)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "type": "LFUCache",
                "size": len(self._nodes),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total else 0.0,
                "evictions": self._evictions,
                "min_freq": self._min_freq,
            }


# ---------------------------------------------------------------------------
# ContentAddressableCache
# ---------------------------------------------------------------------------

class ContentAddressableCache(Generic[V]):
    """Cache keyed by the SHA-256 hash of content.

    Provides automatic deduplication — identical content always maps to the
    same key regardless of how it is referenced.

    Parameters:
        maxsize: Maximum entries.
        default_ttl: TTL in seconds (0 = no expiry).
    """

    def __init__(self, maxsize: int = 1024, default_ttl: float = 0.0) -> None:
        self._inner: LRUCache[str, V] = LRUCache(maxsize=maxsize, default_ttl=default_ttl)

    @staticmethod
    def _content_key(data: bytes) -> str:
        """Compute a SHA-256 hex digest for *data*."""
        return hashlib.sha256(data).hexdigest()

    def get_by_hash(self, content_hash: str, default: Optional[V] = None) -> Optional[V]:
        """Retrieve a value by its pre-computed hash."""
        return self._inner.get(content_hash, default)

    def get(self, data: bytes, default: Optional[V] = None) -> Optional[V]:
        """Retrieve the value associated with the hash of *data*."""
        key = self._content_key(data)
        return self._inner.get(key, default)

    def set(self, data: bytes, value: V, ttl: Optional[float] = None) -> str:
        """Store *value* keyed by the SHA-256 of *data*.

        Returns the content hash.
        """
        key = self._content_key(data)
        self._inner.set(key, value, ttl=ttl)
        return key

    def delete(self, data: bytes) -> bool:
        key = self._content_key(data)
        return self._inner.delete(key)

    def delete_by_hash(self, content_hash: str) -> bool:
        return self._inner.delete(content_hash)

    def clear(self) -> None:
        self._inner.clear()

    def __len__(self) -> int:
        return len(self._inner)

    def stats(self) -> Dict[str, Any]:
        s = self._inner.stats()
        s["type"] = "ContentAddressableCache"
        return s


# ---------------------------------------------------------------------------
# DiskCache
# ---------------------------------------------------------------------------

class DiskCache(Generic[K, V]):
    """Persistent cache using :mod:`shelve` with TTL support.

    Parameters:
        directory: Directory in which to store the shelve database.
        db_name: Base filename for the database.
        default_ttl: Default TTL in seconds (0 = no expiry).
    """

    def __init__(
        self,
        directory: str = ".veriflux_cache",
        db_name: str = "veriflux_disk_cache",
        default_ttl: float = 0.0,
    ) -> None:
        self._dir = pathlib.Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(self._dir / db_name)
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _open(self) -> shelve.Shelf[Any]:
        """Open the shelve database."""
        try:
            return shelve.open(self._db_path, flag="c", protocol=4)
        except Exception as exc:
            raise CacheError(f"Failed to open disk cache: {exc}", cache_name="disk") from exc

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        str_key = str(key)
        with self._lock:
            try:
                db = self._open()
                try:
                    entry = db.get(str_key)
                    if entry is None:
                        self._misses += 1
                        return default
                    value, expiry = entry
                    if expiry and time.monotonic() > expiry:
                        try:
                            del db[str_key]
                        except KeyError:
                            pass
                        self._misses += 1
                        return default
                    self._hits += 1
                    return value
                finally:
                    db.close()
            except CacheError:
                raise
            except Exception as exc:
                raise CacheError(f"Disk cache get failed: {exc}", cache_name="disk") from exc

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = (time.monotonic() + effective_ttl) if effective_ttl > 0 else 0.0
        str_key = str(key)
        with self._lock:
            try:
                db = self._open()
                try:
                    db[str_key] = (value, expiry)
                finally:
                    db.close()
            except Exception as exc:
                raise CacheError(f"Disk cache set failed: {exc}", cache_name="disk") from exc

    def delete(self, key: K) -> bool:
        str_key = str(key)
        with self._lock:
            try:
                db = self._open()
                try:
                    if str_key in db:
                        del db[str_key]
                        return True
                    return False
                finally:
                    db.close()
            except Exception as exc:
                raise CacheError(f"Disk cache delete failed: {exc}", cache_name="disk") from exc

    def clear(self) -> None:
        with self._lock:
            try:
                db = self._open()
                try:
                    db.clear()
                finally:
                    db.close()
            except Exception as exc:
                raise CacheError(f"Disk cache clear failed: {exc}", cache_name="disk") from exc

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            size = 0
            try:
                db = self._open()
                try:
                    size = len(db)
                finally:
                    db.close()
            except Exception:
                pass
            return {
                "type": "DiskCache",
                "size": size,
                "directory": str(self._dir),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total else 0.0,
            }


# ---------------------------------------------------------------------------
# TieredCache — L1 (memory) → L2 (disk)
# ---------------------------------------------------------------------------

class TieredCache(Generic[K, V]):
    """Two-tier cache: fast L1 in-memory (:class:`LRUCache`) backed by a
    slower L2 (:class:`DiskCache`).

    On L1 miss, the value is promoted from L2 to L1.  On L1 eviction the
    entry is **not** automatically demoted; use :meth:`set` to populate both
    tiers, or call :meth:`demote` explicitly.

    Parameters:
        l1_maxsize: Maximum size for the L1 in-memory cache.
        l1_ttl: Default TTL for L1 entries.
        l2_directory: Directory for the L2 disk cache.
        l2_ttl: Default TTL for L2 entries.
    """

    def __init__(
        self,
        l1_maxsize: int = 512,
        l1_ttl: float = 600.0,
        l2_directory: str = ".veriflux_cache",
        l2_ttl: float = 3600.0,
    ) -> None:
        self._l1: LRUCache[K, V] = LRUCache(maxsize=l1_maxsize, default_ttl=l1_ttl)
        self._l2: DiskCache[K, V] = DiskCache(directory=l2_directory, default_ttl=l2_ttl)
        self._promotions = 0
        self._lock = threading.Lock()

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Retrieve from L1, falling back to L2 with promotion."""
        value = self._l1.get(key)
        if value is not None:
            return value
        value = self._l2.get(key)
        if value is not None:
            # Promote to L1
            self._l1.set(key, value)
            with self._lock:
                self._promotions += 1
            return value
        return default

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Store in both L1 and L2."""
        self._l1.set(key, value, ttl=ttl)
        self._l2.set(key, value, ttl=ttl)

    def delete(self, key: K) -> bool:
        r1 = self._l1.delete(key)
        r2 = self._l2.delete(key)
        return r1 or r2

    def clear(self) -> None:
        self._l1.clear()
        self._l2.clear()

    def demote(self, key: K) -> bool:
        """Move *key* from L1 to L2.  Returns ``True`` if the key was in L1."""
        value = self._l1.get(key)
        if value is None:
            return False
        self._l2.set(key, value)
        self._l1.delete(key)
        return True

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "type": "TieredCache",
                "l1": self._l1.stats(),
                "l2": self._l2.stats(),
                "promotions": self._promotions,
            }
