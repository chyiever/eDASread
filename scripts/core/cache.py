"""Simple in-memory caches used by the UI workflow."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheItem(Generic[V]):
    """Cache item wrapper storing byte size for eviction decisions."""

    value: V
    size_bytes: int


class LruByteCache(Generic[K, V]):
    """A small LRU cache bounded by total estimated bytes."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._items: OrderedDict[K, CacheItem[V]] = OrderedDict()
        self._current_bytes = 0

    def clear(self) -> None:
        """Remove all cached values."""
        self._items.clear()
        self._current_bytes = 0

    def get(self, key: K) -> V | None:
        """Return a cached value and mark it as recently used."""
        item = self._items.pop(key, None)
        if item is None:
            return None
        self._items[key] = item
        return item.value

    def put(self, key: K, value: V, size_bytes: int) -> None:
        """Insert a value and evict older entries if needed."""
        previous = self._items.pop(key, None)
        if previous is not None:
            self._current_bytes -= previous.size_bytes

        self._items[key] = CacheItem(value=value, size_bytes=size_bytes)
        self._current_bytes += size_bytes
        self._evict()

    def _evict(self) -> None:
        while self._current_bytes > self._max_bytes and self._items:
            _, item = self._items.popitem(last=False)
            self._current_bytes -= item.size_bytes
