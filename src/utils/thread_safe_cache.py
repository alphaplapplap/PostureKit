"""Thread-safe cache implementations."""

import threading
import sys
from collections import OrderedDict
from typing import Any, Optional, TypeVar, Generic
import logging

logger = logging.getLogger(__name__)

K = TypeVar('K')
V = TypeVar('V')


class ThreadSafeOrderedDict(Generic[K, V]):
    """
    Thread-safe wrapper around OrderedDict with LRU eviction.

    Provides atomic operations and memory-bounded caching.
    """

    def __init__(self, max_size: int = 100, max_memory_mb: Optional[int] = None):
        """
        Initialize thread-safe cache.

        Args:
            max_size: Maximum number of items
            max_memory_mb: Maximum memory usage in MB (optional)
        """
        self._dict: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()  # Reentrant for nested calls
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        self.current_bytes = 0

    def __contains__(self, key: K) -> bool:
        """Thread-safe containment check."""
        with self._lock:
            return key in self._dict

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get item from cache (thread-safe).

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        with self._lock:
            if key in self._dict:
                # Move to end (LRU)
                self._dict.move_to_end(key)
                return self._dict[key]
            return default

    def set(self, key: K, value: V) -> None:
        """
        Set item in cache with automatic eviction (thread-safe).

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Calculate size if memory-bounded
            if self.max_memory_bytes is not None:
                value_size = sys.getsizeof(value)

                # Don't cache values larger than 10% of max memory
                if value_size > self.max_memory_bytes * 0.1:
                    logger.warning(
                        f"Value too large to cache: {value_size / 1024 / 1024:.1f}MB "
                        f"(max {self.max_memory_bytes / 1024 / 1024:.1f}MB)"
                    )
                    return

                # Evict until we have space
                while (
                    self.current_bytes + value_size > self.max_memory_bytes
                    and len(self._dict) > 0
                ):
                    evict_key, evict_val = self._dict.popitem(last=False)
                    self.current_bytes -= sys.getsizeof(evict_val)

                self.current_bytes += value_size

            # Remove old value if updating
            if key in self._dict:
                if self.max_memory_bytes is not None:
                    old_size = sys.getsizeof(self._dict[key])
                    self.current_bytes -= old_size
                self._dict.pop(key)

            # Add new value
            self._dict[key] = value

            # Size-based eviction
            if len(self._dict) > self.max_size:
                evict_key, evict_val = self._dict.popitem(last=False)
                if self.max_memory_bytes is not None:
                    self.current_bytes -= sys.getsizeof(evict_val)

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Remove and return item (thread-safe)."""
        with self._lock:
            if key in self._dict:
                value = self._dict.pop(key)
                if self.max_memory_bytes is not None:
                    self.current_bytes -= sys.getsizeof(value)
                return value
            return default

    def clear(self) -> None:
        """Clear all items (thread-safe)."""
        with self._lock:
            self._dict.clear()
            self.current_bytes = 0

    def __len__(self) -> int:
        """Get cache size (thread-safe)."""
        with self._lock:
            return len(self._dict)

    def move_to_end(self, key: K) -> None:
        """Move item to end of cache (thread-safe)."""
        with self._lock:
            if key in self._dict:
                self._dict.move_to_end(key)

    def items(self):
        """
        Get snapshot of items (thread-safe).

        Returns a copy to prevent external iteration issues.
        """
        with self._lock:
            return list(self._dict.items())

    def keys(self):
        """Get snapshot of keys (thread-safe)."""
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        """Get snapshot of values (thread-safe)."""
        with self._lock:
            return list(self._dict.values())


class ThreadSafeLRUCache(Generic[K, V]):
    """
    Memory-aware LRU cache with thread safety.

    Automatically evicts least recently used items when memory limit reached.
    """

    def __init__(self, max_memory_mb: int = 50):
        """
        Initialize memory-bounded LRU cache.

        Args:
            max_memory_mb: Maximum memory usage in megabytes
        """
        self._cache = ThreadSafeOrderedDict[K, V](
            max_size=10000,  # High size limit (memory is real limit)
            max_memory_mb=max_memory_mb
        )

    def __contains__(self, key: K) -> bool:
        return key in self._cache

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        return self._cache.get(key, default)

    def set(self, key: K, value: V) -> None:
        self._cache.set(key, value)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
