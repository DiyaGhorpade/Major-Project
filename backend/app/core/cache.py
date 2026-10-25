"""Tiny in-process response cache keyed on a cheap "state token" computed from the database.

`get_or_compute(key, token, fn)` returns the cached value while `token` is unchanged. The token is built from
counts / max ids / the latest pipeline run (`repositories.state_token`), so a finished run, an ingestion or a
recommendation change invalidates entries automatically; `clear()` is also called explicitly after writes.
One cache per app (`app.state.cache`), so test apps never share entries. Thread-safe.
"""

import threading
from collections.abc import Callable
from typing import Any


class ResponseCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[Any, Any]] = {}

    def get_or_compute(self, key: str, token: Any, compute: Callable[[], Any]) -> Any:
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None and hit[0] == token:
                return hit[1]
        value = compute()  # outside the lock: two concurrent misses may both compute, which is harmless
        with self._lock:
            self._entries[key] = (token, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
