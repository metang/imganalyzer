"""JSON-RPC handler registry helpers.

This module is intentionally independent of ``imganalyzer.server`` so dispatch
mechanics can be tested without importing the full backend server process.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any


SyncHandler = Callable[[dict[str, Any]], Any]
AsyncHandler = Callable[[int | str | None, dict[str, Any]], None]
TransientErrorPredicate = Callable[[Exception], bool]
Sleeper = Callable[[float], None]


@dataclass
class RpcHandlerRegistry:
    """Registry for JSON-RPC handler maps and retry policy."""

    sync_methods: Mapping[str, SyncHandler]
    async_methods: Mapping[str, AsyncHandler]
    stdio_async_methods: Mapping[str, AsyncHandler]
    lock_retryable_methods: Collection[str]
    is_transient_lock_error: TransientErrorPredicate
    retry_attempts: int = 1
    retry_initial_delay_s: float = 0.0
    sleeper: Sleeper = time.sleep

    def has_sync(self, method: str) -> bool:
        return method in self.sync_methods

    def has_async(self, method: str) -> bool:
        return method in self.async_methods

    def sync_handler(self, method: str) -> SyncHandler:
        return self.sync_methods[method]

    def async_handler(self, method: str) -> AsyncHandler | None:
        return self.async_methods.get(method)

    def stdio_async_handler(self, method: str) -> AsyncHandler | None:
        return self.stdio_async_methods.get(method)

    def call_sync(self, method: str, params: dict[str, Any]) -> Any:
        """Invoke a sync handler, retrying configured transient DB lock errors."""
        handler = self.sync_handler(method)
        if method not in self.lock_retryable_methods:
            return handler(params)

        delay_s = self.retry_initial_delay_s
        for attempt in range(1, max(1, self.retry_attempts) + 1):
            try:
                return handler(params)
            except Exception as exc:
                if not self.is_transient_lock_error(exc) or attempt >= self.retry_attempts:
                    raise
                if delay_s > 0:
                    self.sleeper(delay_s)
                delay_s = min(delay_s * 2, 1.0) if delay_s > 0 else 0.0

        raise RuntimeError("unreachable retry loop exit")

