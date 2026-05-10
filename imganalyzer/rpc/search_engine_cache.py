"""Thread-safe SearchEngine cache used by the JSON-RPC server."""
from __future__ import annotations

import contextlib
import importlib
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any


SearchModuleLoader = Callable[[], ModuleType]
DbPathGetter = Callable[[], Path]


def _default_search_module_loader() -> ModuleType:
    return importlib.import_module("imganalyzer.db.search")


def _default_db_path_getter() -> Path:
    from imganalyzer.db.connection import get_db_path

    return get_db_path()


class SearchEngineCache:
    """Cache one SearchEngine per DB path and SearchEngine class identity.

    The cached engine's SQLite connection is rebound on every acquisition so
    callers can use thread-local connections without rebuilding heavy embedding
    matrices for each RPC request.
    """

    def __init__(
        self,
        search_module_loader: SearchModuleLoader = _default_search_module_loader,
        db_path_getter: DbPathGetter = _default_db_path_getter,
    ) -> None:
        self._search_module_loader = search_module_loader
        self._db_path_getter = db_path_getter
        self._engine: Any = None
        self._key: tuple[str, int] | None = None
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        """Return the lock guarding the cached engine."""
        return self._lock

    def clear(self) -> None:
        """Drop the cached engine and key."""
        with self._lock:
            self._engine = None
            self._key = None

    def acquire(self, conn: Any) -> Any:
        """Return the cached SearchEngine rebound to *conn*.

        Acquires ``lock``; callers must call :meth:`release` in a ``finally``
        block, or use :meth:`context`.
        """
        search_module = self._search_module_loader()
        db_key = (str(self._db_path_getter()), id(search_module.SearchEngine))

        self._lock.acquire()
        try:
            if self._engine is None or self._key != db_key:
                self._engine = search_module.SearchEngine(conn)
                self._key = db_key

            self._engine.conn = conn
            repo = getattr(self._engine, "repo", None)
            if repo is not None:
                repo.conn = conn
        except BaseException:
            self._lock.release()
            raise

        return self._engine

    def release(self) -> None:
        """Release the lock held by :meth:`acquire`."""
        try:
            self._lock.release()
        except RuntimeError:
            # Not held by this thread — ignore so callers can use finally
            # blocks without worrying about early-exit paths before acquisition.
            pass

    @contextlib.contextmanager
    def context(self, conn: Any) -> Iterator[Any]:
        """Context-manager wrapper around :meth:`acquire`."""
        engine = self.acquire(conn)
        try:
            yield engine
        finally:
            self.release()
