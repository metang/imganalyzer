"""Regression tests for B3 (SQL-level pagination) and B4 (batched face clusters).

B3 ensures the search handler does not materialize the full candidate pool
when the final sort can be expressed in SQL. We observe this indirectly by
counting how many rows the SELECT cursor returns from the big joined query:
with SQL LIMIT/OFFSET pushed down, it returns at most ``limit`` rows
(plus one COUNT row for totals), regardless of how many candidates the
search engine produced.

B4 ensures ``_get_face_clusters_for_image_ids`` resolves N images in a
single parameterized IN-list query instead of one-per-image.
"""
from __future__ import annotations

import sqlite3
import sys
from collections.abc import Generator

import pytest

import imganalyzer.server as server

# server.py redirects stdout at import time; restore test stdout.
sys.stdout = server._real_stdout

from tests.test_server_gallery import (  # noqa: E402  (import after stdout fix)
    _SCHEMA_SQL,
    _insert_face_occurrence,
    _insert_processed_image,
)


@pytest.fixture
def gallery_db() -> Generator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    try:
        yield conn
    finally:
        conn.close()


class _RowCountingConn:
    """Connection wrapper that records every executed statement + row count."""

    def __init__(self, inner: sqlite3.Connection) -> None:
        self._inner = inner
        # Each entry: (sql, rows_fetched).
        self.executions: list[tuple[str, int]] = []

    def execute(self, sql, params=()):  # type: ignore[no-untyped-def]
        cursor = self._inner.execute(sql, params)
        return _RowCountingCursor(cursor, sql, self.executions)

    def executemany(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self._inner.executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self._inner.executescript(*args, **kwargs)

    def __getattr__(self, name):  # type: ignore[no-untyped-def]
        return getattr(self._inner, name)


class _RowCountingCursor:
    def __init__(self, cursor, sql, log):  # type: ignore[no-untyped-def]
        self._cursor = cursor
        self._sql = sql
        self._log = log

    def fetchall(self):  # type: ignore[no-untyped-def]
        rows = self._cursor.fetchall()
        self._log.append((self._sql, len(rows)))
        return rows

    def fetchone(self):  # type: ignore[no-untyped-def]
        row = self._cursor.fetchone()
        self._log.append((self._sql, 1 if row is not None else 0))
        return row

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._cursor)

    def __getattr__(self, name):  # type: ignore[no-untyped-def]
        return getattr(self._cursor, name)


def _primary_select_rows(executions: list[tuple[str, int]]) -> list[int]:
    """Row counts for the big joined ``SELECT ... FROM images i ...`` query."""
    return [
        count
        for sql, count in executions
        if "FROM images i" in sql and "COUNT(*)" not in sql.upper()
    ]


def test_b3_sort_by_newest_pushes_limit_to_sql(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With sort_by != relevance, a page of N only fetches ~N main-table rows
    from SQL, even though the fake engine returns many more candidates."""
    for image_id in range(1, 21):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\many\img{image_id}.jpg")
        gallery_db.execute(
            "UPDATE analysis_metadata SET date_time_original = ? WHERE image_id = ?",
            (f"2025-01-{image_id:02d}T12:00:00", image_id),
        )

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(self, query, limit, **_) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
            # Return all 20 candidates with decreasing scores.
            ranked = [{"image_id": i, "score": 1.0 - i * 0.01} for i in range(1, 21)]
            return ranked[:limit]

    import imganalyzer.db.search as search_module

    wrapped = _RowCountingConn(gallery_db)
    monkeypatch.setattr(server, "_get_db", lambda: wrapped)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)

    result = server._handle_search({
        "query": "anything",
        "mode": "hybrid",
        "sortBy": "newest",
        "limit": 3,
        "offset": 0,
    })

    # Correctness: newest 3 are img 20, 19, 18.
    assert [item["image_id"] for item in result["results"]] == [20, 19, 18]
    assert result["total"] == 20
    assert result["hasMore"] is True

    # The main SELECT should have returned at most ``limit`` rows, not 20.
    main_row_counts = _primary_select_rows(wrapped.executions)
    assert main_row_counts, "expected the big joined SELECT to run at least once"
    assert max(main_row_counts) <= 3, (
        f"SQL LIMIT not pushed down for sortBy=newest; fetched {main_row_counts}"
    )


def test_b3_sort_by_relevance_still_ranks_correctly(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: the ``relevance`` branch still preserves engine-ordered fused scores."""
    for image_id in range(1, 6):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\rel\img{image_id}.jpg")

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(self, query, limit, **_) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
            # Intentionally out-of-insert-order to test score-based sort.
            return [
                {"image_id": 3, "score": 0.99},
                {"image_id": 1, "score": 0.80},
                {"image_id": 4, "score": 0.70},
                {"image_id": 5, "score": 0.60},
                {"image_id": 2, "score": 0.40},
            ][:limit]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)

    result = server._handle_search({
        "query": "anything",
        "mode": "hybrid",
        "sortBy": "relevance",
        "limit": 3,
        "offset": 0,
    })
    assert [item["image_id"] for item in result["results"]] == [3, 1, 4]


def test_b4_face_clusters_batched_in_single_query(
    gallery_db: sqlite3.Connection,
) -> None:
    """``_get_face_clusters_for_image_ids`` must resolve N images in ONE query."""
    for image_id in range(1, 6):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\f\img{image_id}.jpg")
        _insert_face_occurrence(
            gallery_db,
            occurrence_id=image_id,
            image_id=image_id,
            identity_name=f"person_{image_id}",
            cluster_id=image_id * 10,
        )

    wrapped = _RowCountingConn(gallery_db)
    clusters = server._get_face_clusters_for_image_ids(wrapped, [1, 2, 3, 4, 5])

    # Exactly one SELECT against face_occurrences (plus the _table_exists probes,
    # which hit sqlite_master, not face_occurrences).
    face_queries = [sql for sql, _ in wrapped.executions if "face_occurrences" in sql]
    assert len(face_queries) == 1, (
        f"expected a single batched face_occurrences query, got {len(face_queries)}: "
        f"{face_queries}"
    )

    # Each image has its own cluster.
    assert set(clusters.keys()) == {1, 2, 3, 4, 5}
    for image_id, entries in clusters.items():
        assert len(entries) == 1
        assert entries[0]["cluster_id"] == image_id * 10


def test_b4_face_clusters_chunks_large_id_lists(
    gallery_db: sqlite3.Connection,
) -> None:
    """ID lists larger than the chunk size are split, but still one query per chunk."""
    # Create 1200 images with face occurrences — forces 3 chunks of 500, 500, 200.
    for image_id in range(1, 1201):
        gallery_db.execute(
            "INSERT INTO images (id, file_path, width, height, file_size) "
            "VALUES (?, ?, 1, 1, 1)",
            (image_id, f"img{image_id}.jpg"),
        )
        gallery_db.execute(
            "INSERT INTO face_occurrences (id, image_id, identity_name, cluster_id) "
            "VALUES (?, ?, ?, ?)",
            (image_id, image_id, "p", image_id % 7 + 1),
        )

    wrapped = _RowCountingConn(gallery_db)
    clusters = server._get_face_clusters_for_image_ids(
        wrapped, list(range(1, 1201))
    )

    face_queries = [sql for sql, _ in wrapped.executions if "face_occurrences" in sql]
    # Exactly ceil(1200/500) = 3 batched queries, not 1200.
    assert len(face_queries) == 3
    assert len(clusters) == 1200


def test_b4_search_page_face_clusters_limited_to_page(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The search handler should only resolve clusters for the returned page,
    not the full candidate pool (avoids the N+1 / whole-candidate-set cost)."""
    # 10 images, each with a face occurrence.
    for image_id in range(1, 11):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\faces\img{image_id}.jpg")
        gallery_db.execute(
            "UPDATE analysis_metadata SET date_time_original = ? WHERE image_id = ?",
            (f"2025-02-{image_id:02d}T10:00:00", image_id),
        )
        _insert_face_occurrence(
            gallery_db,
            occurrence_id=image_id,
            image_id=image_id,
            identity_name="p",
            cluster_id=100 + image_id,
        )

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(self, query, limit, **_):  # type: ignore[no-untyped-def]
            return [{"image_id": i, "score": 1.0 - i * 0.01} for i in range(1, 11)][:limit]

    import imganalyzer.db.search as search_module

    wrapped = _RowCountingConn(gallery_db)
    monkeypatch.setattr(server, "_get_db", lambda: wrapped)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)

    result = server._handle_search({
        "query": "anything",
        "mode": "hybrid",
        "sortBy": "newest",
        "limit": 2,
        "offset": 0,
    })

    # Page has 2 results; each carries its own cluster.
    assert len(result["results"]) == 2
    for record in result["results"]:
        assert record["face_clusters"] is not None
        assert len(record["face_clusters"]) == 1

    # The face_occurrences query for attaching clusters should have returned
    # at most 2 rows (one per paginated image), not 10 (the full candidate pool).
    face_queries = [
        (sql, count)
        for sql, count in wrapped.executions
        if "face_occurrences" in sql and "fo.image_id IN" in sql
    ]
    assert face_queries, "expected the batched face_occurrences lookup to run"
    for _sql, count in face_queries:
        assert count <= 2, (
            f"face cluster lookup pulled {count} rows; "
            "should only cover the 2-image page"
        )
