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
    _insert_search_index_row,
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


def test_search_browse_large_result_uses_bounded_total(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broad browse/search filters should not run exact counts over every row."""
    for image_id in range(1, 6002):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\scale\img{image_id}.jpg")

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)

    result = server._handle_search({"mode": "browse", "limit": 3, "offset": 0})

    assert len(result["results"]) == 3
    assert result["total"] is None
    assert result["hasMore"] is True


def test_search_browse_deep_offset_short_circuits(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Browse-mode deep offsets should not issue a synchronous OFFSET scan."""
    for image_id in range(1, 20):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\offset\img{image_id}.jpg")

    wrapped = _RowCountingConn(gallery_db)
    monkeypatch.setattr(server, "_get_db", lambda: wrapped)

    result = server._handle_search({"mode": "browse", "limit": 3, "offset": 100_000})

    assert result == {"results": [], "total": None, "hasMore": False}
    assert not _primary_select_rows(wrapped.executions)


def test_search_deep_offset_does_not_expand_candidates_without_bound(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large offset must not turn into a huge synchronous search limit."""
    _insert_processed_image(gallery_db, 1, r"E:\Pic\scale\only.jpg")
    seen_limits: list[int] = []

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(self, query, limit, **_):  # type: ignore[no-untyped-def]
            seen_limits.append(int(limit))
            return [{"image_id": 1, "score": 1.0}]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)

    result = server._handle_search(
        {"query": "anything", "mode": "hybrid", "limit": 10, "offset": 100_000}
    )

    assert seen_limits == [5000]
    assert result["results"] == []
    assert result["total"] == 1


def test_search_expanded_terms_are_capped(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expanded query entropy should not fan out into unbounded engine calls."""
    for image_id in range(1, 10):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\terms\img{image_id}.jpg")

    seen_terms: list[str] = []

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(self, query, limit, **_):  # type: ignore[no-untyped-def]
            seen_terms.append(str(query))
            return [{"image_id": len(seen_terms), "score": 1.0 / len(seen_terms)}]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)

    server._handle_search(
        {
            "query": "base",
            "expandedTerms": [f"term-{idx}" for idx in range(20)],
            "mode": "hybrid",
            "limit": 3,
            "offset": 0,
        }
    )

    assert seen_terms == [
        "base",
        "term-0",
        "term-1",
        "term-2",
        "term-3",
        "term-4",
        "term-5",
        "term-6",
    ]


def test_fts_candidate_ids_are_applied_in_sql(
    gallery_db: sqlite3.Connection,
) -> None:
    """Candidate-restricted FTS must find matches beyond the global top-N window."""
    from imganalyzer.db.search import SearchEngine

    for image_id in range(1, 1201):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\fts\img{image_id}.jpg")
        _insert_search_index_row(gallery_db, image_id=image_id, description_text="scale target")

    engine = SearchEngine(gallery_db)
    results = engine.search("scale", mode="text", limit=5, candidate_ids={1000})

    assert [result["image_id"] for result in results] == [1000]


def test_multi_face_search_passes_bounded_limits_to_resolver(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-face search should not resolve each person with limit=None."""
    from imganalyzer.db.search import SearchEngine

    engine = SearchEngine(gallery_db)
    seen_limits: list[tuple[str, int | None]] = []
    rows_by_name = {
        "alice": [
            {"image_id": 1, "file_path": r"E:\Pic\faces\alice.jpg", "score": 1.0},
            {"image_id": 2, "file_path": r"E:\Pic\faces\together.jpg", "score": 1.0},
        ],
        "bob": [
            {"image_id": 2, "file_path": r"E:\Pic\faces\together.jpg", "score": 1.0},
            {"image_id": 3, "file_path": r"E:\Pic\faces\bob.jpg", "score": 1.0},
        ],
    }

    def fake_resolve_face_rows(name: str, limit: int | None = 50) -> list[dict[str, object]]:
        seen_limits.append((name, limit))
        row_limit = 0 if limit is None else int(limit)
        return rows_by_name[name][:row_limit]

    monkeypatch.setattr(engine, "_resolve_face_rows", fake_resolve_face_rows)

    results = engine.search_faces(["alice", "bob"], limit=3, match_mode="all")

    assert seen_limits == [("alice", 12), ("bob", 12)]
    assert [result["image_id"] for result in results] == [2]


def test_multi_face_search_any_mode_keeps_expected_ranked_results(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imganalyzer.db.search import SearchEngine

    engine = SearchEngine(gallery_db)

    def fake_resolve_face_rows(name: str, limit: int | None = 50) -> list[dict[str, object]]:
        assert limit == 8
        rows_by_name = {
            "alice": [
                {"image_id": 1, "file_path": r"E:\Pic\faces\alice.jpg", "score": 1.0},
                {"image_id": 2, "file_path": r"E:\Pic\faces\together.jpg", "score": 1.0},
            ],
            "bob": [
                {"image_id": 2, "file_path": r"E:\Pic\faces\together.jpg", "score": 1.0},
                {"image_id": 3, "file_path": r"E:\Pic\faces\bob.jpg", "score": 1.0},
            ],
        }
        return rows_by_name[name]

    monkeypatch.setattr(engine, "_resolve_face_rows", fake_resolve_face_rows)

    results = engine.search_faces(["alice", "bob"], limit=2, match_mode="any")

    assert [(result["image_id"], result["score"]) for result in results] == [
        (2, 2.0),
        (1, 1.0),
    ]


def test_multi_face_search_passes_bounded_limits_to_repository_lookups(
    gallery_db: sqlite3.Connection,
) -> None:
    from imganalyzer.db.search import SearchEngine

    class FakeRepo:
        def __init__(self) -> None:
            self.person_limits: list[tuple[int, int | None]] = []

        def find_face_identities_by_alias(self, name: str) -> list[dict[str, object]]:
            return []

        def find_persons_by_name(self, name: str) -> list[dict[str, object]]:
            return [{"id": {"alice": 1, "bob": 2}[name], "name": name}]

        def find_clusters_by_label(self, name: str) -> list[dict[str, object]]:
            return []

        def get_images_for_person(
            self,
            person_id: int,
            limit: int | None = 100,
        ) -> list[dict[str, object]]:
            self.person_limits.append((person_id, limit))
            rows_by_person = {
                1: [
                    {"image_id": 1, "file_path": r"E:\Pic\faces\alice.jpg"},
                    {"image_id": 3, "file_path": r"E:\Pic\faces\together.jpg"},
                ],
                2: [
                    {"image_id": 2, "file_path": r"E:\Pic\faces\bob.jpg"},
                    {"image_id": 3, "file_path": r"E:\Pic\faces\together.jpg"},
                ],
            }
            return rows_by_person[person_id]

        def get_images_for_cluster(
            self,
            cluster_id: int,
            limit: int | None = 100,
        ) -> list[dict[str, object]]:
            return []

        def get_images_for_face(
            self,
            name: str,
            limit: int | None = 100,
        ) -> list[dict[str, object]]:
            return []

    engine = SearchEngine(gallery_db)
    fake_repo = FakeRepo()
    engine.repo = fake_repo  # type: ignore[assignment]

    results = engine.search_faces(["alice", "bob"], limit=2, match_mode="all")

    assert fake_repo.person_limits == [(1, 8), (2, 8)]
    assert [result["image_id"] for result in results] == [3]
