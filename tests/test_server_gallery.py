"""Regression tests for gallery SQL filtering in JSON-RPC server handlers."""
from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Generator

import numpy as np
import pytest

import imganalyzer.server as server

# server.py redirects stdout at import time for JSON-RPC mode; restore test stdout.
sys.stdout = server._real_stdout


_SCHEMA_SQL = """
CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    file_size INTEGER
);
CREATE TABLE analysis_metadata (
    image_id INTEGER PRIMARY KEY,
    camera_make TEXT,
    camera_model TEXT,
    lens_model TEXT,
    focal_length REAL,
    f_number REAL,
    exposure_time TEXT,
    iso INTEGER,
    date_time_original TEXT,
    gps_latitude REAL,
    gps_longitude REAL,
    location_city TEXT,
    location_state TEXT,
    location_country TEXT
);
CREATE TABLE analysis_technical (
    image_id INTEGER PRIMARY KEY,
    sharpness_score REAL,
    sharpness_label TEXT,
    exposure_ev REAL,
    exposure_label TEXT,
    noise_level REAL,
    noise_label TEXT,
    snr_db REAL,
    dynamic_range_stops REAL,
    highlight_clipping_pct REAL,
    shadow_clipping_pct REAL,
    avg_saturation REAL,
    dominant_colors TEXT
);
CREATE TABLE analysis_local_ai (
    image_id INTEGER PRIMARY KEY,
    description TEXT,
    scene_type TEXT,
    main_subject TEXT,
    lighting TEXT,
    mood TEXT,
    keywords TEXT,
    detected_objects TEXT,
    face_count INTEGER,
    face_identities TEXT,
    has_people INTEGER,
    ocr_text TEXT
);
CREATE TABLE analysis_blip2 (
    image_id INTEGER PRIMARY KEY,
    description TEXT,
    scene_type TEXT,
    main_subject TEXT,
    lighting TEXT,
    mood TEXT,
    keywords TEXT
);
CREATE TABLE analysis_objects (
    image_id INTEGER PRIMARY KEY,
    detected_objects TEXT,
    has_person INTEGER
);
CREATE TABLE analysis_ocr (
    image_id INTEGER PRIMARY KEY,
    ocr_text TEXT
);
CREATE TABLE analysis_faces (
    image_id INTEGER PRIMARY KEY,
    face_count INTEGER,
    face_identities TEXT
);
CREATE TABLE face_identities (
    id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    display_name TEXT,
    aliases TEXT,
    updated_at TEXT
);
CREATE TABLE face_aliases (
    identity_id INTEGER NOT NULL,
    alias TEXT NOT NULL
);
CREATE TABLE face_persons (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    notes TEXT
);
CREATE TABLE face_cluster_labels (
    cluster_id INTEGER PRIMARY KEY,
    display_name TEXT
);
CREATE TABLE face_occurrences (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL,
    identity_name TEXT,
    cluster_id INTEGER,
    person_id INTEGER
);
CREATE TABLE analysis_cloud_ai (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    description TEXT,
    analyzed_at TEXT
);
CREATE TABLE analysis_aesthetic (
    image_id INTEGER PRIMARY KEY,
    aesthetic_score REAL,
    aesthetic_label TEXT,
    aesthetic_reason TEXT
);
CREATE TABLE embeddings (
    image_id INTEGER,
    embedding_type TEXT,
    vector BLOB
);
"""


@pytest.fixture
def gallery_db() -> Generator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    try:
        yield conn
    finally:
        conn.close()


def _insert_processed_image(conn: sqlite3.Connection, image_id: int, file_path: str) -> None:
    conn.execute(
        "INSERT INTO images (id, file_path, width, height, file_size) VALUES (?, ?, ?, ?, ?)",
        (image_id, file_path, 100, 100, 12345),
    )
    conn.execute("INSERT INTO analysis_metadata (image_id) VALUES (?)", (image_id,))


def _insert_face_identity(
    conn: sqlite3.Connection,
    *,
    identity_id: int,
    canonical_name: str,
    display_name: str,
    aliases: list[str],
) -> None:
    conn.execute(
        """
        INSERT INTO face_identities (id, canonical_name, display_name, aliases, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (identity_id, canonical_name, display_name, json.dumps(aliases), "2026-01-01T00:00:00"),
    )
    conn.executemany(
        "INSERT INTO face_aliases (identity_id, alias) VALUES (?, ?)",
        [(identity_id, alias) for alias in aliases],
    )


def _insert_face_occurrence(
    conn: sqlite3.Connection,
    *,
    occurrence_id: int,
    image_id: int,
    identity_name: str,
    cluster_id: int | None = None,
    person_id: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO face_occurrences (id, image_id, identity_name, cluster_id, person_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (occurrence_id, image_id, identity_name, cluster_id, person_id),
    )


def test_gallery_chunk_folder_filter_works_for_windows_paths(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\2006\01\img1.jpg")

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_gallery_list_images_chunk(
        {
            "folderPath": r"E:\Pic\2006",
            "recursive": True,
            "chunkSize": 300,
            "cursor": None,
        }
    )

    assert result["total"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0]["file_path"] == r"E:\Pic\2006\01\img1.jpg"


def test_gallery_chunk_folder_filter_escapes_like_wildcards(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\2006\100%_done\img1.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\2006\100AAAdone\img2.jpg")

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_gallery_list_images_chunk(
        {
            "folderPath": r"E:\Pic\2006\100%_done",
            "recursive": True,
            "chunkSize": 300,
            "cursor": None,
        }
    )

    assert result["total"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0]["file_path"] == r"E:\Pic\2006\100%_done\img1.jpg"


def test_gallery_chunk_falls_back_to_split_tables_when_local_ai_missing(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery_db.execute(
        "INSERT INTO images (id, file_path, width, height, file_size) VALUES (?, ?, ?, ?, ?)",
        (1, r"E:\Pic\2006\01\img1.jpg", 6000, 4000, 24681012),
    )
    gallery_db.execute(
        """
        INSERT INTO analysis_blip2
            (image_id, description, scene_type, main_subject, lighting, mood, keywords)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "A serene mountain lake at sunset.",
            "landscape",
            "mountain lake",
            "golden hour",
            "calm",
            '["mountain", "lake"]',
        ),
    )
    gallery_db.execute(
        "INSERT INTO analysis_objects (image_id, detected_objects, has_person) VALUES (?, ?, ?)",
        (1, '["mountain", "person"]', 1),
    )
    gallery_db.execute(
        "INSERT INTO analysis_ocr (image_id, ocr_text) VALUES (?, ?)",
        (1, "Trailhead"),
    )
    gallery_db.execute(
        "INSERT INTO analysis_faces (image_id, face_count, face_identities) VALUES (?, ?, ?)",
        (1, 2, '["alice", "bob"]'),
    )
    gallery_db.execute(
        "INSERT INTO analysis_cloud_ai (image_id, description, analyzed_at) VALUES (?, ?, ?)",
        (1, "Cloud view: dramatic mountain range over water.", "2026-01-01T10:00:00"),
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_gallery_list_images_chunk(
        {"folderPath": r"E:\Pic\2006", "recursive": True, "chunkSize": 300, "cursor": None}
    )

    assert result["total"] == 1
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["description"] == "A serene mountain lake at sunset."
    assert item["scene_type"] == "landscape"
    assert item["main_subject"] == "mountain lake"
    assert item["keywords"] == ["mountain", "lake"]
    assert item["detected_objects"] == ["mountain", "person"]
    assert item["face_count"] == 2
    assert item["face_identities"] == ["alice", "bob"]
    assert item["has_people"] is True
    assert item["ocr_text"] == "Trailhead"
    assert item["cloud_description"] == "Cloud view: dramatic mountain range over water."


def test_search_browse_reports_total_and_has_more(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\2006\01\img1.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\2006\01\img2.jpg")
    _insert_processed_image(gallery_db, 3, r"E:\Pic\2006\01\img3.jpg")

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({"mode": "browse", "limit": 2, "offset": 0})

    assert result["total"] == 3
    assert result["hasMore"] is True
    assert [item["image_id"] for item in result["results"]] == [1, 2]


def test_search_ranked_pagination_returns_later_page(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for image_id in range(1, 7):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\search\img{image_id}.jpg")

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(
            self,
            query: str,
            limit: int,
            semantic_weight: float = 0.5,
            mode: str = "hybrid",
        ) -> list[dict[str, object]]:
            ranked = [
                {"image_id": 1, "score": 0.96},
                {"image_id": 2, "score": 0.91},
                {"image_id": 3, "score": 0.84},
                {"image_id": 4, "score": 0.79},
                {"image_id": 5, "score": 0.71},
                {"image_id": 6, "score": 0.62},
            ]
            return ranked[:limit]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)
    result = server._handle_search({"query": "mountain", "mode": "hybrid", "limit": 2, "offset": 2})

    assert result["total"] == 6
    assert result["hasMore"] is True
    assert [item["image_id"] for item in result["results"]] == [3, 4]


def test_search_similar_image_returns_ranked_matches(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for image_id in range(1, 4):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\similar\img{image_id}.jpg")

    gallery_db.executemany(
        "INSERT INTO embeddings (image_id, embedding_type, vector) VALUES (?, ?, ?)",
        [
            (1, "image_clip", np.array([1.0, 0.0], dtype=np.float32).tobytes()),
            (2, "image_clip", np.array([0.9, 0.1], dtype=np.float32).tobytes()),
            (3, "image_clip", np.array([0.0, 1.0], dtype=np.float32).tobytes()),
        ],
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({"similarToImageId": 1, "limit": 2, "offset": 0})

    assert result["total"] == 2
    assert result["hasMore"] is False
    assert [item["image_id"] for item in result["results"]] == [2, 3]
    assert result["results"][0]["score"] > result["results"][1]["score"]


def test_search_people_filters_support_country_recurring_day_and_time_of_day(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\img1.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\people\img2.jpg")
    _insert_processed_image(gallery_db, 3, r"E:\Pic\people\img3.jpg")

    gallery_db.execute(
        "UPDATE analysis_metadata SET location_country = ?, date_time_original = ? WHERE image_id = ?",
        ("US", "2024-02-01T08:15:00", 1),
    )
    gallery_db.execute(
        "UPDATE analysis_metadata SET location_country = ?, date_time_original = ? WHERE image_id = ?",
        ("US", "2025-02-01T18:45:00", 2),
    )
    gallery_db.execute(
        "UPDATE analysis_metadata SET location_country = ?, date_time_original = ? WHERE image_id = ?",
        ("CA", "2025-02-01T08:20:00", 3),
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search(
        {
            "mode": "browse",
            "country": "US",
            "recurringMonthDay": "02-01",
            "timeOfDay": "morning",
            "limit": 10,
            "offset": 0,
        }
    )

    assert result["total"] == 1
    assert result["hasMore"] is False
    assert [item["image_id"] for item in result["results"]] == [1]


def test_search_alias_prompt_routes_to_face_search(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\wyy.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\people\other.jpg")
    _insert_face_identity(
        gallery_db,
        identity_id=1,
        canonical_name="wang_yy",
        display_name="Wang YY",
        aliases=["wyy"],
    )
    gallery_db.executemany(
        """
        INSERT INTO analysis_local_ai (image_id, description, face_count, face_identities, has_people)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (1, "Portrait of Wang YY smiling outdoors.", 1, '["Wang YY"]', 1),
            (2, "Portrait of someone else.", 1, '["Other Person"]', 1),
        ],
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({"query": "WYY", "mode": "hybrid", "limit": 10, "offset": 0})

    assert result["total"] == 1
    assert result["hasMore"] is False
    assert [item["image_id"] for item in result["results"]] == [1]


def test_faces_cluster_relink_updates_label_and_person(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\cluster-a.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\people\cluster-b.jpg")
    gallery_db.execute("INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)", (1, "Chen XC", None))
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=1,
        image_id=1,
        identity_name="chen_xc_child",
        cluster_id=10,
        person_id=None,
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=2,
        image_id=2,
        identity_name="chen_xc_child",
        cluster_id=10,
        person_id=None,
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_faces_cluster_relink(
        {"cluster_id": 10, "display_name": "Chen XC", "person_id": 1, "update_person": True}
    )

    assert result == {"ok": True, "updated": 2}
    label_row = gallery_db.execute(
        "SELECT display_name FROM face_cluster_labels WHERE cluster_id = ?",
        (10,),
    ).fetchone()
    assert label_row["display_name"] == "Chen XC"
    person_ids = gallery_db.execute(
        "SELECT DISTINCT person_id FROM face_occurrences WHERE cluster_id = ?",
        (10,),
    ).fetchall()
    assert {row["person_id"] for row in person_ids} == {1}


def test_search_face_filter_uses_aliases_from_faces_table(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\cxc.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\people\other.jpg")
    _insert_face_identity(
        gallery_db,
        identity_id=1,
        canonical_name="chen_xc",
        display_name="Chen XC",
        aliases=["cxc"],
    )
    gallery_db.executemany(
        "INSERT INTO analysis_faces (image_id, face_count, face_identities) VALUES (?, ?, ?)",
        [
            (1, 1, '["chen_xc"]'),
            (2, 1, '["someone_else"]'),
        ],
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({"query": "CXC", "mode": "hybrid", "limit": 10, "offset": 0})

    assert result["total"] == 1
    assert result["hasMore"] is False
    assert [item["image_id"] for item in result["results"]] == [1]


def test_search_face_filter_matches_alias_across_multiple_identity_rows(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\cxc-child.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\people\cxc-adult.jpg")
    _insert_processed_image(gallery_db, 3, r"E:\Pic\people\other.jpg")
    _insert_face_identity(
        gallery_db,
        identity_id=1,
        canonical_name="chen_xc_child",
        display_name="Chen XC Child",
        aliases=["cxc"],
    )
    _insert_face_identity(
        gallery_db,
        identity_id=2,
        canonical_name="chen_xc_adult",
        display_name="Chen XC Adult",
        aliases=["cxc"],
    )
    gallery_db.executemany(
        "INSERT INTO analysis_faces (image_id, face_count, face_identities) VALUES (?, ?, ?)",
        [
            (1, 1, '["chen_xc_child"]'),
            (2, 1, '["chen_xc_adult"]'),
            (3, 1, '["someone_else"]'),
        ],
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({"query": "CXC", "mode": "hybrid", "limit": 10, "offset": 0})

    assert result["total"] == 2
    assert result["hasMore"] is False
    assert sorted(item["image_id"] for item in result["results"]) == [1, 2]


def test_search_face_filter_matches_person_name_across_multiple_clusters(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\cxc-cluster-a.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\people\cxc-cluster-b.jpg")
    _insert_processed_image(gallery_db, 3, r"E:\Pic\people\other.jpg")
    gallery_db.execute("INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)", (1, "cxc", None))
    gallery_db.executemany(
        "INSERT INTO face_cluster_labels (cluster_id, display_name) VALUES (?, ?)",
        [(10, "CXC"), (20, "CXC")],
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=1,
        image_id=1,
        identity_name="chen_xc_child",
        cluster_id=10,
        person_id=1,
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=2,
        image_id=2,
        identity_name="chen_xc_adult",
        cluster_id=20,
        person_id=1,
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=3,
        image_id=3,
        identity_name="someone_else",
        cluster_id=30,
        person_id=None,
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({"query": "CXC", "mode": "hybrid", "limit": 10, "offset": 0})

    assert result["total"] == 2
    assert result["hasMore"] is False
    assert sorted(item["image_id"] for item in result["results"]) == [1, 2]


def test_search_multi_face_filter_requires_all_selected_people(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\cxc.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\people\wjj.jpg")
    _insert_processed_image(gallery_db, 3, r"E:\Pic\people\cxc-wjj.jpg")
    gallery_db.executemany(
        "INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)",
        [(1, "cxc", None), (2, "wjj", None)],
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=1,
        image_id=1,
        identity_name="chen_xc",
        cluster_id=10,
        person_id=1,
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=2,
        image_id=2,
        identity_name="wang_jj",
        cluster_id=20,
        person_id=2,
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=3,
        image_id=3,
        identity_name="chen_xc",
        cluster_id=10,
        person_id=1,
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=4,
        image_id=3,
        identity_name="wang_jj",
        cluster_id=20,
        person_id=2,
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search(
        {"faces": ["cxc", "wjj"], "faceMatch": "all", "mode": "hybrid", "limit": 10, "offset": 0}
    )

    assert result["total"] == 1
    assert result["hasMore"] is False
    assert [item["image_id"] for item in result["results"]] == [3]


def test_search_resolve_face_query_returns_multiple_faces_and_remaining_text(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery_db.executemany(
        "INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)",
        [(1, "cxc", None), (2, "wjj", None)],
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search_resolve_face_query({"query": "cxc, wjj together at the beach"})

    assert result == {
        "face": "cxc",
        "faces": ["cxc", "wjj"],
        "faceMatch": "all",
        "remainingQuery": "at the beach",
    }


def test_search_alias_prompt_combines_face_and_activity_terms(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for image_id in range(1, 5):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\people\img{image_id}.jpg")

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def resolve_face_query(self, query: str) -> tuple[str | None, str]:
            if query == "wyy playing basketball":
                return "wyy", "playing basketball"
            return None, query

        def resolve_face_queries(self, query: str) -> tuple[list[str], str, str]:
            if query == "wyy playing basketball":
                return ["wyy"], "playing basketball", "all"
            return [], query, "all"

        def search_face(self, name: str, limit: int = 50) -> list[dict[str, object]]:
            assert name == "wyy"
            return [
                {"image_id": 1, "score": 1.0},
                {"image_id": 2, "score": 1.0},
                {"image_id": 4, "score": 1.0},
            ][:limit]

        def search(
            self,
            query: str,
            limit: int,
            semantic_weight: float = 0.5,
            mode: str = "hybrid",
        ) -> list[dict[str, object]]:
            if query != "playing basketball":
                return []
            return [
                {"image_id": 3, "score": 0.95},
                {"image_id": 2, "score": 0.90},
                {"image_id": 4, "score": 0.88},
            ][:limit]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)
    result = server._handle_search(
        {"query": "wyy playing basketball", "mode": "hybrid", "limit": 10, "offset": 0}
    )

    assert result["total"] == 2
    assert result["hasMore"] is False
    assert [item["image_id"] for item in result["results"]] == [2, 4]


def test_search_multi_face_prompt_combines_people_and_activity_terms(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for image_id in range(1, 6):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\people\group{image_id}.jpg")

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def resolve_face_queries(self, query: str) -> tuple[list[str], str, str]:
            if query == "cxc, wjj together at the beach":
                return ["cxc", "wjj"], "at the beach", "all"
            return [], query, "all"

        def resolve_face_query(self, query: str) -> tuple[str | None, str]:
            return None, query

        def search_faces(
            self,
            names: list[str],
            limit: int = 50,
            match_mode: str = "all",
        ) -> list[dict[str, object]]:
            assert names == ["cxc", "wjj"]
            assert match_mode == "all"
            return [
                {"image_id": 1, "score": 2.0},
                {"image_id": 4, "score": 1.8},
            ][:limit]

        def search(
            self,
            query: str,
            limit: int,
            semantic_weight: float = 0.5,
            mode: str = "hybrid",
        ) -> list[dict[str, object]]:
            if query != "at the beach":
                return []
            return [
                {"image_id": 4, "score": 0.98},
                {"image_id": 3, "score": 0.95},
                {"image_id": 1, "score": 0.90},
            ][:limit]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)
    result = server._handle_search(
        {"query": "cxc, wjj together at the beach", "mode": "hybrid", "limit": 10, "offset": 0}
    )

    assert result["total"] == 2
    assert result["hasMore"] is False
    assert [item["image_id"] for item in result["results"]] == [1, 4]


def test_search_best_sort_uses_quality_ranking(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for image_id in range(1, 4):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\best\img{image_id}.jpg")

    gallery_db.executemany(
        """
        INSERT INTO analysis_technical (image_id, sharpness_score, noise_level)
        VALUES (?, ?, ?)
        """,
        [
            (1, 80.0, 0.02),
            (2, 50.0, 0.30),
            (3, 95.0, 0.01),
        ],
    )
    gallery_db.executemany(
        """
        INSERT INTO analysis_aesthetic (image_id, aesthetic_score, aesthetic_label, aesthetic_reason)
        VALUES (?, ?, ?, ?)
        """,
        [
            (1, 8.5, "Excellent", "Balanced composition"),
            (2, 9.0, "Excellent", "Beautiful light"),
            (3, 7.5, "Good", "Sharp with cleaner detail"),
        ],
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({"mode": "browse", "sortBy": "best", "limit": 3, "offset": 0})

    assert result["total"] == 3
    assert [item["image_id"] for item in result["results"]] == [1, 3, 2]


def test_search_expanded_terms_merge_ranked_results(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for image_id in range(1, 4):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\wildlife\img{image_id}.jpg")

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(
            self,
            query: str,
            limit: int,
            semantic_weight: float = 0.5,
            mode: str = "hybrid",
        ) -> list[dict[str, object]]:
            if query == "duck":
                return [{"image_id": 1, "score": 0.9}, {"image_id": 2, "score": 0.8}][:limit]
            if query == "mallard":
                return [{"image_id": 2, "score": 0.95}][:limit]
            if query == "teal":
                return [{"image_id": 3, "score": 0.88}][:limit]
            return []

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)
    result = server._handle_search(
        {
            "query": "duck",
            "expandedTerms": ["mallard", "teal"],
            "mode": "hybrid",
            "limit": 3,
            "offset": 0,
        }
    )

    assert result["total"] == 3
    assert [item["image_id"] for item in result["results"]] == [2, 1, 3]
