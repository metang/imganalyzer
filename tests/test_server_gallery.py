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
CREATE TABLE analysis_caption (
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
CREATE TABLE analysis_perception (
    image_id INTEGER PRIMARY KEY,
    perception_iaa REAL,
    perception_iaa_label TEXT,
    perception_iqa REAL,
    perception_iqa_label TEXT,
    perception_ista REAL,
    perception_ista_label TEXT,
    analyzed_at TEXT
);
CREATE TABLE search_features (
    image_id INTEGER PRIMARY KEY,
    desc_lex TEXT,
    desc_summary TEXT,
    desc_quality REAL,
    keywords_text TEXT,
    objects_text TEXT,
    ocr_text TEXT,
    faces_text TEXT,
    camera_make TEXT,
    camera_model TEXT,
    lens_model TEXT,
    date_time_original TEXT,
    location_city TEXT,
    location_state TEXT,
    location_country TEXT,
    sharpness_score REAL,
    noise_level REAL,
    snr_db REAL,
    dynamic_range_stops REAL,
    perception_iaa REAL,
    perception_iqa REAL,
    perception_ista REAL,
    aesthetic_score REAL,
    face_count INTEGER,
    has_people INTEGER,
    updated_at TEXT
);
CREATE TABLE embeddings (
    image_id INTEGER,
    embedding_type TEXT,
    vector BLOB
);
CREATE VIRTUAL TABLE search_index USING fts5(
    image_id,
    description_text,
    subjects_text,
    keywords_text,
    faces_text,
    exif_text,
    content='',
    tokenize='porter unicode61'
);
"""


@pytest.fixture
def gallery_db() -> Generator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:", isolation_level=None)
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


def _insert_search_index_row(
    conn: sqlite3.Connection,
    *,
    image_id: int,
    description_text: str = "",
    subjects_text: str = "",
    keywords_text: str = "",
    faces_text: str = "",
    exif_text: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO search_index
            (rowid, image_id, description_text, subjects_text, keywords_text, faces_text, exif_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (image_id, str(image_id), description_text, subjects_text, keywords_text, faces_text, exif_text),
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


def test_gallery_chunk_falls_back_to_split_tables_when_caption_missing(
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
            semantic_profile: str | None = None,
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
        INSERT INTO analysis_caption (image_id, description, face_count, face_identities, has_people)
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


def test_search_resolve_face_query_strips_picture_filler_phrase(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery_db.executemany(
        "INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)",
        [(1, "cxc", None), (2, "meng", None)],
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search_resolve_face_query(
        {"query": "group picture 10 or more people. cxc and meng are in the picture"}
    )

    assert result == {
        "face": "cxc",
        "faces": ["cxc", "meng"],
        "faceMatch": "all",
        "remainingQuery": "group picture 10 or more people.",
    }


def test_search_resolve_face_query_allows_partial_person_names(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery_db.execute(
        "INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)",
        (1, "Sun Ting", None),
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search_resolve_face_query({"query": "ting at sunset"})

    assert result == {
        "face": "ting",
        "faces": ["ting"],
        "faceMatch": "any",
        "remainingQuery": "at sunset",
    }


def test_search_partial_person_name_returns_face_results(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\sun-ting.jpg")
    gallery_db.execute(
        "INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)",
        (1, "Sun Ting", None),
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=1,
        image_id=1,
        identity_name="sun_ting",
        cluster_id=10,
        person_id=1,
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({"query": "ting", "mode": "hybrid", "limit": 10, "offset": 0})

    assert result["total"] == 1
    assert result["hasMore"] is False
    assert [item["image_id"] for item in result["results"]] == [1]


def test_fts_match_query_handles_periods_and_boolean_words() -> None:
    from imganalyzer.db.search import _build_fts_match_query

    gallery_db = sqlite3.connect(":memory:")
    try:
        gallery_db.execute(
            """
            CREATE VIRTUAL TABLE search_index USING fts5(
                image_id,
                description_text,
                subjects_text,
                keywords_text,
                faces_text,
                exif_text,
                content='',
                tokenize='porter unicode61'
            )
            """
        )
        match_query = _build_fts_match_query(
            "group picture 10 or more people. cxc and meng are in the picture"
        )
        gallery_db.execute(
            """
            INSERT INTO search_index
                (rowid, image_id, description_text, subjects_text, keywords_text, faces_text, exif_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "1",
                "group picture 10 or more people cxc and meng are in the picture",
                "",
                "",
                "",
                "",
            ),
        )
        rows = gallery_db.execute(
            "SELECT rowid FROM search_index WHERE search_index MATCH ?",
            [match_query],
        ).fetchall()

        assert match_query == (
            '"group" AND "picture" AND "10" AND "or" AND "more" '
            'AND "people" AND "cxc" AND "and" AND "meng" AND "are" '
            'AND "in" AND "the" AND "picture"'
        )
        assert len(rows) == 1
    finally:
        gallery_db.close()


def test_fts_soft_match_query_supports_partial_description_overlap() -> None:
    from imganalyzer.db.search import _build_fts_soft_match_query

    gallery_db = sqlite3.connect(":memory:")
    try:
        gallery_db.execute(
            """
            CREATE VIRTUAL TABLE search_index USING fts5(
                image_id,
                description_text,
                subjects_text,
                keywords_text,
                faces_text,
                exif_text,
                content='',
                tokenize='porter unicode61'
            )
            """
        )
        gallery_db.executemany(
            """
            INSERT INTO search_index
                (rowid, image_id, description_text, subjects_text, keywords_text, faces_text, exif_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    1,
                    "1",
                    (
                        "A woman in a colorful striped dress under a black cardigan sits "
                        "inside a freestanding white bathtub in a modern bathroom."
                    ),
                    "",
                    "",
                    "",
                    "",
                ),
                (2, "2", "Golden sunset over ocean horizon.", "", "", "", ""),
            ],
        )

        soft_query = _build_fts_soft_match_query("striped cardigan bathtub sunset")
        rows = gallery_db.execute(
            """
            SELECT rowid
            FROM search_index
            WHERE search_index MATCH ?
            ORDER BY bm25(search_index, 0.0, 3.8, 3.2, 2.2, 2.6, 0.6)
            """,
            [soft_query],
        ).fetchall()

        assert '"striped"' in soft_query
        assert '"cardigan"' in soft_query
        assert '"bathtub"' in soft_query
        assert [row[0] for row in rows][:1] == [1]
    finally:
        gallery_db.close()


def test_hybrid_search_semantic_primary_keeps_lexical_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imganalyzer.db.search import SearchEngine

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        engine = SearchEngine(conn)
        monkeypatch.setattr(
            engine,
            "_fts_search",
            lambda _query, _limit, **_kw: [{"image_id": 101, "file_path": "/fts.jpg", "score": 0.9}],
        )
        monkeypatch.setattr(
            engine,
            "_semantic_search",
            lambda _query, _limit, _profile=None, **_kw: [{"image_id": 202, "file_path": "/sem.jpg", "score": 0.8}],
        )

        results = engine._hybrid_search("striped cardigan", limit=20, semantic_weight=0.5)
        ids = {item["image_id"] for item in results}
        assert ids == {101, 202}
    finally:
        conn.close()


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
            semantic_profile: str | None = None,
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
            semantic_profile: str | None = None,
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


def test_search_face_and_text_queries_rerank_within_full_face_candidate_set(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for image_id in range(1, 251):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\people\img{image_id}.jpg")

    seen_candidate_ids: list[set[int] | None] = []

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def resolve_face_queries(self, query: str) -> tuple[list[str], str, str]:
            if query == "banban at sunset":
                return ["banban"], "at sunset", "all"
            return [], query, "all"

        def resolve_face_query(self, query: str) -> tuple[str | None, str]:
            return None, query

        def search_face(self, name: str, limit: int | None = 50) -> list[dict[str, object]]:
            assert name == "banban"
            upper = 250 if limit is None or limit <= 0 else min(int(limit), 250)
            return [
                {"image_id": image_id, "score": 1.0}
                for image_id in range(1, upper + 1)
            ]

        def search(
            self,
            query: str,
            limit: int,
            semantic_weight: float = 0.5,
            mode: str = "hybrid",
            semantic_profile: str | None = None,
            candidate_ids: set[int] | None = None,
        ) -> list[dict[str, object]]:
            assert query == "at sunset"
            seen_candidate_ids.append(candidate_ids)
            return [{"image_id": 250, "score": 0.99}]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)
    result = server._handle_search(
        {"query": "banban at sunset", "mode": "hybrid", "limit": 10, "offset": 0}
    )

    assert result["total"] == 1
    assert result["hasMore"] is False
    assert [item["image_id"] for item in result["results"]] == [250]
    assert seen_candidate_ids == [set(range(1, 251))]


def test_search_aesthetic_sort_uses_broader_candidate_pool(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for image_id in range(1, 251):
        _insert_processed_image(gallery_db, image_id, fr"E:\Pic\birds\img{image_id}.jpg")
        gallery_db.execute(
            """
            INSERT INTO analysis_aesthetic (image_id, aesthetic_score, aesthetic_label, aesthetic_reason)
            VALUES (?, ?, ?, ?)
            """,
            (
                image_id,
                9.8 if image_id == 250 else 1.0,
                "high" if image_id == 250 else "low",
                "seeded for ranking test",
            ),
        )

    search_limits: list[int] = []

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(
            self,
            query: str,
            limit: int,
            semantic_weight: float = 0.5,
            mode: str = "hybrid",
            semantic_profile: str | None = None,
        ) -> list[dict[str, object]]:
            assert query == "flock of birds"
            search_limits.append(limit)
            return [
                {
                    "image_id": image_id,
                    "file_path": fr"E:\Pic\birds\img{image_id}.jpg",
                    "score": float(251 - image_id),
                }
                for image_id in range(1, 251)
            ][:limit]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)
    result = server._handle_search(
        {"query": "flock of birds", "mode": "hybrid", "sortBy": "aesthetic", "limit": 1, "offset": 0}
    )

    assert search_limits == [400]
    assert result["total"] == 250
    assert result["hasMore"] is True
    assert [item["image_id"] for item in result["results"]] == [250]


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


def test_search_results_include_face_clusters_when_occurrences_exist(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\clustered.jpg")
    gallery_db.execute("INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)", (7, "Chen XC", None))
    gallery_db.execute(
        "INSERT INTO face_cluster_labels (cluster_id, display_name) VALUES (?, ?)",
        (10, "Chen XC"),
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=1,
        image_id=1,
        identity_name="chen_xc_child",
        cluster_id=10,
        person_id=7,
    )
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=2,
        image_id=1,
        identity_name="chen_xc_child",
        cluster_id=10,
        person_id=7,
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({"mode": "browse", "limit": 10, "offset": 0})

    assert result["total"] == 1
    assert len(result["results"]) == 1
    item = result["results"][0]
    assert item["face_clusters"] == [
        {
            "cluster_id": 10,
            "cluster_label": "Chen XC",
            "person_id": 7,
            "person_name": "Chen XC",
            "face_count": 2,
        }
    ]


def test_image_details_include_face_clusters_when_occurrences_exist(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\clustered-detail.jpg")
    gallery_db.execute("INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)", (3, "Alice", None))
    _insert_face_occurrence(
        gallery_db,
        occurrence_id=11,
        image_id=1,
        identity_name="alice_child",
        cluster_id=22,
        person_id=3,
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_image_details({"image_id": 1})

    assert result["result"] is not None
    clusters = result["result"]["face_clusters"]
    assert clusters == [
        {
            "cluster_id": 22,
            "cluster_label": "Cluster 22",
            "person_id": 3,
            "person_name": "Alice",
            "face_count": 1,
        }
    ]


def test_image_details_face_clusters_null_without_occurrence_table(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery_db.execute("DROP TABLE face_occurrences")
    _insert_processed_image(gallery_db, 1, r"E:\Pic\legacy\no-occurrences.jpg")
    try:
        monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
        result = server._handle_image_details({"image_id": 1})
        assert result["result"] is not None
        assert result["result"]["face_clusters"] is None
    finally:
        gallery_db.execute(
            """
            CREATE TABLE face_occurrences (
                id INTEGER PRIMARY KEY,
                image_id INTEGER NOT NULL,
                identity_name TEXT,
                cluster_id INTEGER,
                person_id INTEGER
            )
            """
        )


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
            semantic_profile: str | None = None,
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


def test_search_rank_preference_recency_maps_to_newest_sort(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\rank\img1.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\rank\img2.jpg")
    gallery_db.execute(
        "UPDATE analysis_metadata SET date_time_original = ? WHERE image_id = ?",
        ("2025-01-01T08:00:00", 1),
    )
    gallery_db.execute(
        "UPDATE analysis_metadata SET date_time_original = ? WHERE image_id = ?",
        ("2026-01-01T08:00:00", 2),
    )

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(
            self,
            query: str,
            limit: int,
            semantic_weight: float = 0.5,
            mode: str = "hybrid",
            semantic_profile: str | None = None,
        ) -> list[dict[str, object]]:
            return [
                {"image_id": 1, "score": 0.99},
                {"image_id": 2, "score": 0.50},
            ][:limit]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)
    result = server._handle_search({
        "query": "city",
        "rankPreference": "recency",
        "limit": 10,
        "offset": 0,
    })

    assert result["total"] == 2
    assert [item["image_id"] for item in result["results"]] == [2, 1]


def test_search_rejects_invalid_rank_preference(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    with pytest.raises(ValueError, match="rankPreference must be one of"):
        server._handle_search({"query": "test", "rankPreference": "fastest"})


def test_search_rejects_invalid_semantic_profile(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    with pytest.raises(ValueError, match="semanticProfile must be one of"):
        server._handle_search({"query": "test", "semanticProfile": "desc_only"})


def test_search_rejects_invalid_must_terms_shape(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    with pytest.raises(ValueError, match="mustTerms must be an array"):
        server._handle_search({"query": "test", "mustTerms": "ball"})
    with pytest.raises(ValueError, match="mustTerms entries must be strings"):
        server._handle_search({"query": "test", "mustTerms": ["ball", 2]})


def test_search_must_terms_support_person_identity_resolution(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\people\cxc.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\people\other.jpg")
    gallery_db.execute("INSERT INTO face_persons (id, name, notes) VALUES (?, ?, ?)", (1, "cxc", None))
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
        identity_name="someone_else",
        cluster_id=20,
        person_id=None,
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({
        "mustTerms": ["cxc"],
        "mode": "hybrid",
        "limit": 10,
        "offset": 0,
    })

    assert result["total"] == 1
    assert [item["image_id"] for item in result["results"]] == [1]


def test_search_passes_semantic_profile_to_engine_search(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\semantic\img1.jpg")
    seen_profiles: list[str | None] = []

    class FakeSearchEngine:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def search(
            self,
            query: str,
            limit: int,
            semantic_weight: float = 0.5,
            mode: str = "hybrid",
            semantic_profile: str | None = None,
        ) -> list[dict[str, object]]:
            seen_profiles.append(semantic_profile)
            return [{"image_id": 1, "score": 0.99}]

    import imganalyzer.db.search as search_module

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    monkeypatch.setattr(search_module, "SearchEngine", FakeSearchEngine)
    result = server._handle_search(
        {
            "query": "portrait",
            "mode": "semantic",
            "semanticProfile": "description_dominant",
            "limit": 10,
            "offset": 0,
        }
    )

    assert result["total"] == 1
    assert [item["image_id"] for item in result["results"]] == [1]
    assert seen_profiles and seen_profiles[0] == "description_dominant"


def test_search_text_mode_handles_partial_description_query(
    gallery_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_processed_image(gallery_db, 1, r"E:\Pic\desc\target.jpg")
    _insert_processed_image(gallery_db, 2, r"E:\Pic\desc\other.jpg")
    _insert_search_index_row(
        gallery_db,
        image_id=1,
        description_text=(
            "A woman sits comfortably inside a large freestanding white bathtub in "
            "a modern bathroom. She wears a colorful striped dress under a black cardigan."
        ),
    )
    _insert_search_index_row(
        gallery_db,
        image_id=2,
        description_text="A sunset landscape over ocean water.",
    )

    monkeypatch.setattr(server, "_get_db", lambda: gallery_db)
    result = server._handle_search({
        "query": "striped cardigan bathtub sunset",
        "mode": "text",
        "limit": 10,
        "offset": 0,
    })

    assert result["total"] == 2
    assert [item["image_id"] for item in result["results"]][:1] == [1]
