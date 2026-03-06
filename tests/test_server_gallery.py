"""Regression tests for gallery SQL filtering in JSON-RPC server handlers."""
from __future__ import annotations

import sqlite3
import sys
from collections.abc import Generator

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
CREATE TABLE analysis_blip2 (image_id INTEGER PRIMARY KEY);
CREATE TABLE analysis_objects (image_id INTEGER PRIMARY KEY);
CREATE TABLE analysis_ocr (image_id INTEGER PRIMARY KEY);
CREATE TABLE analysis_faces (image_id INTEGER PRIMARY KEY);
CREATE TABLE analysis_cloud_ai (image_id INTEGER PRIMARY KEY);
CREATE TABLE analysis_aesthetic (
    image_id INTEGER PRIMARY KEY,
    aesthetic_score REAL,
    aesthetic_label TEXT,
    aesthetic_reason TEXT
);
CREATE TABLE embeddings (image_id INTEGER PRIMARY KEY);
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
