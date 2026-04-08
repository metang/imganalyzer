"""Tests for the storyline module — smart albums, clustering, heroes, and evaluator."""
from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from datetime import datetime, timedelta

import numpy as np
import pytest

from imganalyzer.db.schema import ensure_schema


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_test_db(tmp_path):
    """Create a fresh SQLite DB with schema for testing."""
    db_path = tmp_path / "test_storyline.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def _insert_image(conn, image_id: int, file_path: str = "") -> int:
    """Insert a minimal image row."""
    if not file_path:
        file_path = f"/photos/img_{image_id}.jpg"
    conn.execute(
        "INSERT OR IGNORE INTO images (id, file_path, file_hash, file_size) "
        "VALUES (?, ?, ?, ?)",
        [image_id, file_path, f"hash_{image_id}", 1000],
    )
    return image_id


def _insert_metadata(
    conn,
    image_id: int,
    *,
    dt: str | None = None,
    lat: float | None = None,
    lng: float | None = None,
    city: str | None = None,
    country: str | None = None,
    camera_make: str | None = None,
    geohash: str | None = None,
) -> None:
    """Insert analysis_metadata for an image."""
    conn.execute(
        "INSERT OR REPLACE INTO analysis_metadata "
        "(image_id, date_time_original, gps_latitude, gps_longitude, "
        " location_city, location_country, camera_make, geohash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [image_id, dt, lat, lng, city, country, camera_make, geohash],
    )


def _insert_search_features(
    conn,
    image_id: int,
    *,
    dt: str | None = None,
    country: str | None = None,
    perception_iaa: float = 0.5,
    sharpness: float = 0.5,
    face_count: int = 0,
) -> None:
    """Insert search_features for an image."""
    conn.execute(
        "INSERT OR REPLACE INTO search_features "
        "(image_id, date_time_original, location_country, perception_iaa, "
        " sharpness_score, face_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [image_id, dt, country, perception_iaa, sharpness, face_count],
    )


def _insert_embedding(conn, image_id: int, clip_vec: list[float] | None = None) -> None:
    """Insert a CLIP embedding for an image."""
    if clip_vec is None:
        clip_vec = list(np.random.randn(512).astype(np.float32))
    blob = struct.pack(f"<{len(clip_vec)}f", *clip_vec)
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (image_id, embedding_type, vector) "
        "VALUES (?, 'image_clip', ?)",
        [image_id, blob],
    )


def _insert_face_occurrence(
    conn, image_id: int, person_id: int | None = None, cluster_id: int = 1
) -> None:
    """Insert a face occurrence."""
    emb = np.zeros(512, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO face_occurrences "
        "(image_id, face_idx, embedding, cluster_id, person_id, "
        " bbox_x1, bbox_y1, bbox_x2, bbox_y2) "
        "VALUES (?, 0, ?, ?, ?, 0.1, 0.1, 0.5, 0.5)",
        [image_id, emb, cluster_id, person_id],
    )


def _create_person(conn, name: str) -> int:
    """Create a face_persons entry and return person_id (integer)."""
    cur = conn.execute(
        "INSERT INTO face_persons (name) VALUES (?)",
        [name],
    )
    return cur.lastrowid  # type: ignore[return-value]


def _seed_timeline_album(conn, n_images: int = 20, days_span: int = 30):
    """Seed a test dataset with images spanning time and locations.

    Returns (album_id, person_id, image_ids).
    """
    from imganalyzer.storyline.albums import create_album

    person_id = _create_person(conn, "Alice")

    base_dt = datetime(2024, 1, 1, 10, 0, 0)
    locations = [
        ("New York", "US", 40.7128, -74.0060, "dr5reg"),
        ("Paris", "FR", 48.8566, 2.3522, "u09tvw"),
        ("Tokyo", "JP", 35.6762, 139.6503, "xn76ur"),
    ]

    image_ids = []
    for i in range(n_images):
        iid = i + 1
        _insert_image(conn, iid)

        # Cycle through locations, advance time
        loc_idx = (i * 3) // n_images  # first third NYC, second Paris, third Tokyo
        loc = locations[min(loc_idx, len(locations) - 1)]
        dt = base_dt + timedelta(days=(i * days_span) // n_images, hours=i % 5)

        dt_str = dt.isoformat()
        _insert_metadata(
            conn, iid,
            dt=dt_str, lat=loc[2], lng=loc[3],
            city=loc[0], country=loc[1], geohash=loc[4],
        )
        _insert_search_features(
            conn, iid,
            dt=dt_str, country=loc[1],
            perception_iaa=0.3 + (i % 10) * 0.07,
            sharpness=0.4 + (i % 8) * 0.05,
            face_count=1 if i % 3 == 0 else 0,
        )
        _insert_embedding(conn, iid)
        _insert_face_occurrence(conn, iid, person_id=person_id, cluster_id=1)
        image_ids.append(iid)

    conn.commit()
    return person_id, image_ids


# ── Test: Schema Migration v32 ───────────────────────────────────────────────

class TestSchemaV32:
    def test_tables_created(self, tmp_path):
        conn = _make_test_db(tmp_path)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "smart_albums" in tables
        assert "album_items" in tables
        assert "story_chapters" in tables
        assert "story_moments" in tables
        assert "moment_images" in tables

    def test_schema_version_is_32(self, tmp_path):
        conn = _make_test_db(tmp_path)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == 32


# ── Test: Rule Compiler ──────────────────────────────────────────────────────

class TestRuleCompiler:
    def test_person_rule_single(self, tmp_path):
        from imganalyzer.storyline.rules import compile_rules

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": ["p1"], "mode": "any"}
        ]}
        sql, params = compile_rules(rules)
        assert "face_occurrences" in sql
        assert "p1" in params

    def test_person_rule_cooccurrence(self, tmp_path):
        from imganalyzer.storyline.rules import compile_rules

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": ["p1", "p2"], "mode": "all"}
        ]}
        sql, params = compile_rules(rules)
        # Should have two JOINs for co-occurrence
        assert sql.count("face_occurrences") == 2
        assert "p1" in params
        assert "p2" in params

    def test_date_range_rule(self, tmp_path):
        from imganalyzer.storyline.rules import compile_rules

        rules = {"match": "all", "rules": [
            {"type": "date_range", "start": "2024-01-01", "end": "2024-12-31"}
        ]}
        sql, params = compile_rules(rules)
        assert "date_time_original" in sql
        assert "2024-01-01" in params

    def test_location_rule(self, tmp_path):
        from imganalyzer.storyline.rules import compile_rules

        rules = {"match": "all", "rules": [
            {"type": "location", "country": "US"}
        ]}
        sql, params = compile_rules(rules)
        assert "location_country" in sql

    def test_combined_rules(self, tmp_path):
        from imganalyzer.storyline.rules import compile_rules

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": ["p1"]},
            {"type": "date_range", "start": "2024-01-01"},
            {"type": "location", "country": "US"},
        ]}
        sql, params = compile_rules(rules)
        assert "face_occurrences" in sql
        assert "search_features" in sql
        assert "INTERSECT" in sql

    def test_evaluate_rules_returns_matching_ids(self, tmp_path):
        from imganalyzer.storyline.rules import evaluate_rules

        conn = _make_test_db(tmp_path)
        pid = _create_person(conn, "Bob")
        for i in range(1, 4):
            _insert_image(conn, i)
            _insert_face_occurrence(conn, i, person_id=pid)
        # Image 4 has no face occurrence for pid
        _insert_image(conn, 4)
        conn.commit()

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [pid]}
        ]}
        ids = evaluate_rules(conn, rules)
        assert sorted(ids) == [1, 2, 3]

    def test_any_match_uses_union_semantics(self, tmp_path):
        from imganalyzer.storyline.rules import evaluate_rules

        conn = _make_test_db(tmp_path)
        pid = _create_person(conn, "Bob")

        _insert_image(conn, 1)
        _insert_face_occurrence(conn, 1, person_id=pid)

        _insert_image(conn, 2)
        _insert_search_features(conn, 2, country="US")

        _insert_image(conn, 3)
        _insert_face_occurrence(conn, 3, person_id=pid)
        _insert_search_features(conn, 3, country="US")

        rules = {"match": "any", "rules": [
            {"type": "person", "person_ids": [pid]},
            {"type": "location", "country": "US"},
        ]}
        ids = evaluate_rules(conn, rules)
        assert sorted(ids) == [1, 2, 3]

    def test_empty_rules_returns_nothing(self, tmp_path):
        from imganalyzer.storyline.rules import compile_rules

        sql, params = compile_rules({"match": "all", "rules": []})
        assert "WHERE 0" in sql

    def test_json_string_input(self, tmp_path):
        from imganalyzer.storyline.rules import compile_rules

        rules_json = json.dumps({"match": "all", "rules": [
            {"type": "location", "country": "JP"}
        ]})
        sql, params = compile_rules(rules_json)
        assert "JP" in params


# ── Test: Album CRUD ─────────────────────────────────────────────────────────

class TestAlbumCRUD:
    def test_create_and_get(self, tmp_path):
        from imganalyzer.storyline.albums import create_album, get_album

        conn = _make_test_db(tmp_path)
        pid = _create_person(conn, "Alice")
        for i in range(1, 6):
            _insert_image(conn, i)
            _insert_search_features(conn, i, perception_iaa=0.5 + i * 0.05)
            _insert_face_occurrence(conn, i, person_id=pid)
        conn.commit()

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [pid]}
        ]}
        album = create_album(conn, "Alice's Story", rules, description="Test")
        assert album.name == "Alice's Story"
        assert album.item_count == 5
        assert album.cover_image_id is not None

        fetched = get_album(conn, album.id)
        assert fetched is not None
        assert fetched.id == album.id

    def test_list_albums(self, tmp_path):
        from imganalyzer.storyline.albums import create_album, list_albums

        conn = _make_test_db(tmp_path)
        _insert_image(conn, 1)
        conn.commit()

        create_album(conn, "Album 1", {"match": "all", "rules": []})
        create_album(conn, "Album 2", {"match": "all", "rules": []})
        albums = list_albums(conn)
        assert len(albums) == 2

    def test_delete_album(self, tmp_path):
        from imganalyzer.storyline.albums import create_album, delete_album, get_album

        conn = _make_test_db(tmp_path)
        _insert_image(conn, 1)
        conn.commit()

        album = create_album(conn, "Temp", {"match": "all", "rules": []})
        assert delete_album(conn, album.id)
        assert get_album(conn, album.id) is None

    def test_refresh_membership(self, tmp_path):
        from imganalyzer.storyline.albums import (
            create_album, get_album_image_ids, refresh_membership,
        )

        conn = _make_test_db(tmp_path)
        pid = _create_person(conn, "Alice")
        for i in range(1, 4):
            _insert_image(conn, i)
            _insert_face_occurrence(conn, i, person_id=pid)
        conn.commit()

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [pid]}
        ]}
        album = create_album(conn, "Alice", rules)
        assert len(get_album_image_ids(conn, album.id)) == 3

        # Add a new image
        _insert_image(conn, 4)
        _insert_face_occurrence(conn, 4, person_id=pid)
        conn.commit()

        count = refresh_membership(conn, album.id)
        assert count == 4
        assert len(get_album_image_ids(conn, album.id)) == 4

    def test_check_image_against_rules(self, tmp_path):
        from imganalyzer.storyline.albums import check_image_against_rules

        conn = _make_test_db(tmp_path)
        pid = _create_person(conn, "Alice")
        _insert_image(conn, 1)
        _insert_face_occurrence(conn, 1, person_id=pid)
        _insert_image(conn, 2)  # no face
        conn.commit()

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [pid]}
        ]}
        assert check_image_against_rules(conn, 1, rules) is True
        assert check_image_against_rules(conn, 2, rules) is False


# ── Test: Moment Clustering ──────────────────────────────────────────────────

class TestMomentClustering:
    def test_same_time_same_place_one_moment(self):
        from imganalyzer.storyline.clustering import ImagePoint, cluster_moments

        points = [
            ImagePoint(1, datetime(2024, 1, 1, 10, 0), 40.7, -74.0, "dr5reg"),
            ImagePoint(2, datetime(2024, 1, 1, 10, 10), 40.7, -74.0, "dr5reg"),
            ImagePoint(3, datetime(2024, 1, 1, 10, 20), 40.7, -74.0, "dr5reg"),
        ]
        moments = cluster_moments(points)
        assert len(moments) == 1
        assert len(moments[0].images) == 3

    def test_time_gap_splits_moments(self):
        from imganalyzer.storyline.clustering import ImagePoint, cluster_moments

        points = [
            ImagePoint(1, datetime(2024, 1, 1, 10, 0), 40.7, -74.0, "dr5reg"),
            ImagePoint(2, datetime(2024, 1, 1, 10, 10), 40.7, -74.0, "dr5reg"),
            # 2 hour gap
            ImagePoint(3, datetime(2024, 1, 1, 12, 10), 40.7, -74.0, "dr5reg"),
        ]
        moments = cluster_moments(points)
        assert len(moments) == 2

    def test_location_change_splits_moments(self):
        from imganalyzer.storyline.clustering import ImagePoint, cluster_moments

        points = [
            ImagePoint(1, datetime(2024, 1, 1, 10, 0), 40.7, -74.0, "dr5reg"),
            ImagePoint(2, datetime(2024, 1, 1, 10, 10), 48.8, 2.3, "u09tvw"),
        ]
        moments = cluster_moments(points)
        assert len(moments) == 2

    def test_no_gps_groups_by_time(self):
        from imganalyzer.storyline.clustering import ImagePoint, cluster_moments

        points = [
            ImagePoint(1, datetime(2024, 1, 1, 10, 0), None, None, ""),
            ImagePoint(2, datetime(2024, 1, 1, 10, 15), None, None, ""),
            # 2h gap
            ImagePoint(3, datetime(2024, 1, 1, 12, 15), None, None, ""),
        ]
        moments = cluster_moments(points)
        assert len(moments) == 2

    def test_untimed_images_get_separate_moment(self):
        from imganalyzer.storyline.clustering import ImagePoint, cluster_moments

        points = [
            ImagePoint(1, datetime(2024, 1, 1, 10, 0), None, None, ""),
            ImagePoint(2, None, None, None, ""),
            ImagePoint(3, None, None, None, ""),
        ]
        moments = cluster_moments(points)
        assert len(moments) == 2  # 1 timed + 1 untimed

    def test_empty_input(self):
        from imganalyzer.storyline.clustering import cluster_moments

        assert cluster_moments([]) == []

    def test_moment_timestamps_set(self):
        from imganalyzer.storyline.clustering import ImagePoint, cluster_moments

        t1 = datetime(2024, 1, 1, 10, 0)
        t2 = datetime(2024, 1, 1, 10, 20)
        points = [
            ImagePoint(1, t1, None, None, ""),
            ImagePoint(2, t2, None, None, ""),
        ]
        moments = cluster_moments(points)
        assert moments[0].start_time == t1
        assert moments[0].end_time == t2


# ── Test: Chapter Detection ──────────────────────────────────────────────────

class TestChapterDetection:
    def test_time_gap_creates_chapter(self):
        from imganalyzer.storyline.clustering import (
            ImagePoint, Moment, detect_chapters,
        )

        m1 = Moment([ImagePoint(1, datetime(2024, 1, 1, 10, 0), None, None, "")])
        m2 = Moment([ImagePoint(2, datetime(2024, 1, 1, 11, 0), None, None, "")])
        # 8 hour gap → new chapter
        m3 = Moment([ImagePoint(3, datetime(2024, 1, 1, 19, 0), None, None, "")])

        chapters = detect_chapters([m1, m2, m3], time_gap=timedelta(hours=4))
        assert len(chapters) == 2
        assert len(chapters[0].moments) == 2
        assert len(chapters[1].moments) == 1

    def test_year_boundary_creates_chapter(self):
        from imganalyzer.storyline.clustering import (
            ImagePoint, Moment, detect_chapters,
        )

        m1 = Moment([ImagePoint(1, datetime(2023, 12, 31, 23, 0), None, None, "")])
        m2 = Moment([ImagePoint(2, datetime(2024, 1, 1, 0, 30), None, None, "")])

        chapters = detect_chapters([m1, m2], force_year_breaks=True)
        assert len(chapters) == 2

    def test_distance_gap_creates_chapter(self):
        from imganalyzer.storyline.clustering import (
            ImagePoint, Moment, detect_chapters,
        )

        m1 = Moment([ImagePoint(1, datetime(2024, 1, 1, 10, 0), 40.7, -74.0, "")])
        # Same time but 5000km away
        m2 = Moment([ImagePoint(2, datetime(2024, 1, 1, 11, 0), 48.8, 2.3, "")])

        chapters = detect_chapters([m1, m2], distance_gap_km=50.0)
        assert len(chapters) == 2

    def test_chapter_dates_computed(self):
        from imganalyzer.storyline.clustering import (
            ImagePoint, Moment, detect_chapters,
        )

        t1 = datetime(2024, 6, 1, 10, 0)
        t2 = datetime(2024, 6, 1, 12, 0)
        m1 = Moment([ImagePoint(1, t1, None, None, "")])
        m2 = Moment([ImagePoint(2, t2, None, None, "")])

        chapters = detect_chapters([m1, m2])
        assert chapters[0].start_date == t1
        assert chapters[0].end_date == t2


# ── Test: Chapter Title Generation ───────────────────────────────────────────

class TestChapterTitles:
    def test_title_with_location_and_date(self, tmp_path):
        from imganalyzer.storyline.clustering import (
            Chapter, ImagePoint, Moment, generate_chapter_title,
        )

        conn = _make_test_db(tmp_path)
        _insert_image(conn, 1)
        _insert_metadata(conn, 1, city="Paris", country="FR")
        conn.commit()

        m = Moment([ImagePoint(1, datetime(2024, 6, 15), 48.8, 2.3, "")])
        ch = Chapter([m])
        title = generate_chapter_title(ch, conn)
        assert "Paris" in title
        assert "2024" in title

    def test_title_without_location(self, tmp_path):
        from imganalyzer.storyline.clustering import (
            Chapter, ImagePoint, Moment, generate_chapter_title,
        )

        conn = _make_test_db(tmp_path)
        m = Moment([ImagePoint(1, datetime(2024, 6, 15), None, None, "")])
        ch = Chapter([m])
        title = generate_chapter_title(ch, None)
        assert "Jun" in title or "2024" in title

    def test_untitled_fallback(self):
        from imganalyzer.storyline.clustering import Chapter, Moment, generate_chapter_title

        ch = Chapter([Moment([])])
        title = generate_chapter_title(ch)
        assert title == "Untitled Chapter"


# ── Test: Hero Selection ─────────────────────────────────────────────────────

class TestHeroSelection:
    def test_single_image_moment(self, tmp_path):
        from imganalyzer.storyline.heroes import select_moment_hero

        conn = _make_test_db(tmp_path)
        _insert_image(conn, 1)
        _insert_search_features(conn, 1, perception_iaa=0.8, sharpness=0.7)
        conn.commit()

        hero = select_moment_hero(conn, [1])
        assert hero == 1

    def test_picks_highest_quality(self, tmp_path):
        from imganalyzer.storyline.heroes import select_moment_hero

        conn = _make_test_db(tmp_path)
        for i in range(1, 4):
            _insert_image(conn, i)
            _insert_search_features(
                conn, i,
                perception_iaa=0.3 + i * 0.2,
                sharpness=0.5,
                face_count=1 if i == 3 else 0,
            )
        conn.commit()

        hero = select_moment_hero(conn, [1, 2, 3])
        assert hero == 3  # highest aesthetic + has face

    def test_chapter_heroes_diverse(self, tmp_path):
        from imganalyzer.storyline.heroes import select_chapter_heroes

        conn = _make_test_db(tmp_path)
        # Create 3 moments with distinct embeddings
        for i in range(1, 7):
            _insert_image(conn, i)
            _insert_search_features(conn, i, perception_iaa=0.7, sharpness=0.6)
            # Create distinct embeddings for diversity
            vec = [0.0] * 512
            vec[i % 512] = 1.0
            _insert_embedding(conn, i, vec)
        conn.commit()

        moments = [
            ("m1", [1, 2]),
            ("m2", [3, 4]),
            ("m3", [5, 6]),
        ]
        heroes = select_chapter_heroes(conn, moments)
        assert len(heroes) == 3
        assert all(h in range(1, 7) for h in heroes.values())

    def test_empty_moment(self, tmp_path):
        from imganalyzer.storyline.heroes import select_moment_hero

        conn = _make_test_db(tmp_path)
        assert select_moment_hero(conn, []) is None


# ── Test: Story Generator (end-to-end) ───────────────────────────────────────

class TestStoryGenerator:
    def test_generate_story_basic(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.generator import generate_story

        conn = _make_test_db(tmp_path)
        person_id, image_ids = _seed_timeline_album(conn, n_images=20, days_span=30)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Alice's Story", rules)

        result = generate_story(conn, album.id)
        assert result.image_count == 20
        assert result.moment_count > 0
        assert result.chapter_count > 0

    def test_story_has_chapters_and_moments(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.generator import (
            generate_story, get_story_chapters, get_chapter_moments,
        )

        conn = _make_test_db(tmp_path)
        person_id, _ = _seed_timeline_album(conn, n_images=15, days_span=60)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Test", rules)
        generate_story(conn, album.id)

        chapters = get_story_chapters(conn, album.id)
        assert len(chapters) > 0
        assert chapters[0]["title"] is not None

        moments = get_chapter_moments(conn, chapters[0]["id"])
        assert len(moments) > 0
        assert moments[0]["hero_image_id"] is not None

    def test_regenerate_clears_old_story(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.generator import (
            generate_story, get_story_chapters,
        )

        conn = _make_test_db(tmp_path)
        person_id, _ = _seed_timeline_album(conn, n_images=10, days_span=10)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Test", rules)

        r1 = generate_story(conn, album.id)
        ch1 = get_story_chapters(conn, album.id)

        r2 = generate_story(conn, album.id)
        ch2 = get_story_chapters(conn, album.id)

        # Story was regenerated, chapter IDs should differ
        assert ch1[0]["id"] != ch2[0]["id"]
        assert r2.chapter_count == r1.chapter_count

    def test_empty_album_generates_empty_story(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.generator import generate_story

        conn = _make_test_db(tmp_path)
        rules = {"match": "all", "rules": []}
        album = create_album(conn, "Empty", rules)

        result = generate_story(conn, album.id)
        assert result.image_count == 0
        assert result.chapter_count == 0


# ── Test: Evaluator ──────────────────────────────────────────────────────────

class TestEvaluator:
    def test_evaluator_runs_on_generated_story(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.evaluator import evaluate_story
        from imganalyzer.storyline.generator import generate_story

        conn = _make_test_db(tmp_path)
        person_id, _ = _seed_timeline_album(conn, n_images=20, days_span=30)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Alice", rules)
        generate_story(conn, album.id)

        report = evaluate_story(conn, album.id)
        assert len(report.criteria) > 0
        # Check key criteria exist
        ids = {c.id for c in report.criteria}
        assert "A1" in ids  # precision
        assert "B1" in ids  # temporal tightness
        assert "C1" in ids  # inter-chapter gap
        assert "D4" in ids  # no broken heroes
        assert "E1" in ids  # title present

    def test_evaluator_produces_report_dict(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.evaluator import evaluate_story
        from imganalyzer.storyline.generator import generate_story

        conn = _make_test_db(tmp_path)
        person_id, _ = _seed_timeline_album(conn, n_images=15, days_span=20)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Test", rules)
        generate_story(conn, album.id)

        report = evaluate_story(conn, album.id)
        d = report.to_dict()
        assert "overall_pass" in d
        assert "criteria" in d
        assert isinstance(d["criteria"], dict)

    def test_evaluator_summary_string(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.evaluator import evaluate_story
        from imganalyzer.storyline.generator import generate_story

        conn = _make_test_db(tmp_path)
        person_id, _ = _seed_timeline_album(conn, n_images=10, days_span=15)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Test", rules)
        generate_story(conn, album.id)

        report = evaluate_story(conn, album.id)
        summary = report.summary()
        assert "Story Evaluation" in summary
        assert "A1" in summary

    def test_nonexistent_album(self, tmp_path):
        from imganalyzer.storyline.evaluator import evaluate_story

        conn = _make_test_db(tmp_path)
        report = evaluate_story(conn, "nonexistent")
        assert not report.overall_pass

    def test_rule_idempotency(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.evaluator import evaluate_story
        from imganalyzer.storyline.generator import generate_story

        conn = _make_test_db(tmp_path)
        person_id, _ = _seed_timeline_album(conn, n_images=10, days_span=10)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Idempotent", rules)
        generate_story(conn, album.id)

        report = evaluate_story(conn, album.id)
        a4 = next(c for c in report.criteria if c.id == "A4")
        assert a4.passed, "Rule evaluation should be idempotent"


# ── Test: Presets ────────────────────────────────────────────────────────────

class TestPresets:
    def test_year_in_review(self, tmp_path):
        from imganalyzer.storyline.presets import create_year_in_review

        conn = _make_test_db(tmp_path)
        # Seed images in 2024
        for i in range(1, 6):
            _insert_image(conn, i)
            _insert_search_features(conn, i, dt=f"2024-0{i}-15T12:00:00")
        # Seed one image in 2023 — should NOT be included
        _insert_image(conn, 100)
        _insert_search_features(conn, 100, dt="2023-06-15T12:00:00")

        album = create_year_in_review(conn, year=2024)
        assert album.name == "2024 Year in Review"
        assert album.item_count == 5

    def test_year_in_review_includes_late_dec31(self, tmp_path):
        from imganalyzer.storyline.presets import create_year_in_review

        conn = _make_test_db(tmp_path)
        _insert_image(conn, 1)
        _insert_search_features(conn, 1, dt="2024-12-31T18:30:00")

        album = create_year_in_review(conn, year=2024)
        assert album.item_count == 1

    def test_year_in_review_default_year(self, tmp_path):
        from imganalyzer.storyline.presets import create_year_in_review

        conn = _make_test_db(tmp_path)
        album = create_year_in_review(conn)
        # Should default to previous year
        from datetime import timezone
        expected_year = datetime.now(timezone.utc).year - 1
        assert str(expected_year) in album.name

    def test_person_timeline(self, tmp_path):
        from imganalyzer.storyline.presets import create_person_timeline

        conn = _make_test_db(tmp_path)
        person_id = _create_person(conn, "Bob")

        for i in range(1, 4):
            _insert_image(conn, i)
            _insert_search_features(conn, i, dt=f"2024-01-{i:02d}T12:00:00")
            _insert_face_occurrence(conn, i, person_id=person_id, cluster_id=i)

        album = create_person_timeline(conn, person_id)
        assert "Bob" in album.name
        assert album.item_count == 3

    def test_person_timeline_auto_name(self, tmp_path):
        from imganalyzer.storyline.presets import create_person_timeline

        conn = _make_test_db(tmp_path)
        person_id = _create_person(conn, "Charlie")
        album = create_person_timeline(conn, person_id)
        assert "Charlie" in album.name

    def test_growth_story(self, tmp_path):
        from imganalyzer.storyline.presets import create_growth_story

        conn = _make_test_db(tmp_path)
        person_id = _create_person(conn, "Mia")
        for i in range(1, 3):
            _insert_image(conn, i)
            _insert_search_features(conn, i, dt=f"2024-01-{i:02d}T12:00:00")
            _insert_face_occurrence(conn, i, person_id=person_id, cluster_id=100 + i)

        album = create_growth_story(conn, person_id)
        assert album.item_count == 2
        assert "Growing Up" in album.name

    def test_together_album(self, tmp_path):
        from imganalyzer.storyline.presets import create_together_album

        conn = _make_test_db(tmp_path)
        pid_a = _create_person(conn, "Alice")
        pid_b = _create_person(conn, "Bob")

        # Images with both people
        for i in range(1, 4):
            _insert_image(conn, i)
            _insert_search_features(conn, i, dt=f"2024-01-{i:02d}T12:00:00")
            _insert_face_occurrence(conn, i, person_id=pid_a, cluster_id=i * 10)
            # Need a different face_idx for second person
            emb = np.zeros(512, dtype=np.float32).tobytes()
            conn.execute(
                "INSERT INTO face_occurrences "
                "(image_id, face_idx, embedding, cluster_id, person_id, "
                " bbox_x1, bbox_y1, bbox_x2, bbox_y2) "
                "VALUES (?, 1, ?, ?, ?, 0.5, 0.5, 0.9, 0.9)",
                [i, emb, i * 10 + 1, pid_b],
            )

        album = create_together_album(conn, [pid_a, pid_b])
        assert "Alice" in album.name and "Bob" in album.name
        assert album.item_count == 3

    def test_location_story(self, tmp_path):
        from imganalyzer.storyline.presets import create_location_story

        conn = _make_test_db(tmp_path)
        for i in range(1, 4):
            _insert_image(conn, i)
            _insert_search_features(conn, i, dt=f"2024-0{i}-15T12:00:00", country="JP")

        album = create_location_story(conn, country="JP")
        assert album.item_count == 3
        assert "JP" in album.name

    def test_location_with_city(self, tmp_path):
        from imganalyzer.storyline.presets import create_location_story

        conn = _make_test_db(tmp_path)
        for i in range(1, 3):
            _insert_image(conn, i)
            _insert_search_features(conn, i, dt=f"2024-01-{i:02d}T12:00:00")
            _insert_metadata(conn, i, dt=f"2024-01-{i:02d}T12:00:00", city="Tokyo", country="JP")

        album = create_location_story(conn, country="JP", city="Tokyo")
        assert "Tokyo" in album.name

    def test_on_this_day(self, tmp_path):
        from imganalyzer.storyline.presets import create_on_this_day

        conn = _make_test_db(tmp_path)
        # Images on March 15 across multiple years
        for yr in [2020, 2021, 2022]:
            iid = yr
            _insert_image(conn, iid)
            _insert_search_features(conn, iid, dt=f"{yr}-03-15T10:00:00")

        album = create_on_this_day(conn, month=3, day=15)
        assert "March 15" in album.name
        assert album.item_count == 3

    def test_preset_registry(self):
        from imganalyzer.storyline.presets import PRESET_REGISTRY

        assert "year_in_review" in PRESET_REGISTRY
        assert "on_this_day" in PRESET_REGISTRY
        assert "person_timeline" in PRESET_REGISTRY
        assert "growth_story" in PRESET_REGISTRY
        assert "together" in PRESET_REGISTRY
        assert "location" in PRESET_REGISTRY


# ── Test: Export ─────────────────────────────────────────────────────────────

class TestExport:
    def test_export_html_basic(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.export import export_story_html
        from imganalyzer.storyline.generator import generate_story

        conn = _make_test_db(tmp_path)
        person_id, _ = _seed_timeline_album(conn, n_images=10, days_span=10)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Export Test", rules)
        generate_story(conn, album.id)

        out = tmp_path / "story.html"
        result = export_story_html(
            conn, album.id, out, include_thumbnails=False,
        )
        assert result.exists()
        html = result.read_text(encoding="utf-8")
        assert "Export Test" in html
        assert "<!DOCTYPE html>" in html
        assert "chapter" in html.lower()

    def test_export_nonexistent_album(self, tmp_path):
        conn = _make_test_db(tmp_path)
        out = tmp_path / "bad.html"

        from imganalyzer.storyline.export import export_story_html
        with pytest.raises(ValueError, match="not found"):
            export_story_html(conn, "nonexistent", out)

    def test_export_empty_story(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.export import export_story_html
        from imganalyzer.storyline.generator import generate_story

        conn = _make_test_db(tmp_path)
        rules = {"match": "all", "rules": []}
        album = create_album(conn, "Empty Export", rules)
        generate_story(conn, album.id)

        out = tmp_path / "empty.html"
        result = export_story_html(
            conn, album.id, out, include_thumbnails=False,
        )
        html = result.read_text(encoding="utf-8")
        assert "Empty Export" in html
        assert "0 photos" in html


# ── Test: Incremental ────────────────────────────────────────────────────────

class TestIncremental:
    def test_check_and_add_image_matching(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.incremental import check_and_add_image

        conn = _make_test_db(tmp_path)
        person_id = _create_person(conn, "Dave")

        # Create album for person
        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Dave", rules)

        # Now add a new image matching the rule
        _insert_image(conn, 999)
        _insert_search_features(conn, 999, dt="2024-06-01T12:00:00")
        _insert_face_occurrence(conn, 999, person_id=person_id, cluster_id=999)

        added = check_and_add_image(conn, 999)
        assert album.id in added

        # Verify it's in album_items
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM album_items WHERE album_id=? AND image_id=999",
            [album.id],
        ).fetchone()
        assert row["cnt"] == 1

    def test_check_and_add_image_not_matching(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.incremental import check_and_add_image

        conn = _make_test_db(tmp_path)
        person_id = _create_person(conn, "Eve")

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        create_album(conn, "Eve", rules)

        # Image WITHOUT person_id match
        _insert_image(conn, 888)
        _insert_search_features(conn, 888, dt="2024-06-01T12:00:00")

        added = check_and_add_image(conn, 888)
        assert len(added) == 0

    def test_check_and_add_idempotent(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.incremental import check_and_add_image

        conn = _make_test_db(tmp_path)
        person_id = _create_person(conn, "Fay")

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Fay", rules)

        _insert_image(conn, 777)
        _insert_search_features(conn, 777, dt="2024-06-01T12:00:00")
        _insert_face_occurrence(conn, 777, person_id=person_id, cluster_id=777)

        added1 = check_and_add_image(conn, 777)
        added2 = check_and_add_image(conn, 777)
        # First call adds, second should not error
        assert album.id in added1

    def test_refresh_all_albums(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.incremental import refresh_all_albums

        conn = _make_test_db(tmp_path)
        person_id = _create_person(conn, "Gina")

        # Seed initial images
        for i in range(1, 4):
            _insert_image(conn, i)
            _insert_search_features(conn, i, dt=f"2024-01-{i:02d}T12:00:00")
            _insert_face_occurrence(conn, i, person_id=person_id, cluster_id=i)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Gina", rules)
        assert album.item_count == 3

        # Add 2 more images
        for i in range(4, 6):
            _insert_image(conn, i)
            _insert_search_features(conn, i, dt=f"2024-01-{i:02d}T12:00:00")
            _insert_face_occurrence(conn, i, person_id=person_id, cluster_id=i)

        counts = refresh_all_albums(conn)
        assert album.id in counts
        assert counts[album.id] == 5  # all 5 images now


# ── Test: Narrative ──────────────────────────────────────────────────────────

class TestNarrative:
    def test_ai_narrative_uses_ollama_text_generation(self, tmp_path, monkeypatch):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.generator import generate_story, get_story_chapters
        from imganalyzer.storyline.narrative import generate_all_chapter_narratives

        conn = _make_test_db(tmp_path)
        person_id, image_ids = _seed_timeline_album(conn, n_images=10, days_span=10)

        for image_id in image_ids:
            conn.execute(
                "INSERT OR REPLACE INTO analysis_caption (image_id, description, keywords) VALUES (?, ?, ?)",
                [image_id, f"Photo {image_id} at a family gathering", json.dumps(["family", "travel"])],
            )

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Narrative AI Test", rules)
        generate_story(conn, album.id)

        monkeypatch.setattr(
            "imganalyzer.analysis.ai.ollama.OllamaAI.generate_text",
            lambda self, prompt: "A concise AI summary.",
        )

        updated = generate_all_chapter_narratives(conn, album.id, use_ai=True)
        assert updated > 0

        chapters = get_story_chapters(conn, album.id)
        assert chapters[0]["summary"] == "A concise AI summary."

    def test_heuristic_narrative(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.generator import generate_story, get_story_chapters
        from imganalyzer.storyline.narrative import generate_all_chapter_narratives

        conn = _make_test_db(tmp_path)
        person_id, _ = _seed_timeline_album(conn, n_images=10, days_span=10)

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Narrative Test", rules)
        generate_story(conn, album.id)

        # Use heuristic (no AI) — should always work
        updated = generate_all_chapter_narratives(conn, album.id, use_ai=False)
        assert updated > 0

        chapters = get_story_chapters(conn, album.id)
        for ch in chapters:
            assert ch["summary"] is not None and len(ch["summary"]) > 0

    def test_narrative_on_empty_story(self, tmp_path):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.generator import generate_story
        from imganalyzer.storyline.narrative import generate_all_chapter_narratives

        conn = _make_test_db(tmp_path)
        rules = {"match": "all", "rules": []}
        album = create_album(conn, "Empty Narr", rules)
        generate_story(conn, album.id)

        updated = generate_all_chapter_narratives(conn, album.id, use_ai=False)
        assert updated == 0

    def test_use_ai_false_skips_ollama(self, tmp_path, monkeypatch):
        from imganalyzer.storyline.albums import create_album
        from imganalyzer.storyline.generator import generate_story, get_story_chapters
        from imganalyzer.storyline.narrative import generate_all_chapter_narratives

        conn = _make_test_db(tmp_path)
        person_id, image_ids = _seed_timeline_album(conn, n_images=8, days_span=8)

        for image_id in image_ids:
            conn.execute(
                "INSERT OR REPLACE INTO analysis_caption (image_id, description, keywords) VALUES (?, ?, ?)",
                [image_id, f"Story image {image_id}", json.dumps(["memory"])],
            )

        rules = {"match": "all", "rules": [
            {"type": "person", "person_ids": [person_id]}
        ]}
        album = create_album(conn, "Narrative Heuristic", rules)
        generate_story(conn, album.id)

        def _should_not_run(self, prompt):
            raise AssertionError("AI generation should not be called when use_ai=False")

        monkeypatch.setattr(
            "imganalyzer.analysis.ai.ollama.OllamaAI.generate_text",
            _should_not_run,
        )

        updated = generate_all_chapter_narratives(conn, album.id, use_ai=False)
        assert updated > 0

        chapters = get_story_chapters(conn, album.id)
        assert chapters[0]["summary"]
