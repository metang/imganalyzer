"""Tests for the geocode_cache table and async resolver."""
from __future__ import annotations

import sqlite3

import pytest

from imganalyzer.analysis import geocode_resolver
from imganalyzer.db.schema import SCHEMA_VERSION, ensure_schema


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "test.db"
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    ensure_schema(c)
    yield c
    c.close()


def test_migration_applied(conn):
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    assert row["version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION >= 33

    tbl = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='geocode_cache'"
    ).fetchone()
    assert tbl is not None

    cols = {row["name"] for row in conn.execute("PRAGMA table_info(geocode_cache)")}
    assert {"lat_key", "lon_key", "location_json", "fetched_at"} <= cols


def test_put_and_get_cached_location_round_trip(conn):
    loc = {
        "location_city": "San Francisco",
        "location_state": "California",
        "location_country": "United States",
        "location_country_code": "US",
    }
    geocode_resolver.put_cached_location(conn, 37.77493, -122.41942, loc)

    # Nearby coord within rounding precision (4dp ≈ 11 m) hits same row.
    got = geocode_resolver.get_cached_location(conn, 37.77493, -122.41942)
    assert got == loc

    # Distant coord misses.
    assert geocode_resolver.get_cached_location(conn, 40.0, -74.0) is None


def test_resolve_pending_uses_fetcher_and_updates_metadata(conn):
    # Seed images + analysis_metadata with GPS but no location.
    conn.execute(
        "INSERT INTO images (id, file_path, file_hash) VALUES (1, '/tmp/a.jpg', 'h1')"
    )
    conn.execute(
        "INSERT INTO images (id, file_path, file_hash) VALUES (2, '/tmp/b.jpg', 'h2')"
    )
    # Both images share the same rounded (lat, lon) key.
    conn.execute(
        "INSERT INTO analysis_metadata (image_id, gps_latitude, gps_longitude) "
        "VALUES (1, 37.77493, -122.41942)"
    )
    conn.execute(
        "INSERT INTO analysis_metadata (image_id, gps_latitude, gps_longitude) "
        "VALUES (2, 37.77494, -122.41941)"
    )
    # A third image with different coords; we won't fetch until budget allows.
    conn.execute(
        "INSERT INTO images (id, file_path, file_hash) VALUES (3, '/tmp/c.jpg', 'h3')"
    )
    conn.execute(
        "INSERT INTO analysis_metadata (image_id, gps_latitude, gps_longitude) "
        "VALUES (3, 40.0, -74.0)"
    )
    conn.commit()

    calls: list[tuple[float, float]] = []

    def fake_fetcher(lat: float, lon: float) -> dict[str, str]:
        calls.append((lat, lon))
        if lon < -100.0:
            return {
                "location_city": "San Francisco",
                "location_state": "California",
                "location_country": "United States",
                "location_country_code": "US",
            }
        return {
            "location_city": "New York",
            "location_state": "New York",
            "location_country": "United States",
            "location_country_code": "US",
        }

    updated = geocode_resolver.resolve_pending_locations(
        conn, limit=10, fetcher=fake_fetcher
    )
    assert updated == 3
    # Only 2 unique cache keys → only 2 fetcher calls.
    assert len(calls) == 2

    row1 = conn.execute(
        "SELECT location_city, location_country_code FROM analysis_metadata WHERE image_id = 1"
    ).fetchone()
    assert row1["location_city"] == "San Francisco"
    assert row1["location_country_code"] == "US"

    row3 = conn.execute(
        "SELECT location_city FROM analysis_metadata WHERE image_id = 3"
    ).fetchone()
    assert row3["location_city"] == "New York"

    # Re-running hits the cache; fetcher should NOT be called again.
    # First re-mark location as NULL to force the resolver to look again.
    conn.execute(
        "UPDATE analysis_metadata SET location_city = NULL, location_country = NULL "
        "WHERE image_id = 1"
    )
    conn.commit()
    calls.clear()
    updated2 = geocode_resolver.resolve_pending_locations(
        conn, limit=10, fetcher=fake_fetcher
    )
    assert updated2 == 1
    assert calls == []  # served from geocode_cache


def test_fetcher_failure_does_not_crash(conn):
    conn.execute(
        "INSERT INTO images (id, file_path, file_hash) VALUES (1, '/tmp/a.jpg', 'h1')"
    )
    conn.execute(
        "INSERT INTO analysis_metadata (image_id, gps_latitude, gps_longitude) "
        "VALUES (1, 10.0, 20.0)"
    )
    conn.commit()

    def empty_fetcher(lat: float, lon: float) -> dict[str, str]:
        return {}

    updated = geocode_resolver.resolve_pending_locations(
        conn, limit=10, fetcher=empty_fetcher
    )
    assert updated == 0
    # Cache entry is still written (as an empty dict) so we don't retry immediately.
    row = conn.execute("SELECT location_json FROM geocode_cache").fetchone()
    assert row is not None
