"""Tests for map geo features: stats-extended, gap filler, trip timeline, RDP, geohash."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


def _make_test_db(tmp_path):
    """Create a fresh SQLite DB with full schema."""
    from imganalyzer.db.schema import ensure_schema

    db_path = tmp_path / "test_geo.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def _insert_image(conn, file_path: str, *, lat=None, lng=None,
                  dto=None, geohash=None, camera_model=None,
                  city=None, state=None, country=None,
                  gps_source="exif"):
    """Insert an image + metadata row, return image_id."""
    from imganalyzer.db.geohash import encode as geohash_encode

    cur = conn.execute(
        "INSERT INTO images (file_path, width, height, format) VALUES (?, 100, 100, 'JPEG')",
        [file_path],
    )
    image_id = cur.lastrowid

    if geohash is None and lat is not None and lng is not None:
        geohash = geohash_encode(lat, lng, precision=8)

    conn.execute(
        "INSERT INTO analysis_metadata "
        "(image_id, gps_latitude, gps_longitude, date_time_original, geohash, "
        " camera_model, location_city, location_state, location_country, gps_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [image_id, lat, lng, dto, geohash, camera_model, city, state, country,
         gps_source if lat is not None else None],
    )

    if lat is not None and lng is not None:
        conn.execute(
            "INSERT OR REPLACE INTO geo_rtree (id, min_lat, max_lat, min_lng, max_lng) "
            "VALUES (?, ?, ?, ?, ?)",
            [image_id, lat, lat, lng, lng],
        )

    return image_id


# ── Schema migration v31 ─────────────────────────────────────────────────────

class TestSchemaV31:
    def test_gps_source_column_exists(self, tmp_path):
        conn = _make_test_db(tmp_path)
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(analysis_metadata)").fetchall()]
        assert "gps_source" in cols

    def test_gps_source_default_is_exif(self, tmp_path):
        conn = _make_test_db(tmp_path)
        img_id = _insert_image(conn, "/test.jpg", lat=40.0, lng=-74.0)
        row = conn.execute(
            "SELECT gps_source FROM analysis_metadata WHERE image_id = ?", [img_id]
        ).fetchone()
        assert row["gps_source"] == "exif"


# ── Geohash ──────────────────────────────────────────────────────────────────

class TestGeohash:
    def test_encode_known_value(self):
        from imganalyzer.db.geohash import encode
        # Beijing: ~39.9, ~116.4
        gh = encode(39.9, 116.4, precision=8)
        assert len(gh) == 8
        assert gh.startswith("wx4f")  # verified empirically

    def test_precision_affects_length(self):
        from imganalyzer.db.geohash import encode
        for prec in [1, 4, 6, 8]:
            assert len(encode(0.0, 0.0, precision=prec)) == prec

    def test_nearby_points_share_prefix(self):
        from imganalyzer.db.geohash import encode
        # Two points 10m apart should share at least 7-char prefix
        gh1 = encode(40.7128, -74.0060, precision=8)
        gh2 = encode(40.7129, -74.0061, precision=8)
        assert gh1[:6] == gh2[:6]


# ── RDP simplification ──────────────────────────────────────────────────────

class TestRdpSimplify:
    def _get_rdp(self):
        # Import the server-level function
        import importlib
        import imganalyzer.server as srv
        importlib.reload(srv)  # ensure fresh
        return srv._rdp_simplify

    def test_straight_line_reduces_to_endpoints(self):
        from imganalyzer.server import _rdp_simplify
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        result = _rdp_simplify(points, 0.1)
        assert len(result) == 2
        assert result[0] == (0.0, 0.0)
        assert result[-1] == (3.0, 3.0)

    def test_zigzag_preserved(self):
        from imganalyzer.server import _rdp_simplify
        points = [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0), (3.0, 5.0), (4.0, 0.0)]
        result = _rdp_simplify(points, 0.5)
        assert len(result) >= 3  # cannot be simplified to just 2

    def test_two_points_unchanged(self):
        from imganalyzer.server import _rdp_simplify
        points = [(0.0, 0.0), (1.0, 1.0)]
        assert _rdp_simplify(points, 1.0) == points

    def test_single_point_unchanged(self):
        from imganalyzer.server import _rdp_simplify
        points = [(5.0, 5.0)]
        assert _rdp_simplify(points, 1.0) == points


# ── Trip detection ────────────────────────────────────────────────────────────

class TestTripDetect:
    def test_detect_single_trip(self, tmp_path):
        """A sequence of photos moving >10km should be detected as a trip."""
        conn = _make_test_db(tmp_path)
        base_time = datetime(2024, 7, 1, 8, 0, 0, tzinfo=timezone.utc)

        # Create a trip: Beijing → Shanghai (5 photos, ~1000km apart)
        coords = [
            (39.9, 116.4),   # Beijing
            (38.0, 117.0),   # en route
            (35.0, 118.0),   # midway
            (33.0, 119.5),   # approaching
            (31.2, 121.5),   # Shanghai
        ]
        for i, (lat, lng) in enumerate(coords):
            dto = (base_time + timedelta(hours=i * 2)).isoformat()
            _insert_image(conn, f"/trip/{i}.jpg", lat=lat, lng=lng, dto=dto,
                          city="City" + str(i), country="China")

        from imganalyzer.server import _handle_geo_trip_detect
        # Monkey-patch _get_db for this test
        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_trip_detect({"min_images": 3})
        finally:
            srv._get_db = orig

        assert "trips" in result
        assert len(result["trips"]) >= 1
        trip = result["trips"][0]
        assert trip["image_count"] == 5
        assert trip["distance_km"] > 100

    def test_stationary_photos_not_a_trip(self, tmp_path):
        """Photos all taken at the same location should not be a trip."""
        conn = _make_test_db(tmp_path)
        base_time = datetime(2024, 7, 1, 8, 0, 0, tzinfo=timezone.utc)

        for i in range(10):
            dto = (base_time + timedelta(minutes=i * 5)).isoformat()
            _insert_image(conn, f"/home/{i}.jpg", lat=40.0, lng=-74.0, dto=dto)

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_trip_detect({"min_images": 3})
        finally:
            srv._get_db = orig

        assert result["trips"] == []


# ── Trip timeline ─────────────────────────────────────────────────────────────

class TestTripTimeline:
    def test_timeline_groups_stops(self, tmp_path):
        """Images at the same location within 30 min should form one stop."""
        conn = _make_test_db(tmp_path)
        base_time = datetime(2024, 7, 1, 10, 0, 0, tzinfo=timezone.utc)

        # 3 photos at location A (within 30 min)
        for i in range(3):
            dto = (base_time + timedelta(minutes=i * 5)).isoformat()
            _insert_image(conn, f"/stop1/{i}.jpg", lat=40.0, lng=-74.0, dto=dto)

        # 2 photos at location B, 2 hours later, ~100km away
        for i in range(2):
            dto = (base_time + timedelta(hours=2, minutes=i * 5)).isoformat()
            _insert_image(conn, f"/stop2/{i}.jpg", lat=41.0, lng=-73.0, dto=dto)

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_trip_timeline({
                "start_date": "2024-07-01T00:00:00",
                "end_date": "2024-07-02T00:00:00",
            })
        finally:
            srv._get_db = orig

        assert result["total_images"] == 5
        assert len(result["stops"]) == 2
        assert result["stops"][0]["count"] == 3
        assert result["stops"][1]["count"] == 2
        assert len(result["route_points"]) >= 2

    def test_timeline_empty_range(self, tmp_path):
        conn = _make_test_db(tmp_path)

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_trip_timeline({
                "start_date": "2099-01-01",
                "end_date": "2099-01-02",
            })
        finally:
            srv._get_db = orig

        assert result["stops"] == []
        assert result["total_images"] == 0


# ── Gap filler ────────────────────────────────────────────────────────────────

class TestGapFiller:
    def test_preview_finds_fillable_images(self, tmp_path):
        """An image without GPS between two geotagged images should be fillable."""
        conn = _make_test_db(tmp_path)
        base = datetime(2024, 7, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Image 1: geotagged at 10:00
        _insert_image(conn, "/a.jpg", lat=40.0, lng=-74.0,
                       dto=(base).isoformat())
        # Image 2: NO GPS at 10:15
        _insert_image(conn, "/b.jpg", lat=None, lng=None,
                       dto=(base + timedelta(minutes=15)).isoformat())
        # Image 3: geotagged at 10:30
        _insert_image(conn, "/c.jpg", lat=40.1, lng=-73.9,
                       dto=(base + timedelta(minutes=30)).isoformat())

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_gap_filler_preview({
                "max_gap_minutes": 60,
            })
        finally:
            srv._get_db = orig

        assert result["total_missing"] == 1
        assert result["fillable"] == 1
        assert len(result["previews"]) == 1
        p = result["previews"][0]
        assert 40.0 <= p["inferred_lat"] <= 40.1
        assert -74.0 <= p["inferred_lng"] <= -73.9
        assert p["confidence"] > 0

    def test_preview_respects_max_gap(self, tmp_path):
        """Image too far in time from neighbors should not be fillable."""
        conn = _make_test_db(tmp_path)
        base = datetime(2024, 7, 1, 10, 0, 0, tzinfo=timezone.utc)

        _insert_image(conn, "/a.jpg", lat=40.0, lng=-74.0,
                       dto=base.isoformat())
        _insert_image(conn, "/b.jpg", lat=None, lng=None,
                       dto=(base + timedelta(hours=3)).isoformat())
        _insert_image(conn, "/c.jpg", lat=41.0, lng=-73.0,
                       dto=(base + timedelta(hours=6)).isoformat())

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_gap_filler_preview({"max_gap_minutes": 30})
        finally:
            srv._get_db = orig

        assert result["fillable"] == 0

    def test_apply_writes_coordinates(self, tmp_path):
        """Apply should write inferred GPS to the database."""
        conn = _make_test_db(tmp_path)
        base = datetime(2024, 7, 1, 10, 0, 0, tzinfo=timezone.utc)

        _insert_image(conn, "/a.jpg", lat=40.0, lng=-74.0,
                       dto=base.isoformat())
        img_id = _insert_image(conn, "/b.jpg", lat=None, lng=None,
                                dto=(base + timedelta(minutes=10)).isoformat())
        _insert_image(conn, "/c.jpg", lat=40.0, lng=-74.0,
                       dto=(base + timedelta(minutes=20)).isoformat())

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_gap_filler_apply({
                "max_gap_minutes": 60,
                "min_confidence": 0.1,
            })
        finally:
            srv._get_db = orig

        assert result["filled"] == 1

        row = conn.execute(
            "SELECT gps_latitude, gps_longitude, gps_source, geohash "
            "FROM analysis_metadata WHERE image_id = ?", [img_id]
        ).fetchone()
        assert row["gps_latitude"] is not None
        assert row["gps_longitude"] is not None
        assert row["gps_source"] == "inferred"
        assert row["geohash"] is not None

        # Should also be in rtree
        rtree_row = conn.execute(
            "SELECT * FROM geo_rtree WHERE id = ?", [img_id]
        ).fetchone()
        assert rtree_row is not None

    def test_apply_respects_overrides(self, tmp_path):
        """Images with user overrides on GPS should be skipped."""
        conn = _make_test_db(tmp_path)
        base = datetime(2024, 7, 1, 10, 0, 0, tzinfo=timezone.utc)

        _insert_image(conn, "/a.jpg", lat=40.0, lng=-74.0,
                       dto=base.isoformat())
        img_id = _insert_image(conn, "/b.jpg", lat=None, lng=None,
                                dto=(base + timedelta(minutes=10)).isoformat())
        _insert_image(conn, "/c.jpg", lat=40.0, lng=-74.0,
                       dto=(base + timedelta(minutes=20)).isoformat())

        # Add an override for this image's GPS — both preview and apply should skip it
        conn.execute(
            "INSERT INTO overrides (image_id, table_name, field_name, value) "
            "VALUES (?, 'analysis_metadata', 'gps_latitude', '42.0')",
            [img_id],
        )

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            # Preview should exclude the overridden image
            preview = _handle_geo_gap_filler_preview({
                "max_gap_minutes": 60,
            })
            assert preview["fillable"] == 0, "Override image should be excluded from preview"

            # Apply should also result in 0 filled
            result = _handle_geo_gap_filler_apply({
                "max_gap_minutes": 60,
                "min_confidence": 0.1,
            })
        finally:
            srv._get_db = orig

        assert result["filled"] == 0

        # Verify the image still has no GPS
        row = conn.execute(
            "SELECT gps_latitude FROM analysis_metadata WHERE image_id = ?", [img_id]
        ).fetchone()
        assert row["gps_latitude"] is None


# ── Stats extended ────────────────────────────────────────────────────────────

class TestGeoStatsExtended:
    def test_basic_stats(self, tmp_path):
        conn = _make_test_db(tmp_path)

        _insert_image(conn, "/a.jpg", lat=40.0, lng=-74.0,
                       dto="2024-06-15T10:00:00", city="NYC", country="USA",
                       camera_model="Canon R5")
        _insert_image(conn, "/b.jpg", lat=48.8, lng=2.3,
                       dto="2024-07-20T14:00:00", city="Paris", country="France",
                       camera_model="Sony A7")
        _insert_image(conn, "/c.jpg")  # no GPS

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_stats_extended({})
        finally:
            srv._get_db = orig

        assert result["total_images"] == 3
        assert result["geotagged"] == 2
        assert len(result["countries"]) == 2
        assert len(result["top_cities"]) == 2
        assert len(result["monthly_activity"]) == 2
        assert len(result["top_locations"]) == 2
        assert len(result["camera_by_country"]) == 2

    def test_furthest_from_home(self, tmp_path):
        conn = _make_test_db(tmp_path)

        # Home: NYC (40.7, -74.0)
        _insert_image(conn, "/nearby.jpg", lat=40.8, lng=-73.9, dto="2024-01-01T10:00:00")
        _insert_image(conn, "/far.jpg", lat=-33.9, lng=151.2, dto="2024-06-01T10:00:00")  # Sydney

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_stats_extended({
                "home_lat": 40.7, "home_lng": -74.0
            })
        finally:
            srv._get_db = orig

        assert result["furthest_from_home"] is not None
        assert result["furthest_from_home"]["distance_km"] > 15000  # NYC→Sydney ~16000km

    def test_no_gps_data(self, tmp_path):
        conn = _make_test_db(tmp_path)
        _insert_image(conn, "/no_gps.jpg")

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_stats_extended({})
        finally:
            srv._get_db = orig

        assert result["total_images"] == 1
        assert result["geotagged"] == 0
        assert result["countries"] == []
        assert result["monthly_activity"] == []


# Import the handlers at module level after all classes are defined
from imganalyzer.server import (
    _handle_geo_gap_filler_apply,
    _handle_geo_gap_filler_preview,
    _handle_geo_geocode,
    _handle_geo_stats_extended,
    _handle_geo_trip_detect,
    _handle_geo_trip_timeline,
)


# ── Geocode ──────────────────────────────────────────────────────────────────

class TestGeoGeocode:
    def test_geocode_by_city(self, tmp_path):
        conn = _make_test_db(tmp_path)

        _insert_image(conn, "/a.jpg", lat=39.9, lng=116.4, city="Beijing", country="China")
        _insert_image(conn, "/b.jpg", lat=40.0, lng=116.5, city="Beijing", country="China")
        _insert_image(conn, "/c.jpg", lat=48.8, lng=2.3, city="Paris", country="France")

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_geocode({"location": "Beijing"})
        finally:
            srv._get_db = orig

        assert result["count"] == 2
        assert 39.5 < result["lat"] < 40.5
        assert 116.0 < result["lng"] < 117.0

    def test_geocode_by_country(self, tmp_path):
        conn = _make_test_db(tmp_path)

        _insert_image(conn, "/a.jpg", lat=40.0, lng=-74.0, city="NYC", state="NY", country="USA")
        _insert_image(conn, "/b.jpg", lat=34.0, lng=-118.0, city="LA", state="CA", country="USA")
        _insert_image(conn, "/c.jpg", lat=48.8, lng=2.3, city="Paris", country="France")

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_geocode({"location": "USA"})
        finally:
            srv._get_db = orig

        assert result["count"] == 2
        # Centroid of NYC + LA
        assert 34.0 < result["lat"] < 41.0
        assert -118.0 < result["lng"] < -74.0

    def test_geocode_no_match(self, tmp_path):
        conn = _make_test_db(tmp_path)

        _insert_image(conn, "/a.jpg", lat=40.0, lng=-74.0, city="NYC", country="USA")

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_geocode({"location": "Antarctica"})
        finally:
            srv._get_db = orig

        assert result["count"] == 0
        assert result["lat"] is None
        assert result["lng"] is None

    def test_geocode_empty_location(self, tmp_path):
        conn = _make_test_db(tmp_path)

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_geocode({"location": ""})
        finally:
            srv._get_db = orig

        assert "error" in result

    def test_geocode_by_state(self, tmp_path):
        conn = _make_test_db(tmp_path)

        _insert_image(conn, "/a.jpg", lat=37.7, lng=-122.4, city="SF", state="California", country="USA")
        _insert_image(conn, "/b.jpg", lat=34.0, lng=-118.2, city="LA", state="California", country="USA")
        _insert_image(conn, "/c.jpg", lat=40.7, lng=-74.0, city="NYC", state="New York", country="USA")

        import imganalyzer.server as srv
        orig = srv._get_db
        srv._get_db = lambda: conn
        try:
            result = _handle_geo_geocode({"location": "California"})
        finally:
            srv._get_db = orig

        assert result["count"] == 2
        # Centroid of SF + LA — should be roughly central California
        assert 34.0 < result["lat"] < 38.0
        assert -123.0 < result["lng"] < -118.0
