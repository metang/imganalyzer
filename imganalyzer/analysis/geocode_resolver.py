"""Asynchronous reverse-geocoding resolver.

This module is the async counterpart to :mod:`imganalyzer.analysis.metadata`.
Metadata ingest writes ``gps_latitude`` / ``gps_longitude`` to
``analysis_metadata`` and leaves ``location_*`` columns NULL.  The resolver
scans for such rows, performs rate-limited reverse-geocoding via Nominatim,
persists results to the ``geocode_cache`` table (so it is shared across
distributed workers), and backfills ``analysis_metadata.location_*``.

Follow-up: wire :func:`resolve_pending_locations` into a background loop
from ``server.py`` (out of scope for the current surgical change).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any

from imganalyzer import __version__

log = logging.getLogger(__name__)

# Nominatim usage policy: max 1 request per second.
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_NOMINATIM_MIN_INTERVAL_S = 1.0
_LAST_REQUEST_TS = 0.0

# Cache key resolution: 4 decimal places ≈ 11 m.
_CACHE_PRECISION = 4

_LOCATION_KEYS = (
    "location_city",
    "location_state",
    "location_country",
    "location_country_code",
)


def _cache_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat, _CACHE_PRECISION), round(lon, _CACHE_PRECISION))


def get_cached_location(
    conn: sqlite3.Connection, lat: float, lon: float
) -> dict[str, str] | None:
    """Return the cached reverse-geocode result for ``(lat, lon)`` or ``None``."""
    lat_key, lon_key = _cache_key(lat, lon)
    row = conn.execute(
        "SELECT location_json FROM geocode_cache "
        "WHERE lat_key = ? AND lon_key = ?",
        (lat_key, lon_key),
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["location_json"] if isinstance(row, sqlite3.Row) else row[0])
    except (json.JSONDecodeError, TypeError):
        return None


def put_cached_location(
    conn: sqlite3.Connection, lat: float, lon: float, location: dict[str, str]
) -> None:
    """Upsert a reverse-geocode result into ``geocode_cache``."""
    lat_key, lon_key = _cache_key(lat, lon)
    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache "
        "(lat_key, lon_key, location_json, fetched_at) VALUES (?, ?, ?, ?)",
        (lat_key, lon_key, json.dumps(location), int(time.time())),
    )


def _fetch_from_nominatim(lat: float, lon: float) -> dict[str, str]:
    """Call Nominatim reverse-geocoding (rate-limited to 1 req/s).

    Returns an empty dict on any failure.
    """
    global _LAST_REQUEST_TS
    elapsed = time.monotonic() - _LAST_REQUEST_TS
    if elapsed < _NOMINATIM_MIN_INTERVAL_S:
        time.sleep(_NOMINATIM_MIN_INTERVAL_S - elapsed)

    try:
        import httpx

        resp = httpx.get(
            _NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": f"imganalyzer/{__version__}"},
            timeout=5.0,
        )
        _LAST_REQUEST_TS = time.monotonic()
        resp.raise_for_status()
        data = resp.json()
        addr = data.get("address", {}) or {}
        return {
            "location_city": (
                addr.get("city") or addr.get("town") or addr.get("village") or ""
            ),
            "location_state": addr.get("state", "") or "",
            "location_country": addr.get("country", "") or "",
            "location_country_code": (addr.get("country_code", "") or "").upper(),
        }
    except Exception as exc:
        _LAST_REQUEST_TS = time.monotonic()
        log.warning(
            "geocode_resolver nominatim fetch failed lat=%.6f lon=%.6f "
            "error_type=%s error=%s",
            lat,
            lon,
            type(exc).__name__,
            exc,
        )
        return {}


def resolve_pending_locations(
    conn: sqlite3.Connection,
    limit: int = 50,
    fetcher: Any = None,
) -> int:
    """Resolve up to ``limit`` pending rows that have GPS but no location.

    For each unique rounded ``(lat, lon)``:
      1. Check ``geocode_cache``; if hit, reuse.
      2. Otherwise call ``fetcher(lat, lon)`` (defaults to
         :func:`_fetch_from_nominatim`, rate-limited to 1 req/s) and
         persist the result to ``geocode_cache``.
      3. Update all matching ``analysis_metadata`` rows' ``location_*``
         columns.

    Returns the number of ``analysis_metadata`` rows updated.

    The ``fetcher`` parameter exists to make the function unit-testable
    without hitting the network.
    """
    if fetcher is None:
        fetcher = _fetch_from_nominatim

    rows = conn.execute(
        "SELECT image_id, gps_latitude, gps_longitude FROM analysis_metadata "
        "WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL "
        "AND (location_city IS NULL OR location_city = '') "
        "AND (location_country IS NULL OR location_country = '') "
        "LIMIT ?",
        (limit,),
    ).fetchall()

    updated = 0
    # Group by rounded key so we fetch each unique location at most once.
    by_key: dict[tuple[float, float], list[tuple[int, float, float]]] = {}
    for row in rows:
        image_id = row["image_id"] if isinstance(row, sqlite3.Row) else row[0]
        lat = row["gps_latitude"] if isinstance(row, sqlite3.Row) else row[1]
        lon = row["gps_longitude"] if isinstance(row, sqlite3.Row) else row[2]
        if lat is None or lon is None:
            continue
        by_key.setdefault(_cache_key(lat, lon), []).append((image_id, lat, lon))

    for key, members in by_key.items():
        lat, lon = key
        cached = get_cached_location(conn, lat, lon)
        if cached is None:
            # Use a representative (lat, lon) from the group for the fetch.
            _, rep_lat, rep_lon = members[0]
            location = fetcher(rep_lat, rep_lon)
            put_cached_location(conn, lat, lon, location)
        else:
            location = cached

        if not location:
            continue

        for image_id, _lat, _lon in members:
            conn.execute(
                "UPDATE analysis_metadata SET "
                "location_city = COALESCE(NULLIF(?, ''), location_city), "
                "location_state = COALESCE(NULLIF(?, ''), location_state), "
                "location_country = COALESCE(NULLIF(?, ''), location_country), "
                "location_country_code = COALESCE(NULLIF(?, ''), location_country_code) "
                "WHERE image_id = ?",
                (
                    location.get("location_city", ""),
                    location.get("location_state", ""),
                    location.get("location_country", ""),
                    location.get("location_country_code", ""),
                    image_id,
                ),
            )
            updated += 1

    conn.commit()
    return updated
