"""Time+location clustering for story moments and chapters.

Generalises the trip-stop algorithm from server.py into a reusable engine
with configurable time windows and geohash precision.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from imganalyzer.db.geohash import encode as geohash_encode


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ImagePoint:
    """An image with its temporal and spatial coordinates."""

    image_id: int
    timestamp: datetime | None
    lat: float | None
    lng: float | None
    geohash6: str  # first 6 chars of geohash, "" if no GPS


@dataclass
class Moment:
    """A cluster of images taken at roughly the same time and place."""

    images: list[ImagePoint]
    start_time: datetime | None = None
    end_time: datetime | None = None
    lat: float | None = None
    lng: float | None = None

    def __post_init__(self) -> None:
        if self.images:
            ts = [p.timestamp for p in self.images if p.timestamp]
            if ts:
                self.start_time = min(ts)
                self.end_time = max(ts)
            lats = [p.lat for p in self.images if p.lat is not None]
            lngs = [p.lng for p in self.images if p.lng is not None]
            if lats:
                self.lat = sum(lats) / len(lats)
            if lngs:
                self.lng = sum(lngs) / len(lngs)


@dataclass
class Chapter:
    """A group of moments forming a coherent chapter."""

    moments: list[Moment]
    title: str = ""
    location: str = ""
    start_date: datetime | None = None
    end_date: datetime | None = None

    def __post_init__(self) -> None:
        if self.moments:
            starts = [m.start_time for m in self.moments if m.start_time]
            ends = [m.end_time for m in self.moments if m.end_time]
            if starts:
                self.start_date = min(starts)
            if ends:
                self.end_date = max(ends)


# ── Load image points ────────────────────────────────────────────────────────

def load_album_points(
    conn: sqlite3.Connection,
    album_id: str,
) -> list[ImagePoint]:
    """Load images from album_items with timestamps and GPS data."""
    rows = conn.execute(
        "SELECT ai.image_id, sf.date_time_original, "
        "       am.gps_latitude, am.gps_longitude, am.geohash "
        "FROM album_items ai "
        "LEFT JOIN search_features sf ON sf.image_id = ai.image_id "
        "LEFT JOIN analysis_metadata am ON am.image_id = ai.image_id "
        "WHERE ai.album_id = ? "
        "ORDER BY sf.date_time_original ASC NULLS LAST",
        [album_id],
    ).fetchall()

    points: list[ImagePoint] = []
    for r in rows:
        ts = None
        if r["date_time_original"]:
            try:
                ts = datetime.fromisoformat(r["date_time_original"])
            except (ValueError, TypeError):
                pass

        gh6 = ""
        if r["geohash"]:
            gh6 = r["geohash"][:6]
        elif r["gps_latitude"] is not None and r["gps_longitude"] is not None:
            gh6 = geohash_encode(r["gps_latitude"], r["gps_longitude"], precision=6)

        points.append(ImagePoint(
            image_id=r["image_id"],
            timestamp=ts,
            lat=r["gps_latitude"],
            lng=r["gps_longitude"],
            geohash6=gh6,
        ))
    return points


# ── Moment clustering ────────────────────────────────────────────────────────

def cluster_moments(
    points: list[ImagePoint],
    *,
    time_window: timedelta = timedelta(minutes=30),
) -> list[Moment]:
    """Cluster images into moments by time proximity and geohash-6.

    Two consecutive images belong to the same moment if they are:
    1. Within ``time_window`` of each other, AND
    2. Share the same geohash-6 (both have GPS) OR at least one has no GPS

    Images without timestamps go into a single catch-all moment at the end.
    """
    if not points:
        return []

    # Separate images with/without timestamps
    timed = [p for p in points if p.timestamp is not None]
    untimed = [p for p in points if p.timestamp is None]

    # Sort by timestamp
    timed.sort(key=lambda p: p.timestamp)  # type: ignore[arg-type]

    moments: list[Moment] = []
    if timed:
        current: list[ImagePoint] = [timed[0]]
        for i in range(1, len(timed)):
            prev = timed[i - 1]
            curr = timed[i]

            assert prev.timestamp is not None
            assert curr.timestamp is not None
            close_time = (curr.timestamp - prev.timestamp) <= time_window

            # Spatial check: same area if both have geohash and they match,
            # or if either lacks GPS (benefit of the doubt)
            if prev.geohash6 and curr.geohash6:
                same_area = prev.geohash6 == curr.geohash6
            else:
                same_area = True  # no GPS → group by time only

            if close_time and same_area:
                current.append(curr)
            else:
                moments.append(Moment(images=current))
                current = [curr]
        moments.append(Moment(images=current))

    if untimed:
        moments.append(Moment(images=untimed))

    return moments


# ── Chapter detection ────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Approximate distance between two points in km."""
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def detect_chapters(
    moments: list[Moment],
    *,
    time_gap: timedelta = timedelta(hours=4),
    distance_gap_km: float = 50.0,
    force_year_breaks: bool = True,
) -> list[Chapter]:
    """Group moments into chapters.

    A new chapter starts when:
    1. Time gap between consecutive moments exceeds ``time_gap``, OR
    2. Spatial distance exceeds ``distance_gap_km``, OR
    3. Year changes (if ``force_year_breaks`` is True)
    """
    if not moments:
        return []

    chapters: list[Chapter] = []
    current: list[Moment] = [moments[0]]

    for i in range(1, len(moments)):
        prev = moments[i - 1]
        curr = moments[i]
        should_break = False

        # Time gap check
        if prev.end_time and curr.start_time:
            if (curr.start_time - prev.end_time) > time_gap:
                should_break = True

        # Distance gap check
        if (
            not should_break
            and prev.lat is not None
            and prev.lng is not None
            and curr.lat is not None
            and curr.lng is not None
        ):
            dist = _haversine_km(prev.lat, prev.lng, curr.lat, curr.lng)
            if dist > distance_gap_km:
                should_break = True

        # Year boundary check
        if (
            not should_break
            and force_year_breaks
            and prev.end_time
            and curr.start_time
            and prev.end_time.year != curr.start_time.year
        ):
            should_break = True

        if should_break:
            chapters.append(Chapter(moments=current))
            current = [curr]
        else:
            current.append(curr)

    chapters.append(Chapter(moments=current))
    return chapters


def generate_chapter_title(
    chapter: Chapter,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Generate a heuristic title for a chapter from metadata.

    Uses location names and date ranges from the chapter's images.
    """
    parts: list[str] = []

    # Try to get location from DB if conn available
    location = ""
    if conn is not None and chapter.moments:
        image_ids = []
        for m in chapter.moments:
            image_ids.extend(p.image_id for p in m.images)
        if image_ids:
            placeholders = ",".join("?" for _ in image_ids[:100])
            row = conn.execute(
                f"SELECT location_city, location_state, location_country "
                f"FROM analysis_metadata "
                f"WHERE image_id IN ({placeholders}) "
                f"AND location_city IS NOT NULL "
                f"GROUP BY location_city "
                f"ORDER BY COUNT(*) DESC LIMIT 1",
                image_ids[:100],
            ).fetchone()
            if row:
                city = row["location_city"] or ""
                country = row["location_country"] or ""
                if city:
                    location = f"{city}, {country}" if country else city
                elif country:
                    location = country
    chapter.location = location

    if location:
        parts.append(location)

    # Date range
    if chapter.start_date and chapter.end_date:
        start_str = chapter.start_date.strftime("%b %d")
        if chapter.start_date.year != chapter.end_date.year:
            start_str = chapter.start_date.strftime("%b %d, %Y")
            end_str = chapter.end_date.strftime("%b %d, %Y")
            parts.append(f"{start_str} – {end_str}")
        elif chapter.start_date.month != chapter.end_date.month:
            end_str = chapter.end_date.strftime("%b %d, %Y")
            parts.append(f"{start_str} – {end_str}")
        elif chapter.start_date.day != chapter.end_date.day:
            end_str = chapter.end_date.strftime("%d, %Y")
            parts.append(f"{start_str}–{end_str}")
        else:
            parts.append(chapter.start_date.strftime("%b %d, %Y"))
    elif chapter.start_date:
        parts.append(chapter.start_date.strftime("%b %d, %Y"))

    return " — ".join(parts) if parts else "Untitled Chapter"
