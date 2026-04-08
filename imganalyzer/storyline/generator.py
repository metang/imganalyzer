"""Story generator — orchestrates the full pipeline.

Pipeline: album items → load points → cluster moments → detect chapters
→ select heroes → generate titles → persist to DB.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import timedelta
from typing import Any

from imganalyzer.storyline.clustering import (
    Chapter,
    Moment,
    cluster_moments,
    detect_chapters,
    generate_chapter_title,
    load_album_points,
)
from imganalyzer.storyline.heroes import (
    select_chapter_cover,
    select_chapter_heroes,
)


def generate_story(
    conn: sqlite3.Connection,
    album_id: str,
    *,
    time_window_minutes: int = 30,
    chapter_gap_hours: int = 4,
    chapter_distance_km: float = 50.0,
    force_year_breaks: bool = True,
) -> StoryResult:
    """Generate (or regenerate) the full story for an album.

    1. Load all album image points with timestamps + GPS.
    2. Cluster into moments (time + geohash proximity).
    3. Group moments into chapters (time gaps, distance, year breaks).
    4. Select hero images per moment and chapter.
    5. Generate chapter titles from metadata.
    6. Persist everything to DB.

    Returns a :class:`StoryResult` with counts for evaluation.
    """
    # Clear existing story data for this album
    _clear_story(conn, album_id)

    # Step 1: Load points
    points = load_album_points(conn, album_id)
    if not points:
        conn.execute(
            "UPDATE smart_albums SET chapter_count = 0 WHERE id = ?",
            [album_id],
        )
        conn.commit()
        return StoryResult(album_id=album_id, image_count=0)

    # Step 2: Cluster into moments
    moments = cluster_moments(
        points,
        time_window=timedelta(minutes=time_window_minutes),
    )

    # Step 3: Group into chapters
    chapters = detect_chapters(
        moments,
        time_gap=timedelta(hours=chapter_gap_hours),
        distance_gap_km=chapter_distance_km,
        force_year_breaks=force_year_breaks,
    )

    # Step 4-6: Persist chapters, moments, heroes
    total_moments = 0
    for ch_idx, chapter in enumerate(chapters):
        chapter_id = str(uuid.uuid4())

        # Generate title
        title = generate_chapter_title(chapter, conn)
        chapter.title = title

        # Collect moment data for hero selection
        moment_data: list[tuple[str, list[int]]] = []
        moment_ids: list[str] = []
        for m_idx, moment in enumerate(chapter.moments):
            moment_id = str(uuid.uuid4())
            moment_ids.append(moment_id)
            img_ids = [p.image_id for p in moment.images]
            moment_data.append((moment_id, img_ids))

        # Select heroes for all moments in this chapter
        heroes = select_chapter_heroes(conn, moment_data)

        # Insert chapter (cover set after moments)
        conn.execute(
            "INSERT INTO story_chapters "
            "(id, album_id, title, summary, sort_order, start_date, end_date, "
            " location, image_count, moment_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                chapter_id,
                album_id,
                title,
                None,  # summary generated later (AI or heuristic)
                ch_idx,
                chapter.start_date.isoformat() if chapter.start_date else None,
                chapter.end_date.isoformat() if chapter.end_date else None,
                chapter.location,
                sum(len(m.images) for m in chapter.moments),
                len(chapter.moments),
            ],
        )

        # Insert moments and moment_images
        hero_image_ids: list[int] = []
        for m_idx, moment in enumerate(chapter.moments):
            moment_id = moment_data[m_idx][0]
            hero_id = heroes.get(moment_id)
            if hero_id:
                hero_image_ids.append(hero_id)

            conn.execute(
                "INSERT INTO story_moments "
                "(id, chapter_id, title, sort_order, start_time, end_time, "
                " lat, lng, hero_image_id, image_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    moment_id,
                    chapter_id,
                    None,  # moment titles can be generated on demand
                    m_idx,
                    moment.start_time.isoformat() if moment.start_time else None,
                    moment.end_time.isoformat() if moment.end_time else None,
                    moment.lat,
                    moment.lng,
                    hero_id,
                    len(moment.images),
                ],
            )

            # Insert moment images, sorted by timestamp
            for img_order, pt in enumerate(moment.images):
                conn.execute(
                    "INSERT INTO moment_images "
                    "(moment_id, image_id, sort_order, is_hero) "
                    "VALUES (?, ?, ?, ?)",
                    [moment_id, pt.image_id, img_order, int(pt.image_id == hero_id)],
                )

        # Set chapter cover
        cover_id = select_chapter_cover(conn, hero_image_ids) if hero_image_ids else None
        if cover_id:
            conn.execute(
                "UPDATE story_chapters SET cover_image_id = ? WHERE id = ?",
                [cover_id, chapter_id],
            )

        total_moments += len(chapter.moments)

    # Update album stats
    conn.execute(
        "UPDATE smart_albums SET chapter_count = ?, updated_at = datetime('now') WHERE id = ?",
        [len(chapters), album_id],
    )
    conn.commit()

    return StoryResult(
        album_id=album_id,
        image_count=len(points),
        moment_count=total_moments,
        chapter_count=len(chapters),
    )


def _clear_story(conn: sqlite3.Connection, album_id: str) -> None:
    """Remove all story structure for an album (chapters cascade to moments)."""
    chapter_ids = conn.execute(
        "SELECT id FROM story_chapters WHERE album_id = ?", [album_id]
    ).fetchall()
    for (cid,) in chapter_ids:
        moment_ids = conn.execute(
            "SELECT id FROM story_moments WHERE chapter_id = ?", [cid]
        ).fetchall()
        for (mid,) in moment_ids:
            conn.execute("DELETE FROM moment_images WHERE moment_id = ?", [mid])
        conn.execute("DELETE FROM story_moments WHERE chapter_id = ?", [cid])
    conn.execute("DELETE FROM story_chapters WHERE album_id = ?", [album_id])


# ── Query helpers ────────────────────────────────────────────────────────────

def get_story_chapters(
    conn: sqlite3.Connection,
    album_id: str,
) -> list[dict[str, Any]]:
    """Return all chapters for an album with summary stats."""
    rows = conn.execute(
        "SELECT * FROM story_chapters WHERE album_id = ? ORDER BY sort_order",
        [album_id],
    ).fetchall()
    return [dict(r) for r in rows]


def get_chapter_moments(
    conn: sqlite3.Connection,
    chapter_id: str,
) -> list[dict[str, Any]]:
    """Return all moments in a chapter."""
    rows = conn.execute(
        "SELECT * FROM story_moments WHERE chapter_id = ? ORDER BY sort_order",
        [chapter_id],
    ).fetchall()
    return [dict(r) for r in rows]


def get_moment_images(
    conn: sqlite3.Connection,
    moment_id: str,
) -> list[dict[str, Any]]:
    """Return all images in a moment with ordering."""
    rows = conn.execute(
        "SELECT mi.image_id, mi.sort_order, mi.is_hero, "
        "       sf.date_time_original, sf.perception_iaa "
        "FROM moment_images mi "
        "LEFT JOIN search_features sf ON sf.image_id = mi.image_id "
        "WHERE mi.moment_id = ? ORDER BY mi.sort_order",
        [moment_id],
    ).fetchall()
    return [dict(r) for r in rows]


class StoryResult:
    """Result of story generation for evaluation."""

    def __init__(
        self,
        album_id: str,
        image_count: int = 0,
        moment_count: int = 0,
        chapter_count: int = 0,
    ) -> None:
        self.album_id = album_id
        self.image_count = image_count
        self.moment_count = moment_count
        self.chapter_count = chapter_count

    def __repr__(self) -> str:
        return (
            f"StoryResult(album={self.album_id!r}, "
            f"images={self.image_count}, "
            f"moments={self.moment_count}, "
            f"chapters={self.chapter_count})"
        )
