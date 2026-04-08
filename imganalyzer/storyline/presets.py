"""Built-in smart album presets.

Provides factory functions for common album configurations:
- Year in Review
- On This Day (same month+day across years)
- Growth Story for a person
- Person Timeline
- Location Story
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from imganalyzer.storyline.albums import create_album


def create_year_in_review(
    conn: sqlite3.Connection,
    year: int | None = None,
) -> Any:
    """Create a Year in Review album for the given year (default: last year)."""
    if year is None:
        year = datetime.now(timezone.utc).year - 1

    rules: dict[str, Any] = {
        "match": "all",
        "rules": [
            {
                "type": "date_range",
                "start": f"{year}-01-01",
                "end": f"{year}-12-31T23:59:59",
            }
        ],
    }
    return create_album(
        conn,
        name=f"{year} Year in Review",
        rules=rules,
        description=f"All photos from {year}",
    )


def create_on_this_day(
    conn: sqlite3.Connection,
    month: int | None = None,
    day: int | None = None,
) -> Any:
    """Create an On This Day album for a specific month+day (default: today).

    Note: This uses a date_range rule per year found in the database.
    For a true recurring-date filter, the rule engine would need a
    ``this_day`` rule type (future extension).  For now, we create a
    multi-year union by finding years with images and building per-year
    date ranges.
    """
    now = datetime.now(timezone.utc)
    if month is None:
        month = now.month
    if day is None:
        day = now.day

    # Find years that have images
    rows = conn.execute(
        "SELECT DISTINCT substr(date_time_original, 1, 4) AS yr "
        "FROM search_features "
        "WHERE date_time_original IS NOT NULL "
        "ORDER BY yr"
    ).fetchall()

    years = [int(r["yr"]) for r in rows if r["yr"] and r["yr"].isdigit()]
    if not years:
        years = [now.year]

    # Build rules: one date_range per year for this month+day
    date_rules = []
    for yr in years:
        try:
            target = datetime(yr, month, day)
            date_str = target.strftime("%Y-%m-%d")
            date_rules.append({
                "type": "date_range",
                "start": date_str,
                "end": date_str + "T23:59:59",
            })
        except ValueError:
            continue  # invalid date (e.g., Feb 29 in non-leap year)

    rules: dict[str, Any] = {
        "match": "any",
        "rules": date_rules,
    }

    month_name = datetime(2000, month, day).strftime("%B %d")
    return create_album(
        conn,
        name=f"On This Day — {month_name}",
        rules=rules,
        description=f"Photos from {month_name} across all years",
    )


def create_person_timeline(
    conn: sqlite3.Connection,
    person_id: int,
    person_name: str | None = None,
) -> Any:
    """Create a timeline album for a specific person."""
    if person_name is None:
        row = conn.execute(
            "SELECT name FROM face_persons WHERE id = ?", [person_id]
        ).fetchone()
        person_name = row["name"] if row else f"Person {person_id}"

    rules: dict[str, Any] = {
        "match": "all",
        "rules": [
            {"type": "person", "person_ids": [person_id], "mode": "any"},
        ],
    }
    return create_album(
        conn,
        name=f"The Story of {person_name}",
        rules=rules,
        description=f"All photos featuring {person_name}",
    )


def create_growth_story(
    conn: sqlite3.Connection,
    person_id: int,
    person_name: str | None = None,
) -> Any:
    """Create a growth-story album for a specific person."""
    if person_name is None:
        row = conn.execute(
            "SELECT name FROM face_persons WHERE id = ?", [person_id]
        ).fetchone()
        person_name = row["name"] if row else f"Person {person_id}"

    rules: dict[str, Any] = {
        "match": "all",
        "rules": [
            {"type": "person", "person_ids": [person_id], "mode": "any"},
        ],
    }
    return create_album(
        conn,
        name=f"Growing Up — {person_name}",
        rules=rules,
        description=f"A year-by-year story of {person_name}",
    )


def create_together_album(
    conn: sqlite3.Connection,
    person_ids: list[int],
    person_names: list[str] | None = None,
) -> Any:
    """Create a co-occurrence album for multiple people."""
    if person_names is None:
        person_names = []
        for pid in person_ids:
            row = conn.execute(
                "SELECT name FROM face_persons WHERE id = ?", [pid]
            ).fetchone()
            person_names.append(row["name"] if row else f"Person {pid}")

    names_str = " & ".join(person_names)
    rules: dict[str, Any] = {
        "match": "all",
        "rules": [
            {"type": "person", "person_ids": person_ids, "mode": "all"},
        ],
    }
    return create_album(
        conn,
        name=f"{names_str} Together",
        rules=rules,
        description=f"Photos where {names_str} appear together",
    )


def create_location_story(
    conn: sqlite3.Connection,
    country: str,
    city: str | None = None,
) -> Any:
    """Create a location-based album."""
    rule: dict[str, Any] = {"type": "location", "country": country}
    if city:
        rule["city"] = city

    name = city or country
    rules: dict[str, Any] = {
        "match": "all",
        "rules": [rule],
    }
    return create_album(
        conn,
        name=f"Photos from {name}",
        rules=rules,
        description=f"All photos taken in {name}",
    )


# Registry of all presets for the API
PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    "year_in_review": {
        "name": "Year in Review",
        "description": "All photos from a specific year",
        "params": ["year"],
    },
    "on_this_day": {
        "name": "On This Day",
        "description": "Photos from the same date across all years",
        "params": ["month", "day"],
    },
    "person_timeline": {
        "name": "Person Timeline",
        "description": "All photos of a specific person",
        "params": ["person_id", "person_name"],
    },
    "growth_story": {
        "name": "Growth Story",
        "description": "A year-by-year story for a specific person",
        "params": ["person_id", "person_name"],
    },
    "together": {
        "name": "Together",
        "description": "Co-occurrence album for multiple people",
        "params": ["person_ids", "person_names"],
    },
    "location": {
        "name": "Location Story",
        "description": "All photos from a location",
        "params": ["country", "city"],
    },
}
