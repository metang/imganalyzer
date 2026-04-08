"""Smart-album rule-to-SQL compiler.

Each rule type has a ``compile`` function that returns a SQL fragment and
parameters.  The top-level :func:`compile_rules` builds a complete query
from an album's rule set, returning ``(sql, params)`` ready for
``cursor.execute``.

Design note: The compiler uses ``search_features`` and
``face_occurrences`` with indexed columns for performance.  It never
touches the full search pipeline (FTS5+CLIP+RRF) — that path is too slow
for smart albums with 50K+ images.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass
class CompiledFragment:
    """A SQL fragment with its bound parameters."""

    joins: list[str]
    conditions: list[str]
    params: list[Any]


# ── Individual rule compilers ────────────────────────────────────────────────

def _compile_person(rule: dict[str, Any], idx: int) -> CompiledFragment:
    """Compile a person rule (single or co-occurrence).

    mode='all' → image must contain ALL listed persons (co-occurrence).
    mode='any' → image must contain at least ONE listed person.
    """
    person_ids: list[str] = rule.get("person_ids", [])
    mode = rule.get("mode", "any")

    if not person_ids:
        return CompiledFragment([], [], [])

    joins: list[str] = []
    conditions: list[str] = []
    params: list[Any] = []

    if mode == "all":
        for j, pid in enumerate(person_ids):
            alias = f"fp{idx}_{j}"
            joins.append(
                f"JOIN face_occurrences {alias} "
                f"ON {alias}.image_id = i.id AND {alias}.person_id = ?"
            )
            params.append(pid)
    else:
        alias = f"fp{idx}"
        placeholders = ",".join("?" for _ in person_ids)
        joins.append(
            f"JOIN face_occurrences {alias} "
            f"ON {alias}.image_id = i.id AND {alias}.person_id IN ({placeholders})"
        )
        params.extend(person_ids)

    return CompiledFragment(joins, conditions, params)


def _compile_date_range(rule: dict[str, Any], idx: int) -> CompiledFragment:
    """Compile a date_range rule using search_features.date_time_original."""
    conditions: list[str] = []
    params: list[Any] = []
    start = rule.get("start")
    end = rule.get("end")
    if start:
        conditions.append("sf.date_time_original >= ?")
        params.append(start)
    if end:
        conditions.append("sf.date_time_original <= ?")
        params.append(end)
    return CompiledFragment([], conditions, params)


def _compile_location(rule: dict[str, Any], idx: int) -> CompiledFragment:
    """Compile a location rule (country and/or city)."""
    conditions: list[str] = []
    params: list[Any] = []
    country = rule.get("country")
    city = rule.get("city")
    if country:
        conditions.append("sf.location_country = ?")
        params.append(country)
    if city:
        conditions.append("am.location_city = ?")
        params.append(city)
    return CompiledFragment(
        ["LEFT JOIN analysis_metadata am ON am.image_id = i.id"] if city else [],
        conditions,
        params,
    )


def _compile_keyword(rule: dict[str, Any], idx: int) -> CompiledFragment:
    """Compile a keyword rule using FTS5 search_index."""
    values: list[str] = rule.get("values", [])
    if not values:
        return CompiledFragment([], [], [])

    match_expr = " OR ".join(values)
    alias = f"fts{idx}"
    return CompiledFragment(
        joins=[
            f"JOIN search_index {alias} "
            f"ON {alias}.rowid = i.id AND {alias}.search_index MATCH ?"
        ],
        conditions=[],
        params=[match_expr],
    )


def _compile_scene(rule: dict[str, Any], idx: int) -> CompiledFragment:
    """Compile a scene_type rule."""
    values: list[str] = rule.get("values", [])
    if not values:
        return CompiledFragment([], [], [])
    placeholders = ",".join("?" for _ in values)
    return CompiledFragment(
        joins=["JOIN analysis_caption ac ON ac.image_id = i.id"],
        conditions=[f"ac.scene_type IN ({placeholders})"],
        params=list(values),
    )


def _compile_camera(rule: dict[str, Any], idx: int) -> CompiledFragment:
    """Compile a camera rule (make and/or model)."""
    conditions: list[str] = []
    params: list[Any] = []
    joins: list[str] = []
    make = rule.get("make")
    model = rule.get("model")
    if make or model:
        joins.append(
            "JOIN analysis_metadata cam ON cam.image_id = i.id"
        )
    if make:
        conditions.append("cam.camera_make = ?")
        params.append(make)
    if model:
        conditions.append("cam.camera_model = ?")
        params.append(model)
    return CompiledFragment(joins, conditions, params)


def _compile_min_quality(rule: dict[str, Any], idx: int) -> CompiledFragment:
    """Compile a min_quality rule (aesthetic_score threshold)."""
    threshold = rule.get("aesthetic_score")
    if threshold is None:
        return CompiledFragment([], [], [])
    return CompiledFragment(
        joins=[],
        conditions=["sf.perception_iaa >= ?"],
        params=[threshold],
    )


# ── Registry ─────────────────────────────────────────────────────────────────

_COMPILERS: dict[str, Any] = {
    "person": _compile_person,
    "date_range": _compile_date_range,
    "location": _compile_location,
    "keyword": _compile_keyword,
    "scene": _compile_scene,
    "camera": _compile_camera,
    "min_quality": _compile_min_quality,
}


# ── Top-level compiler ───────────────────────────────────────────────────────

def compile_rules(rules_json: str | dict) -> tuple[str, list[Any]]:
    """Compile a smart-album rule set into a ``(sql, params)`` tuple.

    Parameters
    ----------
    rules_json:
        Either a JSON string or already-parsed dict with keys
        ``match`` (``'all'`` or ``'any'``) and ``rules`` (list of rule
        objects).

    Returns
    -------
    tuple[str, list[Any]]
        A complete SELECT query returning ``image_id`` values, plus bound
        parameters.
    """
    if isinstance(rules_json, str):
        rules_obj = json.loads(rules_json)
    else:
        rules_obj = rules_json

    match_mode = rules_obj.get("match", "all")
    rule_list: list[dict[str, Any]] = rules_obj.get("rules", [])

    if not rule_list:
        return "SELECT id AS image_id FROM images WHERE 0", []

    all_joins: list[str] = []
    all_conditions: list[str] = []
    all_params: list[Any] = []

    # Always join search_features for date/location/quality filtering
    needs_sf = any(
        r.get("type") in ("date_range", "location", "min_quality")
        for r in rule_list
    )

    seen_joins: set[str] = set()

    for idx, rule in enumerate(rule_list):
        rule_type = rule.get("type", "")
        compiler = _COMPILERS.get(rule_type)
        if compiler is None:
            continue

        fragment = compiler(rule, idx)
        for j in fragment.joins:
            # Deduplicate joins by their table alias
            if j not in seen_joins:
                all_joins.append(j)
                seen_joins.add(j)
        all_conditions.extend(fragment.conditions)
        all_params.extend(fragment.params)

    # Build query
    select = "SELECT DISTINCT i.id AS image_id FROM images i"
    if needs_sf:
        select += "\nJOIN search_features sf ON sf.image_id = i.id"

    joins_sql = "\n".join(all_joins)
    if joins_sql:
        select += "\n" + joins_sql

    if all_conditions:
        joiner = " AND " if match_mode == "all" else " OR "
        where = "\nWHERE " + joiner.join(all_conditions)
    else:
        where = ""

    sql = select + where
    return sql, all_params


def evaluate_rules(
    conn: sqlite3.Connection,
    rules_json: str | dict,
) -> list[int]:
    """Evaluate rules and return matching image IDs.

    This is the main entry point for materializing album membership.
    """
    sql, params = compile_rules(rules_json)
    rows = conn.execute(sql, params).fetchall()
    return [r[0] for r in rows]
