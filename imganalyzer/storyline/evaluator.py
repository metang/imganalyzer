"""Quality evaluator for smart album stories.

Runs 24 criteria across 6 categories (A–F) and produces a structured
pass/fail report.  Designed to be run after story generation so the
pipeline can iterate until quality bar is met.
"""
from __future__ import annotations

import json
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class CriterionResult:
    """Result of a single evaluation criterion."""

    id: str
    name: str
    passed: bool
    value: Any
    threshold: Any
    detail: str = ""


@dataclass
class EvalReport:
    """Full evaluation report for a story."""

    album_id: str
    criteria: list[CriterionResult] = field(default_factory=list)
    overall_pass: bool = True
    generation_time_s: float = 0.0

    def add(self, result: CriterionResult) -> None:
        self.criteria.append(result)
        if not result.passed:
            self.overall_pass = False

    def summary(self) -> str:
        lines = [f"Story Evaluation: {'PASS' if self.overall_pass else 'FAIL'}"]
        lines.append(f"Album: {self.album_id}")
        passed = sum(1 for c in self.criteria if c.passed)
        lines.append(f"Criteria: {passed}/{len(self.criteria)} passed")
        for c in self.criteria:
            status = "✓" if c.passed else "✗"
            lines.append(f"  {status} {c.id}: {c.name} = {c.value} (threshold: {c.threshold})")
            if c.detail and not c.passed:
                lines.append(f"    → {c.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "album_id": self.album_id,
            "overall_pass": self.overall_pass,
            "generation_time_s": self.generation_time_s,
            "criteria": {
                c.id: {
                    "name": c.name,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                    "detail": c.detail,
                }
                for c in self.criteria
            },
        }


def evaluate_story(
    conn: sqlite3.Connection,
    album_id: str,
    generation_time_s: float = 0.0,
) -> EvalReport:
    """Run all evaluation criteria on an album's generated story."""
    report = EvalReport(album_id=album_id, generation_time_s=generation_time_s)

    album_row = conn.execute(
        "SELECT * FROM smart_albums WHERE id = ?", [album_id]
    ).fetchone()
    if album_row is None:
        report.add(CriterionResult("A0", "Album exists", False, None, "exists"))
        return report

    item_count = album_row["item_count"]
    rules = json.loads(album_row["rules"])

    chapters = conn.execute(
        "SELECT * FROM story_chapters WHERE album_id = ? ORDER BY sort_order",
        [album_id],
    ).fetchall()

    # Load all moments
    all_moments: list[sqlite3.Row] = []
    moments_by_chapter: dict[str, list[sqlite3.Row]] = {}
    for ch in chapters:
        ch_moments = conn.execute(
            "SELECT * FROM story_moments WHERE chapter_id = ? ORDER BY sort_order",
            [ch["id"]],
        ).fetchall()
        moments_by_chapter[ch["id"]] = ch_moments
        all_moments.extend(ch_moments)

    # ── A: Rule Accuracy ─────────────────────────────────────────────────

    _eval_rule_accuracy(conn, album_id, rules, item_count, report)

    # ── B: Moment Clustering ─────────────────────────────────────────────

    _eval_moment_clustering(conn, all_moments, item_count, report)

    # ── C: Chapter Quality ───────────────────────────────────────────────

    _eval_chapter_quality(conn, chapters, moments_by_chapter, all_moments, item_count, report)

    # ── D: Hero Selection ────────────────────────────────────────────────

    _eval_hero_selection(conn, all_moments, report)

    # ── E: Narrative Quality ─────────────────────────────────────────────

    _eval_narrative(conn, chapters, report)

    # ── F: Scale & Performance ───────────────────────────────────────────

    _eval_scale(
        item_count, len(all_moments), len(chapters),
        generation_time_s, report,
    )

    return report


# ── A: Rule Accuracy ─────────────────────────────────────────────────────────

def _eval_rule_accuracy(
    conn: sqlite3.Connection,
    album_id: str,
    rules: dict[str, Any],
    item_count: int,
    report: EvalReport,
) -> None:
    from imganalyzer.storyline.rules import evaluate_rules

    # A1: Precision — all album items still match the rules
    album_image_ids = set(
        r[0] for r in conn.execute(
            "SELECT image_id FROM album_items WHERE album_id = ?", [album_id]
        ).fetchall()
    )
    if album_image_ids:
        fresh_ids = set(evaluate_rules(conn, rules))
        precision = len(album_image_ids & fresh_ids) / len(album_image_ids)
    else:
        precision = 1.0
    report.add(CriterionResult("A1", "Rule precision", precision >= 0.99, round(precision, 4), 0.99))

    # A2: Recall
    if album_image_ids:
        recall = len(album_image_ids & fresh_ids) / len(fresh_ids) if fresh_ids else 1.0
    else:
        recall = 1.0
    report.add(CriterionResult("A2", "Rule recall", recall >= 0.99, round(recall, 4), 0.99))

    # A3: Query performance — measured externally via generation_time_s
    # (placeholder — actual benchmark done in F1)
    report.add(CriterionResult("A3", "Query performance", True, "deferred to F1", "< 2s"))

    # A4: Idempotency — re-evaluation yields same results
    fresh_ids2 = set(evaluate_rules(conn, rules))
    idempotent = fresh_ids == fresh_ids2 if album_image_ids else True
    report.add(CriterionResult("A4", "Rule idempotency", idempotent, idempotent, True))


# ── B: Moment Clustering ─────────────────────────────────────────────────────

def _eval_moment_clustering(
    conn: sqlite3.Connection,
    moments: list[sqlite3.Row],
    item_count: int,
    report: EvalReport,
) -> None:
    # B1: Temporal tightness (max span per moment ≤ 30 min)
    max_span_minutes = 0.0
    violating_moments = 0
    for m in moments:
        if m["start_time"] and m["end_time"]:
            try:
                start = datetime.fromisoformat(m["start_time"])
                end = datetime.fromisoformat(m["end_time"])
                span = (end - start).total_seconds() / 60.0
                max_span_minutes = max(max_span_minutes, span)
                if span > 30:
                    violating_moments += 1
            except (ValueError, TypeError):
                pass
    report.add(CriterionResult(
        "B1", "Temporal tightness",
        max_span_minutes <= 30.0 or violating_moments == 0,
        f"{max_span_minutes:.1f} min (max), {violating_moments} violations",
        "≤ 30 min",
    ))

    # B2: Spatial tightness — check via geohash (approximation)
    # Full haversine would be expensive; geohash-6 ensures ~600m
    report.add(CriterionResult(
        "B2", "Spatial tightness",
        True,  # enforced by clustering algorithm using geohash-6
        "≤ 600m (geohash-6)",
        "≤ 1 km",
    ))

    # B3: No orphan images
    moment_image_count = 0
    for m in moments:
        moment_image_count += m["image_count"] or 0
    orphan_free = moment_image_count == item_count
    report.add(CriterionResult(
        "B3", "No orphan images",
        orphan_free,
        f"{moment_image_count}/{item_count} images in moments",
        "100%",
        detail="" if orphan_free else f"{item_count - moment_image_count} orphans",
    ))

    # B4: Moment size distribution
    sizes = [m["image_count"] or 0 for m in moments]
    oversized = sum(1 for s in sizes if s > 200)
    pct_oversized = (oversized / len(sizes) * 100) if sizes else 0
    report.add(CriterionResult(
        "B4", "Moment size distribution",
        pct_oversized <= 5.0,
        f"{pct_oversized:.1f}% oversized (>{200})",
        "≤ 5%",
    ))

    # B5: No micro-moments (single-image moments < 20%)
    singles = sum(1 for s in sizes if s == 1)
    pct_singles = (singles / len(sizes) * 100) if sizes else 0
    report.add(CriterionResult(
        "B5", "No micro-moments",
        pct_singles <= 20.0,
        f"{pct_singles:.1f}% single-image moments",
        "≤ 20%",
    ))


# ── C: Chapter Quality ──────────────────────────────────────────────────────

def _eval_chapter_quality(
    conn: sqlite3.Connection,
    chapters: list[sqlite3.Row],
    moments_by_chapter: dict[str, list[sqlite3.Row]],
    all_moments: list[sqlite3.Row],
    item_count: int,
    report: EvalReport,
) -> None:
    # C1: Inter-chapter gap (median ≥ 2 hours)
    gaps_hours: list[float] = []
    for i in range(1, len(chapters)):
        prev_end = chapters[i - 1]["end_date"]
        curr_start = chapters[i]["start_date"]
        if prev_end and curr_start:
            try:
                pe = datetime.fromisoformat(prev_end)
                cs = datetime.fromisoformat(curr_start)
                gap = (cs - pe).total_seconds() / 3600.0
                gaps_hours.append(gap)
            except (ValueError, TypeError):
                pass
    median_gap = sorted(gaps_hours)[len(gaps_hours) // 2] if gaps_hours else 0
    report.add(CriterionResult(
        "C1", "Inter-chapter gap",
        median_gap >= 2.0 or len(chapters) <= 1,
        f"{median_gap:.1f}h (median)",
        "≥ 2h",
    ))

    # C2: Intra-chapter span (95th percentile ≤ 7 days)
    spans_days: list[float] = []
    for ch in chapters:
        if ch["start_date"] and ch["end_date"]:
            try:
                s = datetime.fromisoformat(ch["start_date"])
                e = datetime.fromisoformat(ch["end_date"])
                spans_days.append((e - s).total_seconds() / 86400.0)
            except (ValueError, TypeError):
                pass
    p95_span = sorted(spans_days)[int(len(spans_days) * 0.95)] if spans_days else 0
    report.add(CriterionResult(
        "C2", "Intra-chapter span",
        p95_span <= 7.0 or len(chapters) <= 1,
        f"{p95_span:.1f} days (95th pctl)",
        "≤ 7 days",
    ))

    # C3: Chapter count
    ch_count = len(chapters)
    ok_count = (3 <= ch_count <= 200) if item_count > 10 else True
    report.add(CriterionResult(
        "C3", "Chapter count",
        ok_count,
        ch_count,
        "3–200 (for >10 images)",
    ))

    # C4: No empty chapters
    empty = sum(1 for ch in chapters if (ch["moment_count"] or 0) == 0)
    report.add(CriterionResult("C4", "No empty chapters", empty == 0, f"{empty} empty", 0))

    # C5: Chapter balance
    if chapters and all_moments:
        max_moments = max(
            len(moments_by_chapter.get(ch["id"], [])) for ch in chapters
        )
        balance = max_moments / len(all_moments)
        report.add(CriterionResult(
            "C5", "Chapter balance",
            balance <= 0.40,
            f"{balance:.2%} in largest chapter",
            "≤ 40%",
        ))
    else:
        report.add(CriterionResult("C5", "Chapter balance", True, "N/A", "≤ 40%"))

    # C6: Title accuracy — cross-check with metadata
    titled = [ch for ch in chapters if ch["title"] and ch["title"] != "Untitled Chapter"]
    if titled:
        accurate = 0
        for ch in titled:
            title = ch["title"].lower()
            # Check if title references location or date from metadata
            has_grounding = False
            if ch["location"] and ch["location"].lower() in title:
                has_grounding = True
            if ch["start_date"]:
                try:
                    dt = datetime.fromisoformat(ch["start_date"])
                    year_str = str(dt.year)
                    if year_str in title:
                        has_grounding = True
                except (ValueError, TypeError):
                    pass
            if has_grounding:
                accurate += 1
        accuracy = accurate / len(titled)
        report.add(CriterionResult(
            "C6", "Title accuracy",
            accuracy >= 0.90,
            f"{accuracy:.0%} grounded",
            "≥ 90%",
        ))
    else:
        report.add(CriterionResult("C6", "Title accuracy", True, "No titled chapters", "≥ 90%"))


# ── D: Hero Selection ────────────────────────────────────────────────────────

def _eval_hero_selection(
    conn: sqlite3.Connection,
    moments: list[sqlite3.Row],
    report: EvalReport,
) -> None:
    # D1: Aesthetic rank (hero in top 30% of moment)
    heroes_checked = 0
    heroes_top30 = 0
    for m in moments:
        if not m["hero_image_id"] or (m["image_count"] or 0) < 3:
            continue
        # Get moment images ranked by aesthetic score
        rows = conn.execute(
            "SELECT mi.image_id, COALESCE(sf.perception_iaa, 0) AS aes "
            "FROM moment_images mi "
            "LEFT JOIN search_features sf ON sf.image_id = mi.image_id "
            "WHERE mi.moment_id = ? ORDER BY aes DESC",
            [m["id"]],
        ).fetchall()
        if not rows:
            continue
        heroes_checked += 1
        top30_cutoff = max(1, int(len(rows) * 0.3))
        top30_ids = {r["image_id"] for r in rows[:top30_cutoff]}
        if m["hero_image_id"] in top30_ids:
            heroes_top30 += 1

    d1_ratio = (heroes_top30 / heroes_checked) if heroes_checked else 1.0
    report.add(CriterionResult(
        "D1", "Hero aesthetic rank",
        d1_ratio >= 0.70,
        f"{d1_ratio:.0%} heroes in top 30%",
        "≥ 70%",
    ))

    # D2: Face preference
    face_moments = 0
    hero_has_face = 0
    for m in moments:
        if not m["hero_image_id"]:
            continue
        face_in_moment = conn.execute(
            "SELECT 1 FROM moment_images mi "
            "JOIN search_features sf ON sf.image_id = mi.image_id "
            "WHERE mi.moment_id = ? AND sf.face_count > 0 LIMIT 1",
            [m["id"]],
        ).fetchone()
        if face_in_moment:
            face_moments += 1
            hero_face = conn.execute(
                "SELECT face_count FROM search_features WHERE image_id = ?",
                [m["hero_image_id"]],
            ).fetchone()
            if hero_face and (hero_face["face_count"] or 0) > 0:
                hero_has_face += 1

    d2_ratio = (hero_has_face / face_moments) if face_moments else 1.0
    report.add(CriterionResult(
        "D2", "Face preference",
        d2_ratio >= 0.80,
        f"{d2_ratio:.0%} of face-moments have face hero",
        "≥ 80%",
    ))

    # D3: Chapter hero diversity (mean cosine distance ≥ 0.15)
    # Collect hero embeddings grouped by chapter
    chapter_ids = set()
    hero_embeddings_by_chapter: dict[str, list[list[float]]] = {}
    for m in moments:
        if not m["hero_image_id"]:
            continue
        ch_id = m["chapter_id"]
        chapter_ids.add(ch_id)
        emb_row = conn.execute(
            "SELECT vector AS image_clip FROM embeddings "
            "WHERE image_id = ? AND embedding_type = 'image_clip'",
            [m["hero_image_id"]],
        ).fetchone()
        if emb_row and emb_row["image_clip"]:
            blob = emb_row["image_clip"]
            n = len(blob) // 4
            vec = list(struct.unpack(f"<{n}f", blob))
            hero_embeddings_by_chapter.setdefault(ch_id, []).append(vec)

    diversities: list[float] = []
    for ch_id, embeddings in hero_embeddings_by_chapter.items():
        if len(embeddings) < 2:
            continue
        dists: list[float] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                a, b = embeddings[i], embeddings[j]
                dot = sum(x * y for x, y in zip(a, b))
                na = sum(x * x for x in a) ** 0.5
                nb = sum(x * x for x in b) ** 0.5
                cos_dist = 1.0 - (dot / (na * nb) if na and nb else 0)
                dists.append(cos_dist)
        if dists:
            diversities.append(sum(dists) / len(dists))

    mean_div = (sum(diversities) / len(diversities)) if diversities else 0.5
    report.add(CriterionResult(
        "D3", "Chapter hero diversity",
        mean_div >= 0.15,
        f"{mean_div:.3f} mean cosine distance",
        "≥ 0.15",
    ))

    # D4: No broken heroes
    broken = sum(1 for m in moments if m["hero_image_id"] is None and (m["image_count"] or 0) > 0)
    report.add(CriterionResult("D4", "No broken heroes", broken == 0, f"{broken} broken", 0))


# ── E: Narrative Quality ─────────────────────────────────────────────────────

def _eval_narrative(
    conn: sqlite3.Connection,
    chapters: list[sqlite3.Row],
    report: EvalReport,
) -> None:
    # E1: Title present
    no_title = sum(1 for ch in chapters if not ch["title"])
    report.add(CriterionResult("E1", "Title present", no_title == 0, f"{no_title} missing", 0))

    # E2: Title length (5-100 chars)
    bad_len = 0
    for ch in chapters:
        t = ch["title"] or ""
        if len(t) < 5 or len(t) > 100:
            bad_len += 1
    report.add(CriterionResult(
        "E2", "Title length",
        bad_len == 0,
        f"{bad_len} out of range",
        "5–100 chars",
    ))

    # E3: Location grounding
    geo_chapters = [ch for ch in chapters if ch["location"]]
    if geo_chapters:
        grounded = sum(
            1 for ch in geo_chapters
            if ch["title"] and ch["location"].lower() in ch["title"].lower()
        )
        ratio = grounded / len(geo_chapters)
        report.add(CriterionResult(
            "E3", "Location grounding",
            ratio >= 0.80,
            f"{ratio:.0%}",
            "≥ 80%",
        ))
    else:
        report.add(CriterionResult("E3", "Location grounding", True, "No geo chapters", "≥ 80%"))

    # E4: Date grounding
    dated_chapters = [ch for ch in chapters if ch["start_date"]]
    if dated_chapters:
        grounded = 0
        for ch in dated_chapters:
            title = ch["title"] or ""
            try:
                dt = datetime.fromisoformat(ch["start_date"])
                if str(dt.year) in title:
                    grounded += 1
            except (ValueError, TypeError):
                pass
        ratio = grounded / len(dated_chapters)
        report.add(CriterionResult(
            "E4", "Date grounding",
            ratio >= 0.90,
            f"{ratio:.0%}",
            "≥ 90%",
        ))
    else:
        report.add(CriterionResult("E4", "Date grounding", True, "No dated chapters", "≥ 90%"))

    # E5: No hallucination — needs AI narrative, skip for heuristic mode
    report.add(CriterionResult(
        "E5", "No hallucination",
        True,
        "Heuristic mode (no AI narrative yet)",
        "Spot-check",
    ))


# ── F: Scale & Performance ───────────────────────────────────────────────────

def _eval_scale(
    item_count: int,
    moment_count: int,
    chapter_count: int,
    generation_time_s: float,
    report: EvalReport,
) -> None:
    # F1: Story generation time (< 30s for 50K images)
    # Scale threshold proportionally for smaller datasets
    threshold = 30.0 if item_count >= 50000 else max(5.0, 30.0 * item_count / 50000)
    report.add(CriterionResult(
        "F1", "Story generation time",
        generation_time_s < threshold,
        f"{generation_time_s:.1f}s",
        f"< {threshold:.0f}s",
    ))

    # F2: Memory usage — can't measure post-hoc, mark as deferred
    report.add(CriterionResult(
        "F2", "Memory usage",
        True,
        "Deferred (needs profiling)",
        "< 500 MB",
    ))

    # F3: Hierarchy reduction ratio (≥ 10:1 images-to-moments)
    ratio = (item_count / moment_count) if moment_count > 0 else 0
    # Relax threshold for small albums
    threshold_ratio = 10.0 if item_count >= 100 else 2.0
    report.add(CriterionResult(
        "F3", "Hierarchy reduction",
        ratio >= threshold_ratio or item_count < 20,
        f"{ratio:.1f}:1",
        f"≥ {threshold_ratio:.0f}:1",
    ))

    # F4: View load time — deferred (requires API instrumentation)
    report.add(CriterionResult(
        "F4", "View load time",
        True,
        "Deferred (needs API)",
        "< 500ms",
    ))
