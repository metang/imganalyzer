#!/usr/bin/env python
"""Semantic search diagnostic tool.

Usage:
    conda run -n imganalyzer python scripts/debug_search.py [query] [--limit N]

Sections
--------
1. DB overview     — image counts, embedding coverage, analysis coverage
2. Raw cosine      — top-N raw similarity scores for image_clip and description_clip
3. Staleness       — compare embedding computed_at vs analysis analyzed_at
4. RRF comparison  — SearchEngine output vs raw cosine rankings side-by-side
5. Pagination      — verify search_json_cmd candidate_ids → SQL LIMIT path

All output goes to stderr AND is written to debug_search.log in the repo root.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

# ── Setup: make the repo importable ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Tee logger: writes to stderr + log file simultaneously ───────────────────
LOG_PATH = REPO_ROOT / "debug_search.log"

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _setup_logging() -> _Tee:
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    tee = _Tee(sys.stderr, log_file)
    return tee


OUT: _Tee  # set in main()


def p(*args, **kwargs) -> None:
    """Print to the tee (stderr + log file)."""
    print(*args, **kwargs, file=OUT)


def hr(char: str = "─", width: int = 90) -> None:
    p(char * width)


def section(title: str) -> None:
    p()
    hr("═")
    p(f"  {title}")
    hr("═")
    p()


def subsection(title: str) -> None:
    p()
    hr("─")
    p(f"  {title}")
    hr("─")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trunc(s: str | None, n: int = 80) -> str:
    if not s:
        return "(empty)"
    s = s.replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


def _now_text() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _reconstruct_desc_text(conn, image_id: int) -> str:
    """Reproduce the exact text that _run_embedding would embed today."""
    import json
    parts: list[str] = []

    local = conn.execute(
        "SELECT description, scene_type, main_subject FROM analysis_local_ai WHERE image_id = ?",
        [image_id],
    ).fetchone()
    if local:
        for val in (local["description"], local["scene_type"], local["main_subject"]):
            if val:
                parts.append(val)

    clouds = conn.execute(
        "SELECT description, scene_type, main_subject FROM analysis_cloud_ai WHERE image_id = ?",
        [image_id],
    ).fetchall()
    for c in clouds:
        for val in (c["description"], c["scene_type"], c["main_subject"]):
            if val and val not in parts:
                parts.append(val)

    return " ".join(parts)


# ── Section 1: DB overview ────────────────────────────────────────────────────

def section_overview(conn) -> None:
    section("SECTION 1 — DB OVERVIEW")

    total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    p(f"Total images in DB:          {total_images}")

    has_local  = conn.execute("SELECT COUNT(DISTINCT image_id) FROM analysis_local_ai").fetchone()[0]
    has_cloud  = conn.execute("SELECT COUNT(DISTINCT image_id) FROM analysis_cloud_ai").fetchone()[0]
    has_img_e  = conn.execute("SELECT COUNT(*) FROM embeddings WHERE embedding_type='image_clip'").fetchone()[0]
    has_desc_e = conn.execute("SELECT COUNT(*) FROM embeddings WHERE embedding_type='description_clip'").fetchone()[0]

    p(f"Images with local_ai:        {has_local}")
    p(f"Images with cloud_ai:        {has_cloud}")
    p()
    p(f"Embeddings — image_clip:     {has_img_e}")
    p(f"Embeddings — description_cl: {has_desc_e}")

    # Breakdown: both / only image / only desc / neither
    both = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT image_id FROM embeddings WHERE embedding_type='image_clip'
            INTERSECT
            SELECT image_id FROM embeddings WHERE embedding_type='description_clip'
        )
    """).fetchone()[0]
    only_img = has_img_e - both
    only_desc = has_desc_e - both
    neither = total_images - (both + only_img + only_desc)

    p()
    p("Embedding coverage breakdown:")
    p(f"  both image_clip + description_clip : {both}")
    p(f"  image_clip only                    : {only_img}")
    p(f"  description_clip only              : {only_desc}")
    p(f"  neither (no embedding at all)      : {neither}")

    if neither > 0:
        p()
        p(f"  ⚠  {neither} images have NO embeddings — they are invisible to semantic search.")

    if only_img > 0:
        p()
        p(f"  ⚠  {only_img} images have image_clip but NO description_clip.")
        p("     These are scored only on visual features, not on AI descriptions.")
        p("     This can happen if embeddings were computed before local/cloud AI ran,")
        p("     or before the recent _run_embedding change that skips image_clip when")
        p("     a description is available.")

    if only_desc > 0 and has_img_e == 0:
        p()
        p(f"  ⚠  {only_desc} images have description_clip but NO image_clip at all.")
        p("     Semantic search has NO visual signal — purely text-to-text.")


# ── Section 2: Raw cosine scores ──────────────────────────────────────────────

def section_raw_cosine(conn, query: str, limit: int) -> tuple[list, list]:
    """Returns (image_ranked, desc_ranked) as lists of dicts for later sections."""
    section("SECTION 2 — RAW COSINE SCORES")

    from imganalyzer.embeddings.clip_embedder import (
        CLIPEmbedder, vector_from_bytes, cosine_similarity,
    )

    embedder = CLIPEmbedder()
    visual_query = f"a photo of {query}" if not query.lower().startswith("a photo of") else query
    text_query   = query

    p(f"Query (text→image):  '{visual_query}'")
    p(f"Query (text→text):   '{text_query}'")
    p()

    visual_vec = vector_from_bytes(embedder.embed_text(visual_query))
    text_vec   = vector_from_bytes(embedder.embed_text(text_query))

    img_embs  = conn.execute(
        "SELECT e.image_id, e.vector, e.computed_at, i.file_path "
        "FROM embeddings e JOIN images i ON i.id = e.image_id "
        "WHERE e.embedding_type='image_clip'"
    ).fetchall()

    desc_embs = conn.execute(
        "SELECT e.image_id, e.vector, e.computed_at, i.file_path "
        "FROM embeddings e JOIN images i ON i.id = e.image_id "
        "WHERE e.embedding_type='description_clip'"
    ).fetchall()

    # ── image_clip rankings ───────────────────────────────────────────────────
    import numpy as np
    image_scored = []
    for row in img_embs:
        vec = vector_from_bytes(bytes(row["vector"]))
        sim = cosine_similarity(visual_vec, vec)
        image_scored.append({
            "image_id":   row["image_id"],
            "file_path":  row["file_path"],
            "cosine":     sim,
            "computed_at": row["computed_at"],
        })
    image_scored.sort(key=lambda x: -x["cosine"])

    subsection(f"image_clip  top-{limit}  (query: 'a photo of {query}')")
    p(f"{'Rank':>4}  {'image_id':>8}  {'cosine':>8}  {'computed_at':>20}  description (from DB)")
    hr()
    for rank, r in enumerate(image_scored[:limit]):
        desc = _reconstruct_desc_text(conn, r["image_id"])
        p(f"{rank+1:>4}  {r['image_id']:>8}  {r['cosine']:>8.4f}  {str(r['computed_at']):>20}  {_trunc(desc, 60)}")

    if not image_scored:
        p("  (no image_clip embeddings found)")

    # ── description_clip rankings ─────────────────────────────────────────────
    desc_scored = []
    for row in desc_embs:
        vec = vector_from_bytes(bytes(row["vector"]))
        sim = cosine_similarity(text_vec, vec)
        desc_text_in_db = _reconstruct_desc_text(conn, row["image_id"])
        desc_scored.append({
            "image_id":        row["image_id"],
            "file_path":       row["file_path"],
            "cosine":          sim,
            "computed_at":     row["computed_at"],
            "embedded_text":   desc_text_in_db,
        })
    desc_scored.sort(key=lambda x: -x["cosine"])

    subsection(f"description_clip  top-{limit}  (query: '{query}')")
    p(f"{'Rank':>4}  {'image_id':>8}  {'cosine':>8}  {'computed_at':>20}  embedded text")
    hr()
    for rank, r in enumerate(desc_scored[:limit]):
        p(f"{rank+1:>4}  {r['image_id']:>8}  {r['cosine']:>8.4f}  {str(r['computed_at']):>20}  {_trunc(r['embedded_text'], 60)}")

    if not desc_scored:
        p("  (no description_clip embeddings found)")

    # ── Score distribution stats ──────────────────────────────────────────────
    subsection("Cosine score distribution")
    if image_scored:
        import statistics
        img_vals = [r["cosine"] for r in image_scored]
        p(f"image_clip    — count={len(img_vals)}  min={min(img_vals):.4f}  max={max(img_vals):.4f}  "
          f"mean={statistics.mean(img_vals):.4f}  stdev={statistics.stdev(img_vals) if len(img_vals)>1 else 0:.4f}")
    if desc_scored:
        import statistics
        desc_vals = [r["cosine"] for r in desc_scored]
        p(f"description_cl— count={len(desc_vals)}  min={min(desc_vals):.4f}  max={max(desc_vals):.4f}  "
          f"mean={statistics.mean(desc_vals):.4f}  stdev={statistics.stdev(desc_vals) if len(desc_vals)>1 else 0:.4f}")

    return image_scored, desc_scored


# ── Section 3: Staleness check ────────────────────────────────────────────────

def section_staleness(conn, image_scored: list, desc_scored: list) -> None:
    section("SECTION 3 — STALENESS CHECK")
    p("Comparing embedding computed_at vs analysis analyzed_at for top results.")
    p("A stale embedding was computed BEFORE the latest AI analysis ran,")
    p("meaning the stored vector may not reflect the current description.")
    p()

    # Take top 10 from each list
    check_ids = list({r["image_id"] for r in (image_scored[:10] + desc_scored[:10])})

    p(f"{'image_id':>8}  {'emb_img_at':>20}  {'emb_desc_at':>20}  {'local_ai_at':>20}  {'cloud_ai_at':>20}  stale?")
    hr()

    for image_id in check_ids:
        emb_img = conn.execute(
            "SELECT computed_at FROM embeddings WHERE image_id=? AND embedding_type='image_clip'",
            [image_id],
        ).fetchone()
        emb_desc = conn.execute(
            "SELECT computed_at FROM embeddings WHERE image_id=? AND embedding_type='description_clip'",
            [image_id],
        ).fetchone()
        local_ai = conn.execute(
            "SELECT analyzed_at FROM analysis_local_ai WHERE image_id=?", [image_id]
        ).fetchone()
        cloud_ai = conn.execute(
            "SELECT MAX(analyzed_at) AS analyzed_at FROM analysis_cloud_ai WHERE image_id=?",
            [image_id],
        ).fetchone()

        emb_img_at   = str(emb_img["computed_at"])   if emb_img  else "(none)"
        emb_desc_at  = str(emb_desc["computed_at"])  if emb_desc else "(none)"
        local_ai_at  = str(local_ai["analyzed_at"])  if local_ai else "(none)"
        cloud_ai_at  = str(cloud_ai["analyzed_at"])  if cloud_ai and cloud_ai["analyzed_at"] else "(none)"

        # Determine staleness: description_clip computed before most recent AI analysis?
        stale = False
        relevant_emb_at = emb_desc_at if emb_desc else emb_img_at
        for ai_at in (local_ai_at, cloud_ai_at):
            if ai_at != "(none)" and relevant_emb_at != "(none)":
                if relevant_emb_at < ai_at:
                    stale = True

        flag = "⚠ STALE" if stale else "ok"
        p(f"{image_id:>8}  {emb_img_at:>20}  {emb_desc_at:>20}  {local_ai_at:>20}  {cloud_ai_at:>20}  {flag}")


# ── Section 4: RRF comparison ─────────────────────────────────────────────────

def section_rrf_comparison(conn, query: str, limit: int,
                            image_scored: list, desc_scored: list) -> None:
    section("SECTION 4 — RRF COMPARISON")

    from imganalyzer.db.search import SearchEngine
    engine = SearchEngine(conn)

    p("Running SearchEngine._semantic_search() ...")
    try:
        engine_results = engine._semantic_search(query, limit)
    except Exception as e:
        p(f"  ERROR running _semantic_search: {e}")
        traceback.print_exc(file=OUT)
        return

    # Build rank maps from raw cosine lists
    img_rank_map  = {r["image_id"]: rank for rank, r in enumerate(image_scored)}
    desc_rank_map = {r["image_id"]: rank for rank, r in enumerate(desc_scored)}
    img_cos_map   = {r["image_id"]: r["cosine"] for r in image_scored}
    desc_cos_map  = {r["image_id"]: r["cosine"] for r in desc_scored}

    p()
    p(f"{'RRF':>4}  {'image_id':>8}  {'RRF score':>10}  "
      f"{'img_rank':>9}  {'img_cos':>8}  "
      f"{'desc_rank':>10}  {'desc_cos':>9}  file_path")
    hr()

    for rrf_rank, r in enumerate(engine_results):
        iid = r["image_id"]
        img_r  = img_rank_map.get(iid, -1)
        desc_r = desc_rank_map.get(iid, -1)
        img_c  = img_cos_map.get(iid, 0.0)
        desc_c = desc_cos_map.get(iid, 0.0)
        fname  = Path(r["file_path"]).name if r["file_path"] else "?"
        p(f"{rrf_rank+1:>4}  {iid:>8}  {r['score']:>10.5f}  "
          f"{img_r+1 if img_r>=0 else '-':>9}  {img_c:>8.4f}  "
          f"{desc_r+1 if desc_r>=0 else '-':>10}  {desc_c:>9.4f}  {fname}")

    p()

    # Highlight images that appear in engine results but ranked low on raw cosine
    subsection("Suspicious promotions (low raw cosine → high RRF rank)")
    suspicious = []
    for rrf_rank, r in enumerate(engine_results[:limit//2]):
        iid = r["image_id"]
        img_r  = img_rank_map.get(iid, len(image_scored))
        desc_r = desc_rank_map.get(iid, len(desc_scored))
        if img_r > limit and desc_r > limit:
            suspicious.append((rrf_rank+1, iid, img_r+1, desc_r+1, r["file_path"]))

    if suspicious:
        p(f"Found {len(suspicious)} results in top-{limit//2} RRF that ranked >#{limit} in BOTH raw lists:")
        for rrf_r, iid, ir, dr, fp in suspicious:
            desc = _reconstruct_desc_text(conn, iid)
            p(f"  RRF#{rrf_r}  id={iid}  img_rank={ir}  desc_rank={dr}  {Path(fp).name}")
            p(f"    description: {_trunc(desc, 100)}")
    else:
        p("No suspicious promotions found.")

    # Images with high cosine that didn't make it into RRF results
    subsection("Missed high-cosine images (should have ranked but didn't)")
    engine_ids = {r["image_id"] for r in engine_results}
    missed = []
    for r in image_scored[:limit]:
        if r["image_id"] not in engine_ids:
            missed.append(("image_clip", r["image_id"], r["cosine"], r["file_path"]))
    for r in desc_scored[:limit]:
        if r["image_id"] not in engine_ids:
            missed.append(("desc_clip", r["image_id"], r["cosine"], r["file_path"]))

    if missed:
        p(f"Found {len(missed)} high-cosine images NOT in SearchEngine results (pool cutoff issue?):")
        for src, iid, cos, fp in missed[:20]:
            p(f"  [{src}]  id={iid}  cosine={cos:.4f}  {Path(fp).name}")
    else:
        p("All top raw-cosine images are present in RRF results. Pool size is adequate.")


# ── Section 5: Pagination bug check ──────────────────────────────────────────

def section_pagination(conn, query: str, limit: int = 200) -> None:
    section("SECTION 5 — PAGINATION / ORDERING BUG CHECK")

    from imganalyzer.db.search import SearchEngine
    engine = SearchEngine(conn)

    p(f"Simulating search_json_cmd with query='{query}', limit={limit}")
    p()

    # Step 1: get candidate_ids from search engine (same as CLI does with limit*4)
    search_limit = limit * 4
    p(f"Step 1: SearchEngine.search(limit={search_limit}) ...")
    results = engine.search(query, limit=search_limit, mode="semantic")
    candidate_ids = [r["image_id"] for r in results]
    score_map = {r["image_id"]: r["score"] for r in results}
    p(f"  Returned {len(candidate_ids)} candidate IDs")

    if not candidate_ids:
        p("  ⚠ No results from search engine — nothing to check further.")
        return

    # Step 2: SQL IN query with LIMIT applied BEFORE score sort
    id_placeholders = ",".join("?" * len(candidate_ids))
    sql = f"""
        SELECT i.id AS image_id, i.file_path
        FROM images i
        LEFT JOIN analysis_local_ai la ON la.image_id = i.id
        WHERE i.id IN ({id_placeholders})
        LIMIT ? OFFSET 0
    """
    rows_sql = conn.execute(sql, candidate_ids + [limit]).fetchall()
    sql_ids = [r["image_id"] for r in rows_sql]

    p(f"Step 2: SQL WHERE IN ({len(candidate_ids)} ids) LIMIT {limit}")
    p(f"  SQL returned {len(rows_sql)} rows")

    # Check how many high-score results were cut by LIMIT before re-sort
    if len(sql_ids) < len(candidate_ids):
        missing = set(candidate_ids) - set(sql_ids)
        p(f"  ⚠ SQL LIMIT cut {len(missing)} candidate IDs before score sort!")
        p(f"  This means high-scoring images beyond row #{limit} are DROPPED")
        p(f"  regardless of their search score.")
        p()
        p("  Scores of CUT candidates (should have appeared in results):")
        cut_scores = sorted([(iid, score_map[iid]) for iid in missing], key=lambda x: -x[1])
        for iid, sc in cut_scores[:10]:
            p(f"    image_id={iid}  score={sc:.5f}")
    else:
        p(f"  ✓ All {len(candidate_ids)} candidates survived LIMIT {limit} — no pagination loss.")

    # Step 3: check that Python re-sort matches expected order
    subsection("Score ordering after Python re-sort vs SearchEngine order")
    # Sort sql_ids by score_map
    sql_ids_sorted = sorted(sql_ids, key=lambda iid: -(score_map.get(iid, 0.0)))

    mismatches = 0
    for expected_rank, (engine_result, sql_iid) in enumerate(zip(results, sql_ids_sorted)):
        if engine_result["image_id"] != sql_iid:
            mismatches += 1

    if mismatches == 0:
        p("  ✓ Python re-sort produces the same order as SearchEngine.")
    else:
        p(f"  ⚠ {mismatches} ordering mismatches between SearchEngine and SQL+re-sort.")
        p("  This is expected if SQL LIMIT cut some candidates (see above).")


# ── Section 6: Description quality spot-check ─────────────────────────────────

def section_description_quality(conn, query: str, limit: int) -> None:
    section("SECTION 6 — DESCRIPTION QUALITY SPOT-CHECK")
    p("For top and bottom ranked images (by description_clip cosine),")
    p("print the full description text to assess AI output quality.")
    p()

    from imganalyzer.embeddings.clip_embedder import (
        CLIPEmbedder, vector_from_bytes, cosine_similarity,
    )
    embedder = CLIPEmbedder()
    text_vec = vector_from_bytes(embedder.embed_text(query))

    desc_embs = conn.execute(
        "SELECT e.image_id, e.vector, i.file_path "
        "FROM embeddings e JOIN images i ON i.id = e.image_id "
        "WHERE e.embedding_type='description_clip'"
    ).fetchall()

    if not desc_embs:
        p("  (no description_clip embeddings found)")
        return

    scored = []
    for row in desc_embs:
        vec = vector_from_bytes(bytes(row["vector"]))
        sim = cosine_similarity(text_vec, vec)
        scored.append((sim, row["image_id"], row["file_path"]))
    scored.sort(reverse=True)

    def print_image_detail(rank_label: str, sim: float, image_id: int, file_path: str) -> None:
        import json as _json
        p(f"\n  [{rank_label}]  image_id={image_id}  cosine={sim:.4f}")
        p(f"  file: {Path(file_path).name}")
        local = conn.execute(
            "SELECT description, scene_type, main_subject, keywords FROM analysis_local_ai WHERE image_id=?",
            [image_id],
        ).fetchone()
        if local:
            p(f"  local description : {_trunc(local['description'] or '', 120)}")
            p(f"  local scene_type  : {local['scene_type']}")
            p(f"  local main_subject: {local['main_subject']}")
            try:
                kw = _json.loads(local["keywords"] or "[]")
                p(f"  local keywords    : {', '.join(kw[:10])}")
            except Exception:
                pass
        clouds = conn.execute(
            "SELECT provider, description, scene_type, main_subject FROM analysis_cloud_ai WHERE image_id=?",
            [image_id],
        ).fetchall()
        for c in clouds:
            p(f"  [{c['provider']}] description: {_trunc(c['description'] or '', 120)}")
            p(f"  [{c['provider']}] scene_type : {c['scene_type']}")
            p(f"  [{c['provider']}] main_subj  : {c['main_subject']}")

        embedded_text = _reconstruct_desc_text(conn, image_id)
        p(f"  → would embed text: {_trunc(embedded_text, 140)}")

    subsection(f"TOP {limit} results")
    for rank, (sim, iid, fp) in enumerate(scored[:limit]):
        print_image_detail(f"#{rank+1}", sim, iid, fp)

    subsection(f"BOTTOM {min(5, len(scored))} results")
    for rank, (sim, iid, fp) in enumerate(scored[-5:]):
        actual_rank = len(scored) - 5 + rank + 1
        print_image_detail(f"#{actual_rank}/{len(scored)}", sim, iid, fp)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global OUT

    parser = argparse.ArgumentParser(description="Semantic search diagnostic")
    parser.add_argument("query", nargs="?", default="sunset",
                        help="Search query to diagnose (default: 'sunset')")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of top results to inspect per section (default: 20)")
    parser.add_argument("--skip-clip", action="store_true",
                        help="Skip sections that load the CLIP model (faster, sections 2/3/4/6 skipped)")
    args = parser.parse_args()

    OUT = _setup_logging()

    p("=" * 90)
    p(f"  SEMANTIC SEARCH DIAGNOSTIC  —  query='{args.query}'  limit={args.limit}")
    p(f"  {_now_text()}")
    p(f"  Log file: {LOG_PATH}")
    p("=" * 90)

    from imganalyzer.db.connection import get_db
    conn = get_db()
    p(f"\nDB path: {conn.execute('PRAGMA database_list').fetchone()[2]}")

    # ── Section 1 (no CLIP needed) ─────────────────────────────────────────
    try:
        section_overview(conn)
    except Exception as e:
        p(f"\n[ERROR in section 1] {e}")
        traceback.print_exc(file=OUT)

    if args.skip_clip:
        p("\n--skip-clip set: skipping sections 2–6 (CLIP model not loaded)")
        return

    # ── Sections 2–6 (require CLIP) ────────────────────────────────────────
    image_scored, desc_scored = [], []

    try:
        image_scored, desc_scored = section_raw_cosine(conn, args.query, args.limit)
    except Exception as e:
        p(f"\n[ERROR in section 2] {e}")
        traceback.print_exc(file=OUT)

    try:
        section_staleness(conn, image_scored, desc_scored)
    except Exception as e:
        p(f"\n[ERROR in section 3] {e}")
        traceback.print_exc(file=OUT)

    try:
        section_rrf_comparison(conn, args.query, args.limit, image_scored, desc_scored)
    except Exception as e:
        p(f"\n[ERROR in section 4] {e}")
        traceback.print_exc(file=OUT)

    try:
        section_pagination(conn, args.query)
    except Exception as e:
        p(f"\n[ERROR in section 5] {e}")
        traceback.print_exc(file=OUT)

    try:
        section_description_quality(conn, args.query, min(args.limit, 5))
    except Exception as e:
        p(f"\n[ERROR in section 6] {e}")
        traceback.print_exc(file=OUT)

    p()
    hr("═")
    p("  DIAGNOSTIC COMPLETE")
    hr("═")
    p(f"\nFull log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
