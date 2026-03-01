"""Batch processor — scan folders, register images, enqueue jobs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from rich.console import Console

from imganalyzer.db.queue import JobQueue
from imganalyzer.db.repository import Repository, ALL_MODULES
from imganalyzer.pipeline.modules import compute_file_fingerprint

console = Console()

# Supported image extensions (same as cli.py)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".gif",
    ".heic", ".heif",
    ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf", ".raf", ".rw2",
    ".dng", ".pef", ".srw", ".erf", ".kdc", ".mrw", ".3fr", ".fff",
}


class BatchProcessor:
    """Scan folders, register images in the DB, and enqueue analysis jobs."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.repo = Repository(conn)
        self.queue = JobQueue(conn)

    def ingest(
        self,
        folders: list[Path],
        modules: list[str] | None = None,
        force: bool = False,
        recursive: bool = True,
        compute_hash: bool = True,
        verbose: bool = False,
    ) -> dict[str, int]:
        """Scan folders for images, register in DB, enqueue jobs.

        Returns {registered: N, enqueued: N, skipped: N}.
        """
        target_modules = modules or ALL_MODULES
        stats = {"registered": 0, "enqueued": 0, "skipped": 0}

        # Collect all image files
        all_files: list[Path] = []
        for folder in folders:
            folder = folder.resolve()
            if not folder.is_dir():
                console.print(f"[yellow]Warning: '{folder}' is not a directory, skipping.[/yellow]")
                continue
            if recursive:
                # Single traversal — filter by suffix in Python instead of
                # issuing 52 separate rglob calls (26 extensions × 2 cases).
                all_files.extend(
                    p for p in folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
            else:
                all_files.extend(
                    p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )

        # Deduplicate
        all_files = list(dict.fromkeys(all_files))
        all_files.sort()

        if not all_files:
            console.print("[yellow]No image files found.[/yellow]")
            return stats

        total = len(all_files)
        console.print(f"[cyan]Found {total} image file(s). Registering...[/cyan]")

        # ── Batched ingest ────────────────────────────────────────────────
        # Wrap register + enqueue in explicit transactions (every BATCH_SIZE
        # images) to reduce fsync overhead.  At 500K images × 10 modules
        # the old code issued ~5M individual COMMITs; batching reduces this
        # to ~1000 COMMITs (500x fewer fsyncs).
        BATCH_SIZE = 500
        batch_start = 0

        while batch_start < total:
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = all_files[batch_start:batch_end]

            # Pre-fetch existing paths in one query to avoid N per-image SELECTs
            batch_paths = [str(p.resolve()) for p in batch]
            placeholders = ",".join("?" * len(batch_paths))
            existing_rows = self.conn.execute(
                f"SELECT id, file_path FROM images WHERE file_path IN ({placeholders})",
                batch_paths,
            ).fetchall()
            existing_map = {r["file_path"]: r["id"] for r in existing_rows}

            # Pre-fetch already-analyzed (image_id, module) pairs for this batch
            # to avoid N × M individual is_analyzed queries.
            existing_ids = list(existing_map.values())
            analyzed_set: set[tuple[int, str]] = set()
            if existing_ids and not force:
                # For each analysis table, check which image_ids have rows
                for module_name, table_name in (
                    ("metadata", "analysis_metadata"),
                    ("technical", "analysis_technical"),
                    ("local_ai", "analysis_local_ai"),
                    ("blip2", "analysis_blip2"),
                    ("objects", "analysis_objects"),
                    ("ocr", "analysis_ocr"),
                    ("faces", "analysis_faces"),
                    ("aesthetic", "analysis_aesthetic"),
                ):
                    if module_name not in target_modules:
                        continue
                    id_placeholders = ",".join("?" * len(existing_ids))
                    rows = self.conn.execute(
                        f"SELECT image_id FROM {table_name} "
                        f"WHERE image_id IN ({id_placeholders}) AND analyzed_at IS NOT NULL",
                        existing_ids,
                    ).fetchall()
                    for r in rows:
                        analyzed_set.add((r["image_id"], module_name))

                # Embeddings use a different check (no analyzed_at column)
                if "embedding" in target_modules:
                    id_placeholders = ",".join("?" * len(existing_ids))
                    rows = self.conn.execute(
                        f"SELECT DISTINCT image_id FROM embeddings "
                        f"WHERE image_id IN ({id_placeholders})",
                        existing_ids,
                    ).fetchall()
                    for r in rows:
                        analyzed_set.add((r["image_id"], "embedding"))

                # Cloud AI uses a different check (multiple rows per image)
                if "cloud_ai" in target_modules:
                    id_placeholders = ",".join("?" * len(existing_ids))
                    rows = self.conn.execute(
                        f"SELECT DISTINCT image_id FROM analysis_cloud_ai "
                        f"WHERE image_id IN ({id_placeholders}) AND analyzed_at IS NOT NULL",
                        existing_ids,
                    ).fetchall()
                    for r in rows:
                        analyzed_set.add((r["image_id"], "cloud_ai"))

                # Also check job_queue for done/skipped jobs — these may have
                # no analysis data (e.g. cloud_ai/aesthetic skipped due to
                # has_people guard) but should not be re-enqueued.
                id_placeholders = ",".join("?" * len(existing_ids))
                for mod in target_modules:
                    rows = self.conn.execute(
                        f"SELECT image_id FROM job_queue "
                        f"WHERE image_id IN ({id_placeholders}) AND module = ? "
                        f"AND status IN ('done', 'skipped')",
                        existing_ids + [mod],
                    ).fetchall()
                    for r in rows:
                        analyzed_set.add((r["image_id"], mod))

            # Run the batch inside a single transaction
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for i, path in enumerate(batch):
                    idx = batch_start + i + 1
                    file_path_str = batch_paths[i]

                    # Register or retrieve image
                    image_id = existing_map.get(file_path_str)
                    if image_id is None:
                        file_hash = compute_file_fingerprint(path) if compute_hash else None
                        file_size = path.stat().st_size
                        cur = self.conn.execute(
                            """INSERT INTO images (file_path, file_hash, file_size)
                               VALUES (?, ?, ?)""",
                            [file_path_str, file_hash, file_size],
                        )
                        image_id = cur.lastrowid
                        stats["registered"] += 1

                    # Enqueue analysis jobs
                    for module in target_modules:
                        if not force and (image_id, module) in analyzed_set:
                            stats["skipped"] += 1
                            continue

                        job_id = self.queue.enqueue(
                            image_id=image_id,
                            module=module,
                            priority=_module_priority(module),
                            force=force,
                            _auto_commit=False,
                        )
                        if job_id is not None:
                            stats["enqueued"] += 1
                        else:
                            stats["skipped"] += 1

                    # Emit structured progress line for the Electron UI
                    print(
                        "[PROGRESS] " + json.dumps({
                            "scanned": idx,
                            "total": total,
                            "registered": stats["registered"],
                            "enqueued": stats["enqueued"],
                            "skipped": stats["skipped"],
                            "current": file_path_str,
                        }),
                        flush=True,
                    )

                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

            batch_start = batch_end

        console.print(
            f"[green]Ingest complete.[/green] "
            f"Registered: {stats['registered']}  "
            f"Enqueued: {stats['enqueued']}  "
            f"Skipped: {stats['skipped']}"
        )
        return stats

    def rebuild_module(
        self,
        module: str,
        image_path: str | None = None,
        force: bool = True,
        failed_only: bool = False,
    ) -> int:
        """Re-enqueue a specific module for all (or one) image(s).

        Clears existing analysis data (except overridden fields) and
        re-enqueues the jobs.  Returns count of jobs enqueued.

        Args:
            failed_only: When True, only re-enqueue images that currently have
                         a ``failed`` job for this module in the queue — rather
                         than re-enqueuing every registered image.  Use this
                         for the "Retry failed" button so that already-done
                         jobs are not re-run.
        """
        if module not in ALL_MODULES:
            raise ValueError(
                f"Unknown module: '{module}'. Choose from: {', '.join(ALL_MODULES)}"
            )

        if image_path:
            img = self.repo.get_image_by_path(image_path)
            if img is None:
                raise ValueError(f"Image not found in database: {image_path}")
            image_ids = [img["id"]]
        elif failed_only:
            rows = self.conn.execute(
                "SELECT DISTINCT image_id FROM job_queue WHERE module = ? AND status = 'failed'",
                [module],
            ).fetchall()
            image_ids = [r[0] for r in rows]
        else:
            image_ids = self.repo.iter_image_ids()

        if not image_ids:
            console.print("[yellow]No images found to rebuild.[/yellow]")
            return 0

        console.print(
            f"[cyan]Rebuilding '{module}' for {len(image_ids)} image(s)...[/cyan]"
        )

        # Clear only the failed jobs for this module (keep done/skipped intact)
        if failed_only:
            self.conn.execute(
                "DELETE FROM job_queue WHERE module = ? AND status = 'failed'",
                [module],
            )
            self.conn.commit()
        else:
            self.queue.clear_module(module)

        # Clear analysis data (not overrides) and re-enqueue in a single transaction
        count = 0
        self.conn.execute("BEGIN")
        try:
            for image_id in image_ids:
                self.repo.clear_analysis(image_id, module, commit=False)
                job_id = self.queue.enqueue(
                    image_id=image_id,
                    module=module,
                    priority=_module_priority(module),
                    force=True,
                    _auto_commit=False,
                )
                if job_id is not None:
                    count += 1
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        console.print(
            f"[green]Rebuild queued.[/green] {count} job(s) enqueued for '{module}'."
        )
        return count


def _module_priority(module: str) -> int:
    """Assign default priorities — metadata first, objects highest GPU (gates cloud), embedding last."""
    return {
        "metadata":  100,
        "technical":  90,
        "objects":    85,   # highest GPU priority — unlocks cloud_ai/aesthetic
        "blip2":      80,
        "local_ai":   80,
        "ocr":        78,
        "faces":      77,
        "cloud_ai":   70,
        "aesthetic":  60,
        "embedding":  50,
    }.get(module, 0)
