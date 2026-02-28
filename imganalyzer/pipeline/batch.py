"""Batch processor — scan folders, register images, enqueue jobs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from rich.console import Console

from imganalyzer.db.queue import JobQueue
from imganalyzer.db.repository import Repository, ALL_MODULES
from imganalyzer.pipeline.modules import compute_file_hash

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
                for ext in IMAGE_EXTENSIONS:
                    all_files.extend(folder.rglob(f"*{ext}"))
                    # Also try uppercase
                    all_files.extend(folder.rglob(f"*{ext.upper()}"))
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

        for idx, path in enumerate(all_files, start=1):
            file_path_str = str(path.resolve())

            # Check if already registered
            existing = self.repo.get_image_by_path(file_path_str)
            if existing:
                image_id = existing["id"]
            else:
                file_hash = compute_file_hash(path) if compute_hash else None
                file_size = path.stat().st_size
                image_id = self.repo.register_image(
                    file_path=file_path_str,
                    file_hash=file_hash,
                    file_size=file_size,
                )
                stats["registered"] += 1

            # Enqueue analysis jobs
            for module in target_modules:
                # Cache check: skip if already analyzed and not force
                if not force and self.repo.is_analyzed(image_id, module):
                    stats["skipped"] += 1
                    continue

                job_id = self.queue.enqueue(
                    image_id=image_id,
                    module=module,
                    priority=_module_priority(module),
                    force=force,
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

        # Clear analysis data (not overrides) and re-enqueue
        count = 0
        for image_id in image_ids:
            self.repo.clear_analysis(image_id, module)
            job_id = self.queue.enqueue(
                image_id=image_id,
                module=module,
                priority=_module_priority(module),
                force=True,
            )
            if job_id is not None:
                count += 1

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
