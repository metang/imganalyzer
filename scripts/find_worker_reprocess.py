"""Identify images where technical/faces were processed by a remote worker
(using a resized 1024px image) and should be reprocessed on the master device
for full-resolution accuracy.

Usage:
    python scripts/find_worker_reprocess.py          # dry-run: just count
    python scripts/find_worker_reprocess.py --fix     # re-enqueue for master
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

# Master-only modules: need original full-res image
MASTER_ONLY_MODULES = ("technical", "faces")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find worker-processed jobs that need master reprocessing",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Re-enqueue affected jobs so the master reprocesses them",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path.home() / ".cache" / "imganalyzer" / "imganalyzer.db",
    )
    args = parser.parse_args()

    db = sqlite3.connect(str(args.db))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")

    for module in MASTER_ONLY_MODULES:
        rows = db.execute(
            """SELECT jq.id, jq.image_id, i.file_path, jq.last_node_id
               FROM job_queue jq
               JOIN images i ON i.id = jq.image_id
               WHERE jq.module = ?
                 AND jq.status = 'done'
                 AND jq.last_node_role = 'worker'""",
            [module],
        ).fetchall()

        print(f"\n{module}: {len(rows)} jobs completed by remote workers")
        if rows:
            for r in rows[:10]:
                node = r["last_node_id"] or "unknown"
                name = Path(r["file_path"]).name
                print(f"  job={r['id']} image={r['image_id']} {name} (worker: {node})")
            if len(rows) > 10:
                print(f"  ... and {len(rows) - 10} more")

        if args.fix and rows:
            job_ids = [r["id"] for r in rows]
            ph = ",".join("?" * len(job_ids))
            db.execute(
                f"""UPDATE job_queue
                    SET status = 'pending',
                        attempts = 0,
                        error_message = NULL,
                        skip_reason = NULL,
                        started_at = NULL,
                        completed_at = NULL,
                        last_node_id = NULL,
                        last_node_role = 'force'
                    WHERE id IN ({ph})""",
                job_ids,
            )
            db.commit()
            print(f"  → Re-enqueued {len(job_ids)} jobs for master reprocessing")

    db.close()


if __name__ == "__main__":
    main()
