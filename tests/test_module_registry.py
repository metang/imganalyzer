from __future__ import annotations

import sqlite3
from pathlib import Path


def _make_test_db(tmp_path: Path) -> sqlite3.Connection:
    from imganalyzer.db.schema import ensure_schema

    db_path = tmp_path / "registry.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def test_module_registry_matches_backend_consumers() -> None:
    from imganalyzer.db import repository
    from imganalyzer.pipeline import module_registry as registry
    from imganalyzer.pipeline import scheduler, vram_budget, worker
    from imganalyzer.pipeline.batch import _module_priority

    assert registry.ACTIVE_MODULES == (
        "metadata",
        "technical",
        "caption",
        "objects",
        "faces",
        "perception",
        "embedding",
    )
    assert registry.LEGACY_MODULES == {
        "local_ai",
        "blip2",
        "ocr",
        "cloud_ai",
        "aesthetic",
    }
    assert dict(registry.LEGACY_QUEUE_MODULE_MAP) == {
        "local_ai": "caption",
        "blip2": "caption",
        "cloud_ai": "caption",
        "aesthetic": "perception",
    }
    assert "ocr" not in registry.LEGACY_QUEUE_MODULE_MAP

    assert repository.MODULE_TABLE_MAP == dict(registry.MODULE_TABLE_MAP)
    assert repository.ALL_MODULES == list(registry.ACTIVE_MODULES)

    assert scheduler.GPU_MODULES == registry.GPU_MODULES
    assert scheduler.LOCAL_IO_MODULES == registry.LOCAL_IO_MODULES
    assert scheduler.IO_MODULES == registry.IO_MODULES
    assert scheduler._PREREQUISITES == dict(registry.PREREQUISITES)
    assert scheduler._GPU_PHASES == [list(phase) for phase in registry.GPU_PHASES]

    assert worker.GPU_MODULES == set(registry.GPU_MODULES)
    assert worker.LOCAL_IO_MODULES == set(registry.LOCAL_IO_MODULES)
    assert worker.IO_MODULES == set(registry.IO_MODULES)
    assert worker._PREREQUISITES == dict(registry.PREREQUISITES)
    assert worker._DEPENDENTS == {
        module: list(dependents) for module, dependents in registry.DEPENDENTS.items()
    }
    assert worker._FTS_MODULES == set(registry.FTS_MODULES)
    assert worker.Worker._GPU_BATCH_SIZES == dict(registry.DEFAULT_GPU_BATCH_SIZES)

    assert vram_budget._MODULE_VRAM_GB == dict(registry.MODULE_VRAM_GB)
    assert vram_budget._EXCLUSIVE_MODULES == registry.EXCLUSIVE_GPU_MODULES

    for module in registry.ACTIVE_MODULES:
        assert _module_priority(module) == registry.module_priority(module)
    assert _module_priority("ocr") == 0


def test_module_registry_derived_sets_and_legacy_aliases_are_consistent() -> None:
    from imganalyzer.pipeline import module_registry as registry

    active_names = {module.name for module in registry.MODULES if module.active}
    legacy_names = {module.name for module in registry.MODULES if not module.active}
    remapped_aliases = set(registry.LEGACY_QUEUE_MODULE_MAP)

    assert set(registry.ACTIVE_MODULES) == active_names
    assert registry.LEGACY_MODULES == legacy_names
    assert active_names.isdisjoint(legacy_names)
    assert remapped_aliases <= legacy_names
    assert set(registry.LEGACY_QUEUE_MODULE_MAP.values()) <= active_names
    assert "ocr" in legacy_names
    assert "ocr" not in remapped_aliases

    expected_dependents: dict[str, set[str]] = {}
    for module, prerequisite in registry.PREREQUISITES.items():
        expected_dependents.setdefault(prerequisite, set()).add(module)
    assert {
        prerequisite: set(dependents)
        for prerequisite, dependents in registry.DEPENDENTS.items()
    } == expected_dependents

    assert registry.gpu_phase_labels() == list(registry.GPU_PHASE_LABELS)


def test_registry_legacy_queue_map_remaps_active_jobs_and_preserves_ocr(tmp_path: Path) -> None:
    from imganalyzer.db.queue import JobQueue
    from imganalyzer.db.repository import Repository
    from imganalyzer.pipeline.module_registry import LEGACY_QUEUE_MODULE_MAP

    conn = _make_test_db(tmp_path)
    repo = Repository(conn)
    queue = JobQueue(conn)

    blip2_id = repo.register_image(file_path="/photos/blip2.jpg")
    local_ai_id = repo.register_image(file_path="/photos/local-ai.jpg")
    cloud_ai_id = repo.register_image(file_path="/photos/cloud-ai.jpg")
    aesthetic_id = repo.register_image(file_path="/photos/aesthetic.jpg")
    ocr_id = repo.register_image(file_path="/photos/ocr.jpg")
    duplicate_id = repo.register_image(file_path="/photos/duplicate.jpg")

    queue.enqueue(blip2_id, "blip2")
    queue.enqueue(local_ai_id, "local_ai")
    queue.enqueue(cloud_ai_id, "cloud_ai")
    queue.enqueue(aesthetic_id, "aesthetic")
    queue.enqueue(ocr_id, "ocr")
    queue.enqueue(duplicate_id, "blip2")
    queue.enqueue(duplicate_id, "caption")

    remapped = queue.remap_pending_modules(dict(LEGACY_QUEUE_MODULE_MAP))

    assert remapped == {"updated": 4, "deleted": 1}
    rows = conn.execute(
        """SELECT image_id, module, status
           FROM job_queue
           WHERE status IN ('pending', 'running')
           ORDER BY image_id, module"""
    ).fetchall()

    modules_by_image = {
        int(row["image_id"]): str(row["module"])
        for row in rows
        if int(row["image_id"]) != duplicate_id
    }
    assert modules_by_image == {
        blip2_id: "caption",
        local_ai_id: "caption",
        cloud_ai_id: "caption",
        aesthetic_id: "perception",
        ocr_id: "ocr",
    }

    duplicate_rows = [
        str(row["module"]) for row in rows if int(row["image_id"]) == duplicate_id
    ]
    assert duplicate_rows == ["caption"]
