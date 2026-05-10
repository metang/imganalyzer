"""Canonical metadata for image-analysis pipeline modules.

This module is intentionally data-only so backend layers can import it
without pulling in runners, database connections, or GPU dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ModuleMetadata:
    """Static metadata for one pipeline module."""

    name: str
    table: str
    active: bool
    kind: str
    priority: int = 0
    prerequisite: str | None = None
    queue_remap_to: str | None = None
    vram_gb: float = 0.0
    exclusive_gpu: bool = False
    batch_capable: bool = False
    fts_indexed: bool = False
    master_only: bool = False
    distributed_cache_optional: bool = False


MODULES: tuple[ModuleMetadata, ...] = (
    ModuleMetadata(
        "metadata",
        "analysis_metadata",
        active=True,
        kind="local_io",
        priority=100,
        fts_indexed=True,
        master_only=True,
    ),
    ModuleMetadata(
        "technical",
        "analysis_technical",
        active=True,
        kind="local_io",
        priority=90,
        master_only=True,
    ),
    ModuleMetadata(
        "caption",
        "analysis_caption",
        active=True,
        kind="gpu",
        priority=80,
        vram_gb=8.7,
        fts_indexed=True,
    ),
    ModuleMetadata(
        "local_ai",
        "analysis_caption",
        active=False,
        kind="legacy",
        queue_remap_to="caption",
    ),
    ModuleMetadata(
        "blip2",
        "analysis_blip2",
        active=False,
        kind="legacy",
        queue_remap_to="caption",
    ),
    ModuleMetadata(
        "objects",
        "analysis_objects",
        active=True,
        kind="gpu",
        priority=85,
        vram_gb=2.4,
        batch_capable=True,
    ),
    ModuleMetadata("ocr", "analysis_ocr", active=False, kind="legacy"),
    ModuleMetadata(
        "faces",
        "analysis_faces",
        active=True,
        kind="gpu",
        priority=77,
        prerequisite="objects",
        vram_gb=1.0,
        fts_indexed=True,
        master_only=True,
    ),
    ModuleMetadata(
        "cloud_ai",
        "analysis_cloud_ai",
        active=False,
        kind="legacy",
        queue_remap_to="caption",
    ),
    ModuleMetadata(
        "aesthetic",
        "analysis_aesthetic",
        active=False,
        kind="legacy",
        queue_remap_to="perception",
    ),
    ModuleMetadata(
        "perception",
        "analysis_perception",
        active=True,
        kind="gpu",
        priority=60,
        vram_gb=13.8,
        exclusive_gpu=True,
    ),
    ModuleMetadata(
        "embedding",
        "embeddings",
        active=True,
        kind="gpu",
        priority=50,
        prerequisite="objects",
        vram_gb=0.95,
        batch_capable=True,
        distributed_cache_optional=True,
    ),
)

MODULE_REGISTRY: Mapping[str, ModuleMetadata] = MappingProxyType(
    {module.name: module for module in MODULES}
)
MODULE_TABLE_MAP: Mapping[str, str] = MappingProxyType(
    {module.name: module.table for module in MODULES}
)

ACTIVE_MODULES: tuple[str, ...] = tuple(module.name for module in MODULES if module.active)
LEGACY_MODULES: frozenset[str] = frozenset(
    module.name for module in MODULES if not module.active
)
FULL_RESULT_MODULES: tuple[str, ...] = (
    "metadata",
    "technical",
    "caption",
    "blip2",
    "objects",
    "ocr",
    "faces",
    "aesthetic",
    "perception",
)

MODULE_PRIORITIES: Mapping[str, int] = MappingProxyType(
    {module.name: module.priority for module in MODULES if module.active}
)

GPU_MODULES: frozenset[str] = frozenset(
    module.name for module in MODULES if module.active and module.kind == "gpu"
)
LOCAL_IO_MODULES: frozenset[str] = frozenset(
    module.name for module in MODULES if module.active and module.kind == "local_io"
)
IO_MODULES: frozenset[str] = LOCAL_IO_MODULES

PREREQUISITES: Mapping[str, str] = MappingProxyType(
    {
        module.name: module.prerequisite
        for module in MODULES
        if module.active and module.prerequisite is not None
    }
)

_dependents: dict[str, list[str]] = {}
for _module, _prerequisite in PREREQUISITES.items():
    _dependents.setdefault(_prerequisite, []).append(_module)
DEPENDENTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {module: tuple(dependents) for module, dependents in _dependents.items()}
)

BATCH_CAPABLE_MODULES: frozenset[str] = frozenset(
    module.name for module in MODULES if module.active and module.batch_capable
)
FTS_MODULES: frozenset[str] = frozenset(
    module.name for module in MODULES if module.active and module.fts_indexed
)

MODULE_VRAM_GB: Mapping[str, float] = MappingProxyType(
    {module.name: module.vram_gb for module in MODULES if module.vram_gb > 0.0}
)
EXCLUSIVE_GPU_MODULES: frozenset[str] = frozenset(
    module.name for module in MODULES if module.active and module.exclusive_gpu
)

GPU_PHASES: tuple[tuple[str, ...], ...] = (
    ("caption",),
    ("objects",),
    ("faces", "embedding"),
    ("perception",),
)
GPU_PHASE_LABELS: tuple[str, ...] = tuple(
    f"Phase {index} — {', '.join(phase)}" for index, phase in enumerate(GPU_PHASES)
)
INDEPENDENT_GPU_MODULES: frozenset[str] = frozenset()

DEFAULT_GPU_BATCH_SIZES: Mapping[str, int] = MappingProxyType(
    {
        "objects": 4,
        "embedding": 16,
        "faces": 8,
    }
)

LEGACY_QUEUE_MODULE_MAP: Mapping[str, str] = MappingProxyType(
    {
        module.name: module.queue_remap_to
        for module in MODULES
        if module.queue_remap_to is not None
    }
)

REMOTE_DECODE_PRIORITY_MODULES: tuple[str, ...] = (
    "caption",
    "objects",
    "embedding",
    "perception",
)
DISTRIBUTED_CONTEXT_MODULES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "faces": ("objects",),
        "embedding": ("caption",),
    }
)
DISTRIBUTED_SEARCH_MODULES: frozenset[str] = FTS_MODULES
DISTRIBUTED_MASTER_ONLY_MODULES: frozenset[str] = frozenset(
    module.name for module in MODULES if module.active and module.master_only
)
DISTRIBUTED_CACHE_OPTIONAL_MODULES: frozenset[str] = frozenset(
    module.name
    for module in MODULES
    if module.active and module.distributed_cache_optional
)
DISTRIBUTED_LOCAL_AI_MODULES: frozenset[str] = GPU_MODULES
DISTRIBUTED_ALWAYS_AVAILABLE_MODULES: frozenset[str] = frozenset()
LEASE_TTL_FLOORS_SECONDS: Mapping[str, int] = MappingProxyType(
    {
        "caption": 300,
        "perception": 300,
    }
)


def module_priority(module: str) -> int:
    """Return the default queue priority for *module*."""
    return MODULE_PRIORITIES.get(module, 0)


def gpu_phase_labels() -> list[str]:
    """Return user-facing labels for the configured GPU phases."""
    return list(GPU_PHASE_LABELS)
