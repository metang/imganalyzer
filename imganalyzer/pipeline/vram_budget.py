"""VRAM budget tracker for GPU model co-residency.

Tracks which GPU models are currently loaded and their estimated VRAM
footprint.  The scheduler uses this to decide whether a new model can
be loaded alongside existing ones or must wait for them to unload.

Thread-safe: all public methods are guarded by a reentrant lock so the
scheduler can call ``reserve`` / ``release`` from different GPU threads.
"""

from __future__ import annotations

import threading
from typing import Optional


# Peak VRAM (GB) per module including model weights + batch activations.
# Tuned for the default _GPU_BATCH_SIZES in worker.py.
_MODULE_VRAM_GB: dict[str, float] = {
    "objects":   2.4,   # GroundingDINO mixed fp16/fp32, batch=4
    "blip2":     6.0,   # BLIP-2 FlanT5-XL fp16 + beam search
    "ocr":       1.3,   # TrOCR large-printed fp16
    "faces":     1.0,   # InsightFace buffalo_l ONNX (1 GB arena cap)
    "embedding": 0.95,  # CLIP ViT-L/14 fp16, batch=16
}

# Modules that must run alone (peak VRAM > 50% of a typical budget).
_EXCLUSIVE_MODULES: frozenset[str] = frozenset({"blip2"})

# Default VRAM reservation fraction (matches set_per_process_memory_fraction).
_DEFAULT_FRACTION = 0.70


class VRAMBudget:
    """Thread-safe tracker for GPU memory allocation across modules."""

    def __init__(
        self,
        total_vram_gb: Optional[float] = None,
        fraction: float = _DEFAULT_FRACTION,
    ) -> None:
        self._lock = threading.RLock()
        self._total_gb = total_vram_gb or self._detect_vram()
        self._budget_gb = self._total_gb * fraction
        self._loaded: dict[str, float] = {}  # module -> reserved GB

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def budget_gb(self) -> float:
        """Usable VRAM budget in GB."""
        return self._budget_gb

    @property
    def used_gb(self) -> float:
        """Currently reserved VRAM in GB."""
        with self._lock:
            return sum(self._loaded.values())

    @property
    def free_gb(self) -> float:
        """Remaining VRAM headroom in GB."""
        with self._lock:
            return self._budget_gb - sum(self._loaded.values())

    @property
    def loaded_modules(self) -> list[str]:
        """List of currently loaded GPU module names."""
        with self._lock:
            return list(self._loaded.keys())

    def can_fit(self, module: str) -> bool:
        """Return True if *module* can be loaded within the current budget.

        Also checks exclusive-mode constraints: if an exclusive module is
        already loaded, nothing else can load; if *module* is exclusive,
        nothing else may be loaded.
        """
        vram = _MODULE_VRAM_GB.get(module, 0.0)
        if vram == 0.0:
            return True  # CPU/cloud module — no VRAM needed

        with self._lock:
            # Exclusive constraint
            if module in _EXCLUSIVE_MODULES and self._loaded:
                return False
            if self._loaded.keys() & _EXCLUSIVE_MODULES:
                return False
            # Already loaded (idempotent)
            if module in self._loaded:
                return True
            return (sum(self._loaded.values()) + vram) <= self._budget_gb

    def reserve(self, module: str) -> None:
        """Mark *module* as loaded, reserving its VRAM.

        Raises ``RuntimeError`` if the module would exceed the budget.
        """
        vram = _MODULE_VRAM_GB.get(module, 0.0)
        if vram == 0.0:
            return

        with self._lock:
            if module in self._loaded:
                return  # already reserved
            if not self.can_fit(module):
                raise RuntimeError(
                    f"Cannot load {module} ({vram:.2f} GB): "
                    f"would exceed budget ({self.used_gb:.2f}/{self._budget_gb:.2f} GB used, "
                    f"loaded: {list(self._loaded.keys())})"
                )
            self._loaded[module] = vram

    def release(self, module: str) -> None:
        """Mark *module* as unloaded, freeing its VRAM reservation."""
        with self._lock:
            self._loaded.pop(module, None)

    def is_exclusive(self, module: str) -> bool:
        """Return True if *module* requires exclusive GPU access."""
        return module in _EXCLUSIVE_MODULES

    def vram_for(self, module: str) -> float:
        """Return the estimated peak VRAM (GB) for *module*."""
        return _MODULE_VRAM_GB.get(module, 0.0)

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _detect_vram() -> float:
        """Auto-detect total GPU VRAM in GB.  Falls back to 8 GB."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_mem / (1024 ** 3)
        except Exception:
            pass
        return 8.0

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"VRAMBudget(total={self._total_gb:.1f} GB, "
                f"budget={self._budget_gb:.1f} GB, "
                f"used={self.used_gb:.1f} GB, "
                f"loaded={list(self._loaded.keys())})"
            )
