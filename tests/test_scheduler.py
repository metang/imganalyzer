"""Tests for VRAMBudget and ResourceScheduler."""
from __future__ import annotations

import threading
import pytest

from imganalyzer.pipeline.vram_budget import VRAMBudget, _MODULE_VRAM_GB, _EXCLUSIVE_MODULES
from imganalyzer.pipeline.scheduler import ResourceScheduler, _GPU_PHASES, _PREREQUISITES


# ── VRAMBudget tests ──────────────────────────────────────────────────────────

class TestVRAMBudget:
    def _make(self, total: float = 16.0, fraction: float = 0.70) -> VRAMBudget:
        return VRAMBudget(total_vram_gb=total, fraction=fraction)

    def test_budget_calculation(self):
        vram = self._make(16.0, 0.70)
        assert vram.budget_gb == pytest.approx(11.2)

    def test_empty_state(self):
        vram = self._make()
        assert vram.used_gb == 0.0
        assert vram.free_gb == pytest.approx(11.2)
        assert vram.loaded_modules == []

    def test_reserve_and_release(self):
        vram = self._make()
        vram.reserve("faces")
        assert "faces" in vram.loaded_modules
        assert vram.used_gb == pytest.approx(0.5)

        vram.release("faces")
        assert vram.loaded_modules == []
        assert vram.used_gb == 0.0

    def test_can_fit_small_models(self):
        vram = self._make()
        # All three small models fit together: 0.5 + 1.3 + 0.95 = 2.75 GB
        assert vram.can_fit("faces")
        vram.reserve("faces")
        assert vram.can_fit("ocr")
        vram.reserve("ocr")
        assert vram.can_fit("embedding")
        vram.reserve("embedding")
        assert vram.used_gb == pytest.approx(2.75)

    def test_exclusive_module_alone(self):
        vram = self._make()
        assert vram.can_fit("blip2")
        vram.reserve("blip2")
        assert vram.used_gb == pytest.approx(6.0)
        # Nothing else can load while blip2 is loaded
        assert not vram.can_fit("faces")
        assert not vram.can_fit("embedding")
        assert not vram.can_fit("objects")

    def test_exclusive_blocked_by_existing(self):
        vram = self._make()
        vram.reserve("faces")
        # blip2 can't load while anything else is loaded
        assert not vram.can_fit("blip2")

    def test_reserve_idempotent(self):
        vram = self._make()
        vram.reserve("faces")
        vram.reserve("faces")  # no-op
        assert vram.used_gb == pytest.approx(0.5)

    def test_release_nonexistent(self):
        vram = self._make()
        vram.release("faces")  # no error

    def test_cpu_module_always_fits(self):
        vram = self._make()
        assert vram.can_fit("metadata")
        assert vram.can_fit("cloud_ai")

    def test_reserve_over_budget_raises(self):
        vram = self._make(total=2.0, fraction=0.70)  # 1.4 GB budget
        vram.reserve("embedding")  # 0.95 GB
        with pytest.raises(RuntimeError, match="Cannot load"):
            vram.reserve("ocr")  # 1.3 GB — would exceed 1.4 GB

    def test_is_exclusive(self):
        vram = self._make()
        assert vram.is_exclusive("blip2")
        assert not vram.is_exclusive("faces")
        assert not vram.is_exclusive("embedding")

    def test_vram_for(self):
        vram = self._make()
        assert vram.vram_for("objects") == pytest.approx(2.4)
        assert vram.vram_for("metadata") == 0.0  # CPU module

    def test_thread_safety(self):
        """Reserve/release from multiple threads without crashes."""
        vram = self._make()
        errors: list[Exception] = []

        def _worker(module: str):
            try:
                for _ in range(100):
                    vram.reserve(module)
                    vram.release(module)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=_worker, args=("faces",)),
            threading.Thread(target=_worker, args=("embedding",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ── ResourceScheduler tests ───────────────────────────────────────────────────

class TestResourceScheduler:
    def _make(self, **kwargs) -> ResourceScheduler:
        vram = VRAMBudget(total_vram_gb=16.0, fraction=0.70)
        defaults = {
            "vram_budget": vram,
            "gpu_batch_sizes": {"objects": 4, "blip2": 1, "embedding": 16},
            "default_batch_size": 10,
            "cpu_workers": 4,
            "cloud_workers": 4,
        }
        defaults.update(kwargs)
        return ResourceScheduler(**defaults)

    def test_gpu_phases(self):
        s = self._make()
        assert len(s.gpu_phases) == 3
        assert s.modules_for_phase(0) == ["objects"]
        assert s.modules_for_phase(1) == ["blip2"]
        assert s.modules_for_phase(2) == ["faces", "ocr", "embedding"]

    def test_co_resident_phase(self):
        s = self._make()
        assert not s.is_co_resident_phase(0)  # objects alone
        assert not s.is_co_resident_phase(1)  # blip2 alone
        assert s.is_co_resident_phase(2)       # faces + ocr + embedding

    def test_batch_sizes(self):
        s = self._make()
        assert s.batch_size_for("objects") == 4
        assert s.batch_size_for("blip2") == 1
        assert s.batch_size_for("embedding") == 16
        assert s.batch_size_for("faces") == 10  # falls back to default

    def test_batch_capable(self):
        s = self._make()
        assert s.is_batch_capable("objects")
        assert s.is_batch_capable("blip2")
        assert s.is_batch_capable("embedding")
        assert not s.is_batch_capable("faces")
        assert not s.is_batch_capable("ocr")

    def test_boosted_cloud_workers(self):
        s = self._make(cloud_workers=4, cloud_boost_factor=2)
        assert s.boosted_cloud_workers() == 8

    def test_module_classification(self):
        s = self._make()
        assert s.is_gpu("objects")
        assert s.is_gpu("blip2")
        assert not s.is_gpu("metadata")
        assert s.is_cloud("cloud_ai")
        assert s.is_cloud("aesthetic")
        assert s.is_local_io("metadata")
        assert s.is_local_io("technical")
        assert s.is_io("cloud_ai")
        assert s.is_io("metadata")

    def test_prerequisites(self):
        s = self._make()
        assert s.prerequisite_for("cloud_ai") == "objects"
        assert s.prerequisite_for("faces") == "objects"
        assert s.prerequisite_for("ocr") == "objects"
        assert s.prerequisite_for("aesthetic") == "objects"
        assert s.prerequisite_for("objects") is None
        assert s.prerequisite_for("blip2") is None
        assert s.prerequisite_for("embedding") is None

    def test_shutdown_flag(self):
        event = threading.Event()
        s = self._make(shutdown_event=event)
        assert not s.is_shutdown
        event.set()
        assert s.is_shutdown

    def test_phase_out_of_range(self):
        s = self._make()
        assert s.modules_for_phase(99) == []
        assert not s.is_co_resident_phase(99)


# ── Integration: co-residency fits within VRAM ────────────────────────────────

class TestCoResidencyFitCheck:
    """Verify that the declared co-resident phase actually fits."""

    def test_phase2_fits_in_budget(self):
        vram = VRAMBudget(total_vram_gb=16.0, fraction=0.70)
        phase2_modules = _GPU_PHASES[2]  # faces, ocr, embedding
        total = sum(_MODULE_VRAM_GB.get(m, 0) for m in phase2_modules)
        assert total < vram.budget_gb, (
            f"Phase 2 modules ({phase2_modules}) need {total:.2f} GB "
            f"but budget is {vram.budget_gb:.2f} GB"
        )
        # Actually reserve them all
        for mod in phase2_modules:
            assert vram.can_fit(mod)
            vram.reserve(mod)

    def test_phase2_fits_on_8gb_card(self):
        """Even an 8 GB card should handle co-residency at 70%."""
        vram = VRAMBudget(total_vram_gb=8.0, fraction=0.70)  # 5.6 GB budget
        phase2_modules = _GPU_PHASES[2]
        total = sum(_MODULE_VRAM_GB.get(m, 0) for m in phase2_modules)
        assert total < vram.budget_gb

    def test_blip2_exclusive_in_registry(self):
        """BLIP-2 must be in the exclusive set."""
        assert "blip2" in _EXCLUSIVE_MODULES

    def test_all_gpu_phases_have_known_modules(self):
        """Every module in _GPU_PHASES must have a VRAM entry."""
        for phase in _GPU_PHASES:
            for mod in phase:
                assert mod in _MODULE_VRAM_GB, f"{mod} missing from _MODULE_VRAM_GB"
