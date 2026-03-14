"""Tests for VRAMBudget and ResourceScheduler."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest

from imganalyzer.pipeline.vram_budget import VRAMBudget, _MODULE_VRAM_GB, _EXCLUSIVE_MODULES
from imganalyzer.pipeline.scheduler import ResourceScheduler, _GPU_PHASES


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
        assert vram.used_gb == pytest.approx(1.0)

        vram.release("faces")
        assert vram.loaded_modules == []
        assert vram.used_gb == 0.0

    def test_can_fit_small_models(self):
        vram = self._make()
        # Small models fit together: 1.0 + 0.95 = 1.95 GB
        assert vram.can_fit("faces")
        vram.reserve("faces")
        assert vram.can_fit("embedding")
        vram.reserve("embedding")
        assert vram.used_gb == pytest.approx(1.95)

    def test_exclusive_module_alone(self):
        vram = self._make()
        assert vram.can_fit("perception")

    def test_exclusive_blocked_by_existing(self):
        vram = self._make()
        vram.reserve("faces")
        # perception can't load while anything else is loaded
        assert not vram.can_fit("perception")

    def test_reserve_idempotent(self):
        vram = self._make()
        vram.reserve("faces")
        vram.reserve("faces")  # no-op
        assert vram.used_gb == pytest.approx(1.0)

    def test_release_nonexistent(self):
        vram = self._make()
        vram.release("faces")  # no error

    def test_cpu_module_always_fits(self):
        """Modules with 0 VRAM always fit."""
        vram = self._make(total=8.0, fraction=0.70)
        assert vram.can_fit("metadata")

    def test_reserve_over_budget_raises(self):
        vram = self._make(total=2.0, fraction=0.70)  # 1.4 GB budget
        vram.reserve("embedding")  # 0.95 GB
        with pytest.raises(RuntimeError, match="Cannot load"):
            vram.reserve("faces")  # 1.0 GB — would exceed 1.4 GB

    def test_is_exclusive(self):
        vram = self._make()
        assert vram.is_exclusive("perception")
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
            "gpu_batch_sizes": {"objects": 4, "embedding": 16},
            "default_batch_size": 10,
            "cpu_workers": 4,
        }
        defaults.update(kwargs)
        return ResourceScheduler(**defaults)

    def test_gpu_phases(self):
        s = self._make()
        assert len(s.gpu_phases) == 4
        assert s.modules_for_phase(0) == ["caption"]
        assert s.modules_for_phase(1) == ["objects"]
        assert s.modules_for_phase(2) == ["faces", "embedding"]
        assert s.modules_for_phase(3) == ["perception"]

    def test_co_resident_phase(self):
        s = self._make()
        assert not s.is_co_resident_phase(0)  # caption alone
        assert not s.is_co_resident_phase(1)  # objects alone
        assert s.is_co_resident_phase(2)       # faces + embedding
        assert not s.is_co_resident_phase(3)  # perception alone

    def test_independent_gpu_modules(self):
        s = self._make()
        assert s.independent_gpu_modules() == frozenset()

    def test_batch_sizes(self):
        s = self._make()
        assert s.batch_size_for("objects") == 4
        assert s.batch_size_for("embedding") == 16
        assert s.batch_size_for("faces") == 10  # falls back to default

    def test_batch_capable(self):
        s = self._make()
        assert s.is_batch_capable("objects")
        assert s.is_batch_capable("embedding")
        assert not s.is_batch_capable("faces")

    def test_module_classification(self):
        s = self._make()
        assert s.is_gpu("objects")
        assert s.is_gpu("perception")
        assert not s.is_gpu("metadata")
        assert s.is_local_io("metadata")
        assert s.is_local_io("technical")
        assert s.is_io("metadata")

    def test_prerequisites(self):
        s = self._make()
        assert s.prerequisite_for("faces") == "objects"
        assert s.prerequisite_for("objects") is None
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

    def test_ready_free_vram_threshold_for_exclusive_module(self):
        s = self._make(vram_budget=VRAMBudget(total_vram_gb=16.0, fraction=0.70))
        # Leave ~15% headroom for desktop/compositor reservation.
        assert s._ready_free_vram_gb("perception", s.vram.vram_for("perception")) == pytest.approx(13.6)
        # Non-exclusive modules keep exact threshold.
        assert s._ready_free_vram_gb("objects", 2.4) == pytest.approx(2.4)

    def test_wait_for_vram_ready_accepts_headroom_for_perception(self, monkeypatch):
        s = self._make(vram_budget=VRAMBudget(total_vram_gb=16.0, fraction=0.70))
        monkeypatch.setattr(s, "_cuda_free_gb", lambda: 14.0)
        monkeypatch.setattr(s, "_force_cuda_cleanup", lambda: None)
        # Should not raise: 14 GB free satisfies perception ready threshold.
        s._wait_for_vram_ready("perception")

    def test_unload_happens_before_final_io_collect(self):
        """GPU models should unload before draining trailing IO futures."""
        s = self._make()

        claim_calls = 0
        submit_calls = 0
        call_order: list[str] = []
        pending_future: Future[Any] = Future()

        def _claim_fn(_batch_size: int, module: str) -> list[dict[str, Any]]:
            nonlocal claim_calls
            claim_calls += 1
            if claim_calls == 1:
                return [{"id": 1, "image_id": 1, "module": module}]
            return []

        def _process_batch_fn(
            jobs: list[dict[str, Any]],
            _module: str,
        ) -> dict[str, int]:
            return {"done": len(jobs), "failed": 0, "skipped": 0}

        def _submit_io_fn(
            _local_pool: ThreadPoolExecutor,
            _cloud_pool: ThreadPoolExecutor,
        ) -> dict[Future[Any], dict[str, Any]]:
            nonlocal submit_calls
            submit_calls += 1
            if submit_calls == 1:
                return {pending_future: {"id": 99, "image_id": 99, "module": "metadata"}}
            return {}

        def _collect_fn(futures: dict[Future[Any], dict[str, Any]]) -> None:
            if futures:
                call_order.append("collect")

        def _unload_fn(module: str) -> None:
            call_order.append(f"unload:{module}")

        with (
            ThreadPoolExecutor(max_workers=1) as local_pool,
            ThreadPoolExecutor(max_workers=1) as cloud_pool,
        ):
            s.run_gpu_phase(
                0,  # Phase 0 => ["caption"]
                claim_fn=_claim_fn,
                process_batch_fn=_process_batch_fn,
                process_single_fn=lambda _job: "done",
                submit_io_fn=_submit_io_fn,
                collect_fn=_collect_fn,
                advance_fn=lambda _n: None,
                flush_fn=lambda: None,
                local_pool=local_pool,
                cloud_pool=cloud_pool,
                stats={"done": 0, "failed": 0, "skipped": 0},
                unload_fn=_unload_fn,
            )

        assert "unload:caption" in call_order
        assert "collect" in call_order
        assert call_order.index("unload:caption") < call_order.index("collect")
        assert s.vram.loaded_modules == []

    def test_run_gpu_phase_cancels_queued_io_futures_on_shutdown(self):
        """Pause should cancel queued IO futures instead of running the whole backlog."""
        shutdown = threading.Event()
        s = self._make(shutdown_event=shutdown)

        claim_calls = 0
        submit_calls = 0
        started = threading.Event()
        release = threading.Event()
        cancelled_jobs: list[int] = []
        collected_jobs: list[int] = []
        executed_jobs: list[int] = []

        def _claim_fn(_batch_size: int, module: str) -> list[dict[str, Any]]:
            nonlocal claim_calls
            claim_calls += 1
            if claim_calls == 1:
                return [{"id": 1, "image_id": 1, "module": module}]
            return []

        def _process_batch_fn(
            jobs: list[dict[str, Any]],
            _module: str,
        ) -> dict[str, int]:
            return {"done": len(jobs), "failed": 0, "skipped": 0}

        def _slow(job_id: int) -> str:
            started.set()
            release.wait(timeout=5)
            executed_jobs.append(job_id)
            return "done"

        def _submit_io_fn(
            _local_pool: ThreadPoolExecutor,
            cloud_pool: ThreadPoolExecutor,
        ) -> dict[Future[Any], dict[str, Any]]:
            nonlocal submit_calls
            submit_calls += 1
            if submit_calls > 1:
                return {}
            fut1 = cloud_pool.submit(_slow, 1)
            fut2 = cloud_pool.submit(_slow, 2)
            assert started.wait(timeout=5)
            shutdown.set()
            return {
                fut1: {"id": 11, "image_id": 11, "module": "metadata"},
                fut2: {"id": 12, "image_id": 12, "module": "technical"},
            }

        def _collect_fn(futures: dict[Future[Any], dict[str, Any]]) -> None:
            for fut, job in futures.items():
                assert not fut.cancelled()
                assert fut.result(timeout=5) == "done"
                collected_jobs.append(job["id"])

        def _cancel_futures_fn(futures: dict[Future[Any], dict[str, Any]]) -> None:
            for fut, job in list(futures.items()):
                if fut.cancel():
                    cancelled_jobs.append(job["id"])
                    futures.pop(fut, None)

        def _release_after_cancel() -> None:
            deadline = time.time() + 5
            while time.time() < deadline and not cancelled_jobs:
                time.sleep(0.01)
            release.set()

        releaser = threading.Thread(target=_release_after_cancel, daemon=True)
        releaser.start()

        with (
            ThreadPoolExecutor(max_workers=1) as local_pool,
            ThreadPoolExecutor(max_workers=1) as cloud_pool,
        ):
            s.run_gpu_phase(
                1,  # Phase 1 => ["faces", "embedding"]
                claim_fn=_claim_fn,
                process_batch_fn=_process_batch_fn,
                process_single_fn=lambda _job: "done",
                submit_io_fn=_submit_io_fn,
                collect_fn=_collect_fn,
                advance_fn=lambda _n: None,
                flush_fn=lambda: None,
                local_pool=local_pool,
                cloud_pool=cloud_pool,
                stats={"done": 0, "failed": 0, "skipped": 0},
                unload_fn=lambda _module: None,
                cancel_futures_fn=_cancel_futures_fn,
            )

        releaser.join(timeout=5)
        assert cancelled_jobs == [12]
        assert collected_jobs == [11]
        assert executed_jobs == [1]

    def test_run_io_drain_cancels_queued_futures_on_shutdown(self):
        """IO drain should stop queued work promptly when pause is requested."""
        shutdown = threading.Event()
        s = self._make(shutdown_event=shutdown)

        submit_calls = 0
        started = threading.Event()
        release = threading.Event()
        cancelled_jobs: list[int] = []
        collected_jobs: list[int] = []
        executed_jobs: list[int] = []

        def _slow(job_id: int) -> str:
            started.set()
            release.wait(timeout=5)
            executed_jobs.append(job_id)
            return "done"

        def _submit_io_fn(
            _local_pool: ThreadPoolExecutor,
            cloud_pool: ThreadPoolExecutor,
        ) -> dict[Future[Any], dict[str, Any]]:
            nonlocal submit_calls
            submit_calls += 1
            if submit_calls > 1:
                return {}
            fut1 = cloud_pool.submit(_slow, 1)
            fut2 = cloud_pool.submit(_slow, 2)
            assert started.wait(timeout=5)
            shutdown.set()
            return {
                fut1: {"id": 21, "image_id": 21, "module": "metadata"},
                fut2: {"id": 22, "image_id": 22, "module": "technical"},
            }

        def _collect_fn(futures: dict[Future[Any], dict[str, Any]]) -> None:
            for fut, job in futures.items():
                assert not fut.cancelled()
                assert fut.result(timeout=5) == "done"
                collected_jobs.append(job["id"])

        def _cancel_futures_fn(futures: dict[Future[Any], dict[str, Any]]) -> None:
            for fut, job in list(futures.items()):
                if fut.cancel():
                    cancelled_jobs.append(job["id"])
                    futures.pop(fut, None)

        def _release_after_cancel() -> None:
            deadline = time.time() + 5
            while time.time() < deadline and not cancelled_jobs:
                time.sleep(0.01)
            release.set()

        releaser = threading.Thread(target=_release_after_cancel, daemon=True)
        releaser.start()

        with (
            ThreadPoolExecutor(max_workers=1) as local_pool,
            ThreadPoolExecutor(max_workers=1) as cloud_pool,
        ):
            s.run_io_drain(
                submit_io_fn=_submit_io_fn,
                collect_fn=_collect_fn,
                flush_fn=lambda: None,
                local_pool=local_pool,
                cloud_pool=cloud_pool,
                cancel_futures_fn=_cancel_futures_fn,
            )

        releaser.join(timeout=5)
        assert cancelled_jobs == [22]
        assert collected_jobs == [21]
        assert executed_jobs == [1]


# ── Integration: co-residency fits within VRAM ────────────────────────────────

class TestCoResidencyFitCheck:
    """Verify that the declared co-resident phase actually fits."""

    def test_phase1_fits_in_budget(self):
        vram = VRAMBudget(total_vram_gb=16.0, fraction=0.70)
        phase1_modules = _GPU_PHASES[1]  # faces, embedding
        total = sum(_MODULE_VRAM_GB.get(m, 0) for m in phase1_modules)
        assert total < vram.budget_gb, (
            f"Phase 1 modules ({phase1_modules}) need {total:.2f} GB "
            f"but budget is {vram.budget_gb:.2f} GB"
        )
        # Actually reserve them all
        for mod in phase1_modules:
            assert vram.can_fit(mod)
            vram.reserve(mod)

    def test_phase1_fits_on_8gb_card(self):
        """Even an 8 GB card should handle co-residency at 70%."""
        vram = VRAMBudget(total_vram_gb=8.0, fraction=0.70)  # 5.6 GB budget
        phase1_modules = _GPU_PHASES[1]
        total = sum(_MODULE_VRAM_GB.get(m, 0) for m in phase1_modules)
        assert total < vram.budget_gb

    def test_perception_exclusive_in_registry(self):
        """UniPercept-backed modules must be in the exclusive set."""
        assert "perception" in _EXCLUSIVE_MODULES

    def test_all_gpu_phases_have_known_modules(self):
        """Every module in _GPU_PHASES must have a VRAM entry."""
        for phase in _GPU_PHASES:
            for mod in phase:
                assert mod in _MODULE_VRAM_GB, f"{mod} missing from _MODULE_VRAM_GB"
