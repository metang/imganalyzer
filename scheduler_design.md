# Scheduler and VRAM Allocation Design

This document summarizes the current scheduler and VRAM-allocation design implemented in the codebase.

## 1) Scope and key files

- `imganalyzer/pipeline/worker.py`
- `imganalyzer/pipeline/scheduler.py`
- `imganalyzer/pipeline/vram_budget.py`
- `imganalyzer/db/queue.py`
- `imganalyzer/server.py`
- `imganalyzer/pipeline/batch.py`
- `imganalyzer/pipeline/distributed_worker.py`
- `imganalyzer/analysis/perception.py`
- `imganalyzer/pipeline/modules.py`

## 2) Queue model and job ordering

Jobs are rows in `job_queue` with statuses: `pending`, `running`, `done`, `failed`, `skipped`.

### Enqueue priorities

`BatchProcessor` assigns module priorities at enqueue time (`_module_priority`):

- `metadata`: 100
- `technical`: 90
- `objects`: 85
- `caption`: 80
- `faces`: 77
- `perception`: 60
- `embedding`: 50

Queue claim order is always:

1. `priority DESC`
2. `queued_at ASC`

This is true for both local master claims (`claim`) and leased distributed claims (`claim_leased`).

### Chunk shaping

`get_pending_image_ids()` orders images by number of pending jobs descending (`COUNT(*) DESC`), so early chunks contain images with the most unfinished work.

## 3) Master scheduling loop (worker)

The master worker (`Worker._run_loop`) is the top-level orchestrator.

### Anti-idle sweep behavior

`_pending_image_ids_with_running_wait()` does not treat `pending=0` as completion when `running>0`.
It waits, periodically tries `release_expired_leases()`, and only exits when both pending and running are empty.
This prevents the master from pausing while distributed workers still hold leases.

### Sweep/chunk/mini-batch hierarchy

For each sweep:

1. Build chunk list (`chunk_size` or single chunk).
2. For each chunk, optionally split into mini-batches (`_mini_batch_size`, target ~50-100 images).
3. For each mini-batch, run all scheduler GPU phases in order.
4. Run IO drain.
5. Retry the same chunk while pending jobs remain.

Mini-batch interleaving is used so fully analyzed results appear earlier (instead of draining all caption work first).

### Perception cooldown

If a perception phase fails due to VRAM readiness timeout, the worker defers perception for a cooldown window (`_defer_perception_phase`) and continues other modules, then retries later.

## 4) ResourceScheduler design

`ResourceScheduler` controls phase execution and phase-local concurrency.

### GPU phases

Current phases:

1. `["caption"]`
2. `["objects"]`
3. `["faces", "embedding"]` (co-resident phase)
4. `["perception"]` (exclusive)

`INDEPENDENT_GPU_MODULES` is currently empty (all active GPU modules are in phase pipeline).

### Batched vs single-image execution

- Batch-capable modules in scheduler: `objects`, `embedding`.
- Other modules run single-image flow.
- In multi-module phases, each module gets its own thread; optional prefetch pipeline feeds non-batch modules.

### IO coexistence

While GPU work runs, scheduler continuously:

- submits local IO jobs (`metadata`, `technical`)
- reaps completed futures
- flushes periodic side effects (FTS/XMP hooks)

On phase exit, scheduler unloads phase models before final IO collection and performs CUDA cleanup.

## 5) VRAM allocation model

`VRAMBudget` is a thread-safe reservation system (RLock) with static per-module VRAM estimates.

### Static module VRAM estimates (`_MODULE_VRAM_GB`)

- `caption`: `8.7`
- `objects`: `2.4`
- `faces`: `1.0`
- `embedding`: `0.95`
- `perception`: `13.8`

### Budget fraction

Usable budget is:

- `budget_gb = total_vram_gb * 0.70`

This aligns with worker-level CUDA memory fraction setting:

- `torch.cuda.set_per_process_memory_fraction(0.70)` when CUDA is available.

### Exclusivity

`_EXCLUSIVE_MODULES = {"perception"}`.

Rules:

- If an exclusive module is loaded, no other GPU module may load.
- If loading an exclusive module, it must be alone.
- Exclusive modules are allowed to fit against physical `total_gb` (not strict 70% budget), because they run alone.

### Reserve/release semantics

- `reserve(module)` records module VRAM in `_loaded` if `can_fit`.
- `release(module)` removes reservation.
- CPU modules (not in `_MODULE_VRAM_GB`) are treated as 0 VRAM.

## 6) Runtime VRAM readiness gate

Before reserving a phase module, scheduler runs `_wait_for_vram_ready(module)`:

1. Query free CUDA memory (`torch.cuda.mem_get_info`).
2. Compute required free threshold:
   - Non-exclusive: exact `needed_gb`.
   - Exclusive: `min(needed_gb, total_gb - desktop_reserve_gb)`, where
     `desktop_reserve_gb = max(2.0, total_gb * 0.15)`.
3. Force cleanup (`gc.collect`, `empty_cache`, optional `torch.cuda.ipc_collect`) while waiting.
4. Timeout after 120s with explicit error.

This headroom rule avoids deadlock on desktop GPUs where compositor/driver permanently occupies part of VRAM.

## 7) Distributed scheduling behavior

Distributed workers lease jobs via `jobs/claim` (`server.py` + `queue.claim_leased`).

### Lease controls

- Max active leases per worker: `3`.
- Leases are heartbeated (`jobs/heartbeat`) and reclaimed when expired.
- `claim_leased` supports:
  - module filter
  - supported-modules list filter
  - module affinity (`prefer_module`)
  - chunk affinity (`prefer_image_ids`)

### Master reservation policy

When coordinator run is active and master has no current running jobs, `jobs/claim` reserves at least one eligible pending job for master and limits remote claim size accordingly.
This prevents workers from leasing all pending jobs and forcing master into coordinating-only idle mode.

### Prerequisite handling

During claim validation and local execution, prerequisites are enforced from worker-side prerequisite map:

- `faces` depends on `objects`
- `embedding` depends on `objects`

If prerequisite is not done:

- If prerequisite failed/skipped: dependent is marked skipped.
- Otherwise dependent is released/deferred back to pending.

## 8) Perception-specific VRAM assumptions

Perception runtime uses UniPercept 4-bit NF4 quantization (`analysis/perception.py`) and model load includes:

- `max_memory={0: "14GiB", "cpu": "24GiB"}`

That runtime behavior is reflected in the scheduler estimate (`13.8 GB`) and exclusive handling.

## 9) Status/telemetry surfaces related to scheduler

`status` RPC includes:

- module status counts
- total pending/running/done/failed/skipped
- chunk info (`index`, `total`, per-module chunk counts)
- node-level running jobs (master + workers)
- module average ms (`module_avg_processing_ms(last_n=100)`)

`module_avg_processing_ms` uses done-only jobs and computes average from each module’s latest `N` completed rows using `(completed_at - started_at)`.

## 10) Summarization: high-level design, considerations, and trade-offs

### High-level design

The system uses a hybrid scheduler:

- Priority queue ordering at ingest/claim time.
- Phase-based GPU execution (`caption -> objects -> faces+embedding -> perception`).
- Concurrent IO draining (`metadata`, `technical`) while GPU phases run.
- Distributed lease-based workers with coordinator-side guardrails to keep the master device active.
- Chunk + mini-batch interleaving to improve time-to-first-fully-analyzed results.

In short, it favors predictable progress and stable memory behavior over aggressive opportunistic scheduling.

### Design considerations

The implementation optimizes for:

- **Master utilization**: avoid "coordinating-only" idle periods when queue work exists.
- **VRAM safety**: explicit reservation, exclusivity, readiness checks, and forced cleanup.
- **Desktop realism**: headroom for compositor/driver reservations on display-attached GPUs.
- **Dependency correctness**: prerequisite-aware claim/defer/skip behavior.
- **Operational resilience**: lease heartbeat, expired-lease reclaim, stale-job recovery, retry paths.
- **UI observability**: chunk info, node activity, and recent timing/throughput stats exposed via `status`.

### Trade-offs made in the design

- **Static VRAM estimates vs adaptive runtime models**  
  Static values are simple and deterministic, but require periodic retuning as models/configs change.

- **Phase barriers vs fully dynamic DAG scheduling**  
  Phases reduce complexity and VRAM chaos, but can leave throughput on the table versus a fine-grained dependency scheduler.

- **Master-reserve policy vs maximum worker burst throughput**  
  Reserving pending jobs for master improves local utilization, but may slightly reduce short-term remote-worker saturation.

- **Small lease cap (`3`) vs fewer coordinator round trips**  
  Limits hoarding and improves fairness, but increases claim/heartbeat frequency and RPC chatter.

- **Chunk affinity vs global queue fairness**  
  Improves chunk completion latency/UX, but can temporarily deprioritize jobs outside the active chunk.

### Pros and cons

#### Pros

- Predictable VRAM behavior with lower risk of OOM thrash.
- Better master-device utilization under distributed load.
- Faster perceived progress through chunk/mini-batch interleaving.
- Strong failure recovery (lease reclaim, stale recovery, prerequisite-aware deferral).
- Clear operational telemetry for UI and debugging.

#### Cons

- Scheduler logic is spread across worker, server, queue, and distributed worker code (higher maintenance complexity).
- Static VRAM numbers can become stale and require manual calibration.
- Phase sequencing can underutilize hardware in edge cases where cross-phase overlap might be possible.
- SQLite lock contention remains a practical constraint at high concurrency.
