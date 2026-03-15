# Pipeline Scheduler and VRAM Architecture

This document describes the current scheduler implementation used by both the
local coordinator worker and distributed workers.

## 1) Active modules and legacy compatibility

Active pipeline modules:

- `metadata`
- `technical`
- `caption`
- `objects`
- `faces`
- `perception`
- `embedding`

Legacy queue/module names are migrated for active jobs:

- `blip2` -> `caption`
- `cloud_ai` -> `caption`
- `local_ai` -> `caption`
- `aesthetic` -> `perception`

Migration is applied in both the local worker run loop and coordinator runtime
reconciliation so resume works with old queues.

## 2) Scheduler ownership by component

Scheduling logic is split by responsibility:

- `imganalyzer/pipeline/unified_scheduler.py`
  - Central claim policy kernel.
  - Worker control states (`active`, `pause-drain`, `pause-immediate`, `paused`).
  - Capability filtering (including CUDA-gated `perception`).
  - Module affinity epoch + ETA-aware preference via per-worker EWMA timings.

- `imganalyzer/db/queue.py`
  - Persistent queue and lease primitives (enqueue/claim/lease heartbeat/complete/fail).
  - Runtime reconciliation primitives (`reconcile_runtime_state`, `recover_stale`).

- `imganalyzer/pipeline/worker.py`
  - Local master worker execution loop.
  - Chunk and mini-batch orchestration.
  - Anti-idle behavior while remote leases are still running.

- `imganalyzer/pipeline/scheduler.py`
  - GPU phase execution engine.
  - In-phase concurrency and IO/GPU interleaving.
  - VRAM-ready gating and phase model load/unload boundaries.

- `imganalyzer/pipeline/distributed_worker.py`
  - Remote worker claim/execute/report loop.
  - Capability probing and pause/resume honoring.

- `imganalyzer/server.py`
  - RPC control plane (`workers/*`, `jobs/*`).
  - Delegates claim policy decisions to `compute_claim_policy`.
  - One-time startup reconciliation.

## 3) Queue and claim model

Queue statuses: `pending`, `running`, `done`, `failed`, `skipped`.

Priority order at enqueue:

- `metadata` 100
- `technical` 90
- `objects` 85
- `caption` 80
- `faces` 77
- `perception` 60
- `embedding` 50

Prerequisites:

- `faces` depends on `objects`
- `embedding` depends on `objects`

Distributed claims are lease-based:

- Max active leases per worker: `3`
- Lease TTL is refreshed by worker heartbeat
- Expired leases are reclaimed and returned to `pending`

`compute_claim_policy(...)` applies in one place:

1. Block claims for paused workers.
2. Enforce active-lease cap.
3. Filter by module filters + worker capabilities.
4. Reserve at least one eligible pending job for master when run is active and master has no local running jobs.
5. Prefer active chunk image IDs when available.
6. Choose preferred module using epoch stickiness and per-worker timing EWMA.

## 4) Local execution model (chunk-first)

The local worker processes queue work in chunked sweeps:

1. Build pending image set.
2. Split into chunks (`chunk_size`) when configured.
3. Split each large chunk into mini-batches (`~50-100` images).
4. Run all GPU phases for each mini-batch before moving on.
5. Drain IO work throughout.

GPU phases:

- Phase 0: `caption`
- Phase 1: `objects`
- Phase 2: `faces`, `embedding` (co-resident)
- Phase 3: `perception` (exclusive)

Important runtime behaviors:

- Anti-idle loop waits for `running` jobs to finish even when local `pending` reaches zero.
- Perception VRAM timeout triggers cooldown deferral so other modules keep progressing.
- Chunk context is exposed so remote claims can prefer the same chunk.

## 5) VRAM model

`imganalyzer/pipeline/vram_budget.py` uses static module estimates:

- `caption`: `8.7 GB`
- `objects`: `2.4 GB`
- `faces`: `1.0 GB`
- `embedding`: `0.95 GB`
- `perception`: `13.8 GB`

Policy:

- Scheduler budget is `70%` of detected VRAM.
- `perception` is exclusive (`_EXCLUSIVE_MODULES = {"perception"}`).
- Exclusive modules can fit against physical VRAM when running alone.

Before loading each module, scheduler checks runtime free VRAM and performs
allocator cleanup. For exclusive modules it uses a desktop headroom rule so
display/driver reservations do not deadlock the phase transition.

## 6) Distributed worker behavior

Distributed workers:

- Probe available modules at startup.
- Advertise capabilities in `workers/register` (`cuda`, `mps`, `supportedModules`).
- Claim jobs through coordinator policy.
- Heartbeat worker state + lease extensions.
- Honor coordinator pause states immediately.

Hardware-aware routing:

- `perception` is blocked on workers that explicitly report `cuda=false`.
- Workers only receive modules present in `supportedModules` when provided.

## 7) Resilience and resume

Recovery paths:

- Startup/runtime reconciliation remaps legacy module names and repairs
  queue/lease invariants.
- Dangling lease rows are removed.
- Running jobs without valid leases are re-queued.
- Stale worker rows are marked offline.
- Worker startup can release stale leases for its own worker id.

These steps make pause/resume/restart robust without losing queue progress.

## 8) Telemetry used by scheduling/UI

`status` includes:

- Per-module counts by status
- Global totals (`pending`, `running`, `done`, `failed`, `skipped`)
- Chunk progress context (`index`, `total`, module counts)
- Node-level running jobs
- `module_avg_processing_ms(last_n=100)`

`module_avg_processing_ms` is computed from each module's latest 100 `done` jobs
using `(completed_at - started_at)` in SQL.

## 9) Required checklist for scheduler/module changes

When adding/removing/renaming modules:

1. Update scheduler + worker/module dispatch + repository map + VRAM budget.
2. Update distributed worker capability probe + UI module labels/selectors.
3. Add legacy remap mapping in both worker run loop and coordinator reconciliation,
   and run queue remap for existing pending/running jobs.

Skipping any of these can leave jobs permanently unclaimable.

## 10) Trade-offs

Pros:

- Policy centralization in `unified_scheduler.py` improves consistency.
- Chunk-first execution improves time-to-first-fully-processed results.
- Lease + reconciliation model is resilient to crashes and restarts.
- Hardware-aware routing prevents impossible assignments.

Cons:

- Runtime adapters still span multiple components.
- Static VRAM estimates need periodic retuning.
- SQLite remains a practical throughput bottleneck at very high concurrency.
