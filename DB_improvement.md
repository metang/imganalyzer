# SQLite Database Design: Lock Contention Analysis & Improvement Plan

> **Generated**: 2026-03-27 | **Analyzed by**: Claude Opus 4.6, GPT-5.3-Codex, GPT-5.4
> **Scope**: `imganalyzer/db/`, `imganalyzer/pipeline/worker.py`, `imganalyzer/server.py`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture](#2-current-architecture)
3. [Connection Architecture](#3-connection-architecture)
4. [Hot Path Analysis](#4-hot-path-analysis)
5. [Lock Contention Points](#5-lock-contention-points)
6. [Transaction Analysis](#6-transaction-analysis)
7. [Indexing Analysis](#7-indexing-analysis)
8. [Concurrent Access Matrix](#8-concurrent-access-matrix)
9. [Face/Cluster Operations Deep Dive](#9-facecluster-operations-deep-dive)
10. [PRAGMA & Configuration Audit](#10-pragma--configuration-audit)
11. [Current Design: Pros](#11-current-design-pros)
12. [Current Design: Cons](#12-current-design-cons)
13. [Top 5 Improvement Opportunities](#13-top-5-improvement-opportunities)

---

## 1. Executive Summary

The imganalyzer backend uses **SQLite with WAL mode** as its single database, storing job queues, analysis results, face data, embeddings, FTS5 search indexes, and profiler data — all in one file (`~/.cache/imganalyzer/imganalyzer.db`).

The system employs a multi-threaded architecture where the JSON-RPC server (main thread), worker thread (daemon), IO thread pool workers, HTTP server threads, ingest threads, and clustering threads all access the same SQLite database concurrently. WAL mode allows concurrent reads, but **only one writer at a time** — all write transactions are serialized.

**Key findings across all analyses:**
- The **FTS5 periodic flush** holds write locks for 250ms–2.5s, blocking all other writers
- The **clustering thread** has **no `busy_timeout`** set (0ms), causing immediate failures on lock contention
- **`upsert_*` methods** are not internally atomic (DELETE + INSERT without transaction wrapper)
- The **`_handle_jobs_claim` loop** creates bursts of `BEGIN IMMEDIATE` transactions
- **Ingest batches** of 500 images hold the write lock for 50–500ms
- **Inconsistent `busy_timeout`** values across connections (0ms, 5s, 30s)

---

## 2. Current Architecture

### Threading Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PROCESS: imganalyzer                           │
│                                                                     │
│  ┌─────────────────────────────────────┐                            │
│  │  Main Thread (JSON-RPC stdio loop)  │                            │
│  │  Connection: thread-local singleton │                            │
│  │  busy_timeout: 30,000ms             │                            │
│  │  Operations: status polls, search,  │                            │
│  │    face ops, gallery, workers mgmt  │                            │
│  └─────────────────────────────────────┘                            │
│                                                                     │
│  ┌─────────────────────────────────────┐                            │
│  │  HTTP Threads (ThreadingHTTPServer) │                            │
│  │  Each: own conn via _get_db()       │                            │
│  │  busy_timeout: 30,000ms             │                            │
│  │  Operations: distributed worker     │                            │
│  │    claim, complete, heartbeat       │                            │
│  └─────────────────────────────────────┘                            │
│                                                                     │
│  ┌─────────────────────────────────────┐                            │
│  │  Worker Thread ("rpc-run", daemon)  │                            │
│  │  Connection: fresh per-thread       │                            │
│  │  busy_timeout: 30,000ms             │                            │
│  │  ┌───────────────────────────────┐  │                            │
│  │  │ IO ThreadPool workers         │  │                            │
│  │  │ Each: fresh conn via          │  │                            │
│  │  │   _get_thread_db()            │  │                            │
│  │  │ busy_timeout: 30,000ms        │  │                            │
│  │  │ Ops: metadata/technical jobs  │  │                            │
│  │  └───────────────────────────────┘  │                            │
│  └─────────────────────────────────────┘                            │
│                                                                     │
│  ┌─────────────────────────────────────┐                            │
│  │  Ingest Thread (daemon)             │                            │
│  │  Connection: fresh, busy_t=30s      │                            │
│  │  Ops: batch image insert + enqueue  │                            │
│  └─────────────────────────────────────┘                            │
│                                                                     │
│  ┌─────────────────────────────────────┐                            │
│  │  Clustering Thread (daemon)         │                            │
│  │  Connection: fresh                  │                            │
│  │  ⚠️ busy_timeout: NOT SET (0ms!)    │                            │
│  │  ⚠️ isolation_level: default        │                            │
│  │  Ops: cluster_faces() bulk writes   │                            │
│  └─────────────────────────────────────┘                            │
│                                                                     │
│  ┌─────────────────────────────────────┐                            │
│  │  Pre-decode Buffer (from main)      │                            │
│  │  Connection: ephemeral, read-only   │                            │
│  │  ⚠️ busy_timeout: NOT SET           │                            │
│  │  Ops: SELECT pending job_queue      │                            │
│  └─────────────────────────────────────┘                            │
│                                                                     │
│  ┌─────────────────────────────────────┐                            │
│  │  DecodedImageStore (separate DB)    │                            │
│  │  File: cache_index.db               │                            │
│  │  No contention with main DB         │                            │
│  └─────────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  REMOTE: Distributed Worker (separate machine)                      │
│  sqlite3.connect(":memory:") — no file contention                  │
│  All DB access proxied via HTTP → coordinator                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Connection Architecture

### All Connection Creation Sites

| Location | File:Line | `busy_timeout` | `isolation_level` | `check_same_thread` | Purpose |
|---|---|---|---|---|---|
| `get_db()` singleton | `connection.py:35` | 5,000ms | `None` | `True` (default) | CLI-only singleton |
| Server bootstrap | `server.py:148` | 30,000ms | `None` | `False` | Schema init (closed immediately) |
| Server thread-local | `server.py:162` | 30,000ms | `None` | `False` | Main/HTTP thread requests |
| Ingest thread | `server.py:759` | 30,000ms | default | `False` | Batch ingest |
| Worker thread | `server.py:884` | 30,000ms | `None` | `False` | Worker main conn |
| Worker IO threads | `worker.py:247` | 30,000ms | `None` | `False` | Per-thread worker conn |
| Pre-decode buffer | `server.py:430` | **NOT SET (0)** | default | `False` | Read pending jobs |
| Clustering thread | `server.py:3739` | **NOT SET (0)** | default | `False` | Face clustering |
| Distributed worker | `distributed_worker.py:689` | N/A | `None` | `False` | In-memory sandbox |

### Key Observations

- **Inconsistent `busy_timeout`**: Values range from 0ms (clustering, pre-decode) to 5s (CLI) to 30s (server/worker). The 0ms timeout on clustering is a critical bug — any concurrent write causes immediate failure.
- **Inconsistent `isolation_level`**: Some connections use `isolation_level=None` (explicit transaction control), others use default (Python manages transactions). This creates different autocommit behaviors across threads.
- **No connection pooling**: Each thread creates a fresh connection. Thread-local storage provides reuse within a thread, but there's no pool with limits or idle management.

---

## 4. Hot Path Analysis

During batch processing of N images × M modules:

### Tier 1: Per-Job Operations (N×M frequency)

| Operation | File:Line | Transaction Type | Lock Duration | Frequency |
|---|---|---|---|---|
| `JobQueue.claim()` | `queue.py:133-189` | `BEGIN IMMEDIATE` | <1ms | N×M/batch_size |
| `JobQueue.mark_done()` | `queue.py:501-509` | Auto-commit | <1ms | N×M |
| `Repository.upsert_*()` | `repository.py:284-449` | **None (caller wraps)** | <1ms per stmt | N×M |
| `Repository.is_analyzed()` | `repository.py:260-282` | Read-only | N/A | N×M |

### Tier 2: Per-Image Operations (N frequency)

| Operation | File:Line | Transaction Type | Lock Duration |
|---|---|---|---|
| `Repository.get_image()` | `repository.py:211-213` | Read-only | N/A |
| Prerequisite check | `queue.py:560-568` | Read-only | N/A |
| `_emit_result()` | `worker.py:119-143` | No DB | N/A |

### Tier 3: Periodic Operations

| Operation | File:Line | Transaction Type | Lock Duration | Frequency |
|---|---|---|---|---|
| FTS5 flush (50 images) | `worker.py:1370-1387` | `BEGIN IMMEDIATE` | **250ms–2.5s** | Every 60s |
| Status poll (`stats()`) | `queue.py:770-781` | Read-only | N/A | Every 1s |
| Lease heartbeat | `queue.py:366-390` | `BEGIN IMMEDIATE` | <1ms | Per active worker, periodic |

### Tier 4: One-Time / On-Demand Operations

| Operation | File:Line | Transaction Type | Lock Duration |
|---|---|---|---|
| Ingest batch (500 imgs) | `batch.py:151-205` | `BEGIN IMMEDIATE` | **50–500ms** |
| `cluster_faces()` | `repository.py:2182+` | Individual auto-commits | **Seconds–minutes** |
| `split_cluster()` | `repository.py:988-1097` | Individual + final commit | Moderate |

---

## 5. Lock Contention Points

### 🔴 CRITICAL: FTS5 Flush Holds Write Lock for Seconds

**File**: `worker.py:1370-1383`

```python
self.conn.execute("BEGIN IMMEDIATE")
try:
    for image_id in chunk:  # up to 50 images
        self.repo.update_search_artifacts(image_id)
    self.conn.commit()
```

Each `update_search_artifacts()` performs multiple SELECTs across 8+ analysis tables, then FTS5 DELETE + INSERT, plus `search_features` upsert. **Estimated lock hold: 250ms–2.5s per batch of 50 images.** During this window, ALL other writers (queue claims, analysis saves, distributed completions) are blocked.

### 🔴 CRITICAL: Clustering Thread Missing `busy_timeout`

**File**: `server.py:3739-3741`

```python
conn = sqlite3.connect(str(db_path), check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
conn.row_factory = sqlite3.Row
# ⚠️ No busy_timeout, no isolation_level=None, no synchronous=NORMAL
```

If `cluster_faces()` tries to write while any other writer holds the lock, it **fails immediately** with `database is locked`. This is the most likely cause of the reported face clustering lock errors.

### 🟡 HIGH: `_handle_jobs_claim` Transaction Burst

**File**: `server.py:1513-1599`

Each distributed worker claim request can execute multiple `BEGIN IMMEDIATE` transactions in a tight loop:
- `claim_leased()` — SELECT + UPDATE + lease INSERT
- Per-job: `mark_skipped_leased()` or `release_leased()` — each a separate `BEGIN IMMEDIATE`

With multiple distributed workers polling simultaneously, this creates high write contention.

### 🟡 HIGH: `upsert_*` Methods Not Internally Atomic

**File**: `repository.py:284-449`

Pattern: `DELETE FROM table WHERE image_id = ?` → `INSERT INTO table (...) VALUES (...)` with no transaction wrapper. Each statement auto-commits with `isolation_level=None`. If a crash occurs between DELETE and INSERT, data is lost. Callers typically wrap in `_transaction`, but this is not enforced.

### 🟢 MODERATE: Ingest Batch Size

**File**: `batch.py:151-205`

500 images × ~7 modules = ~4000 rows in one `BEGIN IMMEDIATE` transaction. Lock held for 50–500ms. Well-designed for throughput, but blocks all other writers during the window.

---

## 6. Transaction Analysis

### Transaction Duration Summary

| Method | File:Line | Type | Duration Estimate | Contention Risk |
|---|---|---|---|---|
| `claim()` | `queue.py:161-185` | `BEGIN IMMEDIATE` | <1ms | Medium (frequent) |
| `claim_leased()` | `queue.py:253-299` | `BEGIN IMMEDIATE` | 1-5ms | **High** (loops) |
| `mark_done()` | `queue.py:502-509` | Auto-commit | <1ms | Low |
| `mark_done_leased()` | `queue.py:427-444` | `BEGIN IMMEDIATE` | <1ms | Medium |
| `heartbeat_lease()` | `queue.py:373-390` | `BEGIN IMMEDIATE` | <1ms | Medium |
| `release_expired_leases()` | `queue.py:307-330` | `BEGIN IMMEDIATE` | 1-10ms | Medium |
| `reconcile_runtime_state()` | `queue.py:692-735` | `BEGIN IMMEDIATE` | 10-50ms | Low (startup) |
| `remap_pending_modules()` | `queue.py:645-674` | `BEGIN IMMEDIATE` | 10-100ms | Low (startup) |
| `_handle_jobs_complete` | `server.py:1741-1794` | `BEGIN IMMEDIATE` | 5-50ms | **High** |
| Ingest batch | `batch.py:151-205` | `BEGIN IMMEDIATE` | 50-500ms | Medium |
| FTS5 flush | `worker.py:1372-1383` | `BEGIN IMMEDIATE` | **250ms-2.5s** | **Critical** |
| `cluster_faces()` | `repository.py:2182+` | Auto-commit per stmt | **Seconds-minutes** | **Critical** |
| `split_cluster()` | `repository.py:988-1097` | Auto-commit + commit | Moderate | **High** |

### Transaction Pattern Issues

1. **No nested transaction support**: SQLite doesn't support nested transactions. With `isolation_level=None`, accidentally calling `BEGIN` while already in a transaction will fail.
2. **Implicit autocommit**: Repository `upsert_*` methods rely on callers for transaction boundaries. Some callers wrap properly (`modules.py _transaction`), others may not.
3. **Long-held locks**: FTS flush (2.5s) and clustering (minutes) far exceed the recommended <100ms transaction guideline for SQLite.

---

## 7. Indexing Analysis

### Current Indexes

#### `job_queue` (hottest table)
| Index | Columns | Purpose |
|---|---|---|
| `idx_job_queue_status` | `(status, priority DESC)` | Status filtering |
| `idx_job_queue_image_module` | `(image_id, module)` | Uniqueness + lookup |
| `idx_job_queue_claim` | `(status, module, priority DESC, queued_at ASC)` | Claim ordering |
| `idx_job_queue_node_status` | `(last_node_role, last_node_id, status, completed_at)` | Node filtering |
| UNIQUE | `(image_id, module)` | Constraint |

#### `face_occurrences`
| Index | Columns |
|---|---|
| `idx_face_occ_image` | `(image_id)` |
| `idx_face_occ_cluster` | `(cluster_id)` |
| `idx_face_occ_identity` | `(identity_name)` |
| `idx_face_occ_cluster_identity` | `(cluster_id, identity_name)` |
| `idx_face_occ_cluster_person` | `(cluster_id, person_id)` |
| `idx_face_occurrences_person_id` | `(person_id)` |

#### Other notable indexes
| Table | Index | Columns |
|---|---|---|
| `images` | `idx_images_file_path` | `(file_path)` |
| `images` | `idx_images_file_hash` | `(file_hash)` |
| `embeddings` | `idx_embeddings_image_type` | `(image_id, embedding_type)` |
| `job_leases` | `idx_job_leases_worker` | `(worker_id)` |
| `job_leases` | `idx_job_leases_expiry` | `(lease_expires_at)` |
| `search_features` | Multiple | `(date, country, face_count, quality)` |

### Indexing Gaps

1. **`(status, image_id)` on `job_queue`**: `pending_count(image_ids=...)` and `running_count(image_ids=...)` filter by status + image_id set. Current indexes lead with `(status, module)` or `(status, priority)`. A `(status, image_id)` index would help chunk-scoped queries.

2. **Dynamic ORDER BY in `claim_leased()`**: `CASE WHEN image_id IN (...) THEN 0 ELSE 1 END` expressions cannot use indexes. With 500K+ pending jobs, this forces a full scan + sort.

3. **`face_occurrences.embedding IS NOT NULL`**: `cluster_faces()` and embedding cache both filter on non-null embeddings. A partial index `CREATE INDEX ... ON face_occurrences(id) WHERE embedding IS NOT NULL` would help.

4. **`(module, status)` for `stats()`**: The `GROUP BY module, status` query would benefit from an index with `module` as the leading column. Current indexes lead with `status`.

---

## 8. Concurrent Access Matrix

| Writer Operation | vs claim() | vs mark_done() | vs upsert_*() | vs FTS flush | vs Status poll | vs Search |
|---|---|---|---|---|---|---|
| **claim() / claim_leased()** | ⚠️ Serialized | ⚠️ One waits | ⚠️ One waits | ⚠️ One waits | ✅ WAL read | ✅ WAL read |
| **mark_done()** | ⚠️ One waits | ⚠️ Serialized | ⚠️ One waits | ⚠️ One waits | ✅ | ✅ |
| **upsert_*() (analysis)** | ⚠️ One waits | ⚠️ One waits | ⚠️ Serialized | ⚠️ One waits | ✅ | ✅ |
| **FTS5 flush (50 imgs)** | ❌ Blocks 0.25–2.5s | ❌ Blocks | ❌ Blocks | N/A | ✅ | ⚠️ May read stale |
| **Ingest batch (500 imgs)** | ❌ Blocks 50–500ms | ❌ Blocks | ❌ Blocks | ❌ Blocks | ✅ | ✅ |
| **cluster_faces()** | ⚠️ Many small writes | ⚠️ Interleaved | ⚠️ Interleaved | ⚠️ | ✅ | ✅ |

**Legend**: ✅ No conflict (WAL read) | ⚠️ Short wait (busy_timeout handles) | ❌ Extended lock hold

### Most Dangerous Conflict Patterns

1. **FTS flush vs. distributed worker `jobs/complete`**: FTS holds write lock 0.25–2.5s. All distributed completions must wait.
2. **FTS flush vs. local worker `claim()`**: Local job claims blocked during FTS flush.
3. **Clustering vs. everything**: Without `busy_timeout`, clustering fails immediately when any writer is active.
4. **Ingest vs. everything**: 500-image batch blocks all writers for up to 500ms.
5. **Multiple distributed `jobs/claim` requests**: Each triggers a loop of `BEGIN IMMEDIATE` transactions, serializing against each other and the local worker.

---

## 9. Face/Cluster Operations Deep Dive

Face operations are specifically reported as problematic. Analysis reveals several compounding issues:

### `cluster_faces()` — Full Recluster
- **Location**: `repository.py:2182+`, invoked from `server.py:3726-3745`
- **Pattern**: Clears ALL cluster IDs (`UPDATE face_occurrences SET cluster_id = NULL`), then reassigns via individual UPDATEs per occurrence
- **Transaction**: Individual auto-commits (no wrapping transaction!)
- **Connection**: `server.py:3739` — **missing `busy_timeout`**
- **Risk**: Any concurrent write causes immediate `database is locked`. A crash mid-operation leaves partial clustering state.
- **Duration**: Seconds to minutes depending on face count

### `split_cluster()` — Split Mixed Clusters
- **Location**: `repository.py:988-1097`
- **Pattern**: Reads all embeddings → in-memory clustering → bulk UPDATEs per sub-cluster → single `conn.commit()`
- **Risk**: With `isolation_level=None`, each UPDATE auto-commits. The final `commit()` is a no-op. Data is written incrementally, not atomically.

### `link_cluster_to_person()` / `unlink_cluster_from_person()`
- **Location**: `repository.py:774-790`
- **Pattern**: Single UPDATE + commit. Low risk, fast.

### `merge_faces()` — Identity Merge
- **Location**: `repository.py:1592-1645`
- **Pattern**: Multiple UPDATEs across `face_embeddings`, `face_identities`, `face_aliases` + final commit
- **Risk**: Non-atomic with auto-commit. A failure mid-merge leaves inconsistent identity data.

### `propagate_person_labels()`
- **Location**: `repository.py:1540-1590`
- **Pattern**: SELECT all → loop through clusters → conditional UPDATEs
- **Risk**: Multiple independent auto-commits. Concurrent face operations could interleave.

### Key Issue: Face Ops During Processing

When a user triggers face operations (merge, split, recluster) while image processing is active:
- Worker thread's analysis saves and queue operations compete for write locks
- Face operations on the clustering thread have **0ms busy_timeout**
- Result: Immediate `database is locked` failures on the clustering thread
- The worker's 30s `busy_timeout` means it eventually succeeds, but face ops fail fast

---

## 10. PRAGMA & Configuration Audit

### Currently Configured

| PRAGMA | CLI (`connection.py`) | Server/Worker | Clustering Thread | Best Practice |
|---|---|---|---|---|
| `journal_mode` | WAL ✅ | WAL ✅ | WAL ✅ | WAL ✅ |
| `synchronous` | NORMAL ✅ | NORMAL ✅ | **default (FULL)** ⚠️ | NORMAL |
| `foreign_keys` | ON ✅ | ON ✅ | **NOT SET** ⚠️ | ON |
| `busy_timeout` | 5,000ms | 30,000ms | **NOT SET (0ms)** 🔴 | 5,000–30,000ms |
| `isolation_level` | `None` ✅ | `None` ✅ | **default** ⚠️ | `None` |

### Not Configured (Recommended)

| PRAGMA | Current | Recommended | Impact |
|---|---|---|---|
| `wal_autocheckpoint` | 1000 (default) | 2000–5000 during batch | Fewer checkpoint stalls |
| `cache_size` | -2000 (default ~2MB) | -32000 (~32MB) | Fewer disk reads for hot tables |
| `mmap_size` | 0 (default) | 256MB+ | Faster reads via memory-mapped I/O |
| `journal_size_limit` | -1 (unlimited) | 64MB | Prevent WAL file bloat |
| `PRAGMA optimize` | Not called | On connection close | Maintain query planner stats |

---

## 11. Current Design: Pros

### Consensus across all analyses:

1. **WAL mode consistently enabled**: All connections set `journal_mode=WAL`. Concurrent readers never block on writers. UI status polls and search queries work during processing. *(connection.py:37, worker.py:254, server.py:155)*

2. **Thread-local connections in server**: `_get_db()` returns thread-local connections via `threading.local()`, correctly preventing cross-thread sharing. Each HTTP handler thread gets its own connection. *(server.py:175-181)*

3. **Explicit `BEGIN IMMEDIATE` for atomic claims**: Queue claim operations use `BEGIN IMMEDIATE` with proper `try/except/rollback`, preventing phantom reads in the SELECT-then-UPDATE claim pattern. *(queue.py:159-189, 253-303)*

4. **Explicit transaction control (`isolation_level=None`)**: Disabling Python's implicit transaction management eliminates "cannot start a transaction within a transaction" errors. Transaction boundaries are explicit and visible. *(connection.py:34-35)*

5. **Generous `busy_timeout` (30s) on main connections**: Most connections retry for up to 30s, which handles typical transient contention. *(server.py:274, worker.py:257)*

6. **Application-level lock retry with backoff**: `_LOCK_RETRY_ATTEMPTS=4` with exponential backoff in the worker provides a second layer of resilience beyond SQLite's `busy_timeout`. *(worker.py:95-96, 1213-1316)*

7. **Batched ingest for throughput**: 500 images per transaction reduces fsync overhead from millions of commits to thousands. *(batch.py:86-205)*

8. **Batched FTS flush with dirty tracking**: Instead of rebuilding FTS after every module completion (N×M calls), dirty images are tracked in `_fts_dirty` and flushed every 60s in batches of 50. *(worker.py:1339-1397)*

9. **Distributed workers use in-memory sandboxes**: Remote workers use `:memory:` databases, completely eliminating file lock contention from remote processing. All coordinator DB access is proxied via HTTP. *(distributed_worker.py:689)*

10. **Clean schema migration system**: Version-based migrations (27 versions) with per-version commits and forward-compatible schema evolution. *(schema.py)*

---

## 12. Current Design: Cons

### Consensus across all analyses:

1. **🔴 Inconsistent `busy_timeout` across connections**: Values range from 0ms (clustering: `server.py:3739`, pre-decode: `server.py:430`) to 5s (CLI: `connection.py:40`) to 30s (server/worker). The 0ms timeout on the clustering thread is the most likely root cause of face clustering lock errors.

2. **🔴 FTS5 flush holds write lock for too long**: Batches of 50 images × multi-table reads + FTS operations hold the write lock for 250ms–2.5s. During this window, ALL other writers (queue operations, analysis saves, distributed completions) are blocked. *(worker.py:1370-1383)*

3. **🔴 `cluster_faces()` is non-transactional with broken connection config**: Performs global `UPDATE face_occurrences SET cluster_id = NULL` followed by per-occurrence UPDATEs — all as individual auto-commits (no wrapping transaction). Combined with 0ms `busy_timeout`, this is the primary source of face clustering failures. *(repository.py:2182+, server.py:3739)*

4. **🟡 `upsert_*` methods not internally atomic**: DELETE + INSERT without transaction wrapper. Relies on callers for atomicity. If any code path forgets the wrapper, a crash between DELETE and INSERT loses data. *(repository.py:284-449)*

5. **🟡 `_handle_jobs_claim` creates transaction bursts**: Each distributed claim request executes multiple `BEGIN IMMEDIATE` transactions in a tight loop — one for the claim, plus one per invalid job for skip/release. With multiple workers polling simultaneously, this serializes heavily. *(server.py:1513-1599)*

6. **🟡 Ingest batch size too large**: 500 images × 7 modules = ~4000 rows in one transaction. Lock held for 50–500ms, blocking all other writers. *(batch.py:151-205)*

7. **🟡 No WAL checkpoint management**: Default `wal_autocheckpoint=1000` pages. During heavy batch processing, the WAL file can grow significantly, and passive checkpoints may fail to keep up. No explicit checkpoint scheduling or size limits configured.

8. **🟢 No `PRAGMA optimize`**: Query planner statistics are never updated. Over time, as data distribution changes, the query planner may make suboptimal choices.

9. **🟢 Single-file database**: All data (queue, analysis, faces, embeddings, FTS, profiler) in one file. Large databases (500K+ images) may have long checkpoint times and WAL bloat.

10. **🟢 Status polling scans full `job_queue`**: `stats()` does `GROUP BY module, status` across the entire table every 1s. At 3.5M rows, this is expensive even with indexes.

---

## 13. Top 5 Improvement Opportunities

### Opportunity 1: Fix Clustering Connection & Make Face Operations Atomic

**Impact**: 🔴 Critical — eliminates the primary reported bug
**Complexity**: Low
**Files**: `server.py:3739`, `repository.py` (clustering + face methods)

#### Problem
The clustering thread creates a connection without `busy_timeout`, `isolation_level=None`, or `synchronous=NORMAL`. Any concurrent write causes immediate `database is locked`. Additionally, `cluster_faces()` writes individual auto-commits, leaving partial state on crash.

#### Solution
```python
# server.py:3739 — Fix connection creation:
conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None, check_same_thread=False)
conn.row_factory = sqlite3.Row
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("PRAGMA foreign_keys=ON")
conn.execute(f"PRAGMA busy_timeout={_DB_BUSY_TIMEOUT_MS}")
```

Additionally, wrap `cluster_faces()` and `split_cluster()` in explicit `BEGIN IMMEDIATE...COMMIT` transactions for atomicity. Consider batching the face occurrence updates (e.g., 100 per transaction) to limit lock hold time while maintaining crash safety.

#### Expected Impact
- Eliminates immediate `database is locked` failures during face clustering
- Prevents partial clustering state on crash
- Face operations will wait up to 30s for locks instead of failing instantly

---

### Opportunity 2: Reduce FTS5 Flush Lock Duration

**Impact**: 🔴 High — eliminates the longest lock hold, unblocking all other writers
**Complexity**: Low
**Files**: `worker.py:1370-1387`

#### Problem
FTS flush holds `BEGIN IMMEDIATE` for 50 images × multi-table reads + FTS operations = 250ms–2.5s. During this window, ALL other writers are blocked.

#### Solution A: Per-Image Micro-Transactions
```python
# Instead of one big transaction for 50 images:
for image_id in fts_snapshot:
    try:
        self.conn.execute("BEGIN IMMEDIATE")
        self.repo.update_search_artifacts(image_id)
        self.conn.commit()
    except Exception:
        self.conn.rollback()
        failed_ids.append(image_id)
```
Lock hold: ~5-50ms per image instead of 250ms–2.5s per batch.

#### Solution B: Smaller Batches (compromise)
Reduce batch size from 50 to 5–10 images per transaction. Still amortizes overhead but limits lock hold to ~25-500ms.

#### Solution C: Deferred Async Queue (best long-term)
Move search artifact rebuilds to a dedicated low-priority queue that runs during idle periods or with adaptive batching based on lock contention signals.

#### Expected Impact
- Reduces maximum lock hold time from 2.5s to <50ms (Solution A) or <500ms (Solution B)
- Unblocks queue claims, analysis saves, and distributed completions during FTS updates
- Tradeoff: More fsyncs, but with `synchronous=NORMAL` the cost is negligible

---

### Opportunity 3: Batch `_handle_jobs_claim` Operations

**Impact**: 🟡 High — significant throughput improvement for distributed processing
**Complexity**: Medium
**Files**: `server.py:1513-1599`, `queue.py` (new batch methods)

#### Problem
Each distributed claim request triggers a loop of `claim_leased()` + per-job `mark_skipped_leased()` / `release_leased()` — each a separate `BEGIN IMMEDIATE` transaction. N+1 transactions per claim request.

#### Solution
Batch the post-claim validation into a single transaction:
```python
# After claim_leased() returns claimed jobs:
to_skip, to_release, valid = [], [], []
for job in claimed:
    if should_skip(job): to_skip.append(...)
    elif should_release(job): to_release.append(...)
    else: valid.append(job)

# Batch skip/release in one transaction
if to_skip or to_release:
    conn.execute("BEGIN IMMEDIATE")
    for job_id, token, reason in to_skip:
        # batch SQL operations...
    for job_id, token in to_release:
        # batch SQL operations...
    conn.commit()
```

Reduces N+1 transactions to 2 (one claim, one batch cleanup).

#### Expected Impact
- 50-80% reduction in write transactions during distributed claim handling
- Less contention between multiple distributed workers polling simultaneously
- Faster claim response times

---

### Opportunity 4: Standardize Connection Configuration & Add WAL Management

**Impact**: 🟡 Medium — stability improvement + tail latency reduction
**Complexity**: Low–Medium
**Files**: `db/connection.py` (new factory), `server.py`, `worker.py`

#### Problem
Connection creation is copy-pasted across 8+ locations with inconsistent PRAGMAs. No WAL checkpoint management.

#### Solution A: Centralized Connection Factory
```python
# db/connection.py — new function:
def create_connection(
    path: Path | None = None,
    busy_timeout_ms: int = 30000,
) -> sqlite3.Connection:
    """Create a properly configured SQLite connection."""
    db_path = path or get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.execute("PRAGMA cache_size=-32000")  # 32MB cache
    return conn
```

Replace all 8+ connection creation sites with `create_connection()`.

#### Solution B: WAL Checkpoint Management
Add periodic WAL checkpoint during batch processing idle periods:
```python
# In worker's periodic flush or server maintenance loop:
conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
```

Configure `wal_autocheckpoint` to a higher value (e.g., 5000) during batch processing to reduce checkpoint-induced stalls, then reset after.

#### Expected Impact
- Eliminates the class of "missing busy_timeout" bugs
- Single point of configuration for future PRAGMA changes
- Controlled WAL growth prevents unbounded file size

---

### Opportunity 5: Reduce Ingest Batch Transaction Size & Add Covering Index for Status Polls

**Impact**: 🟢 Medium — reduced lock hold during ingest + faster UI polling
**Complexity**: Low
**Files**: `pipeline/batch.py`, `db/schema.py`

#### Problem A: Ingest Lock Duration
500 images per `BEGIN IMMEDIATE` holds the write lock for 50–500ms. During large imports, this blocks all other writers.

#### Solution A: Adaptive Batch Size
Reduce ingest batch from 500 to 50–200 images. Use adaptive sizing based on whether processing is active:
```python
batch_size = 50 if self._is_processing_active() else 500
```

#### Problem B: Status Poll Performance
`stats()` scans the full `job_queue` with `GROUP BY module, status`. At millions of rows, this is slow.

#### Solution B: Covering Index + Materialized Stats
```sql
-- Add covering index for the stats() query:
CREATE INDEX IF NOT EXISTS idx_job_queue_module_status
    ON job_queue(module, status);
```

For even better performance, maintain an in-memory or DB-cached stats counter table updated incrementally on each claim/complete/fail, avoiding the GROUP BY entirely. The existing `_status_cache` with short TTL (already partially implemented at `server.py:624-630`) should be extended to all status requests.

#### Expected Impact
- Ingest lock hold reduced from 500ms to 50–100ms
- Status poll from O(N) scan to O(1) cached lookup
- Better UI responsiveness during batch processing

---

## 14. Cross-Model Agreement & Discrepancies

> **Models**: Claude Opus 4.6 (completed, 319s, 61 tool calls), GPT-5.3-Codex (completed, 336s, 142 tool calls), GPT-5.4 (completed, 30,823s, 96 tool calls — very thorough, included in full comparison)

### 14.1 Commonly Agreed Issues (All Three Models)

These issues were independently identified by all three models with consistent severity assessment:

| # | Issue | Opus | Codex | GPT-5.4 | Confidence |
|---|---|---|---|---|---|
| 1 | **Inconsistent `busy_timeout`** across connections | ✅ (0ms clustering, pre-decode) | ✅ (mixed 5s/30s) | ✅ (standardize to 30s) | ✅✅✅ Unanimous |
| 2 | **FTS5 flush / search rebuild holds lock too long** | ✅ CRITICAL (250ms–2.5s) | ✅ Top-5 hotspot | ✅ + found duplicate rebuild bug | ✅✅✅ Unanimous |
| 3 | **`cluster_faces()` global rewrite is problematic** | ✅ Non-transactional | ✅ Major hotspot | ✅ Heavy contention | ✅✅✅ Unanimous |
| 4 | **Claim operations create write contention** | ✅ Tight loop | ✅ Claim loops | ✅ Missing claim index | ✅✅✅ Unanimous |
| 5 | **Repository multi-statement ops not atomic** | ✅ upsert_* (CRITICAL) | ✅ (implicit) | ✅ + update_image() bug | ✅✅✅ Unanimous |
| 6 | **WAL mode + thread-local connections = good** | ✅ Pro | ✅ Pro | ✅ Pro | ✅✅✅ Unanimous |
| 7 | **`BEGIN IMMEDIATE` for claims = correct** | ✅ Pro | ✅ Pro | ✅ Pro | ✅✅✅ Unanimous |
| 8 | **Lock retry with backoff = well-designed** | ✅ Pro | ✅ Pro | ✅ Pro | ✅✅✅ Unanimous |
| 9 | **Distributed workers use in-memory sandboxes** | ✅ Pro | ✅ (confirmed) | ✅ Pro | ✅✅✅ Unanimous |
| 10 | **No WAL checkpoint management** | Not covered | ✅ Improvement #5 | ✅ Con #8 + WAL growth observed | ✅✅ Two models |

### 14.2 Unique Findings by GPT-5.4 (Not Found by Other Models)

GPT-5.4's extended analysis (30,823s) uncovered several critical issues the other models missed:

#### 🔴 CRITICAL: `update_image()` Commits Inside Outer Transactions
**File**: `repository.py:221-227`

`update_image()` unconditionally calls `self.conn.commit()`. When called from within `_handle_jobs_complete()` (which starts `BEGIN IMMEDIATE` at `server.py:1741`), this **prematurely commits the outer transaction**. The remainder of result persistence proceeds outside any transaction — breaking atomicity.

This is a **correctness bug** that neither Opus nor Codex identified.

#### 🟡 HIGH: Duplicate Search Rebuild on Face Writes
**Files**: `passes/faces.py:55`, `worker.py:1254-1258`

The faces pass calls `repo.update_search_artifacts(image_id)` immediately inside its transaction. Then the worker *also* marks the image as FTS-dirty, triggering a second rebuild during periodic flush. Double writes = double lock time, and the immediate rebuild makes the face transaction significantly longer than necessary.

#### 🟡 HIGH: Missing Claim Index (Dropped by Migration)
**File**: `schema.py:342-345` (created), `schema.py:395-436` (table recreation drops it)

The `idx_job_queue_claim` index was created in schema v8 but the table recreation in a later migration may have dropped it. GPT-5.4 checked the live query plan and found the claim query falls back to `idx_job_queue_status` + temp B-tree sort. This makes every claim operation slower than intended.

#### 🟡 MEDIUM: "Read" APIs That Actually Write
**Files**: `server.py:3606-3608`, `server.py:3805-3825`

`faces/crop` and `faces/crop-batch` write thumbnail cache data, creating surprise contention from UI browsing during pipeline processing.

#### 🟢 LOW: N+1 Query Pattern in `get_impure_clusters()`
**File**: `repository.py:948-986`

Queries all candidate clusters, then calls `compute_cluster_purity()` for each one individually — repeated DB roundtrips + Python vector math.

#### 🟢 LOW: Missing `face_embeddings(identity_id)` Index
`get_face_embeddings()` does a full table scan without an index on `identity_id`.

### 14.3 Discrepancies Between Models

| # | Topic | Opus Position | Codex Position | GPT-5.4 Position | Assessment |
|---|---|---|---|---|---|
| **1** | **Clustering `busy_timeout=0` severity** | LOW | Significant | Agrees (standardize all) | **CRITICAL** — all agree it's a problem, Opus underweights it |
| **2** | **Hotspot #1 ranking** | FTS flush (duration) | Ingest batch (freq×duration) | Missing claim index (performance) | Different valid perspectives; all three issues matter |
| **3** | **`upsert_*` atomicity** | CRITICAL | Not flagged | Flagged + found `update_image()` bug | Opus + GPT-5.4 agree; GPT-5.4 finds concrete broken transaction |
| **4** | **FTS fix approach** | Micro-txns (simple) | Dedicated queue (architectural) | Remove duplicate rebuild first | **GPT-5.4 most targeted** — fix the double-rebuild before architectural changes |
| **5** | **Missing PRAGMAs** | Not audited | `cache_size`, `mmap_size`, etc. | `wal_autocheckpoint`, `journal_size_limit` | Codex + GPT-5.4 both flag; complementary recommendations |
| **6** | **Redundant indexes** | Not mentioned | Not mentioned | Flagged — UNIQUE autoindexes overlap with explicit indexes | **GPT-5.4 unique finding** |
| **7** | **"Read" APIs that write** | Not mentioned | Not mentioned | `faces/crop*` writes surprise-contend | **GPT-5.4 unique finding** |
| **8** | **`update_image()` breaks outer tx** | Not found | Not found | CRITICAL correctness bug | **GPT-5.4 unique finding** — verified in code |
| **9** | **Duplicate face search rebuild** | Not found | Not found | HIGH — immediate + deferred = double writes | **GPT-5.4 unique finding** |
| **10** | **Missing claim index (dropped)** | Not found | Not found | Con #1 — live query plan confirms temp sort | **GPT-5.4 unique finding** — verified against live DB |

### 14.4 Synthesis: Confidence-Weighted Priority

Combining all three models' analyses with agreement weighting:

| Priority | Issue | Models Agreeing | Recommended Action |
|---|---|---|---|
| **P0** | Clustering thread `busy_timeout=0` | All 3 | Fix immediately — 5-line change |
| **P0** | `update_image()` breaks outer tx | GPT-5.4 (verified) | Fix immediately — remove unconditional commit |
| **P1** | FTS5 flush lock duration (2.5s) | All 3 | Reduce to per-image or small-batch txns |
| **P1** | Duplicate face search rebuild | GPT-5.4 (verified) | Remove immediate rebuild from faces pass |
| **P2** | `cluster_faces()` non-transactional | All 3 | Wrap in explicit transaction with batch commits |
| **P2** | Missing claim index (dropped by migration) | GPT-5.4 (verified) | Re-add in schema v28 migration |
| **P3** | Claim loop transaction bursts | All 3 | Batch post-validation into single tx |
| **P4** | Ingest batch size too large | Opus + Codex | Adaptive batch size (50–200) |
| **P5** | Standardize connection config | All 3 | Connection factory function |
| **P6** | Missing PRAGMAs (cache_size, mmap_size) | Codex + GPT-5.4 | Add performance PRAGMAs |
| **P7** | WAL checkpoint management | Codex + GPT-5.4 | Add scheduled passive checkpoints |
| **P8** | "Read" APIs that write (faces/crop) | GPT-5.4 only | Document or defer cache writes |
| **P9** | Incremental face clustering | Codex only | Long-term architectural improvement |

---

## Appendix: Hotspot Ranking (Combined Analysis)

Ranked by **frequency × lock duration × conflict surface**:

| Rank | Operation | Frequency | Lock Duration | Conflict Risk |
|---|---|---|---|---|
| 1 | FTS5 flush (50 images) | Every 60s | 250ms–2.5s | **Critical** |
| 2 | `cluster_faces()` (global rewrite) | On-demand | Seconds–minutes | **Critical** |
| 3 | `claim_leased()` loop | Per distributed claim | 1-5ms × N | **High** |
| 4 | `jobs/complete` payload persist | Per distributed completion | 5-50ms | **High** |
| 5 | Ingest batch (500 images) | Per import | 50-500ms | **Medium-High** |
| 6 | `release_expired_leases()` | Periodic | 1-10ms | Medium |
| 7 | `split_cluster()` | On-demand | Moderate | Medium |
| 8 | `reconcile_runtime_state()` | Startup | 10-50ms | Low |
| 9 | `remap_pending_modules()` | Startup | 10-100ms | Low |
| 10 | Individual `mark_done()` | Per job | <1ms | Low |
