# imganalyzer - Agent Rules

## Project Overview
Electron + React 18 + TypeScript + Tailwind CSS frontend (`imganalyzer-app/`) with a Python backend (`imganalyzer/`) communicating via JSON-RPC over stdin/stdout. The Python backend runs as a persistent child process spawned by Electron.

See `ARCHITECTURE.md` for full project structure.

## Critical Rules

### 1. SQLite: NEVER reuse connections across threads
**This is the #1 recurring bug in this codebase.**

- `imganalyzer/db/connection.py` provides `get_db()` which returns a **singleton** connection created in the calling thread.
- Python's `sqlite3` module defaults to `check_same_thread=True`, meaning a connection created in Thread A **cannot** be used in Thread B. Doing so raises: `"SQLite objects created in a thread can only be used in that same thread"`.
- The JSON-RPC server runs in the main thread, but the batch worker runs in a **daemon thread** (via `threading.Thread`). The worker **must not** call `get_db()` — it must create its own `sqlite3.connect()` with `check_same_thread=False`.
- Always use WAL mode (`PRAGMA journal_mode=WAL`) when creating cross-thread connections to avoid locking conflicts with the main thread's connection.
- When adding ANY new code that touches the database from a non-main thread, always create a fresh connection using `get_db_path()` + `sqlite3.connect(..., check_same_thread=False)`.

**Pattern to follow:**
```python
import sqlite3
from imganalyzer.db.connection import get_db_path

# In a worker/background thread:
db_path = get_db_path()
conn = sqlite3.connect(str(db_path), check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
conn.row_factory = sqlite3.Row
```

**Pattern to AVOID:**
```python
from imganalyzer.db.connection import get_db
# WRONG in a background thread - will crash or silently fail
db = get_db()
```

### 2. signal.signal() only works in the main thread
- `signal.signal()` raises `ValueError: signal only works in main thread of the main interpreter` when called from any non-main thread.
- The worker's `run()` method is called from a daemon thread in server mode (but from the main thread in CLI mode).
- Always guard `signal.signal()` calls: `if threading.current_thread() is threading.main_thread():`

### 3. stdout is reserved for JSON-RPC messages
- `sys.stdout` is redirected to `sys.stderr` at startup in `server.py` (line ~53).
- All JSON-RPC communication uses `_real_stdout` (captured before redirect).
- **Never use `print()` in server mode** — it goes to stderr. Use `_send()` / `_send_notification()` for JSON-RPC, or `sys.stderr.write()` for debug output.
- The `_send_lock` (threading.Lock) must be held for ALL writes to `_real_stdout` to prevent interleaved JSON from concurrent threads.

### 4. Python module-level name binding defeats monkey-patching
- Monkey-patching `builtins.print` does NOT affect modules that already imported `print` at the module level (which is all of them — `print` is resolved at compile time).
- Use direct callback references (e.g., `worker._result_notify = callback`) instead of monkey-patching builtins.

### 5. ThreadPoolExecutor futures must be collected
- The worker uses `ThreadPoolExecutor` for IO/cloud analysis modules.
- Futures from the executor must be explicitly collected and their results/exceptions handled.
- Uncollected futures with exceptions are silently swallowed.
- Always emit result notifications for both success AND failure cases.

## Architecture Reminders

### Notification Pipeline (5 stages)
```
worker.py _emit_result()
  -> callback _result_notify(payload)
  -> server.py _send_notification("run/result", payload)
  -> _send() -> _real_stdout (JSON-RPC)
  -> python-rpc.ts handleLine() -> globalNotificationCb()
  -> batch.ts notification handler -> emitResult() -> IPC batch:result
  -> renderer onBatchResult -> React state -> LiveResultsFeed
```

### Progress updates vs Result notifications
- **Progress** (bars, counts): DB polling via `rpc.call('status', {})` every 1s — completely separate from notifications.
- **Results** (per-image): JSON-RPC notifications (`run/result`) pushed from worker through the pipeline above.
- These are independent systems. Progress can work while results are broken (and vice versa).

### Key file locations
- Python server: `imganalyzer/server.py`
- Batch worker: `imganalyzer/pipeline/worker.py`
- DB connection: `imganalyzer/db/connection.py`
- Job queue: `imganalyzer/db/queue.py`
- Electron batch handler: `imganalyzer-app/src/main/batch.ts`
- JSON-RPC transport: `imganalyzer-app/src/main/python-rpc.ts`
- React batch hook: `imganalyzer-app/src/renderer/hooks/useBatchProcess.ts`
- LiveResultsFeed: `imganalyzer-app/src/renderer/components/LiveResultsFeed.tsx`

## Code Style
- Python: Follow existing patterns in the codebase. Use type hints. Use `typer` for CLI.
- TypeScript: Strict mode. Use React hooks. Tailwind for styling.
- Test with `pytest` for Python, no test framework set up for the Electron app yet.
