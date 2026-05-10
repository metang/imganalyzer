# Copilot Instructions — imganalyzer

## Build, Test, Lint

### Python backend (`imganalyzer/`)

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test
pytest tests/test_imganalyzer.py::TestClassName::test_method_name

# Lint
ruff check imganalyzer/

# Type check
mypy imganalyzer/
```

Config: `ruff` line-length 100, target Python 3.10. See `pyproject.toml` for full settings.

### Electron frontend (`imganalyzer-app/`)

```bash
cd imganalyzer-app
npm install
npm run dev      # Dev mode with hot reload
npm run build    # Production build
```

No frontend test framework is configured.

## Architecture

Electron + React 18 + TypeScript + Tailwind frontend (`imganalyzer-app/`) with a Python backend (`imganalyzer/`) communicating via **JSON-RPC 2.0 over stdin/stdout**. Electron spawns `python -m imganalyzer.server` as a single persistent child process at startup — there is no per-call subprocess overhead.

The Python package also works standalone as a CLI via `typer` (`imganalyzer.cli`).

### Communication flow

```
Renderer (React) → IPC → Main process (Electron)
  → python-rpc.ts → stdin → server.py (JSON-RPC)
  → stdout → python-rpc.ts → IPC → Renderer
```

**Progress** (bars, counts) uses DB polling via `rpc.call('status', {})` every 1s. **Results** (per-image) are pushed as JSON-RPC notifications (`run/result`) through a 5-stage pipeline from `worker.py` → `server.py` → `python-rpc.ts` → `batch.ts` → React state. These are independent systems.

### Batch pipeline

Chunk-first four-phase GPU processing for time-to-first-complete results and GPU memory efficiency:
1. **Phase 0**: `caption` (Qwen 3.5 via Ollama, ~8.7 GB)
2. **Phase 1**: `objects` (GroundingDINO, batch=4, unlocks dependents)
3. **Phase 2**: `faces` + `embedding` (co-resident when VRAM allows)
4. **Phase 3**: `perception` (UniPercept, CUDA-only, exclusive)

Module priority order: `metadata(100) > technical(90) > objects(85) > caption(80) > faces(77) > perception(60) > embedding(50)`.

Prerequisites: `faces` and `embedding` depend on `objects` completing first. Legacy queue names are remapped (`local_ai`/`blip2`/`cloud_ai` → `caption`, `aesthetic` → `perception`).

### Ownership map

- Backend module metadata is canonical in `imganalyzer\pipeline\module_registry.py`: active/legacy modules, table map, priorities, prerequisites, GPU phases, VRAM, queue remaps, and distributed flags.
- Frontend module metadata is canonical in `imganalyzer-app\src\shared\moduleMetadata.ts`: labels, pass selector order, result/progress labels, and legacy retry aliases.
- RPC seams live in `imganalyzer\rpc\handler_registry.py` (method lookup + transient SQLite-lock retry) and `imganalyzer\rpc\search_engine_cache.py` (thread-safe SearchEngine reuse/rebind). `server.py` owns public JSON-RPC names/transports.
- Repo hygiene: generated root logs, caches, build outputs, `*.egg-info`, `__pycache__`, and generated `model-eval` reports are ignored; source, tests, fixtures, lockfiles, and canonical docs stay tracked.

### Database

SQLite with WAL mode at `~/.cache/imganalyzer/imganalyzer.db`. Schema managed by sequential migrations in `db/schema.py`. The `overrides` table protects user-edited fields from being overwritten during re-analysis.

## Critical Conventions

### SQLite: NEVER reuse connections across threads

This is the #1 recurring bug. The JSON-RPC server runs in the main thread; the batch worker runs in a **daemon thread**.

- Main thread: use `get_db()` from `imganalyzer/db/connection.py` (thread-local singleton).
- Background/worker threads: create a fresh connection with `check_same_thread=False` and WAL mode:

```python
import sqlite3
from imganalyzer.db.connection import get_db_path

db_path = get_db_path()
conn = sqlite3.connect(str(db_path), check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
conn.row_factory = sqlite3.Row
```

**Never** call `get_db()` from a background thread.

### stdout is reserved for JSON-RPC

`sys.stdout` is redirected to `sys.stderr` at startup in `server.py`. All JSON-RPC uses `_real_stdout` (captured before redirect). Never use `print()` in server mode. The `_send_lock` must be held for all writes to `_real_stdout`.

### signal.signal() only works in the main thread

Guard all `signal.signal()` calls:
```python
if threading.current_thread() is threading.main_thread():
    signal.signal(...)
```

### ThreadPoolExecutor futures must be collected

The worker/scheduler uses `ThreadPoolExecutor` for local I/O and per-module execution tasks. Uncollected futures with exceptions are silently swallowed. Always handle results/exceptions and emit result notifications for both success and failure.

### Python style

- Type hints on all functions.
- `typer` for CLI commands.
- Follow existing patterns in the codebase.

### TypeScript style

- Strict mode (`strict: true` in tsconfig).
- React hooks for state management. Tailwind CSS for styling.
- IPC types declared in `src/renderer/global.d.ts`.
