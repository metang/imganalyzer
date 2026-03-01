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

Two-phase processing for GPU memory efficiency:
1. **Phase 1**: Drain all `objects` jobs (GroundingDINO) to set `has_person`/`has_text` flags, then unload the model.
2. **Phase 2**: Sequential GPU passes (blip2, ocr, faces, embedding) with model load/unload between each, plus a concurrent `ThreadPoolExecutor` for cloud AI.

Module priority order: `metadata(100) > technical(90) > objects(85) > blip2(80) > ocr(78) > faces(77) > cloud_ai(70) > aesthetic(60) > embedding(50)`.

Prerequisites: `cloud_ai`, `aesthetic`, `ocr`, and `faces` all depend on `objects` completing first.

### Database

SQLite with WAL mode at `~/.cache/imganalyzer/imganalyzer.db`. Schema managed by sequential migrations in `db/schema.py`. The `overrides` table protects user-edited fields from being overwritten during re-analysis.

for ## Critical Conventions

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

Uncollected futures with exceptions are silently swallowed. Always handle results/exceptions and emit result notifications for both success and failure.

### Python style

- Type hints on all functions.
- `typer` for CLI commands.
- Follow existing patterns in the codebase.

### TypeScript style

- Strict mode (`strict: true` in tsconfig).
- React hooks for state management. Tailwind CSS for styling.
- IPC types declared in `src/renderer/global.d.ts`.
