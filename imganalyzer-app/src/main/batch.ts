/**
 * batch.ts — IPC handlers for the batch processing pipeline.
 *
 * All analysis work is delegated to the Python `imganalyzer` CLI via
 * `conda run -n imganalyzer python -m imganalyzer.cli <cmd>`.  This module
 * manages subprocess lifecycle, progress polling, and event forwarding to
 * the renderer.
 *
 * IPC channels (main ↔ renderer):
 *   Invoke (renderer → main):
 *     batch:ingest    — scan folder, register images, enqueue jobs
 *     batch:start     — start `imganalyzer run`
 *     batch:pause     — kill run process (queue preserved in DB)
 *     batch:resume    — re-spawn run with saved config
 *     batch:stop      — kill run + clear pending/running jobs from DB
 *
 *   Events (main → renderer):
 *     batch:tick             — polled stats every 1 s (BatchStats)
 *     batch:result           — per-job completion (BatchResult)
 *     batch:ingest-line      — raw ingest stdout lines
 *     batch:ingest-progress  — structured ingest progress (BatchIngestProgress)
 */

import { ipcMain, BrowserWindow } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import { join } from 'path'

// ── Constants ────────────────────────────────────────────────────────────────

const PKG_ROOT = process.env.IMGANALYZER_PKG_ROOT || 'D:\\Code\\imganalyzer'
const CONDA_ENV = 'imganalyzer'
const POLL_INTERVAL_MS = 1000
// Sliding window for images/sec computation (ms)
const RATE_WINDOW_MS = 10_000
// How many recent completion durations to average for avgMs
const AVG_WINDOW = 100

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SessionConfig {
  folder: string
  modules: string[]
  workers: number
  cloudWorkers: number
  cloudProvider: string
  recursive: boolean
  noHash: boolean
}

export interface BatchModuleStats {
  pending: number
  running: number
  done: number
  failed: number
  skipped: number
}

export interface BatchStats {
  status: BatchStatus
  totalImages: number
  modules: Partial<Record<string, BatchModuleStats>>
  totals: { pending: number; running: number; done: number; failed: number; skipped: number }
  avgMsPerImage: number
  imagesPerSec: number
  estimatedMs: number
  elapsedMs: number
}

export type BatchStatus =
  | 'idle'
  | 'ingesting'
  | 'running'
  | 'paused'
  | 'done'
  | 'stopped'
  | 'error'

export interface BatchResult {
  path: string
  module: string
  status: 'done' | 'failed' | 'skipped'
  durationMs: number
  error?: string
}

export interface BatchIngestProgress {
  scanned: number
  total: number
  registered: number
  enqueued: number
  skipped: number
  current: string
}

// ── Module-scope state ────────────────────────────────────────────────────────

let mainWin: BrowserWindow | null = null
let runProcess: ChildProcess | null = null
let pollTimer: NodeJS.Timeout | null = null
let sessionConfig: SessionConfig | null = null
let batchStatus: BatchStatus = 'idle'
let sessionStartMs = 0

// Sliding window of { timestamp, durationMs } for rate + avg computation
const completionWindow: Array<{ ts: number; durationMs: number }> = []

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Spawn a conda CLI command, returning the child process. */
function condaSpawn(cliArgs: string[], opts: { cwd?: string } = {}): ChildProcess {
  return spawn(
    'conda',
    [
      'run', '-n', CONDA_ENV, '--no-capture-output',
      'python', '-m', 'imganalyzer.cli',
      ...cliArgs,
    ],
    {
      cwd: opts.cwd ?? PKG_ROOT,
      env: {
        ...process.env,
        HF_HUB_DISABLE_SYMLINKS_WARNING: '1',
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1',
      },
      // Pipe all streams so we can read stdout/stderr
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  )
}

/** Run a CLI command to completion and return stdout. */
function condaExec(cliArgs: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(
      'conda',
      [
        'run', '-n', CONDA_ENV, '--no-capture-output',
        'python', '-m', 'imganalyzer.cli',
        ...cliArgs,
      ],
      {
        cwd: PKG_ROOT,
        env: {
          ...process.env,
          HF_HUB_DISABLE_SYMLINKS_WARNING: '1',
          PYTHONIOENCODING: 'utf-8',
          PYTHONUTF8: '1',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      }
    )

    let stdout = ''
    proc.stdout?.on('data', (chunk: Buffer) => { stdout += chunk.toString('utf8') })
    proc.stderr?.on('data', () => { /* swallow stderr */ })

    proc.on('close', (code) => {
      if (code !== 0) reject(new Error(`CLI exited with code ${code}`))
      else resolve(stdout)
    })
    proc.on('error', reject)

    // Safety timeout
    const timer = setTimeout(() => {
      try { proc.kill() } catch { /* ignore */ }
      reject(new Error('condaExec timed out'))
    }, 30_000)
    proc.on('close', () => clearTimeout(timer))
  })
}

/** Emit a batch:tick event to the renderer with current stats. */
function emitTick(stats: BatchStats): void {
  mainWin?.webContents?.send('batch:tick', stats)
}

/** Emit a batch:result event to the renderer. */
function emitResult(result: BatchResult): void {
  mainWin?.webContents?.send('batch:result', result)
}

/** Emit a batch:ingest-progress event to the renderer. */
function emitIngestProgress(progress: BatchIngestProgress): void {
  mainWin?.webContents?.send('batch:ingest-progress', progress)
}

/** Record a completion and maintain the sliding window. */
function recordCompletion(durationMs: number): void {
  const now = Date.now()
  completionWindow.push({ ts: now, durationMs })
  // Evict entries older than RATE_WINDOW_MS
  const cutoff = now - RATE_WINDOW_MS
  while (completionWindow.length > 0 && completionWindow[0].ts < cutoff) {
    completionWindow.shift()
  }
  // Keep the window bounded
  if (completionWindow.length > AVG_WINDOW * 2) {
    completionWindow.splice(0, completionWindow.length - AVG_WINDOW * 2)
  }
}

/** Compute derived metrics from the completion window. */
function computeMetrics(
  pending: number,
  workers: number,
  cloudWorkers: number
): { imagesPerSec: number; avgMsPerImage: number; estimatedMs: number } {
  const now = Date.now()
  const cutoff = now - RATE_WINDOW_MS
  const recent = completionWindow.filter((e) => e.ts >= cutoff)

  const imagesPerSec = recent.length > 0 ? recent.length / (RATE_WINDOW_MS / 1000) : 0

  const lastN = completionWindow.slice(-AVG_WINDOW)
  const avgMsPerImage =
    lastN.length > 0
      ? lastN.reduce((sum, e) => sum + e.durationMs, 0) / lastN.length
      : 0

  const effectiveWorkers = Math.max(1, workers + cloudWorkers)
  const estimatedMs =
    avgMsPerImage > 0 && pending > 0
      ? (pending * avgMsPerImage) / effectiveWorkers
      : 0

  return { imagesPerSec, avgMsPerImage, estimatedMs }
}

/** Poll `imganalyzer status --json` and emit a batch:tick. */
async function doPoll(): Promise<void> {
  try {
    const raw = await condaExec(['status', '--json'])
    // The JSON is the first line that parses successfully
    const line = raw.split('\n').find((l) => l.trim().startsWith('{'))
    if (!line) return
    const data = JSON.parse(line) as {
      total_images: number
      modules: Record<string, Record<string, number>>
      totals: Record<string, number>
    }

    const workers = sessionConfig?.workers ?? 1
    const cloudWorkers = sessionConfig?.cloudWorkers ?? 4
    const pending = data.totals.pending ?? 0
    const metrics = computeMetrics(pending, workers, cloudWorkers)

    const stats: BatchStats = {
      status: batchStatus,
      totalImages: data.total_images,
      modules: data.modules as Partial<Record<string, BatchModuleStats>>,
      totals: {
        pending:  data.totals.pending  ?? 0,
        running:  data.totals.running  ?? 0,
        done:     data.totals.done     ?? 0,
        failed:   data.totals.failed   ?? 0,
        skipped:  data.totals.skipped  ?? 0,
      },
      avgMsPerImage: metrics.avgMsPerImage,
      imagesPerSec:  metrics.imagesPerSec,
      estimatedMs:   metrics.estimatedMs,
      elapsedMs:     sessionStartMs > 0 ? Date.now() - sessionStartMs : 0,
    }

    emitTick(stats)
  } catch {
    // Polling errors are non-fatal — just skip this tick
  }
}

/** Start the 1-second poll loop. */
function startPolling(): void {
  stopPolling()
  pollTimer = setInterval(() => { void doPoll() }, POLL_INTERVAL_MS)
}

/** Stop the poll loop. */
function stopPolling(): void {
  if (pollTimer !== null) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

/** Attach [RESULT] line parser to a run process stdout. */
function attachResultParser(proc: ChildProcess): void {
  let buf = ''
  proc.stdout?.on('data', (chunk: Buffer) => {
    buf += chunk.toString()
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (trimmed.startsWith('[RESULT] ')) {
        try {
          const payload = JSON.parse(trimmed.slice('[RESULT] '.length)) as {
            path: string
            module: string
            status: 'done' | 'failed' | 'skipped'
            ms: number
            error?: string
          }
          const result: BatchResult = {
            path:       payload.path,
            module:     payload.module,
            status:     payload.status,
            durationMs: payload.ms,
            error:      payload.error,
          }
          emitResult(result)
          if (payload.status === 'done') {
            recordCompletion(payload.ms)
          }
        } catch {
          // Malformed [RESULT] line — ignore
        }
      }
    }
  })
}

/** Kill the run process if alive. */
function killRunProcess(): void {
  if (runProcess) {
    try { runProcess.kill('SIGTERM') } catch { /* ignore */ }
    runProcess = null
  }
}

// ── IPC Handlers ──────────────────────────────────────────────────────────────

export function registerBatchHandlers(win: BrowserWindow): void {
  mainWin = win

  // ── batch:ingest ──────────────────────────────────────────────────────────
  ipcMain.handle(
    'batch:ingest',
    async (
      _evt,
      folder: string,
      modules: string[],
      recursive: boolean,
      noHash: boolean
    ): Promise<{ registered: number; enqueued: number; skipped: number }> => {
      batchStatus = 'ingesting'
      // Reset sliding window for a fresh session
      completionWindow.length = 0
      sessionStartMs = 0

      const args: string[] = ['ingest', folder, '--modules', modules.join(',')]
      if (!recursive) args.push('--no-recursive')
      if (noHash)     args.push('--no-hash')

      return new Promise((resolve, reject) => {
        const proc = condaSpawn(args)
        ingestProcess = proc

        let combined = ''
        let stdoutBuf = ''

        const processLines = (chunk: Buffer) => {
          const text = chunk.toString()
          combined += text
          stdoutBuf += text
          const lines = stdoutBuf.split('\n')
          stdoutBuf = lines.pop() ?? ''
          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed) continue
            // Parse structured [PROGRESS] lines and emit as ingest-progress events
            if (trimmed.startsWith('[PROGRESS] ')) {
              try {
                const payload = JSON.parse(trimmed.slice('[PROGRESS] '.length)) as BatchIngestProgress
                emitIngestProgress(payload)
              } catch {
                // Malformed [PROGRESS] line — fall through to forward as raw line
              }
            }
            // Always forward as a raw ingest line too (for debug / fallback log)
            mainWin?.webContents?.send('batch:ingest-line', line)
          }
        }

        proc.stdout?.on('data', processLines)
        proc.stderr?.on('data', processLines)

        proc.on('close', (code) => {
          ingestProcess = null
          if (code !== 0) {
            reject(new Error(`ingest exited with code ${code}`))
            return
          }
          // Parse "Ingest complete. Registered: N  Enqueued: N  Skipped: N"
          const m = combined.match(
            /Registered:\s*(\d+)\s+Enqueued:\s*(\d+)\s+Skipped:\s*(\d+)/i
          )
          resolve({
            registered: m ? parseInt(m[1]) : 0,
            enqueued:   m ? parseInt(m[2]) : 0,
            skipped:    m ? parseInt(m[3]) : 0,
          })
        })

        proc.on('error', reject)
      })
    }
  )

  // ── batch:start ───────────────────────────────────────────────────────────
  ipcMain.handle(
    'batch:start',
    async (
      _evt,
      folder: string,
      modules: string[],
      workers: number,
      cloudProvider = 'copilot',
      recursive = true,
      noHash = false,
      cloudWorkers = 4
    ): Promise<void> => {
      killRunProcess()

      sessionConfig = { folder, modules, workers, cloudWorkers, cloudProvider, recursive, noHash }
      sessionStartMs = Date.now()
      completionWindow.length = 0
      batchStatus = 'running'

      const args: string[] = [
        'run',
        '--workers',       String(workers),
        '--cloud-workers', String(cloudWorkers),
        '--cloud',         cloudProvider,
        '--no-xmp',
        '--verbose',
      ]

      const proc = condaSpawn(args)
      runProcess = proc
      attachResultParser(proc)

      proc.stderr?.on('data', () => { /* swallow — Rich output goes to stderr */ })

      proc.on('close', (code) => {
        runProcess = null
        if (batchStatus === 'running') {
          // Emit one final tick with done/error status so UI can show completion,
          // then reset to idle so a fresh session is possible without stale state.
          const finalStatus: BatchStatus = code === 0 ? 'done' : 'error'
          batchStatus = finalStatus
          void doPoll().then(() => {
            // Small delay so the renderer can render the final tick before we reset
            setTimeout(() => { batchStatus = 'idle' }, 3000)
          })
        }
        stopPolling()
      })

      proc.on('error', () => {
        runProcess = null
        if (batchStatus === 'running') {
          batchStatus = 'error'
          void doPoll().then(() => {
            setTimeout(() => { batchStatus = 'idle' }, 3000)
          })
        }
        stopPolling()
      })

      startPolling()
    }
  )

  // ── batch:pause ───────────────────────────────────────────────────────────
  ipcMain.handle('batch:pause', async (): Promise<void> => {
    killRunProcess()
    stopPolling()
    batchStatus = 'paused'
    // Emit one tick so the UI updates immediately
    void doPoll()
  })

  // ── batch:resume ──────────────────────────────────────────────────────────
  ipcMain.handle('batch:resume', async (): Promise<void> => {
    if (!sessionConfig) return
    const { workers, cloudWorkers, cloudProvider } = sessionConfig

    const args: string[] = [
      'run',
      '--workers',       String(workers),
      '--cloud-workers', String(cloudWorkers),
      '--cloud',         cloudProvider,
      '--no-xmp',
      '--verbose',
    ]

    const proc = condaSpawn(args)
    runProcess = proc
    batchStatus = 'running'
    attachResultParser(proc)

    proc.stderr?.on('data', () => { /* swallow */ })

    proc.on('close', (code) => {
      runProcess = null
      if (batchStatus === 'running') {
        batchStatus = code === 0 ? 'done' : 'error'
        void doPoll().then(() => {
          setTimeout(() => { batchStatus = 'idle' }, 3000)
        })
      }
      stopPolling()
    })

    proc.on('error', () => {
      runProcess = null
      if (batchStatus === 'running') {
        batchStatus = 'error'
        void doPoll().then(() => {
          setTimeout(() => { batchStatus = 'idle' }, 3000)
        })
      }
      stopPolling()
    })

    startPolling()
  })

  // ── batch:stop ────────────────────────────────────────────────────────────
  ipcMain.handle('batch:stop', async (_evt, folder: string): Promise<void> => {
    killRunProcess()
    stopPolling()
    batchStatus = 'stopped'

    // Clear pending + running jobs for this folder from the DB
    try {
      await condaExec([
        'queue-clear', folder,
        '--status', 'pending,running',
      ])
    } catch {
      // Best-effort — log but don't throw
    }

    // Emit a final stopped tick
    void doPoll()

    // Reset session so a fresh start is possible
    sessionConfig = null
    completionWindow.length = 0
    sessionStartMs = 0
    batchStatus = 'idle'
  })

  // ── batch:check-pending ───────────────────────────────────────────────────
  // Returns the number of pending + running jobs in the DB so the renderer
  // can decide whether to auto-resume on startup.
  ipcMain.handle('batch:check-pending', async (): Promise<{ pending: number; running: number }> => {
    try {
      const raw = await condaExec(['status', '--json'])
      const line = raw.split('\n').find((l) => l.trim().startsWith('{'))
      if (!line) return { pending: 0, running: 0 }
      const data = JSON.parse(line) as { totals: Record<string, number> }
      return {
        pending: data.totals.pending ?? 0,
        running: data.totals.running ?? 0,
      }
    } catch {
      return { pending: 0, running: 0 }
    }
  })

  // ── batch:resume-pending ──────────────────────────────────────────────────
  // Spawns `imganalyzer run` to drain whatever is already in the queue,
  // without needing a prior ingest.  Uses sessionConfig if available
  // (same session), or sensible defaults (workers=1, no cloud) otherwise.
  ipcMain.handle(
    'batch:resume-pending',
    async (_evt, workers = 1, cloudProvider = 'copilot', cloudWorkers = 4): Promise<void> => {
      if (batchStatus === 'running') return  // already running

      killRunProcess()

      const w  = sessionConfig?.workers      ?? workers
      const cw = sessionConfig?.cloudWorkers ?? cloudWorkers
      const cloud = sessionConfig?.cloudProvider ?? cloudProvider

      sessionStartMs = Date.now()
      completionWindow.length = 0
      batchStatus = 'running'

      const args: string[] = [
        'run',
        '--workers',       String(w),
        '--cloud-workers', String(cw),
        '--cloud',         cloud,
        '--no-xmp',
        '--verbose',
      ]

      const proc = condaSpawn(args)
      runProcess = proc
      attachResultParser(proc)

      proc.stderr?.on('data', () => { /* swallow */ })

      proc.on('close', (code) => {
        runProcess = null
        if (batchStatus === 'running') {
          const finalStatus: BatchStatus = code === 0 ? 'done' : 'error'
          batchStatus = finalStatus
          void doPoll().then(() => {
            setTimeout(() => { batchStatus = 'idle' }, 3000)
          })
        }
        stopPolling()
      })

      proc.on('error', () => {
        runProcess = null
        if (batchStatus === 'running') {
          batchStatus = 'error'
          void doPoll().then(() => {
            setTimeout(() => { batchStatus = 'idle' }, 3000)
          })
        }
        stopPolling()
      })

      startPolling()
    }
  )

  // ── batch:retry-failed ────────────────────────────────────────────────────
  // Re-enqueues all failed jobs (across the modules that have failures) then
  // re-spawns the worker.  The renderer passes the list of module keys that
  // currently have failures so we only call `rebuild` for those modules.
  ipcMain.handle(
    'batch:retry-failed',
    async (_evt, modules: string[]): Promise<void> => {
      if (batchStatus === 'running') return  // don't interfere with an active run

      killRunProcess()
      stopPolling()

      // Re-enqueue only the failed jobs for each affected module
      for (const mod of modules) {
        try {
          await condaExec(['rebuild', mod, '--failed-only'])
        } catch {
          // Best-effort: continue to next module even if one rebuild fails
        }
      }

      // Spawn the worker to process the newly-enqueued jobs
      const w  = sessionConfig?.workers      ?? 1
      const cw = sessionConfig?.cloudWorkers ?? 4
      const cloud = sessionConfig?.cloudProvider ?? 'copilot'

      sessionStartMs = Date.now()
      completionWindow.length = 0
      batchStatus = 'running'

      const args: string[] = [
        'run',
        '--workers',       String(w),
        '--cloud-workers', String(cw),
        '--cloud',         cloud,
        '--no-xmp',
        '--verbose',
      ]

      const proc = condaSpawn(args)
      runProcess = proc
      attachResultParser(proc)

      proc.stderr?.on('data', () => { /* swallow */ })

      proc.on('close', (code) => {
        runProcess = null
        if (batchStatus === 'running') {
          const finalStatus: BatchStatus = code === 0 ? 'done' : 'error'
          batchStatus = finalStatus
          void doPoll().then(() => {
            setTimeout(() => { batchStatus = 'idle' }, 3000)
          })
        }
        stopPolling()
      })

      proc.on('error', () => {
        runProcess = null
        if (batchStatus === 'running') {
          batchStatus = 'error'
          void doPoll().then(() => {
            setTimeout(() => { batchStatus = 'idle' }, 3000)
          })
        }
        stopPolling()
      })

      startPolling()
    }
  )

  // ── batch:queue-clear-all ─────────────────────────────────────────────────
  // Wipes every job in the queue regardless of status.  Refuses to run while
  // a batch is in progress — the caller must stop first.
  ipcMain.handle('batch:queue-clear-all', async (): Promise<{ deleted: number }> => {
    if (batchStatus === 'running' || batchStatus === 'paused') {
      throw new Error('Cannot clear queue while a batch is running. Stop the batch first.')
    }

    killRunProcess()
    stopPolling()

    const raw = await condaExec(['queue-clear', '--status', 'all'])
    // Parse "Cleared N job(s)" from the CLI output
    const match = raw.match(/Cleared (\d+) job/)
    const deleted = match ? parseInt(match[1], 10) : 0

    // Reset server-side state
    sessionConfig = null
    completionWindow.length = 0
    sessionStartMs = 0
    batchStatus = 'idle'

    // Emit one final poll tick so the renderer resets its counters
    void doPoll()

    return { deleted }
  })
}

// Keep a reference to the ingest process so future code can cancel it if needed
let ingestProcess: ChildProcess | null = null
