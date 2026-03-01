/**
 * batch.ts — IPC handlers for the batch processing pipeline.
 *
 * Delegates to the persistent Python JSON-RPC server via `python-rpc.ts`.
 * This eliminates the 1-3s conda subprocess overhead per operation and
 * removes the need for status polling via subprocess.
 *
 * IPC channels (main <-> renderer):
 *   Invoke (renderer -> main):
 *     batch:ingest    — scan folder, register images, enqueue jobs
 *     batch:start     — start `imganalyzer run`
 *     batch:pause     — cancel run (queue preserved in DB)
 *     batch:resume    — re-start run with saved config
 *     batch:stop      — cancel run + clear pending/running jobs from DB
 *
 *   Events (main -> renderer):
 *     batch:tick             — polled stats every 1 s (BatchStats)
 *     batch:result           — per-job completion (BatchResult)
 *     batch:ingest-line      — raw ingest stdout lines
 *     batch:ingest-progress  — structured ingest progress (BatchIngestProgress)
 */

import { ipcMain, BrowserWindow } from 'electron'
import { rpc, ensureServerRunning, setNotificationListener, shutdownServer } from './python-rpc'

// ── Constants ────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 1000
// Max age for completion-window entries used by avgMsPerImage (ms)
const COMPLETION_WINDOW_MS = 10_000
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
  imagesPerSec: number
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
  keywords?: string[]
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
let pollTimer: NodeJS.Timeout | null = null
let sessionConfig: SessionConfig | null = null
let batchStatus: BatchStatus = 'idle'
let sessionStartMs = 0
let isRunActive = false
let currentRunId = 0
let idleTimer: ReturnType<typeof setTimeout> | null = null

// Sliding window of { timestamp, durationMs, module } for avg-per-image computation
const completionWindow: Array<{ ts: number; durationMs: number; module: string }> = []

// Session-lifetime counters for average img/s (total done / elapsed).
let sessionCompletions = 0
const moduleCompletions: Record<string, number> = {}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Reset all completion counters for a fresh session. */
function resetSessionCounters(): void {
  completionWindow.length = 0
  sessionCompletions = 0
  for (const key of Object.keys(moduleCompletions)) {
    delete moduleCompletions[key]
  }
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

/** Record a completion and maintain the sliding window + session counters. */
function recordCompletion(durationMs: number, module: string): void {
  const now = Date.now()
  completionWindow.push({ ts: now, durationMs, module })
  // Evict entries older than COMPLETION_WINDOW_MS
  const cutoff = now - COMPLETION_WINDOW_MS
  while (completionWindow.length > 0 && completionWindow[0].ts < cutoff) {
    completionWindow.shift()
  }
  // Keep the window bounded
  if (completionWindow.length > AVG_WINDOW * 2) {
    completionWindow.splice(0, completionWindow.length - AVG_WINDOW * 2)
  }
  // Session-lifetime counters for average speed
  sessionCompletions++
  moduleCompletions[module] = (moduleCompletions[module] ?? 0) + 1
}

/** Compute derived metrics using session-average speed. */
function computeMetrics(
  pending: number,
  workers: number,
  cloudWorkers: number
): { imagesPerSec: number; avgMsPerImage: number; estimatedMs: number } {
  // Average img/s over the entire run (survives pauses / completion)
  const elapsedSec = sessionStartMs > 0 ? (Date.now() - sessionStartMs) / 1000 : 0
  const imagesPerSec = elapsedSec > 0 && sessionCompletions > 0
    ? sessionCompletions / elapsedSec
    : 0

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

/** Compute per-module images/sec as session-average (total done / elapsed). */
function computeModuleSpeeds(): Record<string, number> {
  const elapsedSec = sessionStartMs > 0 ? (Date.now() - sessionStartMs) / 1000 : 0
  if (elapsedSec <= 0) return {}

  const speeds: Record<string, number> = {}
  for (const [mod, count] of Object.entries(moduleCompletions)) {
    speeds[mod] = count / elapsedSec
  }
  return speeds
}

/** Poll status via RPC (no subprocess) and emit a batch:tick. */
async function doPoll(): Promise<void> {
  try {
    const data = await rpc.call('status', {}) as {
      total_images: number
      modules: Record<string, Record<string, number>>
      totals: Record<string, number>
    }

    const workers = sessionConfig?.workers ?? 1
    const cloudWorkers = sessionConfig?.cloudWorkers ?? 4
    const pending = data.totals.pending ?? 0
    const metrics = computeMetrics(pending, workers, cloudWorkers)
    const moduleSpeeds = computeModuleSpeeds()

    // Merge per-module speed into each module's stats
    const modulesWithSpeed: Partial<Record<string, BatchModuleStats>> = {}
    for (const [mod, modStats] of Object.entries(data.modules)) {
      modulesWithSpeed[mod] = {
        ...(modStats as unknown as BatchModuleStats),
        imagesPerSec: moduleSpeeds[mod] ?? 0,
      }
    }

    const stats: BatchStats = {
      status: batchStatus,
      totalImages: data.total_images,
      modules: modulesWithSpeed,
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

/** Set up the notification listener for streaming run/ingest results. */
function setupNotificationListener(): void {
  setNotificationListener((notif) => {
    const { method, params } = notif
    const p = params as Record<string, unknown>

    switch (method) {
      case 'ingest/progress':
        emitIngestProgress(p as unknown as BatchIngestProgress)
        mainWin?.webContents?.send('batch:ingest-line', JSON.stringify(p))
        break

      case 'run/result': {
        const rawKw = p.keywords
        const keywords = Array.isArray(rawKw)
          ? rawKw as string[]
          : typeof rawKw === 'string'
            ? rawKw.split(',').map((s: string) => s.trim()).filter(Boolean)
            : undefined
        const result: BatchResult = {
          path:       (p.path as string) ?? '',
          module:     (p.module as string) ?? '',
          status:     p.status as 'done' | 'failed' | 'skipped',
          durationMs: (p.ms as number) ?? 0,
          error:      p.error as string | undefined,
          keywords,
        }
        emitResult(result)
        if (result.status === 'done') {
          recordCompletion(result.durationMs, result.module)
        }
        break
      }

      case 'run/done': {
        const runId = currentRunId
        if (batchStatus !== 'running') break
        isRunActive = false
        stopPolling()
        void rpc.call('status', {}).then((data: any) => {
          if (currentRunId !== runId) return
          const pending = data?.totals?.pending ?? 0
          const running = data?.totals?.running ?? 0
          if (pending + running > 0) {
            batchStatus = 'paused'
          } else {
            batchStatus = 'done'
            if (idleTimer) clearTimeout(idleTimer)
            idleTimer = setTimeout(() => { batchStatus = 'idle' }, 3000)
          }
          void doPoll()
        }).catch(() => {
          if (currentRunId !== runId) return
          batchStatus = 'done'
          void doPoll()
          if (idleTimer) clearTimeout(idleTimer)
          idleTimer = setTimeout(() => { batchStatus = 'idle' }, 3000)
        })
        break
      }

      case 'run/error': {
        const runId = currentRunId
        if (batchStatus !== 'running') break
        isRunActive = false
        stopPolling()
        void rpc.call('status', {}).then((data: any) => {
          if (currentRunId !== runId) return
          const pending = data?.totals?.pending ?? 0
          const running = data?.totals?.running ?? 0
          if (pending + running > 0) {
            batchStatus = 'paused'
          } else {
            batchStatus = 'error'
            if (idleTimer) clearTimeout(idleTimer)
            idleTimer = setTimeout(() => { batchStatus = 'idle' }, 3000)
          }
          void doPoll()
        }).catch(() => {
          if (currentRunId !== runId) return
          batchStatus = 'error'
          void doPoll()
          if (idleTimer) clearTimeout(idleTimer)
          idleTimer = setTimeout(() => { batchStatus = 'idle' }, 3000)
        })
        break
      }
    }
  })
}

/** Kill all background work and stop polling. Called on app quit. */
export async function killAllBatchProcesses(): Promise<void> {
  stopPolling()
  if (isRunActive) {
    try {
      await rpc.call('cancel_run', {})
    } catch { /* ignore */ }
    isRunActive = false
  }
  await shutdownServer()
}

// ── IPC Handlers ──────────────────────────────────────────────────────────────

export function registerBatchHandlers(win: BrowserWindow): void {
  mainWin = win
  setupNotificationListener()

  // Ensure the server is started as soon as the app opens
  void ensureServerRunning().catch((err) => {
    console.error('Failed to start Python server:', err)
  })

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
      // Reset counters for a fresh session
      resetSessionCounters()
      sessionStartMs = 0

      try {
        await ensureServerRunning()
        const result = await rpc.call('ingest', {
          folders: [folder],
          modules: modules.join(','),
          recursive,
          computeHash: !noHash,
        }) as { registered?: number; enqueued?: number; skipped?: number }
        return {
          registered: result.registered ?? 0,
          enqueued: result.enqueued ?? 0,
          skipped: result.skipped ?? 0,
        }
      } catch (err) {
        throw new Error(`Ingest failed: ${err}`)
      }
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
      sessionConfig = { folder, modules, workers, cloudWorkers, cloudProvider, recursive, noHash }
      sessionStartMs = Date.now()
      resetSessionCounters()
      currentRunId++
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
      batchStatus = 'running'
      isRunActive = true

      try {
        await ensureServerRunning()
        await rpc.call('run', {
          workers,
          cloudWorkers,
          cloudProvider,
          noXmp: true,
          verbose: true,
        })
      } catch (err) {
        isRunActive = false
        batchStatus = 'error'
        if (idleTimer) clearTimeout(idleTimer)
        idleTimer = setTimeout(() => { batchStatus = 'idle' }, 5000)
        return
      }

      startPolling()
    }
  )

  // ── batch:pause ───────────────────────────────────────────────────────────
  ipcMain.handle('batch:pause', async (): Promise<void> => {
    try {
      await rpc.call('cancel_run', {})
    } catch { /* ignore */ }
    isRunActive = false
    stopPolling()
    batchStatus = 'paused'
    // Emit one tick so the UI updates immediately
    void doPoll()
  })

  // ── batch:resume ──────────────────────────────────────────────────────────
  ipcMain.handle('batch:resume', async (): Promise<void> => {
    // Use session config if available, otherwise fall back to sensible defaults
    // (e.g. after crash recovery when sessionConfig was never populated).
    const w     = sessionConfig?.workers       ?? 1
    const cw    = sessionConfig?.cloudWorkers  ?? 4
    const cloud = sessionConfig?.cloudProvider ?? 'copilot'

    // When there's no sessionConfig, we're recovering from a crash/restart —
    // use staleTimeout=0 to reclaim all stuck 'running' jobs immediately.
    const needsStaleRecovery = !sessionConfig

    currentRunId++
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
    batchStatus = 'running'
    isRunActive = true

    try {
      await ensureServerRunning()
      const runParams: Record<string, unknown> = {
        workers: w,
        cloudWorkers: cw,
        cloudProvider: cloud,
        noXmp: true,
        verbose: true,
      }
      if (needsStaleRecovery) {
        runParams.staleTimeout = 0
      }
      await rpc.call('run', runParams)
    } catch (err) {
      isRunActive = false
      batchStatus = 'error'
      if (idleTimer) clearTimeout(idleTimer)
      idleTimer = setTimeout(() => { batchStatus = 'idle' }, 5000)
      return
    }

    startPolling()
  })

  // ── batch:stop ────────────────────────────────────────────────────────────
  ipcMain.handle('batch:stop', async (_evt, folder: string): Promise<void> => {
    // Cancel the active run
    try {
      await rpc.call('cancel_run', {})
    } catch { /* ignore */ }
    isRunActive = false
    stopPolling()
    batchStatus = 'stopped'

    // Clear pending + running jobs for this folder from the DB
    try {
      await rpc.call('queue_clear', {
        folder,
        status: 'pending,running',
      })
    } catch { /* ignore */ }

    // Emit a final stopped tick
    void doPoll()

    // Reset session so a fresh start is possible
    sessionConfig = null
    resetSessionCounters()
    sessionStartMs = 0
    batchStatus = 'idle'
  })

  // ── batch:check-pending ───────────────────────────────────────────────────
  ipcMain.handle('batch:check-pending', async (): Promise<{ pending: number; running: number }> => {
    try {
      await ensureServerRunning()
      const data = await rpc.call('status', {}) as {
        totals: Record<string, number>
      }
      return {
        pending: data.totals.pending ?? 0,
        running: data.totals.running ?? 0,
      }
    } catch {
      return { pending: 0, running: 0 }
    }
  })

  // ── batch:resume-pending ──────────────────────────────────────────────────
  ipcMain.handle(
    'batch:resume-pending',
    async (_evt, workers = 1, cloudProvider = 'copilot', cloudWorkers = 4): Promise<void> => {
      if (batchStatus === 'running') return

      const w  = sessionConfig?.workers      ?? workers
      const cw = sessionConfig?.cloudWorkers ?? cloudWorkers
      const cloud = sessionConfig?.cloudProvider ?? cloudProvider

      // Populate sessionConfig if it wasn't set (crash recovery / fresh start)
      // so that subsequent batch:resume calls have it available.
      if (!sessionConfig) {
        sessionConfig = {
          folder: '',
          modules: [],
          workers: w,
          cloudWorkers: cw,
          cloudProvider: cloud,
          recursive: true,
          noHash: false,
        }
      }

      sessionStartMs = Date.now()
      resetSessionCounters()
      currentRunId++
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
      batchStatus = 'running'
      isRunActive = true

      try {
        await ensureServerRunning()
        // Use staleTimeout=0 to recover ALL stuck 'running' jobs from a
        // previous crash, regardless of how recently they were claimed.
        await rpc.call('run', {
          workers: w,
          cloudWorkers: cw,
          cloudProvider: cloud,
          noXmp: true,
          verbose: true,
          staleTimeout: 0,
        })
      } catch (err) {
        isRunActive = false
        batchStatus = 'error'
        if (idleTimer) clearTimeout(idleTimer)
        idleTimer = setTimeout(() => { batchStatus = 'idle' }, 5000)
        return
      }

      startPolling()
    }
  )

  // ── batch:retry-failed ────────────────────────────────────────────────────
  ipcMain.handle(
    'batch:retry-failed',
    async (_evt, modules: string[]): Promise<void> => {
      if (batchStatus === 'running') return

      // Re-enqueue only the failed jobs for each affected module
      for (const mod of modules) {
        try {
          await rpc.call('rebuild', { module: mod, failedOnly: true })
        } catch { /* ignore */ }
      }

      // Spawn the worker to process the newly-enqueued jobs
      const w  = sessionConfig?.workers      ?? 1
      const cw = sessionConfig?.cloudWorkers ?? 4
      const cloud = sessionConfig?.cloudProvider ?? 'copilot'

      sessionStartMs = Date.now()
      resetSessionCounters()
      currentRunId++
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
      batchStatus = 'running'
      isRunActive = true

      try {
        await ensureServerRunning()
        await rpc.call('run', {
          workers: w,
          cloudWorkers: cw,
          cloudProvider: cloud,
          noXmp: true,
          verbose: true,
        })
      } catch (err) {
        isRunActive = false
        batchStatus = 'error'
        if (idleTimer) clearTimeout(idleTimer)
        idleTimer = setTimeout(() => { batchStatus = 'idle' }, 5000)
        return
      }

      startPolling()
    }
  )

  // ── batch:queue-clear-all ─────────────────────────────────────────────────
  // Only clears pending/running/failed jobs. Done and skipped rows are
  // preserved so that re-ingest correctly skips already-processed images
  // (especially modules skipped at runtime like cloud_ai/aesthetic with
  // has_people guard, which have no analysis-table data).
  ipcMain.handle('batch:queue-clear-all', async (): Promise<{ deleted: number }> => {
    if (batchStatus === 'running' || batchStatus === 'paused') {
      throw new Error('Cannot clear queue while a batch is running. Stop the batch first.')
    }

    stopPolling()

    const result = await rpc.call('queue_clear', {
      status: 'pending,running,failed',
    }) as { deleted: number }

    // Reset server-side state
    sessionConfig = null
    resetSessionCounters()
    sessionStartMs = 0
    batchStatus = 'idle'

    // Emit one final poll tick so the renderer resets its counters
    void doPoll()

    return result
  })

  // ── batch:queue-clear-done ────────────────────────────────────────────────
  // Removes completed (done + skipped) jobs from the queue.
  // Safe to call while idle — does not affect pending/running/failed jobs.
  // Note: after clearing, re-ingest will re-enqueue these images.
  ipcMain.handle('batch:queue-clear-done', async (): Promise<{ deleted: number }> => {
    if (batchStatus === 'running') {
      throw new Error('Cannot clear completed jobs while a batch is running. Stop the batch first.')
    }

    const result = await rpc.call('queue_clear', {
      status: 'done,skipped',
    }) as { deleted: number }

    // Refresh stats so the renderer sees updated counts
    void doPoll()

    return result
  })
}
