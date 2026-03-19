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
 *     batch:pause-target  — pause coordinator/master/specific worker
 *     batch:resume-target — resume coordinator/master/specific worker
 *
 *   Events (main -> renderer):
 *     batch:tick             — polled stats every 1 s (BatchStats)
 *     batch:result           — per-job completion (BatchResult)
 *     batch:ingest-line      — raw ingest stdout lines
 *     batch:ingest-progress  — structured ingest progress (BatchIngestProgress)
 */

import { ipcMain, BrowserWindow } from 'electron'
import { rpc, ensureServerRunning, setNotificationListener, shutdownServer } from './python-rpc'
import { getCoordinatorStatus, startCoordinator, stopCoordinator } from './coordinator'
import { getAppSettings } from './settings'
import type { CoordinatorStatus } from './settings'

// ── Constants ────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 1000
// Max age for completion-window entries used by avgMsPerImage (ms)
const COMPLETION_WINDOW_MS = 10_000
// How many recent completion durations to average for avgMs
const AVG_WINDOW = 100
// How many completed chunk durations to keep for avg chunk completion time
const CHUNK_AVG_WINDOW = 20

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SessionConfig {
  folder: string
  modules: string[]
  workers: number
  recursive: boolean
  noHash: boolean
  forceReprocess: boolean
  profile: boolean
}

export interface BatchModuleStats {
  pending: number
  running: number
  done: number
  failed: number
  skipped: number
  imagesPerSec: number
  avgMsPerImage: number
}

export interface BatchQueueSummary {
  totalPasses: number
  activePasses: number
  completedPasses: number
  remainingPasses: number
  remainingJobs: number
}

export interface BatchActiveModule {
  module: string
  count: number
}

export interface BatchNode {
  id: string
  role: 'master' | 'worker'
  label: string
  status: string
  desiredState?: string
  stateReason?: string | null
  platform?: string
  lastHeartbeat?: string | null
  lastResultAt?: string | null
  runningJobs: number
  completedJobs: number
  doneJobs: number
  failedJobs: number
  skippedJobs: number
  imagesPerSec: number
  avgMsPerImage: number
  capabilities?: Record<string, unknown>
  activeModules: BatchActiveModule[]
}

export interface BatchStats {
  status: BatchStatus
  monitorOnly: boolean
  coordinator: CoordinatorStatus
  totalImages: number
  modules: Partial<Record<string, BatchModuleStats>>
  totals: { pending: number; running: number; done: number; failed: number; skipped: number }
  avgMsPerImage: number
  imagesPerSec: number
  estimatedMs: number
  elapsedMs: number
  chunkAvgCompletionMs: number
  chunkElapsedMs: number
  chunkEstimatedMs: number
  queue: BatchQueueSummary
  nodes: BatchNode[]
  chunk?: { size: number; index: number; total: number; modules: Record<string, number> }
}

export type BatchStatus =
  | 'idle'
  | 'ingesting'
  | 'running'
  | 'paused'
  | 'done'
  | 'stopped'
  | 'error'

export type BatchPauseMode = 'pause-drain' | 'pause-immediate'

export interface BatchControlTarget {
  scope: 'coordinator' | 'master' | 'worker'
  workerId?: string
}

export interface BatchResult {
  id: string
  jobId?: number
  path: string
  module: string
  status: 'done' | 'failed' | 'skipped'
  durationMs: number
  error?: string
  keywords?: string[]
  nodeId: string
  nodeRole: 'master' | 'worker'
  nodeLabel: string
  completedAt?: string
}

export interface BatchIngestProgress {
  scanned: number
  total: number
  registered: number
  enqueued: number
  skipped: number
  current: string
}

interface CompletionEntry {
  ts: number
  durationMs: number
  module: string
  nodeId: string
}

interface NodeCounters {
  done: number
  failed: number
  skipped: number
  lastResultAt: string | null
}

interface ServerWorkerNode {
  id: string
  displayName?: string
  platform?: string
  capabilities?: Record<string, unknown>
  status?: string
  desiredState?: string
  stateReason?: string | null
  lastHeartbeat?: string | null
  runningJobs?: number
  activeModules?: BatchActiveModule[]
}

interface ServerRecentResult {
  jobId: number
  path: string
  module: string
  status: 'done' | 'failed' | 'skipped'
  durationMs?: number
  error?: string | null
  completedAt?: string
  nodeId?: string
  nodeRole?: 'master' | 'worker'
  nodeLabel?: string
}

interface ServerChunkInfo {
  size: number
  index: number
  total: number
  modules: Record<string, number>
}

interface ServerStatusPayload {
  total_images: number
  modules: Record<string, Record<string, number>>
  module_avg_ms?: Record<string, number>
  chunk?: ServerChunkInfo | null
  totals: Record<string, number>
  remaining_images?: number
  nodes?: {
    master?: {
      id?: string
      role?: 'master'
      displayName?: string
      platform?: string
      desiredState?: string
      stateReason?: string | null
      runningJobs?: number
      activeModules?: BatchActiveModule[]
    }
    workers?: ServerWorkerNode[]
  }
  recent_results?: ServerRecentResult[]
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
let monitorOnly = false
let pollInFlight = false
// Consecutive polls where remainingPasses was 0 while monitorOnly is active.
// We require several zero readings before concluding the run is truly done,
// to avoid stopping polling on transient zero snapshots.
let monitorZeroStreak = 0
const MONITOR_ZERO_THRESHOLD = 5

const MASTER_NODE_ID = 'master'
const MASTER_NODE_LABEL = 'Master device'
const SYNTHETIC_RESULT_WINDOW_SLOP_MS = POLL_INTERVAL_MS * 2

// Sliding window of done results for avg-per-image computation.
const completionWindow: CompletionEntry[] = []
const processedStatusResultKeys = new Set<string>()
const nodeCounters = new Map<string, NodeCounters>()
let nextResultSequence = 0

// Chunk-based ETA: throughput from the last completed chunk.
// passesPerSec is derived from all workers' combined throughput during
// that chunk (wall-clock time includes parallel worker contributions).
let lastChunkPassesPerSec = 0
const chunkCompletionWindowMs: number[] = []
let currentChunkKey: string | null = null
let currentChunkStartMs = 0

// ── Helpers ───────────────────────────────────────────────────────────────────

function normalizePauseMode(mode: unknown): BatchPauseMode {
  return mode === 'pause-immediate' ? 'pause-immediate' : 'pause-drain'
}

function resolveTargetWorkerId(target: BatchControlTarget): string {
  if (target.scope === 'master') return MASTER_NODE_ID
  if (target.scope === 'worker') {
    const workerId = target.workerId?.trim()
    if (!workerId) throw new Error('workerId is required when scope is worker')
    return workerId
  }
  throw new Error(`Unsupported target scope: ${target.scope}`)
}

/** Reset all completion counters for a fresh session. */
function resetSessionCounters(): void {
  completionWindow.length = 0
  processedStatusResultKeys.clear()
  nodeCounters.clear()
  lastChunkPassesPerSec = 0
  chunkCompletionWindowMs.length = 0
  currentChunkKey = null
  currentChunkStartMs = 0
  monitorZeroStreak = 0
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

/** Remove outdated entries from the rolling completion window. */
function pruneCompletionWindow(now: number): void {
  const cutoff = now - COMPLETION_WINDOW_MS
  while (completionWindow.length > 0 && completionWindow[0].ts < cutoff) {
    completionWindow.shift()
  }
}

function trackNodeCounters(result: BatchResult): void {
  const current = nodeCounters.get(result.nodeId) ?? {
    done: 0,
    failed: 0,
    skipped: 0,
    lastResultAt: null,
  }

  if (result.status === 'done') current.done += 1
  else if (result.status === 'failed') current.failed += 1
  else current.skipped += 1

  current.lastResultAt = result.completedAt ?? new Date().toISOString()
  nodeCounters.set(result.nodeId, current)
}

/** Record a completion and maintain the sliding window + session counters. */
function recordCompletion(durationMs: number, module: string, nodeId: string): void {
  const now = Date.now()
  completionWindow.push({ ts: now, durationMs, module, nodeId })
  pruneCompletionWindow(now)
  if (completionWindow.length > AVG_WINDOW * 2) {
    completionWindow.splice(0, completionWindow.length - AVG_WINDOW * 2)
  }
}

function trackResult(result: BatchResult, emit = true): void {
  trackNodeCounters(result)
  if (result.status === 'done') {
    recordCompletion(result.durationMs, result.module, result.nodeId)
  }
  if (emit) emitResult(result)
}

function getStatusResultKey(result: ServerRecentResult): string {
  return `${result.jobId}:${result.completedAt ?? ''}:${result.status}`
}

function nextLocalResultId(): string {
  nextResultSequence += 1
  return `local:${nextResultSequence}`
}

function parseResultStatus(raw: unknown): BatchResult['status'] {
  if (raw === 'done' || raw === 'failed' || raw === 'skipped') return raw
  return 'failed'
}

function parseDurationMs(raw: unknown): number {
  if (typeof raw === 'number') {
    return Number.isFinite(raw) ? raw : 0
  }

  if (typeof raw === 'string') {
    const trimmed = raw.trim()
    if (!trimmed) return 0
    const parsed = Number(trimmed)
    return Number.isFinite(parsed) ? parsed : 0
  }

  return 0
}

function parseResultKeywords(raw: unknown): string[] | undefined {
  if (Array.isArray(raw)) {
    const cleaned = raw
      .map((value) => String(value).trim())
      .filter((value) => value.length > 0)
    return cleaned.length > 0 ? cleaned : undefined
  }

  if (typeof raw === 'string') {
    const trimmed = raw.trim()
    if (!trimmed) return undefined

    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const parsed = JSON.parse(trimmed)
        if (Array.isArray(parsed)) {
          const cleaned = parsed
            .map((value) => String(value).trim())
            .filter((value) => value.length > 0)
          return cleaned.length > 0 ? cleaned : undefined
        }
      } catch {
        // Not valid JSON; fall back to comma-split parsing below.
      }
    }

    const split = trimmed
      .split(',')
      .map((value) => value.trim())
      .filter((value) => value.length > 0)
    return split.length > 0 ? split : undefined
  }

  return undefined
}

function parseServerTimestamp(value?: string | null): number | null {
  if (!value) return null
  const isoLike = value.includes('T') ? value : value.replace(' ', 'T')
  const normalized = /(?:Z|[+-]\d{2}:\d{2})$/.test(isoLike) ? isoLike : `${isoLike}Z`
  const parsed = Date.parse(normalized)
  return Number.isFinite(parsed) ? parsed : null
}

function syncRecentResults(recentResults: ServerRecentResult[]): void {
  if (recentResults.length === 0) return

  const minimumTs =
    sessionStartMs > 0
      ? sessionStartMs - SYNTHETIC_RESULT_WINDOW_SLOP_MS
      : 0

  const ordered = [...recentResults].reverse()
  for (const item of ordered) {
    const key = getStatusResultKey(item)
    if (processedStatusResultKeys.has(key)) continue
    processedStatusResultKeys.add(key)

    const completedTs = parseServerTimestamp(item.completedAt)
    if (minimumTs > 0 && completedTs !== null && completedTs < minimumTs) continue

    if (!monitorOnly && (item.nodeRole ?? 'master') === 'master') continue

    trackResult(
      {
        id: key,
        jobId: item.jobId,
        path: item.path,
        module: item.module,
        status: item.status,
        durationMs: item.durationMs ?? 0,
        error: item.error ?? undefined,
        nodeId: item.nodeId ?? MASTER_NODE_ID,
        nodeRole: item.nodeRole ?? 'master',
        nodeLabel: item.nodeLabel ?? MASTER_NODE_LABEL,
        completedAt: item.completedAt,
      },
      true
    )
  }
}

/** Compute rate using a rolling time span from the oldest entry in the window. */
function computeRollingRate(entries: CompletionEntry[], now: number): number {
  if (entries.length === 0) return 0
  const spanMs = Math.max(1000, now - entries[0].ts)
  return entries.length / (spanMs / 1000)
}

/** Compute derived metrics using rolling done-only speed. */
function computeMetrics(
  remainingPasses: number,
  workers: number,
): { imagesPerSec: number; avgMsPerImage: number; estimatedMs: number } {
  const now = Date.now()
  pruneCompletionWindow(now)
  const imagesPerSec = computeRollingRate(completionWindow, now)

  const lastN = completionWindow.slice(-AVG_WINDOW)
  const avgMsPerImage =
    lastN.length > 0
      ? lastN.reduce((sum, e) => sum + e.durationMs, 0) / lastN.length
      : 0

  // Prefer chunk-based throughput for ETA (accounts for all workers'
  // combined contributions during the last chunk).
  const effectiveWorkers = Math.max(1, workers)
  let estimatedMs: number
  if (lastChunkPassesPerSec > 0 && remainingPasses > 0) {
    estimatedMs = (remainingPasses / lastChunkPassesPerSec) * 1000
  } else if (imagesPerSec > 0 && remainingPasses > 0) {
    estimatedMs = (remainingPasses / imagesPerSec) * 1000
  } else if (avgMsPerImage > 0 && remainingPasses > 0) {
    estimatedMs = (remainingPasses * avgMsPerImage) / effectiveWorkers
  } else {
    estimatedMs = 0
  }

  return { imagesPerSec, avgMsPerImage, estimatedMs }
}

/** Compute per-module rolling speed + avg latency from done events. */
function computeModuleMetrics(): Record<string, { imagesPerSec: number; avgMsPerImage: number }> {
  const now = Date.now()
  pruneCompletionWindow(now)
  const byModule: Record<string, CompletionEntry[]> = {}
  for (const entry of completionWindow) {
    if (!byModule[entry.module]) byModule[entry.module] = []
    byModule[entry.module].push(entry)
  }
  const metrics: Record<string, { imagesPerSec: number; avgMsPerImage: number }> = {}
  for (const [mod, entries] of Object.entries(byModule)) {
    const lastN = entries.slice(-AVG_WINDOW)
    const avgMsPerImage =
      lastN.length > 0
        ? lastN.reduce((sum, e) => sum + e.durationMs, 0) / lastN.length
        : 0
    metrics[mod] = {
      imagesPerSec: computeRollingRate(entries, now),
      avgMsPerImage,
    }
  }
  return metrics
}

function computeNodeMetrics(): Record<string, { imagesPerSec: number; avgMsPerImage: number }> {
  const now = Date.now()
  pruneCompletionWindow(now)
  const byNode: Record<string, CompletionEntry[]> = {}
  for (const entry of completionWindow) {
    if (!byNode[entry.nodeId]) byNode[entry.nodeId] = []
    byNode[entry.nodeId].push(entry)
  }

  const metrics: Record<string, { imagesPerSec: number; avgMsPerImage: number }> = {}
  for (const [nodeId, entries] of Object.entries(byNode)) {
    const lastN = entries.slice(-AVG_WINDOW)
    const avgMsPerImage =
      lastN.length > 0
        ? lastN.reduce((sum, e) => sum + e.durationMs, 0) / lastN.length
        : 0
    metrics[nodeId] = {
      imagesPerSec: computeRollingRate(entries, now),
      avgMsPerImage,
    }
  }
  return metrics
}

function buildBatchNodes(data: ServerStatusPayload): BatchNode[] {
  const nodeMetrics = computeNodeMetrics()
  const masterMeta = data.nodes?.master
  const masterCounts = nodeCounters.get(MASTER_NODE_ID)
  const masterRunningJobs = masterMeta?.runningJobs ?? 0

  const masterStatus =
    monitorOnly
      ? 'monitoring'
      : masterRunningJobs > 0
        ? 'running'
        : batchStatus === 'running'
          ? 'coordinating'
          : batchStatus

  const masterNode: BatchNode = {
    id: MASTER_NODE_ID,
    role: 'master',
    label: masterMeta?.displayName ?? MASTER_NODE_LABEL,
    status: masterStatus,
    desiredState: masterMeta?.desiredState ?? 'active',
    stateReason: masterMeta?.stateReason ?? null,
    platform: masterMeta?.platform,
    runningJobs: masterRunningJobs,
    completedJobs: (masterCounts?.done ?? 0) + (masterCounts?.failed ?? 0) + (masterCounts?.skipped ?? 0),
    doneJobs: masterCounts?.done ?? 0,
    failedJobs: masterCounts?.failed ?? 0,
    skippedJobs: masterCounts?.skipped ?? 0,
    imagesPerSec: nodeMetrics[MASTER_NODE_ID]?.imagesPerSec ?? 0,
    avgMsPerImage: nodeMetrics[MASTER_NODE_ID]?.avgMsPerImage ?? 0,
    lastResultAt: masterCounts?.lastResultAt ?? null,
    activeModules: masterMeta?.activeModules ?? [],
  }

  const workerNodes = (data.nodes?.workers ?? []).map((worker): BatchNode => {
    const workerCounts = nodeCounters.get(worker.id)
    return {
      id: worker.id,
      role: 'worker',
      label: worker.displayName ?? worker.id,
      status: (worker.runningJobs ?? 0) > 0 ? 'running' : (worker.status ?? 'idle'),
      desiredState: worker.desiredState ?? 'active',
      stateReason: worker.stateReason ?? null,
      platform: worker.platform,
      lastHeartbeat: worker.lastHeartbeat ?? null,
      lastResultAt: workerCounts?.lastResultAt ?? null,
      runningJobs: worker.runningJobs ?? 0,
      completedJobs: (workerCounts?.done ?? 0) + (workerCounts?.failed ?? 0) + (workerCounts?.skipped ?? 0),
      doneJobs: workerCounts?.done ?? 0,
      failedJobs: workerCounts?.failed ?? 0,
      skippedJobs: workerCounts?.skipped ?? 0,
      imagesPerSec: nodeMetrics[worker.id]?.imagesPerSec ?? 0,
      avgMsPerImage: nodeMetrics[worker.id]?.avgMsPerImage ?? 0,
      capabilities: worker.capabilities,
      activeModules: worker.activeModules ?? [],
    }
  })

  return [masterNode, ...workerNodes]
}

/** Poll status via RPC (no subprocess) and emit a batch:tick. */
async function doPoll(): Promise<void> {
  // Re-entrance guard: skip if previous poll is still in flight to avoid
  // piling up RPC calls when the server is slow to respond.
  if (pollInFlight) return
  pollInFlight = true
  try {
    const data = await rpc.call('status', {}) as ServerStatusPayload

    const workers = sessionConfig?.workers ?? 1
    const pending = data.totals.pending ?? 0
    const running = data.totals.running ?? 0
    const remainingPasses = pending + running
    const activePasses = running

    // Auto-heal stale paused UI state whenever the full queue still has work.
    // This keeps the run active based on global queue state, not chunk-local state.
    if (batchStatus === 'paused' && remainingPasses > 0) {
      batchStatus = 'running'
      monitorOnly = false
      isRunActive = true
    }

    const shouldSyncRecentResults =
      monitorOnly ||
      batchStatus === 'running' ||
      batchStatus === 'paused' ||
      batchStatus === 'done' ||
      batchStatus === 'error'

    if (shouldSyncRecentResults) {
      syncRecentResults(data.recent_results ?? [])
    }

    const metrics = computeMetrics(remainingPasses, workers)
    const moduleMetrics = computeModuleMetrics()
    const now = Date.now()
    const chunk = data.chunk ?? undefined

    let chunkRemainingPasses = 0
    let chunkElapsedMs = 0
    if (chunk && chunk.total > 0) {
      const nextChunkKey = `${chunk.index}:${chunk.total}:${chunk.size}`
      if (currentChunkKey !== nextChunkKey) {
        currentChunkKey = nextChunkKey
        currentChunkStartMs = now
      }
      chunkRemainingPasses = Object.values(chunk.modules).reduce((sum, count) => sum + count, 0)
      chunkElapsedMs = currentChunkStartMs > 0 ? now - currentChunkStartMs : 0
    } else {
      currentChunkKey = null
      currentChunkStartMs = 0
    }

    const chunkAvgCompletionMs =
      chunkCompletionWindowMs.length > 0
        ? chunkCompletionWindowMs.reduce((sum, ms) => sum + ms, 0) / chunkCompletionWindowMs.length
        : 0

    let chunkEstimatedMs = 0
    if (chunkRemainingPasses > 0) {
      if (lastChunkPassesPerSec > 0) {
        chunkEstimatedMs = (chunkRemainingPasses / lastChunkPassesPerSec) * 1000
      } else if (metrics.imagesPerSec > 0) {
        chunkEstimatedMs = (chunkRemainingPasses / metrics.imagesPerSec) * 1000
      }
    }

    if (monitorOnly) {
      if (remainingPasses > 0) {
        batchStatus = 'running'
        monitorZeroStreak = 0
      } else {
        // Require several consecutive zero readings before concluding work
        // is done.  Transient zeros (lease gaps, stale recovery windows)
        // won't trigger premature stop.
        monitorZeroStreak++
        const hasActiveWorkers = (data.nodes?.workers ?? []).some(
          (w: any) => w.status === 'online' && (w.runningJobs > 0 || w.pendingJobs > 0),
        )
        if (hasActiveWorkers) {
          // Workers still active — keep polling regardless of zero readings
          monitorZeroStreak = 0
        }
        if (
          monitorZeroStreak >= MONITOR_ZERO_THRESHOLD &&
          (batchStatus === 'running' || batchStatus === 'paused')
        ) {
          batchStatus = 'done'
          monitorZeroStreak = 0
          stopPolling()
          if (idleTimer) clearTimeout(idleTimer)
          idleTimer = setTimeout(() => {
            batchStatus = 'idle'
            monitorOnly = false
          }, 3000)
        }
      }
    }

    // Merge per-module speed into each module's stats
    const moduleAvgMs: Record<string, number> = data.module_avg_ms ?? {}
    const modulesWithSpeed: Partial<Record<string, BatchModuleStats>> = {}
    for (const [mod, modStats] of Object.entries(data.modules)) {
      modulesWithSpeed[mod] = {
        ...(modStats as unknown as BatchModuleStats),
        imagesPerSec: moduleMetrics[mod]?.imagesPerSec ?? 0,
        avgMsPerImage: moduleAvgMs[mod] ?? 0,
      }
    }

    const totals = {
      pending: data.totals.pending ?? 0,
      running: data.totals.running ?? 0,
      done: data.totals.done ?? 0,
      failed: data.totals.failed ?? 0,
      skipped: data.totals.skipped ?? 0,
    }
    const completedPasses = totals.done + totals.failed + totals.skipped
    const totalPasses = remainingPasses + completedPasses

    const stats: BatchStats = {
      status: batchStatus,
      monitorOnly,
      coordinator: getCoordinatorStatus(),
      totalImages: data.total_images,
      modules: modulesWithSpeed,
      totals,
      avgMsPerImage: metrics.avgMsPerImage,
      imagesPerSec: metrics.imagesPerSec,
      estimatedMs: metrics.estimatedMs,
      elapsedMs: sessionStartMs > 0 ? now - sessionStartMs : 0,
      chunkAvgCompletionMs,
      chunkElapsedMs,
      chunkEstimatedMs,
      queue: {
        totalPasses,
        activePasses,
        completedPasses,
        remainingPasses,
        remainingJobs: data.remaining_images ?? 0,
      },
      nodes: buildBatchNodes(data),
      chunk,
    }

    emitTick(stats)
  } catch (err) {
    // Log poll errors so they're visible in dev-tools / stderr instead of
    // being silently swallowed — critical for diagnosing dashboard freezes.
    console.error('[batch] doPoll error:', err)
  } finally {
    pollInFlight = false
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
        const payloadData =
          typeof p.data === 'object' && p.data !== null
            ? p.data as Record<string, unknown>
            : null
        const path =
          typeof p.path === 'string'
            ? p.path
            : typeof payloadData?.path === 'string'
              ? payloadData.path
              : ''
        const moduleName =
          typeof p.module === 'string'
            ? p.module
            : typeof payloadData?.module === 'string'
              ? payloadData.module
              : ''
        const status = parseResultStatus(p.status ?? payloadData?.status)
        const durationMs = parseDurationMs(p.ms ?? payloadData?.ms)
        const error =
          typeof p.error === 'string'
            ? p.error
            : typeof payloadData?.error === 'string'
              ? payloadData.error
              : undefined
        const keywords =
          parseResultKeywords(p.keywords) ??
          parseResultKeywords(p.keyword) ??
          parseResultKeywords(payloadData?.keywords) ??
          parseResultKeywords(payloadData?.keyword)
        const nodeId = typeof p.nodeId === 'string' ? p.nodeId : MASTER_NODE_ID
        const nodeRole = p.nodeRole === 'worker' ? 'worker' as const : 'master' as const
        const nodeLabel = typeof p.nodeLabel === 'string' ? p.nodeLabel : MASTER_NODE_LABEL
        const result: BatchResult = {
          id: nextLocalResultId(),
          path,
          module: moduleName,
          status,
          durationMs,
          error,
          keywords,
          nodeId,
          nodeRole,
          nodeLabel,
          completedAt: new Date().toISOString(),
        }
        trackResult(result)

        // Safety net: if polling stopped but we're still receiving results,
        // the run isn't really done — restart polling.
        if (monitorOnly && pollTimer === null) {
          console.warn('[batch] run/result received while polling stopped — restarting')
          monitorZeroStreak = 0
          batchStatus = 'running'
          startPolling()
        }
        break
      }

      case 'run/chunk_done': {
        const durationMs = (p.durationMs as number) ?? 0
        const passesCompleted = (p.passesCompleted as number) ?? 0
        if (durationMs > 0) {
          chunkCompletionWindowMs.push(durationMs)
          if (chunkCompletionWindowMs.length > CHUNK_AVG_WINDOW) {
            chunkCompletionWindowMs.shift()
          }
        }
        if (durationMs > 0 && passesCompleted > 0) {
          lastChunkPassesPerSec = passesCompleted / (durationMs / 1000)
        }
        break
      }

      case 'run/done': {
        const runId = currentRunId
        if (batchStatus !== 'running') break
        isRunActive = false
        const wasPaused = !!(p as any)?.paused
        void rpc.call('status', {}).then((data: any) => {
          if (currentRunId !== runId) return
          const masterRunning = data?.nodes?.master?.runningJobs ?? 0
          const pending = data?.totals?.pending ?? 0
          const running = data?.totals?.running ?? 0
          if (masterRunning > 0) {
            batchStatus = 'running'
            monitorOnly = false
            isRunActive = true
          } else if (pending + running > 0) {
            if (wasPaused) {
              batchStatus = 'paused'
              monitorOnly = false
            } else {
              // Master finished local work but distributed workers still
              // have active jobs — keep polling so the UI stays live.
              monitorOnly = true
              monitorZeroStreak = 0
            }
          } else {
            batchStatus = 'done'
            stopPolling()
            if (idleTimer) clearTimeout(idleTimer)
            idleTimer = setTimeout(() => {
              batchStatus = 'idle'
              monitorOnly = false
            }, 3000)
          }
          void doPoll()
        }).catch(() => {
          if (currentRunId !== runId) return
          batchStatus = 'done'
          stopPolling()
          void doPoll()
          if (idleTimer) clearTimeout(idleTimer)
          idleTimer = setTimeout(() => {
            batchStatus = 'idle'
            monitorOnly = false
          }, 3000)
        })
        break
      }

      case 'run/error': {
        const runId = currentRunId
        if (batchStatus !== 'running') break
        isRunActive = false
        void rpc.call('status', {}).then((data: any) => {
          if (currentRunId !== runId) return
          const masterRunning = data?.nodes?.master?.runningJobs ?? 0
          const pending = data?.totals?.pending ?? 0
          const running = data?.totals?.running ?? 0
          if (masterRunning > 0) {
            batchStatus = 'running'
            monitorOnly = false
            isRunActive = true
          } else if (pending + running > 0) {
            batchStatus = 'paused'
            monitorOnly = false
          } else {
            batchStatus = 'error'
            stopPolling()
            if (idleTimer) clearTimeout(idleTimer)
            idleTimer = setTimeout(() => {
              batchStatus = 'idle'
              monitorOnly = false
            }, 3000)
          }
          void doPoll()
        }).catch(() => {
          if (currentRunId !== runId) return
          batchStatus = 'error'
          stopPolling()
          void doPoll()
          if (idleTimer) clearTimeout(idleTimer)
          idleTimer = setTimeout(() => {
            batchStatus = 'idle'
            monitorOnly = false
          }, 3000)
        })
        break
      }

      case 'faces/clustering-done':
        mainWin?.webContents?.send('faces:clustering-done', p)
        break
    }
  })
}

/** Kill all background work and stop polling. Called on app quit. */
export async function killAllBatchProcesses(): Promise<void> {
  stopPolling()
  monitorOnly = false
  monitorZeroStreak = 0
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
      noHash: boolean,
      forceReprocess = false
    ): Promise<{ registered: number; enqueued: number; skipped: number }> => {
      batchStatus = 'ingesting'
      monitorOnly = false
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
          force: forceReprocess,
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
      recursive = true,
      noHash = false,
      profile = false,
      chunkSize = 500,
      forceReprocess = false
    ): Promise<void> => {
      sessionConfig = { folder, modules, workers, recursive, noHash, forceReprocess, profile }
      sessionStartMs = Date.now()
      resetSessionCounters()
      currentRunId++
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
      batchStatus = 'running'
      monitorOnly = false
      isRunActive = true

      try {
        await ensureServerRunning()
        await rpc.call('run', {
          workers,
          noXmp: true,
          verbose: true,
          staleTimeout: 0,
          force: forceReprocess,
          profile,
          chunkSize,
        })
      } catch (err) {
        isRunActive = false
        batchStatus = 'error'
        if (idleTimer) clearTimeout(idleTimer)
        idleTimer = setTimeout(() => {
          batchStatus = 'idle'
          monitorOnly = false
        }, 5000)
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
    batchStatus = 'paused'
    monitorOnly = false
    // Emit one tick so the UI updates immediately
    void doPoll()
  })

  // ── batch:resume ──────────────────────────────────────────────────────────
  ipcMain.handle('batch:resume', async (): Promise<void> => {
    // Use session config if available, otherwise fall back to sensible defaults
    // (e.g. after crash recovery when sessionConfig was never populated).
    const w     = sessionConfig?.workers       ?? 1

    currentRunId++
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
    batchStatus = 'running'
    monitorOnly = false
    isRunActive = true

    try {
      await ensureServerRunning()
      // Always use staleTimeout=0 on resume: pausing leaves in-progress jobs
      // stuck as 'running' in the DB, and they need immediate recovery.
      const runParams: Record<string, unknown> = {
        workers: w,
        noXmp: true,
        verbose: true,
        staleTimeout: 0,
        force: sessionConfig?.forceReprocess ?? false,
        profile: sessionConfig?.profile ?? false,
      }
      await rpc.call('run', runParams)
    } catch (err) {
      isRunActive = false
      batchStatus = 'error'
      if (idleTimer) clearTimeout(idleTimer)
      idleTimer = setTimeout(() => {
        batchStatus = 'idle'
        monitorOnly = false
      }, 5000)
      return
    }

    startPolling()
  })

  // ── batch:pause-target ────────────────────────────────────────────────────
  ipcMain.handle(
    'batch:pause-target',
    async (_evt, target: BatchControlTarget, mode?: BatchPauseMode): Promise<void> => {
      const pauseMode = normalizePauseMode(mode)
      if (!target || typeof target !== 'object') {
        throw new Error('target is required')
      }

      if (target.scope === 'coordinator') {
        const coordinator = getCoordinatorStatus()
        if (coordinator.state === 'running' || coordinator.state === 'starting') {
          await stopCoordinator()
        }
        await doPoll()
        return
      }

      await ensureServerRunning()
      const workerId = resolveTargetWorkerId(target)
      await rpc.call('workers/pause', {
        workerId,
        mode: pauseMode,
        reason: 'paused from Running tab',
      })

      if (workerId === MASTER_NODE_ID && batchStatus === 'running') {
        batchStatus = 'paused'
        monitorOnly = false
        isRunActive = false
      }
      await doPoll()
    }
  )

  // ── batch:resume-target ───────────────────────────────────────────────────
  ipcMain.handle(
    'batch:resume-target',
    async (_evt, target: BatchControlTarget): Promise<void> => {
      if (!target || typeof target !== 'object') {
        throw new Error('target is required')
      }

      if (target.scope === 'coordinator') {
        const settings = await getAppSettings()
        if (!settings.distributed.enabled) {
          throw new Error('Enable distributed coordinator in Settings before resuming it.')
        }
        await startCoordinator(settings.distributed)
        await doPoll()
        return
      }

      await ensureServerRunning()
      const workerId = resolveTargetWorkerId(target)
      await rpc.call('workers/resume', { workerId })
      await doPoll()
    }
  )

  // ── batch:stop ────────────────────────────────────────────────────────────
  ipcMain.handle('batch:stop', async (_evt, folder: string): Promise<void> => {
    // Cancel the active run
    try {
      await rpc.call('cancel_run', {})
    } catch { /* ignore */ }
    isRunActive = false
    stopPolling()
    batchStatus = 'stopped'
    monitorOnly = false

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
    async (_evt, workers = 1): Promise<void> => {
      if (batchStatus === 'running') return

      const w  = sessionConfig?.workers ?? workers

      // Populate sessionConfig if it wasn't set (crash recovery / fresh start)
      // so that subsequent batch:resume calls have it available.
      if (!sessionConfig) {
        sessionConfig = {
          folder: '',
          modules: [],
          workers: w,
          recursive: true,
          noHash: false,
          forceReprocess: false,
          profile: false,
        }
      }

      sessionStartMs = Date.now()
      resetSessionCounters()
      currentRunId++
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
      batchStatus = 'running'
      monitorOnly = false
      isRunActive = true

      try {
        await ensureServerRunning()
        // Use staleTimeout=0 to recover ALL stuck 'running' jobs from a
        // previous crash, regardless of how recently they were claimed.
        await rpc.call('run', {
          workers: w,
          noXmp: true,
          verbose: true,
          staleTimeout: 0,
          profile: sessionConfig?.profile ?? false,
        })
      } catch (err) {
        isRunActive = false
        batchStatus = 'error'
        if (idleTimer) clearTimeout(idleTimer)
        idleTimer = setTimeout(() => {
          batchStatus = 'idle'
          monitorOnly = false
        }, 5000)
        return
      }

      startPolling()
    }
  )

  // ── batch:monitor-existing ────────────────────────────────────────────────
  ipcMain.handle('batch:monitor-existing', async (): Promise<boolean> => {
    try {
      if (isRunActive) return false
      await ensureServerRunning()
      const data = await rpc.call('status', {}) as {
        totals: Record<string, number>
      }
      const running = data.totals.running ?? 0
      if (running <= 0) return false

      // Only enter monitor-only mode when a distributed worker is online;
      // otherwise the running jobs are stale from a crashed local session
      // and should be resumed locally via resumePending().
      let hasOnlineWorker = false
      try {
        const wl = await rpc.call('workers/list', {}) as { workers?: Array<{ status?: string }> }
        hasOnlineWorker = (wl.workers ?? []).some((w) => w.status === 'online')
      } catch { /* workers table may not exist yet */ }
      if (!hasOnlineWorker) return false

      sessionStartMs = Date.now()
      resetSessionCounters()
      currentRunId++
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
      batchStatus = 'running'
      monitorOnly = true
      monitorZeroStreak = 0
      isRunActive = false
      startPolling()
      void doPoll()
      return true
    } catch {
      return false
    }
  })

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
      const w  = sessionConfig?.workers ?? 1

      sessionStartMs = Date.now()
      resetSessionCounters()
      currentRunId++
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
      batchStatus = 'running'
      monitorOnly = false
      isRunActive = true

      try {
        await ensureServerRunning()
        await rpc.call('run', {
          workers: w,
          noXmp: true,
          verbose: true,
          staleTimeout: 0,
          profile: sessionConfig?.profile ?? false,
        })
      } catch (err) {
        isRunActive = false
        batchStatus = 'error'
        if (idleTimer) clearTimeout(idleTimer)
        idleTimer = setTimeout(() => {
          batchStatus = 'idle'
          monitorOnly = false
        }, 5000)
        return
      }

      startPolling()
    }
  )

  // ── batch:rebuild-module ────────────────────────────────────────────────
  // Re-enqueue a module for ALL images (force=true). If the worker is
  // already running, the new jobs are picked up automatically.
  ipcMain.handle(
    'batch:rebuild-module',
    async (_evt, module: string): Promise<void> => {
      await ensureServerRunning()
      await rpc.call('rebuild', { module, force: true })

      // If a worker is already running it will pick up the new jobs —
      // no need to spawn another one.
      if (batchStatus === 'running') return

      const w  = sessionConfig?.workers ?? 1

      sessionStartMs = Date.now()
      resetSessionCounters()
      currentRunId++
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
      batchStatus = 'running'
      monitorOnly = false
      isRunActive = true

      try {
        await rpc.call('run', {
          workers: w,
          noXmp: true,
          verbose: true,
          staleTimeout: 0,
          profile: sessionConfig?.profile ?? false,
        })
      } catch (err) {
        isRunActive = false
        batchStatus = 'error'
        if (idleTimer) clearTimeout(idleTimer)
        idleTimer = setTimeout(() => {
          batchStatus = 'idle'
          monitorOnly = false
        }, 5000)
        return
      }

      startPolling()
    }
  )

  // ── batch:queue-clear-all ─────────────────────────────────────────────────
  // Only clears pending/running/failed jobs. Done and skipped rows are
  // preserved so that re-ingest correctly skips already-processed images
  // (especially modules skipped at runtime like perception with
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
    monitorOnly = false

    // Emit one final poll tick so the renderer resets its counters
    void doPoll()

    return result
  })

  // ── batch:queue-clear-done ────────────────────────────────────────────────
  // Removes completed (done + skipped) jobs from the queue.
  // Safe to call in any state — only touches terminal-status rows, never
  // pending/running jobs, so it cannot interfere with an active batch.
  // Note: after clearing, re-ingest will re-enqueue these images.
  ipcMain.handle('batch:queue-clear-done', async (): Promise<{ deleted: number }> => {
    const result = await rpc.call('queue_clear', {
      status: 'done,skipped',
    }) as { deleted: number }

    // Refresh stats so the renderer sees updated counts
    await doPoll()

    return result
  })
}
