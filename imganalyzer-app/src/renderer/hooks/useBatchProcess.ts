import { useState, useCallback, useEffect, useRef } from 'react'
import type {
  BatchControlTarget,
  BatchPauseMode,
  BatchStats,
  BatchResult,
  BatchStatus,
  BatchIngestProgress,
} from '../global'

// ── Pass config ───────────────────────────────────────────────────────────────

/** All CLI module keys selectable by the user. */
export const ALL_MODULE_KEYS = [
  'metadata',
  'technical',
  'caption',
  'objects',
  'faces',
  'perception',
  'embedding',
] as const

export type ModuleKey = (typeof ALL_MODULE_KEYS)[number]

/** Default empty stats object. */
function emptyStats(): BatchStats {
  return {
    status: 'idle',
    monitorOnly: false,
    coordinator: { state: 'stopped', pid: null, url: null, lastError: null },
    totalImages: 0,
    modules: {},
    totals: { pending: 0, running: 0, done: 0, failed: 0, skipped: 0 },
    avgMsPerImage: 0,
    imagesPerSec: 0,
    estimatedMs: 0,
    elapsedMs: 0,
    chunkAvgCompletionMs: 0,
    chunkElapsedMs: 0,
    chunkEstimatedMs: 0,
    queue: {
      totalPasses: 0,
      activePasses: 0,
      completedPasses: 0,
      remainingPasses: 0,
      remainingJobs: 0,
    },
    nodes: [],
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export interface BatchConfig {
  folder: string
  modules: ModuleKey[]
  workers: number
  recursive: boolean
  noHash: boolean
  forceReprocess: boolean
  profile: boolean
  chunkSize: number
}

export interface UseBatchProcessReturn {
  /** Current aggregate stats from the last poll tick. */
  stats: BatchStats
  /** Live per-job results (capped at 200). */
  results: BatchResult[]
  /** Raw lines emitted by the ingest subprocess. */
  ingestLines: string[]
  /** Structured ingest progress (scanned/total/registered/enqueued/skipped/current). */
  ingestProgress: BatchIngestProgress | null
  /** Summary returned after ingest completes. */
  ingestSummary: { registered: number; enqueued: number; skipped: number } | null
  /** Human-readable error if something went wrong. */
  error: string | null

  startBatch(config: BatchConfig): Promise<void>
  /** Resume any pending/running jobs left over from a previous session. */
  resumePending(workers?: number): Promise<boolean>
  /** Monitor existing jobs already being processed elsewhere (for example by a distributed worker). */
  monitorExisting(): Promise<boolean>
  /** Re-enqueue all failed jobs for the given modules and re-run. */
  retryFailed(modules: string[]): Promise<void>
  /** Re-enqueue ALL images for a single module (force rebuild) and run. */
  rebuildModule(module: string): Promise<void>
  /** Wipe the entire job queue and reset to idle. Returns number of deleted jobs. */
  clearQueue(): Promise<number>
  /** Remove completed (done + skipped) jobs from the queue. Returns number of deleted jobs. */
  clearCompleted(): Promise<number>
  pause(): Promise<void>
  pauseTarget(target: BatchControlTarget, mode?: BatchPauseMode): Promise<void>
  resume(): Promise<void>
  resumeTarget(target: BatchControlTarget): Promise<void>
  removeWorker(workerId: string): Promise<void>
  stop(folder: string): Promise<void>
}

const MAX_RESULTS = 200
const MAX_INGEST_LINES = 500

export function useBatchProcess(): UseBatchProcessReturn {
  const [stats, setStats] = useState<BatchStats>(emptyStats())
  const [results, setResults] = useState<BatchResult[]>([])
  const [ingestLines, setIngestLines] = useState<string[]>([])
  const [ingestProgress, setIngestProgress] = useState<BatchIngestProgress | null>(null)
  const [ingestSummary, setIngestSummary] = useState<{
    registered: number
    enqueued: number
    skipped: number
  } | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Keep unsubscribe refs so we can clean up on unmount
  const unsubTickRef = useRef<(() => void) | null>(null)
  const unsubResultRef = useRef<(() => void) | null>(null)
  const unsubIngestRef = useRef<(() => void) | null>(null)
  const unsubIngestProgressRef = useRef<(() => void) | null>(null)

  // Subscribe to events once on mount
  useEffect(() => {
    unsubTickRef.current = window.api.onBatchTick((s) => {
      setStats((prev) => ({
        ...s,
        coordinator: s.coordinator ?? prev.coordinator ?? emptyStats().coordinator,
        nodes: Array.isArray(s.nodes) ? s.nodes : [],
      }))
    })
    unsubResultRef.current = window.api.onBatchResult((r) => {
      setResults((prev) => {
        // Below cap: simple prepend (cheap — single spread).
        if (prev.length < MAX_RESULTS) {
          return [r, ...prev]
        }
        // At cap: single allocation of exactly MAX_RESULTS. Avoids the
        // [r, ...prev] + slice(0, 200) double-allocation which at 10 results/s
        // produced ~4000 throwaway array slots per second.
        const next = new Array<BatchResult>(MAX_RESULTS)
        next[0] = r
        for (let i = 0; i < MAX_RESULTS - 1; i++) next[i + 1] = prev[i]
        return next
      })
    })
    unsubIngestRef.current = window.api.onBatchIngestLine((line) => {
      setIngestLines((prev) => {
        const next = [...prev, line]
        return next.length > MAX_INGEST_LINES ? next.slice(-MAX_INGEST_LINES) : next
      })
    })
    unsubIngestProgressRef.current = window.api.onBatchIngestProgress((p) => {
      setIngestProgress(p)
    })

    return () => {
      unsubTickRef.current?.()
      unsubResultRef.current?.()
      unsubIngestRef.current?.()
      unsubIngestProgressRef.current?.()
    }
  }, [])

  const startBatch = useCallback(async (config: BatchConfig) => {
    setError(null)
    setResults([])
    setIngestLines([])
    setIngestProgress(null)
    setIngestSummary(null)

    // Update status optimistically so UI shows "ingesting" immediately
    setStats((prev) => ({ ...prev, status: 'ingesting' as BatchStatus, monitorOnly: false }))

    try {
      const summary = await window.api.batchIngest(
        config.folder,
        config.modules,
        config.recursive,
        config.noHash,
        config.forceReprocess
      )
      setIngestSummary(summary)

      if (summary.enqueued === 0) {
        // Nothing to run — go back to idle
        setStats((prev) => ({ ...prev, status: 'idle' as BatchStatus, monitorOnly: false }))
        return
      }

      await window.api.batchStart(
        config.folder,
        config.modules,
        config.workers,
        config.recursive,
        config.noHash,
        config.profile,
        config.chunkSize,
        config.forceReprocess
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStats((prev) => ({ ...prev, status: 'error' as BatchStatus, monitorOnly: false }))
    }
  }, [])

  const pause = useCallback(async () => {
    try {
      await window.api.batchPause()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const pauseTarget = useCallback(async (target: BatchControlTarget, mode: BatchPauseMode = 'pause-drain') => {
    try {
      await window.api.batchPauseTarget(target, mode)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const resume = useCallback(async () => {
    try {
      await window.api.batchResume()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const resumeTarget = useCallback(async (target: BatchControlTarget) => {
    try {
      await window.api.batchResumeTarget(target)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const removeWorker = useCallback(async (workerId: string) => {
    try {
      await window.api.batchRemoveWorker(workerId)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const stop = useCallback(async (folder: string) => {
    try {
      await window.api.batchStop(folder)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  /**
   * Check for leftover pending/running jobs from a previous session and, if
   * any exist, re-spawn the worker process.  Returns true if jobs were found
   * and the worker was resumed, false otherwise.
   */
  const resumePending = useCallback(async (workers?: number): Promise<boolean> => {
    try {
      const { pending, running } = await window.api.batchCheckPending()
      if (pending + running === 0) return false

      // Optimistically jump to 'running' so the UI phase transitions immediately.
      // The 1-second poll tick will overwrite with real stats.
      setStats((prev) => ({ ...prev, status: 'running' as BatchStatus, monitorOnly: false }))

      await window.api.batchResumePending(workers)
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return false
    }
  }, [])

  const monitorExisting = useCallback(async (): Promise<boolean> => {
    try {
      return await window.api.batchMonitorExisting()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return false
    }
  }, [])

  const retryFailed = useCallback(async (modules: string[]) => {
    try {
      // Optimistically jump to 'running' so the UI stays in the active phase
      setStats((prev) => ({ ...prev, status: 'running' as BatchStatus, monitorOnly: false }))
      await window.api.batchRetryFailed(modules)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStats((prev) => ({ ...prev, status: 'error' as BatchStatus, monitorOnly: false }))
    }
  }, [])

  const rebuildModule = useCallback(async (module: string) => {
    try {
      setStats((prev) => ({ ...prev, status: 'running' as BatchStatus, monitorOnly: false }))
      await window.api.batchRebuildModule(module)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStats((prev) => ({ ...prev, status: 'error' as BatchStatus, monitorOnly: false }))
    }
  }, [])

  const clearQueue = useCallback(async (): Promise<number> => {
    try {
      const { deleted } = await window.api.batchQueueClearAll()
      // Reset all local state
      setStats(emptyStats())
      setResults([])
      setIngestLines([])
      setIngestProgress(null)
      setIngestSummary(null)
      setError(null)
      return deleted
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return 0
    }
  }, [])

  const clearCompleted = useCallback(async (): Promise<number> => {
    try {
      const { deleted } = await window.api.batchQueueClearDone()
      return deleted
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return 0
    }
  }, [])

  return {
    stats,
    results,
    ingestLines,
    ingestProgress,
    ingestSummary,
    error,
    startBatch,
    monitorExisting,
    resumePending,
    retryFailed,
    rebuildModule,
    clearQueue,
    clearCompleted,
    pause,
    pauseTarget,
    resume,
    resumeTarget,
    removeWorker,
    stop,
  }
}
