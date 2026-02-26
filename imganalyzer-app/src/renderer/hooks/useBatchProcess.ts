import { useState, useCallback, useEffect, useRef } from 'react'
import type { BatchStats, BatchResult, BatchStatus, BatchIngestProgress } from '../global'

// ── Pass config ───────────────────────────────────────────────────────────────

/** All CLI module keys selectable by the user. */
export const ALL_MODULE_KEYS = [
  'metadata',
  'technical',
  'local_ai',
  'cloud_ai',
  'aesthetic',
  'embedding',
] as const

export type ModuleKey = (typeof ALL_MODULE_KEYS)[number]

/** Default empty stats object. */
function emptyStats(): BatchStats {
  return {
    status: 'idle',
    totalImages: 0,
    modules: {},
    totals: { pending: 0, running: 0, done: 0, failed: 0, skipped: 0 },
    avgMsPerImage: 0,
    imagesPerSec: 0,
    estimatedMs: 0,
    elapsedMs: 0,
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export interface BatchConfig {
  folder: string
  modules: ModuleKey[]
  workers: number
  cloudProvider: string
  recursive: boolean
  noHash: boolean
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
  resumePending(workers?: number, cloudProvider?: string): Promise<boolean>
  /** Re-enqueue all failed jobs for the given modules and re-run. */
  retryFailed(modules: string[]): Promise<void>
  pause(): Promise<void>
  resume(): Promise<void>
  stop(folder: string): Promise<void>
}

const MAX_RESULTS = 200

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
    unsubTickRef.current = window.api.onBatchTick((s) => setStats(s))
    unsubResultRef.current = window.api.onBatchResult((r) => {
      setResults((prev) => {
        const next = [r, ...prev]
        return next.length > MAX_RESULTS ? next.slice(0, MAX_RESULTS) : next
      })
    })
    unsubIngestRef.current = window.api.onBatchIngestLine((line) => {
      setIngestLines((prev) => [...prev, line])
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
    setStats((prev) => ({ ...prev, status: 'ingesting' as BatchStatus }))

    try {
      const summary = await window.api.batchIngest(
        config.folder,
        config.modules,
        config.recursive,
        config.noHash
      )
      setIngestSummary(summary)

      if (summary.enqueued === 0) {
        // Nothing to run — go back to idle
        setStats((prev) => ({ ...prev, status: 'idle' as BatchStatus }))
        return
      }

      await window.api.batchStart(
        config.folder,
        config.modules,
        config.workers,
        config.cloudProvider,
        config.recursive,
        config.noHash
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStats((prev) => ({ ...prev, status: 'error' as BatchStatus }))
    }
  }, [])

  const pause = useCallback(async () => {
    try {
      await window.api.batchPause()
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
  const resumePending = useCallback(async (workers?: number, cloudProvider?: string): Promise<boolean> => {
    try {
      const { pending, running } = await window.api.batchCheckPending()
      if (pending + running === 0) return false

      // Optimistically jump to 'running' so the UI phase transitions immediately.
      // The 1-second poll tick will overwrite with real stats.
      setStats((prev) => ({ ...prev, status: 'running' as BatchStatus }))

      await window.api.batchResumePending(workers, cloudProvider)
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return false
    }
  }, [])

  const retryFailed = useCallback(async (modules: string[]) => {
    try {
      // Optimistically jump to 'running' so the UI stays in the active phase
      setStats((prev) => ({ ...prev, status: 'running' as BatchStatus }))
      await window.api.batchRetryFailed(modules)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStats((prev) => ({ ...prev, status: 'error' as BatchStatus }))
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
    resumePending,
    retryFailed,
    pause,
    resume,
    stop,
  }
}
