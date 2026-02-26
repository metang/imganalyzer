import { useState, useCallback, useEffect, useRef } from 'react'
import { useBatchProcess } from '../hooks/useBatchProcess'
import { PassSelector, defaultPassSelectorValue, resolveModuleKeys } from './PassSelector'
import type { PassSelectorValue } from './PassSelector'
import { ProgressDashboard } from './ProgressDashboard'
import { LiveResultsFeed } from './LiveResultsFeed'
import { ConfirmStopDialog } from './ConfirmStopDialog'
import type { BatchIngestProgress } from '../global'

// ── Phase routing ─────────────────────────────────────────────────────────────

type Phase = 'config' | 'ingesting' | 'active'

function derivePhase(status: string): Phase {
  switch (status) {
    case 'ingesting': return 'ingesting'
    case 'running':
    case 'paused':
    case 'done':
    case 'error':
      return 'active'
    default:
      return 'config'
  }
}

// ── Config panel ──────────────────────────────────────────────────────────────

interface ConfigPanelProps {
  folder: string
  onFolderChange(path: string): void
  passSel: PassSelectorValue
  onPassSelChange(v: PassSelectorValue): void
  onStart(): void
  error: string | null
  /** Set when a previous ingest returned 0 enqueued jobs (all already done). */
  nothingToRun: boolean
}

function ConfigPanel({
  folder,
  onFolderChange,
  passSel,
  onPassSelChange,
  onStart,
  error,
  nothingToRun,
}: ConfigPanelProps) {
  const canStart = folder.trim() !== '' && passSel.selectedKeys.size > 0

  return (
    <div className="flex flex-col gap-5 max-w-2xl w-full mx-auto py-6 px-4">
      <h2 className="text-base font-semibold text-neutral-200">Batch Processing</h2>

      {/* Folder input */}
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">
          Source folder
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={folder}
            onChange={(e) => onFolderChange(e.target.value)}
            placeholder="C:\Photos\2024"
            className="
              flex-1 px-3 py-2 rounded-lg bg-neutral-800 border border-neutral-700
              text-sm text-neutral-200 placeholder-neutral-600
              focus:outline-none focus:border-blue-500
            "
          />
          <button
            onClick={async () => {
              const picked = await window.api.openFolder()
              if (picked) onFolderChange(picked)
            }}
            className="
              px-3 py-2 rounded-lg bg-neutral-700 text-sm text-neutral-200
              hover:bg-neutral-600 transition-colors shrink-0
            "
          >
            Browse
          </button>
        </div>
      </div>

      {/* Pass selector */}
      <PassSelector value={passSel} onChange={onPassSelChange} />

      {nothingToRun && (
        <p className="text-sm text-yellow-400 bg-yellow-900/20 border border-yellow-800 rounded-lg px-3 py-2">
          All images in this folder have already been processed for the selected passes.
          Nothing was enqueued.
        </p>
      )}

      {error && (
        <p className="text-sm text-red-400 bg-red-900/20 border border-red-800 rounded-lg px-3 py-2">
          {error}
        </p>
      )}

      <button
        onClick={onStart}
        disabled={!canStart}
        className="
          self-start px-5 py-2 rounded-lg text-sm font-medium transition-colors
          bg-blue-700 text-white
          disabled:opacity-30 disabled:cursor-not-allowed
          enabled:hover:bg-blue-600
        "
      >
        Start batch
      </button>
    </div>
  )
}

// ── Ingest panel ──────────────────────────────────────────────────────────────

interface IngestPanelProps {
  progress: BatchIngestProgress | null
}

function IngestPanel({ progress }: IngestPanelProps) {
  const pct = progress && progress.total > 0
    ? Math.round((progress.scanned / progress.total) * 100)
    : 0

  // Strip directory — show only the filename
  const currentName = progress?.current
    ? progress.current.replace(/^.*[\\/]/, '')
    : ''

  return (
    <div className="flex flex-col gap-4 max-w-2xl w-full mx-auto py-6 px-4">
      <h2 className="text-base font-semibold text-neutral-200">Scanning folder…</h2>

      {/* Spinner + label */}
      <div className="flex items-center gap-2 text-sm text-neutral-400">
        <div className="w-3.5 h-3.5 border-2 border-neutral-600 border-t-blue-400 rounded-full animate-spin shrink-0" />
        {progress
          ? `Scanned ${progress.scanned.toLocaleString()} of ${progress.total.toLocaleString()} images`
          : 'Registering images and queuing jobs…'
        }
      </div>

      {/* Progress bar */}
      <div className="w-full bg-neutral-800 rounded-full h-2 overflow-hidden">
        <div
          className="h-2 bg-blue-500 rounded-full transition-all duration-200"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Stats row */}
      {progress && (
        <div className="grid grid-cols-4 gap-2 text-center">
          {(
            [
              ['Scanned',    progress.scanned],
              ['Registered', progress.registered],
              ['Enqueued',   progress.enqueued],
              ['Skipped',    progress.skipped],
            ] as [string, number][]
          ).map(([label, value]) => (
            <div key={label} className="bg-neutral-800 rounded-lg py-2 px-1">
              <div className="text-base font-semibold text-neutral-100 tabular-nums">
                {value.toLocaleString()}
              </div>
              <div className="text-xs text-neutral-500 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Current file being processed */}
      {currentName && (
        <p className="text-xs text-neutral-500 truncate" title={progress?.current}>
          {currentName}
        </p>
      )}
    </div>
  )
}

// ── Active panel (running / paused / done / error) ────────────────────────────

interface ActivePanelProps {
  stats: ReturnType<typeof useBatchProcess>['stats']
  results: ReturnType<typeof useBatchProcess>['results']
  ingestSummary: ReturnType<typeof useBatchProcess>['ingestSummary']
  /** Non-null when we auto-resumed a previous session on mount. */
  resumeBanner: string | null
  onDismissBanner(): void
  onPause(): void
  onResume(): void
  onRequestStop(): void
  onRetryFailed(modules: string[]): void
}

function ActivePanel({
  stats,
  results,
  ingestSummary,
  resumeBanner,
  onDismissBanner,
  onPause,
  onResume,
  onRequestStop,
  onRetryFailed,
}: ActivePanelProps) {
  return (
    <div className="flex flex-col gap-4 h-full overflow-hidden px-4 py-4">
      {/* Auto-resume banner */}
      {resumeBanner && (
        <div className="flex items-center justify-between gap-2 text-sm text-blue-300 bg-blue-900/20 border border-blue-800 rounded-lg px-3 py-2">
          <span>{resumeBanner}</span>
          <button
            onClick={onDismissBanner}
            className="text-blue-400 hover:text-blue-200 transition-colors text-xs shrink-0"
            aria-label="Dismiss"
          >
            Dismiss
          </button>
        </div>
      )}

      {ingestSummary && (
        <p className="text-xs text-neutral-500">
          Ingest: {ingestSummary.registered.toLocaleString()} registered,{' '}
          {ingestSummary.enqueued.toLocaleString()} enqueued,{' '}
          {ingestSummary.skipped.toLocaleString()} skipped
        </p>
      )}

      <ProgressDashboard
        stats={stats}
        onPause={onPause}
        onResume={onResume}
        onStop={onRequestStop}
        onRetryFailed={onRetryFailed}
      />

      <div className="flex-1 flex flex-col min-h-0 border-t border-neutral-800 pt-3">
        <p className="text-xs text-neutral-500 mb-1.5">
          Live results (last 200)
        </p>
        <LiveResultsFeed results={results} />
      </div>
    </div>
  )
}

// ── Root BatchView ────────────────────────────────────────────────────────────

interface BatchViewProps {
  initialFolder?: string
}

export function BatchView({ initialFolder = '' }: BatchViewProps) {
  const batch = useBatchProcess()
  const { stats, results, ingestProgress, ingestSummary, error } = batch

  const [folder, setFolder] = useState(initialFolder)
  const [passSel, setPassSel] = useState<PassSelectorValue>(defaultPassSelectorValue)
  const [showStopDialog, setShowStopDialog] = useState(false)
  const [resumeBanner, setResumeBanner] = useState<string | null>(null)

  // Auto-resume any pending/running jobs from a previous session (runs once on mount)
  const didCheckRef = useRef(false)
  useEffect(() => {
    if (didCheckRef.current) return
    didCheckRef.current = true
    ;(async () => {
      const resumed = await batch.resumePending(passSel.workers, passSel.cloudProvider)
      if (resumed) {
        setResumeBanner('Resuming unfinished jobs from a previous session…')
      }
    })()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const phase = derivePhase(stats.status)

  const handleStart = useCallback(async () => {
    // Resolve UI keys → deduplicated CLI module keys
    const modules = resolveModuleKeys(passSel.selectedKeys)
    await batch.startBatch({
      folder,
      modules,
      workers: passSel.workers,
      cloudProvider: passSel.cloudProvider,
      recursive: passSel.recursive,
      noHash: passSel.noHash,
    })
  }, [batch, folder, passSel])

  const handleRequestStop = useCallback(() => {
    setShowStopDialog(true)
  }, [])

  const handleConfirmStop = useCallback(async () => {
    setShowStopDialog(false)
    await batch.stop(folder)
  }, [batch, folder])

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {phase === 'config' && (
        <div className="flex-1 overflow-y-auto">
          <ConfigPanel
            folder={folder}
            onFolderChange={setFolder}
            passSel={passSel}
            onPassSelChange={setPassSel}
            onStart={handleStart}
            error={error}
            nothingToRun={ingestSummary !== null && ingestSummary.enqueued === 0}
          />
        </div>
      )}

      {phase === 'ingesting' && (
        <div className="flex-1 overflow-y-auto">
          <IngestPanel progress={ingestProgress} />
        </div>
      )}

      {phase === 'active' && (
        <div className="flex-1 flex flex-col min-h-0">
          <ActivePanel
            stats={stats}
            results={results}
            ingestSummary={ingestSummary}
            resumeBanner={resumeBanner}
            onDismissBanner={() => setResumeBanner(null)}
            onPause={batch.pause}
            onResume={batch.resume}
            onRequestStop={handleRequestStop}
            onRetryFailed={batch.retryFailed}
          />
        </div>
      )}

      {showStopDialog && (
        <ConfirmStopDialog
          onConfirm={handleConfirmStop}
          onCancel={() => setShowStopDialog(false)}
        />
      )}
    </div>
  )
}
