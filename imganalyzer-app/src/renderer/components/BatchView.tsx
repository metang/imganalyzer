import { useState, useCallback, useEffect, useRef } from 'react'
import { PassSelector, defaultPassSelectorValue, resolveModuleKeys } from './PassSelector'
import type { PassSelectorValue } from './PassSelector'
import { ProgressDashboard } from './ProgressDashboard'
import { LiveResultsFeed } from './LiveResultsFeed'
import { ConfirmStopDialog } from './ConfirmStopDialog'
import { SectionHeading, SurfaceCard, UiButton } from './ui'
import type { UseBatchProcessReturn } from '../hooks/useBatchProcess'
import type { BatchIngestProgress, BatchResult } from '../global'

// ── Shared props type ─────────────────────────────────────────────────────────

/** Props accepted by both BatchConfigView and BatchRunView. */
export interface BatchViewProps {
  batch: UseBatchProcessReturn
  /** Pre-fill the folder path from the gallery tab. */
  initialFolder?: string
  /** Called when a batch starts so the parent can react if needed. */
  onBatchStarted?(): void
}

// ── Config panel ──────────────────────────────────────────────────────────────

interface ConfigPanelProps {
  folder: string
  onFolderChange(path: string): void
  passSel: PassSelectorValue
  onPassSelChange(v: PassSelectorValue): void
  profile: boolean
  onProfileChange(v: boolean): void
  onStart(): void
  error: string | null
  nothingToRun: boolean
}

function ConfigPanel({
  folder,
  onFolderChange,
  passSel,
  onPassSelChange,
  profile,
  onProfileChange,
  onStart,
  error,
  nothingToRun,
}: ConfigPanelProps) {
  const canStart = folder.trim() !== '' && passSel.selectedKeys.size > 0

  return (
    <div className="flex flex-col gap-5 max-w-3xl w-full mx-auto py-6 px-4">
      <SurfaceCard tone="accent">
        <SectionHeading
          eyebrow="Batch processing"
          title="Queue up a focused analysis run"
          description="Choose a folder, select the passes you care about, and let the workstation handle the heavy lifting."
        />

        {/* Folder input */}
        <div className="mt-4 flex flex-col gap-1.5">
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
                flex-1 rounded-xl border border-neutral-700 bg-black/20 px-3 py-2
                text-sm text-neutral-200 placeholder-neutral-600
                focus:border-cyan-500 focus:outline-none
              "
            />
            <UiButton
              onClick={async () => {
                const picked = await window.api.openFolder()
                if (picked) onFolderChange(picked)
              }}
              variant="secondary"
            >
              Browse
            </UiButton>
          </div>
        </div>

        {/* Pass selector */}
        <div className="mt-4">
          <PassSelector value={passSel} onChange={onPassSelChange} />
        </div>

        {nothingToRun && (
          <p className="mt-4 text-sm text-yellow-300 bg-yellow-900/20 border border-yellow-800 rounded-xl px-3 py-2">
            All images in this folder have already been processed for the selected passes.
            Nothing was enqueued.
          </p>
        )}

        {error && (
          <p className="mt-4 text-sm text-red-300 bg-red-900/20 border border-red-800 rounded-xl px-3 py-2">
            {error}
          </p>
        )}

        {/* Profiler toggle */}
        <label className="mt-4 flex items-center gap-2 text-xs text-neutral-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={profile}
            onChange={(e) => onProfileChange(e.target.checked)}
            className="rounded border-neutral-600 bg-neutral-800 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-0"
          />
          Enable profiler (performance analysis)
        </label>

        <div className="mt-4 flex flex-wrap gap-2">
          <UiButton
            onClick={onStart}
            disabled={!canStart}
            variant="primary"
          >
            Start batch
          </UiButton>
        </div>
      </SurfaceCard>
    </div>
  )
}

// ── Ingest panel ──────────────────────────────────────────────────────────────

function IngestPanel({ progress }: { progress: BatchIngestProgress | null }) {
  const pct = progress && progress.total > 0
    ? Math.round((progress.scanned / progress.total) * 100)
    : 0

  const currentName = progress?.current
    ? progress.current.replace(/^.*[\\/]/, '')
    : ''

  return (
    <div className="flex flex-col gap-4 max-w-2xl w-full mx-auto py-6 px-4">
      <SurfaceCard tone="accent">
        <SectionHeading
          eyebrow="Ingesting"
          title="Scanning folder…"
          description="Registering images and queueing passes for analysis."
        />

        <div className="mt-4 flex items-center gap-2 text-sm text-neutral-400">
          <div className="w-3.5 h-3.5 border-2 border-neutral-600 border-t-cyan-400 rounded-full animate-spin shrink-0" />
          {progress
            ? `Scanned ${progress.scanned.toLocaleString()} of ${progress.total.toLocaleString()} images`
            : 'Registering images and queueing jobs…'
          }
        </div>

        <div className="mt-3 w-full bg-neutral-800 rounded-full h-2 overflow-hidden">
          <div
            className="h-2 bg-cyan-500 rounded-full transition-all duration-200"
            style={{ width: `${pct}%` }}
          />
        </div>

        {progress && (
          <div className="mt-4 grid grid-cols-4 gap-2 text-center">
            {(
              [
                ['Scanned',    progress.scanned],
                ['Registered', progress.registered],
                ['Enqueued',   progress.enqueued],
                ['Skipped',    progress.skipped],
              ] as [string, number][]
            ).map(([label, value]) => (
              <div key={label} className="rounded-xl border border-neutral-800 bg-black/15 py-2 px-1">
                <div className="text-base font-semibold text-neutral-100 tabular-nums">
                  {value.toLocaleString()}
                </div>
                <div className="text-xs text-neutral-500 mt-0.5">{label}</div>
              </div>
            ))}
          </div>
        )}

        {currentName && (
          <p className="mt-3 text-xs text-neutral-500 truncate" title={progress?.current}>
            {currentName}
          </p>
        )}
      </SurfaceCard>
    </div>
  )
}

function LiveErrorPanel({ error, results }: { error: string | null; results: BatchResult[] }) {
  const failures = results.filter((result) => result.status === 'failed').slice(0, 5)

  if (!error && failures.length === 0) return null

  return (
    <SurfaceCard tone="danger">
      <div className="text-sm font-semibold text-red-300">Live errors</div>
      <p className="mt-1 text-xs text-red-200/80">
        New failures appear here immediately so you can confirm what is breaking.
      </p>

      {error && (
        <div className="mt-3 rounded-lg border border-red-800/80 bg-red-950/40 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}

      {failures.length > 0 && (
        <div className="mt-3 flex flex-col gap-2">
          {failures.map((result) => (
            <div
              key={result.id}
              className="rounded-lg border border-red-900/70 bg-black/20 px-3 py-2 text-xs"
            >
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="font-semibold text-red-300">{result.nodeLabel}</span>
                <span className="text-neutral-400">{result.module}</span>
                <span className="text-neutral-500">
                  {result.path.replace(/^.*[\\/]/, '')}
                </span>
              </div>
              {result.error && (
                <div className="mt-1 whitespace-pre-wrap break-words text-red-100">
                  {result.error}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </SurfaceCard>
  )
}

// ── BatchConfigView ───────────────────────────────────────────────────────────

/**
 * Renders the config + ingest phases.
 * When the user starts a batch, calls onBatchStarted so the parent can
 * respond without this component owning navigation state.
 */
export function BatchConfigView({ batch, initialFolder = '', onBatchStarted }: BatchViewProps) {
  const { stats, ingestProgress, ingestSummary, error } = batch

  const [folder, setFolder] = useState(initialFolder)
  const [passSel, setPassSel] = useState<PassSelectorValue>(defaultPassSelectorValue)
  const [profile, setProfile] = useState(false)
  const [chunkSize, setChunkSize] = useState(500)

  useEffect(() => {
    window.api.getAppSettings().then((bundle) => {
      setChunkSize(bundle.settings.processing?.chunkSize ?? 500)
    }).catch(() => {})
  }, [])

  const phase = stats.status === 'ingesting' ? 'ingesting' : 'config'

  const handleStart = useCallback(async () => {
    const modules = resolveModuleKeys(passSel.selectedKeys)
    batch.startBatch({
      folder,
      modules,
      workers: passSel.workers,
      recursive: passSel.recursive,
      noHash: passSel.noHash,
      forceReprocess: passSel.forceReprocess,
      profile,
      chunkSize,
    }).then(() => {
      // startBatch resolves after ingest; parent can react via onBatchStarted
      // if it wants to surface that state elsewhere.
    })
    onBatchStarted?.()
  }, [batch, folder, passSel, profile, chunkSize, onBatchStarted])

  if (phase === 'ingesting') {
    return (
      <div className="flex-1 overflow-y-auto">
        <IngestPanel progress={ingestProgress} />
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <ConfigPanel
        folder={folder}
        onFolderChange={setFolder}
        passSel={passSel}
        onPassSelChange={setPassSel}
        profile={profile}
        onProfileChange={setProfile}
        onStart={handleStart}
        error={error}
        nothingToRun={ingestSummary !== null && ingestSummary.enqueued === 0}
      />
    </div>
  )
}

// ── BatchRunView ──────────────────────────────────────────────────────────────

interface BatchRunViewProps extends BatchViewProps {
  /** Non-null when we auto-resumed a previous session on mount. */
  resumeBanner: string | null
  onDismissBanner(): void
}

/**
 * Renders the active/running phase: ProgressDashboard + LiveResultsFeed.
 * Also handles the ConfirmStopDialog overlay.
 */
export function BatchRunView({
  batch,
  initialFolder = '',
  resumeBanner,
  onDismissBanner,
}: BatchRunViewProps) {
  const { stats, results, ingestSummary, error } = batch

  const [showStopDialog, setShowStopDialog] = useState(false)
  const pausedByPauseAllRef = useRef<{
    coordinatorPaused: boolean
    pausedWorkerIds: string[]
    masterPaused: boolean
  }>({
    coordinatorPaused: false,
    pausedWorkerIds: [],
    masterPaused: false,
  })
  // Derive the folder from initialFolder (passed from App which tracks gallery folder)
  const folder = initialFolder

  const handleConfirmStop = useCallback(async () => {
    setShowStopDialog(false)
    await batch.stop(folder)
  }, [batch, folder])

  const handlePause = useCallback(async () => {
    const snapshot = {
      coordinatorPaused: false,
      pausedWorkerIds: [] as string[],
      masterPaused: false,
    }

    const coordinatorState = stats.coordinator?.state ?? 'stopped'
    if (coordinatorState === 'running' || coordinatorState === 'starting') {
      await batch.pauseTarget({ scope: 'coordinator' }, 'pause-drain')
      snapshot.coordinatorPaused = true
    }

    const nodes = Array.isArray(stats.nodes) ? stats.nodes : []
    for (const node of nodes) {
      if (node.role !== 'worker') continue
      await batch.pauseTarget({ scope: 'worker', workerId: node.id }, 'pause-drain')
      snapshot.pausedWorkerIds.push(node.id)
    }

    await batch.pauseTarget({ scope: 'master' }, 'pause-drain')
    snapshot.masterPaused = true
    pausedByPauseAllRef.current = snapshot
  }, [batch, stats.coordinator?.state, stats.nodes])

  // Combined resume handler: use batch.resume() when paused (has sessionConfig),
  // otherwise use resumePending() which works without sessionConfig.
  const handleResume = useCallback(async () => {
    const pausedTargets = pausedByPauseAllRef.current
    if (pausedTargets.coordinatorPaused) {
      await batch.resumeTarget({ scope: 'coordinator' })
    }
    for (const workerId of pausedTargets.pausedWorkerIds) {
      await batch.resumeTarget({ scope: 'worker', workerId })
    }
    if (pausedTargets.masterPaused) {
      await batch.resumeTarget({ scope: 'master' })
    }

    if (stats.status === 'paused') {
      await batch.resume()
    } else {
      await batch.resumePending()
    }
    pausedByPauseAllRef.current = {
      coordinatorPaused: false,
      pausedWorkerIds: [],
      masterPaused: false,
    }
  }, [batch, stats.status])

  const handleClearQueue = useCallback(async () => {
    await batch.clearQueue()
  }, [batch])

  const handleClearCompleted = useCallback(async () => {
    await batch.clearCompleted()
  }, [batch])

  return (
    <div className="relative h-full overflow-y-auto">
      <div className="flex min-h-full flex-col gap-4 px-4 py-4">

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
          onPause={handlePause}
          onResume={handleResume}
          onPauseTarget={batch.pauseTarget}
          onResumeTarget={batch.resumeTarget}
          onRemoveWorker={batch.removeWorker}
          onStop={() => setShowStopDialog(true)}
          onRetryFailed={batch.retryFailed}
          onClearQueue={handleClearQueue}
          onClearCompleted={handleClearCompleted}
        />

        <LiveErrorPanel error={error} results={results} />

        <section className="flex min-h-[220px] max-h-[38vh] flex-col rounded-xl border border-neutral-800 bg-neutral-900/30 p-3">
          <p className="text-xs text-neutral-500 mb-1.5">
            Live results (last 200)
          </p>
          <LiveResultsFeed results={results} />
        </section>
      </div>

      {showStopDialog && (
        <ConfirmStopDialog
          onConfirm={handleConfirmStop}
          onCancel={() => setShowStopDialog(false)}
        />
      )}
    </div>
  )
}
