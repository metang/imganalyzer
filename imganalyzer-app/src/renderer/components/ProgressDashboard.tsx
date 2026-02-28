import type { BatchStats, BatchModuleStats } from '../global'

interface Props {
  stats: BatchStats
  onPause(): void
  onResume(): void
  onStop(): void
  onRetryFailed(modules: string[]): void
  onClearQueue(): void
}

// ── Formatting helpers ────────────────────────────────────────────────────────

function fmtMs(ms: number): string {
  if (ms <= 0) return '—'
  if (ms < 1000) return `${Math.round(ms)}ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  const rem = Math.round(s % 60)
  return `${m}m ${rem}s`
}

function fmtRate(imgPerSec: number): string {
  if (imgPerSec <= 0) return '—'
  return `${imgPerSec.toFixed(1)} img/s`
}

const MODULE_LABELS: Record<string, string> = {
  metadata:  'Metadata',
  technical: 'Technical',
  local_ai:  'Local AI',
  blip2:     'Caption (BLIP-2)',
  objects:   'Objects (DINO)',
  ocr:       'OCR (TrOCR)',
  faces:     'Faces',
  cloud_ai:  'Cloud AI',
  aesthetic: 'Aesthetic',
  embedding: 'Embeddings',
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ModuleRow({ name, stats }: { name: string; stats: BatchModuleStats }) {
  const total   = stats.pending + stats.running + stats.done + stats.failed + stats.skipped
  const complete = stats.done + stats.failed + stats.skipped
  const pct = total > 0 ? Math.round((complete / total) * 100) : 0

  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-20 shrink-0 text-neutral-400">{MODULE_LABELS[name] ?? name}</span>
      <div className="flex-1 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-600 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-8 text-right text-neutral-500">{pct}%</span>
      <span className="w-28 text-right text-neutral-600">
        {complete} / {total}
        {stats.failed > 0 && (
          <span className="text-red-500 ml-1">({stats.failed} err)</span>
        )}
        {stats.skipped > 0 && (
          <span className="text-yellow-600 ml-1">({stats.skipped} skip)</span>
        )}
      </span>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function ProgressDashboard({ stats, onPause, onResume, onStop, onRetryFailed, onClearQueue }: Props) {
  const { status, totals, modules, imagesPerSec, avgMsPerImage, estimatedMs, elapsedMs } = stats

  const totalJobs  = totals.pending + totals.running + totals.done + totals.failed + totals.skipped
  const complete   = totals.done + totals.failed + totals.skipped
  const overallPct = totalJobs > 0 ? Math.round((complete / totalJobs) * 100) : 0

  const isRunning = status === 'running'
  const isPaused  = status === 'paused'

  // Modules with at least one failure — passed to onRetryFailed
  const failedModules = Object.entries(modules)
    .filter(([, s]) => s && s.failed > 0)
    .map(([mod]) => mod)

  const canRetry      = failedModules.length > 0 && !isRunning
  const canClearQueue = !isRunning && !isPaused && totalJobs > 0

  return (
    <div className="flex flex-col gap-4">

      {/* ── Overall progress bar ────────────────────────────────────────────── */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between text-xs text-neutral-400">
          <span>{complete.toLocaleString()} / {totalJobs.toLocaleString()} jobs</span>
          <span>{overallPct}%</span>
        </div>
        <div className="h-2 bg-neutral-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              status === 'error' ? 'bg-red-600' :
              status === 'done'  ? 'bg-emerald-600' :
              'bg-blue-600'
            }`}
            style={{ width: `${overallPct}%` }}
          />
        </div>
      </div>

      {/* ── Per-module breakdown ─────────────────────────────────────────────── */}
      {Object.keys(modules).length > 0 && (
        <div className="flex flex-col gap-1.5">
          {Object.entries(modules).map(([mod, s]) =>
            s ? <ModuleRow key={mod} name={mod} stats={s} /> : null
          )}
        </div>
      )}

      {/* ── Stats row ────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-4 gap-2 text-xs">
        {[
          { label: 'Rate',    value: fmtRate(imagesPerSec) },
          { label: 'Avg/img', value: fmtMs(avgMsPerImage) },
          { label: 'ETA',     value: fmtMs(estimatedMs) },
          { label: 'Elapsed', value: fmtMs(elapsedMs) },
        ].map(({ label, value }) => (
          <div key={label} className="flex flex-col gap-0.5 bg-neutral-800/60 rounded-lg px-3 py-2">
            <span className="text-neutral-500">{label}</span>
            <span className="font-mono text-neutral-200">{value}</span>
          </div>
        ))}
      </div>

      {/* ── Control buttons ──────────────────────────────────────────────────── */}
      <div className="flex gap-2 flex-wrap">
        {isRunning && (
          <button
            onClick={onPause}
            className="
              px-4 py-1.5 rounded-lg text-sm text-neutral-200
              bg-neutral-700 hover:bg-neutral-600 transition-colors
            "
          >
            Pause
          </button>
        )}
        {isPaused && (
          <button
            onClick={onResume}
            className="
              px-4 py-1.5 rounded-lg text-sm text-neutral-200
              bg-blue-700 hover:bg-blue-600 transition-colors
            "
          >
            Resume
          </button>
        )}
        {(isRunning || isPaused) && (
          <button
            onClick={onStop}
            className="
              px-4 py-1.5 rounded-lg text-sm text-red-300
              bg-neutral-800 border border-neutral-700
              hover:bg-red-900/40 hover:border-red-700 transition-colors
            "
          >
            Stop
          </button>
        )}
        {canRetry && (
          <button
            onClick={() => onRetryFailed(failedModules)}
            className="
              px-4 py-1.5 rounded-lg text-sm text-orange-300
              bg-neutral-800 border border-neutral-700
              hover:bg-orange-900/30 hover:border-orange-700 transition-colors
            "
            title={`Retry ${totals.failed} failed job${totals.failed !== 1 ? 's' : ''} across: ${failedModules.join(', ')}`}
          >
            Retry failed ({totals.failed})
          </button>
        )}
        {canClearQueue && (
          <button
            onClick={onClearQueue}
            className="
              px-4 py-1.5 rounded-lg text-sm text-neutral-400
              bg-neutral-800 border border-neutral-700
              hover:bg-red-900/30 hover:border-red-800 hover:text-red-300 transition-colors
            "
            title="Delete all jobs from the queue and reset to idle"
          >
            Clear queue
          </button>
        )}
      </div>
    </div>
  )
}
