import type { BatchStats, BatchModuleStats } from '../global'

interface Props {
  stats: BatchStats
  onPause(): void
  onResume(): void
  onStop(): void
  onRetryFailed(modules: string[]): void
  onClearQueue(): void
  onClearCompleted(): void
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
  return `${imgPerSec.toFixed(1)}`
}

const MODULE_LABELS: Record<string, string> = {
  metadata:  'Metadata',
  technical: 'Technical',
  local_ai:  'Local AI',
  blip2:     'BLIP-2 Caption',
  objects:   'Objects (DINO)',
  ocr:       'OCR',
  faces:     'Faces',
  cloud_ai:  'Cloud AI',
  aesthetic: 'Aesthetic',
  embedding: 'Embeddings',
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ModuleTableRow({ name, stats }: { name: string; stats: BatchModuleStats }) {
  const total    = stats.pending + stats.running + stats.done + stats.failed + stats.skipped
  const complete = stats.done + stats.failed + stats.skipped
  const pct      = total > 0 ? Math.round((complete / total) * 100) : 0

  return (
    <tr className="text-xs">
      {/* Module name */}
      <td className="py-1 pr-3 text-neutral-400 whitespace-nowrap">
        {MODULE_LABELS[name] ?? name}
      </td>
      {/* Progress bar */}
      <td className="py-1 pr-3 w-full">
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 rounded-full transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-neutral-500 w-8 text-right shrink-0">{pct}%</span>
        </div>
      </td>
      {/* Speed */}
      <td className="py-1 px-2 text-right font-mono text-neutral-300 whitespace-nowrap">
        {fmtRate(stats.imagesPerSec)}
      </td>
      {/* Processed (done) */}
      <td className="py-1 px-2 text-right font-mono text-neutral-300 whitespace-nowrap">
        {stats.done.toLocaleString()}
      </td>
      {/* Skipped */}
      <td className={`py-1 px-2 text-right font-mono whitespace-nowrap ${stats.skipped > 0 ? 'text-yellow-500' : 'text-neutral-600'}`}>
        {stats.skipped > 0 ? stats.skipped.toLocaleString() : '—'}
      </td>
      {/* Error */}
      <td className={`py-1 px-2 text-right font-mono whitespace-nowrap ${stats.failed > 0 ? 'text-red-500' : 'text-neutral-600'}`}>
        {stats.failed > 0 ? stats.failed.toLocaleString() : '—'}
      </td>
      {/* Total */}
      <td className="py-1 pl-2 text-right font-mono text-neutral-500 whitespace-nowrap">
        {total.toLocaleString()}
      </td>
    </tr>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function ProgressDashboard({ stats, onPause, onResume, onStop, onRetryFailed, onClearQueue, onClearCompleted }: Props) {
  const { status, totals, modules, imagesPerSec, avgMsPerImage, estimatedMs, elapsedMs } = stats

  const totalJobs  = totals.pending + totals.running + totals.done + totals.failed + totals.skipped
  const complete   = totals.done + totals.failed + totals.skipped
  const overallPct = totalJobs > 0 ? Math.round((complete / totalJobs) * 100) : 0

  const isRunning = status === 'running'
  const isPaused  = status === 'paused'
  const hasPending = totals.pending > 0

  // Show Resume when paused OR when there are pending jobs but the worker
  // is no longer running (e.g. worker exited / finished early / error state).
  const showResume = isPaused || (!isRunning && hasPending)

  // Modules with at least one failure — passed to onRetryFailed
  const failedModules = Object.entries(modules)
    .filter(([, s]) => s && s.failed > 0)
    .map(([mod]) => mod)

  const canRetry      = failedModules.length > 0 && !isRunning
  const canClearQueue = !isRunning && !isPaused && totalJobs > 0
  const canClearCompleted = !isRunning && (totals.done + totals.skipped) > 0

  const moduleEntries = Object.entries(modules).filter(
    (entry): entry is [string, BatchModuleStats] => entry[1] != null
  )

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

      {/* ── Per-module table ─────────────────────────────────────────────────── */}
      {moduleEntries.length > 0 && (
        <table className="w-full border-collapse">
          <thead>
            <tr className="text-[10px] uppercase tracking-wider text-neutral-600">
              <th className="py-1 pr-3 text-left font-medium">Module</th>
              <th className="py-1 pr-3 text-left font-medium">Progress</th>
              <th className="py-1 px-2 text-right font-medium whitespace-nowrap">img/s</th>
              <th className="py-1 px-2 text-right font-medium">Processed</th>
              <th className="py-1 px-2 text-right font-medium">Skipped</th>
              <th className="py-1 px-2 text-right font-medium">Error</th>
              <th className="py-1 pl-2 text-right font-medium">Total</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800/50">
            {moduleEntries.map(([mod, s]) => (
              <ModuleTableRow key={mod} name={mod} stats={s} />
            ))}
          </tbody>
        </table>
      )}

      {/* ── Stats row ────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-4 gap-2 text-xs">
        {[
          { label: 'Rate',    value: imagesPerSec > 0 ? fmtRate(imagesPerSec) + ' img/s' : '—' },
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
        {showResume && (
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
        {canClearCompleted && (
          <button
            onClick={onClearCompleted}
            className="
              px-4 py-1.5 rounded-lg text-sm text-neutral-400
              bg-neutral-800 border border-neutral-700
              hover:bg-neutral-700/50 hover:border-neutral-600 hover:text-neutral-200 transition-colors
            "
            title={`Remove ${totals.done + totals.skipped} completed job${totals.done + totals.skipped !== 1 ? 's' : ''} from the queue`}
          >
            Clear completed ({totals.done + totals.skipped})
          </button>
        )}
      </div>
    </div>
  )
}
