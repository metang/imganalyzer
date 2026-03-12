import type { BatchStats, BatchModuleStats, BatchNode, BatchActiveModule } from '../global'

interface Props {
  stats: BatchStats
  onPause(): void
  onResume(): void
  onStop(): void
  onRetryFailed(modules: string[]): void
  onClearQueue(): void
  onClearCompleted(): void
}

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

function statusTone(status: string): string {
  switch (status) {
    case 'running':
      return 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
    case 'monitoring':
    case 'coordinating':
      return 'bg-blue-500/15 text-blue-300 border-blue-500/30'
    case 'paused':
      return 'bg-yellow-500/15 text-yellow-300 border-yellow-500/30'
    case 'error':
      return 'bg-red-500/15 text-red-300 border-red-500/30'
    case 'done':
      return 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
    case 'online':
      return 'bg-neutral-700/60 text-neutral-200 border-neutral-600'
    default:
      return 'bg-neutral-800 text-neutral-300 border-neutral-700'
  }
}

const MODULE_LABELS: Record<string, string> = {
  metadata: 'Metadata',
  technical: 'Technical',
  local_ai: 'Local AI',
  blip2: 'Caption & Keywords',
  objects: 'Objects (DINO)',
  faces: 'Faces',
  cloud_ai: 'Caption & Keywords',
  aesthetic: 'Aesthetic',
  embedding: 'Embeddings',
}

function formatModuleLabel(module: string): string {
  return MODULE_LABELS[module] ?? module
}

function SummaryCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 px-3 py-3">
      <div className="text-[11px] uppercase tracking-wider text-neutral-500">{label}</div>
      <div className="mt-1 font-mono text-base text-neutral-100">{value}</div>
      {hint && <div className="mt-1 text-xs text-neutral-500">{hint}</div>}
    </div>
  )
}

function QueuePill({
  label,
  value,
  tone,
}: {
  label: string
  value: number
  tone?: 'warning' | 'danger'
}) {
  const toneClass =
    tone === 'danger'
      ? 'border-red-800/80 text-red-300'
      : tone === 'warning'
        ? 'border-yellow-800/80 text-yellow-300'
        : 'border-neutral-800 text-neutral-300'

  return (
    <div className={`rounded-full border px-3 py-1.5 text-xs ${toneClass}`}>
      <span className="text-neutral-500">{label}</span>
      <span className="ml-2 font-mono">{value.toLocaleString()}</span>
    </div>
  )
}

function ModuleTableRow({ name, stats }: { name: string; stats: BatchModuleStats }) {
  const total = stats.pending + stats.running + stats.done + stats.failed + stats.skipped
  const complete = stats.done + stats.failed + stats.skipped
  const pct = total > 0 ? Math.round((complete / total) * 100) : 0

  return (
    <tr className="text-xs">
      <td className="py-1.5 pr-3 text-neutral-400 whitespace-nowrap">
        {MODULE_LABELS[name] ?? name}
      </td>
      <td className="py-1.5 pr-3 min-w-[180px]">
        <div className="flex items-center gap-2">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-neutral-800">
            <div
              className="h-full rounded-full bg-blue-600 transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="w-8 shrink-0 text-right text-neutral-500">{pct}%</span>
        </div>
      </td>
      <td className="px-2 py-1.5 text-right font-mono text-neutral-300 whitespace-nowrap">
        {fmtRate(stats.imagesPerSec)}
      </td>
      <td className="px-2 py-1.5 text-right font-mono text-neutral-400 whitespace-nowrap">
        {fmtMs(stats.avgMsPerImage)}
      </td>
      <td className="px-2 py-1.5 text-right font-mono text-neutral-300 whitespace-nowrap">
        {stats.pending.toLocaleString()}
      </td>
      <td className="px-2 py-1.5 text-right font-mono text-neutral-300 whitespace-nowrap">
        {stats.running.toLocaleString()}
      </td>
      <td className="px-2 py-1.5 text-right font-mono text-neutral-300 whitespace-nowrap">
        {stats.done.toLocaleString()}
      </td>
      <td className={`px-2 py-1.5 text-right font-mono whitespace-nowrap ${stats.failed > 0 ? 'text-red-400' : 'text-neutral-600'}`}>
        {stats.failed > 0 ? stats.failed.toLocaleString() : '—'}
      </td>
      <td className={`pl-2 py-1.5 text-right font-mono whitespace-nowrap ${stats.skipped > 0 ? 'text-yellow-400' : 'text-neutral-600'}`}>
        {stats.skipped > 0 ? stats.skipped.toLocaleString() : '—'}
      </td>
    </tr>
  )
}

function ActivePassChips({
  modules,
  emptyLabel,
}: {
  modules: BatchActiveModule[]
  emptyLabel?: string | null
}) {
  if (modules.length === 0) {
    return emptyLabel ? <span className="text-xs text-neutral-500">{emptyLabel}</span> : null
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {modules.map((item) => (
        <span
          key={`${item.module}:${item.count}`}
          className="rounded-full border border-blue-800/70 bg-blue-900/20 px-2.5 py-1 text-[11px] text-blue-200"
          title={`${item.count} running ${formatModuleLabel(item.module)} pass${item.count === 1 ? '' : 'es'}`}
        >
          {formatModuleLabel(item.module)}
          {item.count > 1 ? ` x${item.count}` : ''}
        </span>
      ))}
    </div>
  )
}

const NODE_COLORS = [
  { bar: 'bg-blue-500', dot: 'bg-blue-400', text: 'text-blue-300' },
  { bar: 'bg-emerald-500', dot: 'bg-emerald-400', text: 'text-emerald-300' },
  { bar: 'bg-purple-500', dot: 'bg-purple-400', text: 'text-purple-300' },
  { bar: 'bg-amber-500', dot: 'bg-amber-400', text: 'text-amber-300' },
  { bar: 'bg-rose-500', dot: 'bg-rose-400', text: 'text-rose-300' },
  { bar: 'bg-cyan-500', dot: 'bg-cyan-400', text: 'text-cyan-300' },
]

function NodeContribution({ nodes }: { nodes: BatchNode[] }) {
  const totalDone = nodes.reduce((sum, n) => sum + n.completedJobs, 0)
  if (totalDone === 0 && nodes.every((n) => n.runningJobs === 0)) return null

  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4">
      <div className="mb-3 text-sm font-semibold text-neutral-100">Worker contribution</div>

      {totalDone > 0 && (
        <div className="mb-3 flex h-3 overflow-hidden rounded-full bg-neutral-800">
          {nodes.map((node, i) => {
            const pct = totalDone > 0 ? (node.completedJobs / totalDone) * 100 : 0
            if (pct === 0) return null
            return (
              <div
                key={node.id}
                className={`${NODE_COLORS[i % NODE_COLORS.length].bar} transition-all duration-500`}
                style={{ width: `${pct}%` }}
                title={`${node.label}: ${node.completedJobs.toLocaleString()} (${Math.round(pct)}%)`}
              />
            )
          })}
        </div>
      )}

      <div className="grid gap-1.5">
        {nodes.map((node, i) => {
          const pct = totalDone > 0 ? (node.completedJobs / totalDone) * 100 : 0
          const color = NODE_COLORS[i % NODE_COLORS.length]
          return (
            <div key={node.id} className="flex items-center gap-3 text-xs">
              <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${color.dot}`} />
              <span className="min-w-[120px] text-neutral-200">{node.label}</span>
              <span className={`rounded-full border px-2 py-0.5 text-[11px] ${statusTone(node.status)}`}>
                {node.status}
              </span>
              <span className="ml-auto font-mono text-neutral-300">
                {node.completedJobs.toLocaleString()}
              </span>
              {totalDone > 0 && (
                <span className="w-10 text-right font-mono text-neutral-500">
                  {Math.round(pct)}%
                </span>
              )}
              <span className="w-16 text-right font-mono text-neutral-400">
                {fmtRate(node.imagesPerSec)}
              </span>
              <ActivePassChips modules={node.activeModules} />
            </div>
          )
        })}
      </div>
    </section>
  )
}

export function ProgressDashboard({
  stats,
  onPause,
  onResume,
  onStop,
  onRetryFailed,
  onClearQueue,
  onClearCompleted,
}: Props) {
  const {
    status,
    monitorOnly,
    queue,
    totals,
    modules,
    imagesPerSec,
    avgMsPerImage,
    estimatedMs,
    elapsedMs,
    nodes,
  } = stats

  const totalPasses = queue.totalPasses
  const complete = queue.completedPasses
  const overallPct = totalPasses > 0 ? Math.round((complete / totalPasses) * 100) : 0

  const isRunning = status === 'running'
  const isPaused = status === 'paused'
  const hasPending = totals.pending > 0
  const showResume = isPaused || (!isRunning && hasPending)

  const failedModules = Object.entries(modules)
    .filter(([, s]) => s && s.failed > 0)
    .map(([mod]) => mod)

  const canRetry = failedModules.length > 0 && !isRunning
  const canClearQueue = !isRunning && !isPaused && totalPasses > 0
  const canClearCompleted = (totals.done + totals.skipped) > 0

  const moduleEntries = Object.entries(modules).filter(
    (entry): entry is [string, BatchModuleStats] => entry[1] != null
  )

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-neutral-400">
          <span>
            {complete.toLocaleString()} / {totalPasses.toLocaleString()} passes complete
          </span>
          <span>{overallPct}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-neutral-800">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              status === 'error'
                ? 'bg-red-600'
                : status === 'done'
                  ? 'bg-emerald-600'
                  : 'bg-blue-600'
            }`}
            style={{ width: `${overallPct}%` }}
          />
        </div>
      </div>

      {monitorOnly && (
        <p className="rounded-lg border border-blue-800 bg-blue-900/20 px-3 py-2 text-xs text-blue-300">
          Monitoring distributed worker progress. Pause/Resume control the local worker only.
        </p>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <SummaryCard label="Remaining jobs" value={queue.remainingJobs.toLocaleString()} />
        <SummaryCard label="Remaining passes" value={queue.remainingPasses.toLocaleString()} />
        <SummaryCard label="Done rate" value={fmtRate(imagesPerSec)} />
        <SummaryCard
          label="ETA"
          value={fmtMs(estimatedMs)}
          hint={estimatedMs > 0 ? 'Based on recent throughput' : 'Waiting for enough samples'}
        />
        <SummaryCard label="Elapsed" value={fmtMs(elapsedMs)} />
        <SummaryCard label="Avg/pass" value={fmtMs(avgMsPerImage)} />
      </div>

      <section className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4">
        <div className="mb-3 text-sm font-semibold text-neutral-100">Queue status</div>
        <div className="flex flex-wrap gap-2">
          <QueuePill label="Pending" value={totals.pending} />
          <QueuePill label="Running" value={totals.running} />
          <QueuePill label="Done" value={totals.done} />
          <QueuePill label="Failed" value={totals.failed} tone="danger" />
          <QueuePill label="Skipped" value={totals.skipped} tone="warning" />
        </div>
      </section>

      <NodeContribution nodes={nodes} />

      {moduleEntries.length > 0 && (
        <section className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4">
          <div className="mb-3 text-sm font-semibold text-neutral-100">Remaining passes by module</div>
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-neutral-600">
                  <th className="py-1 pr-3 text-left font-medium">Module</th>
                  <th className="py-1 pr-3 text-left font-medium">Progress</th>
                  <th className="py-1 px-2 text-right font-medium whitespace-nowrap">done/s</th>
                  <th className="py-1 px-2 text-right font-medium whitespace-nowrap">Avg ms</th>
                  <th className="py-1 px-2 text-right font-medium">Pending</th>
                  <th className="py-1 px-2 text-right font-medium">Running</th>
                  <th className="py-1 px-2 text-right font-medium">Done</th>
                  <th className="py-1 px-2 text-right font-medium">Failed</th>
                  <th className="py-1 pl-2 text-right font-medium">Skipped</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/70">
                {moduleEntries.map(([mod, moduleStats]) => (
                  <ModuleTableRow key={mod} name={mod} stats={moduleStats} />
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <div className="flex flex-wrap gap-2">
        {!monitorOnly && isRunning && (
          <button
            onClick={onPause}
            className="rounded-lg bg-neutral-700 px-4 py-1.5 text-sm text-neutral-200 transition-colors hover:bg-neutral-600"
          >
            Pause
          </button>
        )}
        {!monitorOnly && showResume && (
          <button
            onClick={onResume}
            className="rounded-lg bg-blue-700 px-4 py-1.5 text-sm text-neutral-200 transition-colors hover:bg-blue-600"
          >
            Resume
          </button>
        )}
        {(isRunning || isPaused) && (
          <button
            onClick={onStop}
            className="rounded-lg border border-neutral-700 bg-neutral-800 px-4 py-1.5 text-sm text-red-300 transition-colors hover:border-red-700 hover:bg-red-900/40"
          >
            Stop
          </button>
        )}
        {canRetry && (
          <button
            onClick={() => onRetryFailed(failedModules)}
            className="rounded-lg border border-neutral-700 bg-neutral-800 px-4 py-1.5 text-sm text-orange-300 transition-colors hover:border-orange-700 hover:bg-orange-900/30"
            title={`Retry ${totals.failed} failed pass${totals.failed !== 1 ? 'es' : ''} across: ${failedModules.join(', ')}`}
          >
            Retry failed ({totals.failed})
          </button>
        )}
        {canClearQueue && (
          <button
            onClick={onClearQueue}
            className="rounded-lg border border-neutral-700 bg-neutral-800 px-4 py-1.5 text-sm text-neutral-400 transition-colors hover:border-red-800 hover:bg-red-900/30 hover:text-red-300"
            title="Delete all jobs from the queue and reset to idle"
          >
            Clear queue
          </button>
        )}
        {canClearCompleted && (
          <button
            onClick={onClearCompleted}
            className="rounded-lg border border-neutral-700 bg-neutral-800 px-4 py-1.5 text-sm text-neutral-400 transition-colors hover:border-neutral-600 hover:bg-neutral-700/50 hover:text-neutral-200"
            title={`Remove ${totals.done + totals.skipped} completed pass${totals.done + totals.skipped !== 1 ? 'es' : ''} from the queue`}
          >
            Clear completed ({totals.done + totals.skipped})
          </button>
        )}
      </div>
    </div>
  )
}
