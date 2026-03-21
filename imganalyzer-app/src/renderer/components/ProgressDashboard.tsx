import type {
  BatchActiveModule,
  BatchControlTarget,
  BatchModuleStats,
  BatchNode,
  BatchPauseMode,
  BatchStats,
} from '../global'

interface Props {
  stats: BatchStats
  onPause(): void
  onResume(): void
  onPauseTarget(target: BatchControlTarget, mode?: BatchPauseMode): void
  onResumeTarget(target: BatchControlTarget): void
  onStop(): void
  onRetryFailed(modules: string[]): void
  onClearQueue(): void
  onClearCompleted(): void
}

function fmtMs(ms: number | undefined): string {
  if (!ms || ms <= 0) return '—'
  if (ms < 1000) return `${Math.round(ms)}ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  const rem = Math.round(s % 60)
  return `${m}m ${rem}s`
}

function fmtRate(imgPerSec: number | undefined): string {
  if (!imgPerSec || imgPerSec <= 0) return '—'
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
  caption: 'Caption',
  objects: 'Objects (DINO)',
  faces: 'Faces',
  perception: 'Perception',
  embedding: 'Embeddings',
}

const LEGACY_RETRY_MODULE_MAP: Record<string, string> = {
  blip2: 'caption',
  cloud_ai: 'caption',
  local_ai: 'caption',
  aesthetic: 'perception',
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

function CachePill({
  preDecode,
}: {
  preDecode: { done: number; failed: number; total: number; running: boolean }
}) {
  return (
    <div className="rounded-full border border-cyan-800/70 px-3 py-1.5 text-xs text-cyan-300">
      <span className="text-neutral-500">Decoded</span>
      <span className="ml-2 font-mono">
        {preDecode.done.toLocaleString()} / {preDecode.total.toLocaleString()}
      </span>
      {preDecode.running && (
        <span className="ml-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400" />
      )}
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

function PreDecodeProgress({
  preDecode,
}: {
  preDecode: { done: number; failed: number; total: number; running: boolean }
}) {
  const cached = preDecode.done
  const pct = preDecode.total > 0 ? Math.round((cached / preDecode.total) * 100) : 0
  const remaining = preDecode.total - cached - preDecode.failed

  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-neutral-100">
          Image cache
          {preDecode.running && (
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-cyan-400" title="Pre-decoding in progress" />
          )}
        </div>
        <span className="text-xs font-mono text-neutral-500">{pct}%</span>
      </div>
      <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-neutral-800">
        <div
          className="h-full rounded-full bg-cyan-600 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex flex-wrap gap-3 text-xs text-neutral-400">
        <span>
          <span className="font-mono text-neutral-200">{cached.toLocaleString()}</span>
          <span className="ml-1 text-neutral-500">cached</span>
        </span>
        {remaining > 0 && (
          <span>
            <span className="font-mono text-neutral-300">{remaining.toLocaleString()}</span>
            <span className="ml-1 text-neutral-500">remaining</span>
          </span>
        )}
        {preDecode.failed > 0 && (
          <span>
            <span className="font-mono text-red-400">{preDecode.failed.toLocaleString()}</span>
            <span className="ml-1 text-neutral-500">failed</span>
          </span>
        )}
        <span className="ml-auto text-neutral-600">
          of {preDecode.total.toLocaleString()} images
        </span>
      </div>
    </section>
  )
}

function ChunkProgress({
  chunk,
}: {
  chunk: { size: number; index: number; total: number; modules: Record<string, number> }
}) {
  const moduleEntries = Object.entries(chunk.modules)
    .filter(([, cnt]) => cnt > 0)
    .sort(([, a], [, b]) => b - a)
  const totalRemaining = moduleEntries.reduce((s, [, c]) => s + c, 0)

  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="text-sm font-semibold text-neutral-100">
          Current chunk
        </span>
        <span className="text-xs font-mono text-neutral-500">
          {chunk.index + 1} / {chunk.total}
          <span className="ml-2 text-neutral-600">({chunk.size} images)</span>
        </span>
      </div>

      {moduleEntries.length > 0 ? (
        <div className="grid gap-1.5">
          {moduleEntries.map(([mod, count]) => (
            <div key={mod} className="flex items-center gap-3 text-xs">
              <span className="min-w-[90px] text-neutral-400">
                {MODULE_LABELS[mod] ?? mod}
              </span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-neutral-800">
                <div
                  className="h-full rounded-full bg-cyan-600/70 transition-all duration-300"
                  style={{ width: `${Math.min(100, (count / chunk.size) * 100)}%` }}
                />
              </div>
              <span className="w-12 shrink-0 text-right font-mono text-neutral-300">
                {count.toLocaleString()}
              </span>
            </div>
          ))}
          <div className="mt-1 text-right text-[11px] font-mono text-neutral-500">
            {totalRemaining.toLocaleString()} remaining
          </div>
        </div>
      ) : (
        <p className="text-xs text-neutral-500">All jobs in this chunk completed</p>
      )}
    </section>
  )
}

export function ProgressDashboard({
  stats,
  onPause,
  onResume,
  onPauseTarget,
  onResumeTarget,
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
    estimatedMs,
    elapsedMs,
    chunkAvgCompletionMs,
    chunkElapsedMs,
    chunkEstimatedMs,
    nodes,
  } = stats

  const totalPasses = queue.totalPasses
  const complete = queue.completedPasses
  const overallPct = totalPasses > 0 ? Math.round((complete / totalPasses) * 100) : 0

  const isRunning = status === 'running'
  const isPaused = status === 'paused'
  const hasPending = totals.pending > 0
  const showResume = isPaused || (!isRunning && hasPending)

  const moduleEntries = Object.entries(modules).filter(
    (entry): entry is [string, BatchModuleStats] => entry[1] != null
  )

  const failedModules = Array.from(
    new Set(
      moduleEntries
        .filter(([, s]) => s && s.failed > 0)
        .map(([mod]) => LEGACY_RETRY_MODULE_MAP[mod] ?? mod)
    )
  )

  const canRetry = failedModules.length > 0 && !isRunning
  const canClearQueue = !isRunning && !isPaused && totalPasses > 0
  const canClearCompleted = (totals.done + totals.skipped) > 0
  const coordinatorState = stats.coordinator?.state ?? 'stopped'
  const coordinatorCanPause = coordinatorState === 'running' || coordinatorState === 'starting'
  const masterNode = nodes.find((node) => node.role === 'master') ?? null
  const workerNodes = nodes.filter((node) => node.role === 'worker')

  const isNodePaused = (node: BatchNode): boolean => {
    const desired = node.desiredState ?? 'active'
    return desired === 'pause-drain' || desired === 'pause-immediate' || desired === 'paused'
  }

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
          Monitoring distributed worker progress. Use target controls below to pause/resume nodes.
        </p>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <SummaryCard
          label="Avg chunk completion"
          value={fmtMs(chunkAvgCompletionMs)}
          hint={
            chunkAvgCompletionMs > 0
              ? 'Average of recent completed chunks'
              : 'Waiting for first chunk completion'
          }
        />
        <SummaryCard
          label="Current chunk runtime"
          value={fmtMs(chunkElapsedMs)}
          hint={stats.chunk ? `Chunk ${stats.chunk.index + 1} in progress` : 'No active chunk'}
        />
        <SummaryCard
          label="Chunk ETA"
          value={fmtMs(chunkEstimatedMs)}
          hint={chunkEstimatedMs > 0 ? 'Based on chunk throughput' : 'Waiting for enough samples'}
        />
        <SummaryCard
          label="ETA"
          value={fmtMs(estimatedMs)}
          hint={estimatedMs > 0 ? 'Based on recent throughput' : 'Waiting for enough samples'}
        />
        <SummaryCard label="Elapsed" value={fmtMs(elapsedMs)} />
      </div>

      <section className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4">
        <div className="mb-3 text-sm font-semibold text-neutral-100">Queue status</div>
        <div className="flex flex-wrap gap-2">
          <QueuePill label="Pending" value={totals.pending} />
          <QueuePill label="Running" value={totals.running} />
          <QueuePill label="Done" value={totals.done} />
          <QueuePill label="Failed" value={totals.failed} tone="danger" />
          <QueuePill label="Skipped" value={totals.skipped} tone="warning" />
          {stats.preDecode && stats.preDecode.total > 0 && (
            <CachePill preDecode={stats.preDecode} />
          )}
        </div>
      </section>

      {stats.preDecode && stats.preDecode.total > 0 && (
        <PreDecodeProgress preDecode={stats.preDecode} />
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <NodeContribution nodes={nodes} />
        {stats.chunk && stats.chunk.total > 0 && (
          <ChunkProgress chunk={stats.chunk} />
        )}
      </div>

      {moduleEntries.length > 0 && (
        <section className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4">
          <div className="mb-3 text-sm font-semibold text-neutral-100">Remaining passes by module</div>
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-neutral-600">
                  <th className="py-1 pr-3 text-left font-medium">Module</th>
                  <th className="py-1 pr-3 text-left font-medium">Progress</th>
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

      <section className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="text-sm font-semibold text-neutral-100">Pause targets</div>
          <div className="text-[11px] text-neutral-500">Fine control for coordinator, master worker, and remotes</div>
        </div>

        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-neutral-800/80 bg-black/20 px-3 py-2 text-xs">
            <span className="min-w-[120px] text-neutral-200">Coordinator</span>
            <span className={`rounded-full border px-2 py-0.5 text-[11px] ${statusTone(coordinatorState)}`}>
              {coordinatorState}
            </span>
            <span className="text-neutral-500">
              {stats.coordinator?.lastError ? `Error: ${stats.coordinator.lastError}` : 'Distributed job router'}
            </span>
            <button
              onClick={() =>
                coordinatorCanPause
                  ? onPauseTarget({ scope: 'coordinator' }, 'pause-drain')
                  : onResumeTarget({ scope: 'coordinator' })
              }
              className={`ml-auto rounded-md px-3 py-1 text-[11px] transition-colors ${
                coordinatorCanPause
                  ? 'bg-neutral-700 text-neutral-200 hover:bg-neutral-600'
                  : 'bg-blue-700 text-neutral-100 hover:bg-blue-600'
              }`}
            >
              {coordinatorCanPause ? 'Pause' : 'Resume'}
            </button>
          </div>

          {masterNode && (
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-neutral-800/80 bg-black/20 px-3 py-2 text-xs">
              <span className="min-w-[120px] text-neutral-200">{masterNode.label}</span>
              <span className={`rounded-full border px-2 py-0.5 text-[11px] ${statusTone(masterNode.status)}`}>
                {masterNode.status}
              </span>
              <span className="text-neutral-500">
                desired: {masterNode.desiredState ?? 'active'}
                {masterNode.stateReason ? ` (${masterNode.stateReason})` : ''}
              </span>
              <button
                onClick={() =>
                  isNodePaused(masterNode)
                    ? onResumeTarget({ scope: 'master' })
                    : onPauseTarget({ scope: 'master' }, 'pause-drain')
                }
                className={`ml-auto rounded-md px-3 py-1 text-[11px] transition-colors ${
                  isNodePaused(masterNode)
                    ? 'bg-blue-700 text-neutral-100 hover:bg-blue-600'
                    : 'bg-neutral-700 text-neutral-200 hover:bg-neutral-600'
                }`}
              >
                {isNodePaused(masterNode) ? 'Resume' : 'Pause'}
              </button>
            </div>
          )}

          {workerNodes.map((node) => (
            <div
              key={node.id}
              className="flex flex-wrap items-center gap-2 rounded-lg border border-neutral-800/80 bg-black/20 px-3 py-2 text-xs"
            >
              <span className="min-w-[120px] text-neutral-200">{node.label}</span>
              <span className={`rounded-full border px-2 py-0.5 text-[11px] ${statusTone(node.status)}`}>
                {node.status}
              </span>
              <span className="text-neutral-500">
                desired: {node.desiredState ?? 'active'}
                {node.stateReason ? ` (${node.stateReason})` : ''}
              </span>
              <button
                onClick={() =>
                  isNodePaused(node)
                    ? onResumeTarget({ scope: 'worker', workerId: node.id })
                    : onPauseTarget({ scope: 'worker', workerId: node.id }, 'pause-drain')
                }
                className={`ml-auto rounded-md px-3 py-1 text-[11px] transition-colors ${
                  isNodePaused(node)
                    ? 'bg-blue-700 text-neutral-100 hover:bg-blue-600'
                    : 'bg-neutral-700 text-neutral-200 hover:bg-neutral-600'
                }`}
              >
                {isNodePaused(node) ? 'Resume' : 'Pause'}
              </button>
            </div>
          ))}
        </div>
      </section>

      <div className="flex flex-wrap gap-2">
        {!monitorOnly && isRunning && (
          <button
            onClick={onPause}
            className="rounded-lg bg-neutral-700 px-4 py-1.5 text-sm text-neutral-200 transition-colors hover:bg-neutral-600"
          >
            Pause all
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
