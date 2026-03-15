import { useMemo } from 'react'
import type { BatchResult } from '../global'

interface Props {
  results: BatchResult[]
}

const STATUS_COLORS: Record<BatchResult['status'], string> = {
  done:    'text-emerald-400',
  failed:  'text-red-400',
  skipped: 'text-yellow-400',
}

const MODULE_LABELS: Record<string, string> = {
  metadata: 'metadata',
  technical: 'technical',
  caption: 'caption',
  objects: 'objects',
  faces: 'faces',
  perception: 'perception',
  embedding: 'embedding',
}

const STATUS_LABELS: Record<BatchResult['status'], string> = {
  done:    'done',
  failed:  'fail',
  skipped: 'skip',
}

/** Distinct hues for worker nodes on a dark background. */
const NODE_PALETTE = [
  { text: 'text-sky-400',     dot: 'bg-sky-400' },
  { text: 'text-violet-400',  dot: 'bg-violet-400' },
  { text: 'text-amber-400',   dot: 'bg-amber-400' },
  { text: 'text-teal-400',    dot: 'bg-teal-400' },
  { text: 'text-rose-400',    dot: 'bg-rose-400' },
  { text: 'text-lime-400',    dot: 'bg-lime-400' },
  { text: 'text-fuchsia-400', dot: 'bg-fuchsia-400' },
  { text: 'text-cyan-400',    dot: 'bg-cyan-400' },
  { text: 'text-orange-400',  dot: 'bg-orange-400' },
  { text: 'text-indigo-400',  dot: 'bg-indigo-400' },
]

const MASTER_STYLE = { text: 'text-blue-400', dot: 'bg-blue-400' }

function djb2(s: string): number {
  let h = 5381
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0
  return h
}

/** Build a stable nodeId → palette mapping from the current result set. */
function buildNodeColorMap(results: BatchResult[]): Map<string, typeof MASTER_STYLE> {
  const map = new Map<string, typeof MASTER_STYLE>()
  for (const r of results) {
    if (map.has(r.nodeId)) continue
    if (r.nodeRole === 'master') {
      map.set(r.nodeId, MASTER_STYLE)
    } else {
      const idx = Math.abs(djb2(r.nodeId)) % NODE_PALETTE.length
      map.set(r.nodeId, NODE_PALETTE[idx])
    }
  }
  return map
}

/**
 * Scrollable live feed of the last 200 per-job results.
 * Newest entries appear at the top.
 */
export function LiveResultsFeed({ results }: Props) {
  const nodeColors = useMemo(() => buildNodeColorMap(results), [results])

  if (results.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-600 text-xs">
        No results yet
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto text-xs font-mono">
      {results.map((r) => {
        const nc = nodeColors.get(r.nodeId) ?? MASTER_STYLE
        return (
        <div
          key={r.id}
          className="flex items-baseline gap-2 px-3 py-0.5 hover:bg-neutral-800/50 transition-colors"
        >
          {/* Status badge — fixed width */}
          <span className={`w-9 shrink-0 font-semibold uppercase ${STATUS_COLORS[r.status]}`}>
            {STATUS_LABELS[r.status]}
          </span>
          {/* Node label with per-node color */}
          <span
            className={`hidden items-center gap-1.5 w-28 shrink-0 truncate md:inline-flex ${nc.text}`}
            title={r.nodeLabel}
          >
            <span className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${nc.dot}`} />
            {r.nodeLabel}
          </span>
          {/* Module name — fixed width */}
          <span className="w-20 shrink-0 text-neutral-500">{MODULE_LABELS[r.module] ?? r.module}</span>
          {/* Filename — shrinks to make room for error */}
          <span
            className="shrink-0 text-neutral-300 truncate max-w-[180px]"
            title={r.path}
          >
            {r.path.replace(/\\/g, '/').split('/').pop()}
          </span>
          {/* Error message — takes remaining space, truncates with full text on hover */}
          {r.error ? (
            <span
              className="flex-1 text-red-400 truncate min-w-0 cursor-help"
              title={r.error}
            >
              {r.error}
            </span>
          ) : r.keywords && r.keywords.length > 0 ? (
            <span
              className="flex-1 text-neutral-500 truncate min-w-0 cursor-help"
              title={Array.isArray(r.keywords) ? r.keywords.join(', ') : String(r.keywords)}
            >
              {Array.isArray(r.keywords) ? r.keywords.join(', ') : String(r.keywords)}
            </span>
          ) : (
            <span className="flex-1 min-w-0" />
          )}
          {/* Duration — pinned to right */}
          {r.durationMs > 0 && (
            <span className="shrink-0 text-neutral-600">
              {r.durationMs < 1000
                ? `${r.durationMs}ms`
                : `${(r.durationMs / 1000).toFixed(1)}s`}
            </span>
          )}
        </div>
        )
      })}
    </div>
  )
}
