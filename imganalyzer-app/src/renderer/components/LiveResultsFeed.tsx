import type { BatchResult } from '../global'

interface Props {
  results: BatchResult[]
}

const STATUS_COLORS: Record<BatchResult['status'], string> = {
  done:    'text-emerald-400',
  failed:  'text-red-400',
  skipped: 'text-yellow-400',
}

const STATUS_LABELS: Record<BatchResult['status'], string> = {
  done:    'done',
  failed:  'fail',
  skipped: 'skip',
}

/**
 * Scrollable live feed of the last 200 per-job results.
 * Newest entries appear at the top.
 */
export function LiveResultsFeed({ results }: Props) {
  if (results.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-neutral-600 text-xs">
        No results yet
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto text-xs font-mono">
      {results.map((r, i) => (
        <div
          key={i}
          className="flex items-baseline gap-2 px-3 py-0.5 hover:bg-neutral-800/50 transition-colors"
        >
          {/* Status badge — fixed width */}
          <span className={`w-9 shrink-0 font-semibold uppercase ${STATUS_COLORS[r.status]}`}>
            {STATUS_LABELS[r.status]}
          </span>
          {/* Module name — fixed width */}
          <span className="w-20 shrink-0 text-neutral-500">{r.module}</span>
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
              title={r.keywords.join(', ')}
            >
              {r.keywords.join(', ')}
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
      ))}
    </div>
  )
}
