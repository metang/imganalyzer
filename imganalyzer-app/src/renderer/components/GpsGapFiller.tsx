import { useState, useCallback } from 'react'
import type { GapFillerPreviewItem } from '../global'

interface Props {
  open: boolean
  onClose: () => void
  onApplied?: () => void
}

type Step = 'idle' | 'scanning' | 'preview' | 'applying' | 'done'

export function GpsGapFiller({ open, onClose, onApplied }: Props) {
  const [step, setStep] = useState<Step>('idle')
  const [maxGap, setMaxGap] = useState(60)
  const [minConfidence, setMinConfidence] = useState(0.5)
  const [fillable, setFillable] = useState(0)
  const [totalMissing, setTotalMissing] = useState(0)
  const [previews, setPreviews] = useState<GapFillerPreviewItem[]>([])
  const [result, setResult] = useState<{ filled: number; skipped_override: number; skipped_low_confidence: number } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleScan = useCallback(async () => {
    setStep('scanning')
    setError(null)
    try {
      const res = await window.api.geoGapFillerPreview({ max_gap_minutes: maxGap })
      if (res.error) {
        setError(res.error)
        setStep('idle')
        return
      }
      setFillable(res.fillable)
      setTotalMissing(res.total_missing)
      setPreviews(res.previews)
      setStep('preview')
    } catch (err) {
      setError(String(err))
      setStep('idle')
    }
  }, [maxGap])

  const handleApply = useCallback(async () => {
    setStep('applying')
    setError(null)
    try {
      const res = await window.api.geoGapFillerApply({ max_gap_minutes: maxGap, min_confidence: minConfidence })
      if (res.error) {
        setError(res.error)
        setStep('preview')
        return
      }
      setResult(res)
      setStep('done')
      onApplied?.()
    } catch (err) {
      setError(String(err))
      setStep('preview')
    }
  }, [maxGap, minConfidence, onApplied])

  const handleClose = () => {
    setStep('idle')
    setPreviews([])
    setResult(null)
    setError(null)
    onClose()
  }

  const aboveThreshold = previews.filter((p) => p.confidence >= minConfidence).length
  const estimatedFillable = fillable > previews.length
    ? Math.round((aboveThreshold / previews.length) * fillable)
    : aboveThreshold

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl w-[600px] max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-neutral-800">
          <h2 className="text-base font-semibold text-neutral-100">🛰️ GPS Gap Filler</h2>
          <button onClick={handleClose} className="text-neutral-400 hover:text-neutral-200 text-lg">×</button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {error && (
            <div className="text-red-400 text-xs bg-red-900/20 p-3 rounded">{error}</div>
          )}

          {/* Step: Idle / Configure */}
          {(step === 'idle' || step === 'scanning') && (
            <div className="space-y-4">
              <p className="text-sm text-neutral-300">
                Scan your library for images without GPS coordinates and infer locations
                from temporally adjacent geotagged photos.
              </p>
              <div className="space-y-2">
                <label className="text-xs text-neutral-400 block">
                  Maximum time gap for interpolation
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={5}
                    max={240}
                    step={5}
                    value={maxGap}
                    onChange={(e) => setMaxGap(Number(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-sm text-neutral-200 w-20 text-right tabular-nums">{maxGap} min</span>
                </div>
              </div>
              <button
                onClick={handleScan}
                disabled={step === 'scanning'}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 disabled:text-neutral-400 text-white text-sm rounded-lg transition-colors"
              >
                {step === 'scanning' ? 'Scanning…' : 'Scan Library'}
              </button>
            </div>
          )}

          {/* Step: Preview */}
          {step === 'preview' && (
            <div className="space-y-4">
              {/* Summary */}
              <div className="bg-neutral-800/60 rounded-lg p-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-neutral-400">Images without GPS</span>
                  <span className="text-neutral-200 font-medium">{totalMissing.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-neutral-400">Can estimate location for</span>
                  <span className="text-blue-400 font-medium">{fillable.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-neutral-400">Above confidence threshold</span>
                  <span className="text-emerald-400 font-medium">~{estimatedFillable.toLocaleString()}</span>
                </div>
              </div>

              {/* Confidence slider */}
              <div className="space-y-2">
                <label className="text-xs text-neutral-400 block">
                  Minimum confidence threshold
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    value={minConfidence}
                    onChange={(e) => setMinConfidence(Number(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-sm text-neutral-200 w-12 text-right tabular-nums">
                    {(minConfidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Preview table */}
              {previews.length > 0 && (
                <div className="border border-neutral-800 rounded-lg overflow-hidden">
                  <div className="text-xs text-neutral-500 px-3 py-1.5 bg-neutral-800/40">
                    Showing {Math.min(previews.length, 50)} of {fillable.toLocaleString()} fillable images
                  </div>
                  <div className="max-h-48 overflow-y-auto">
                    <table className="w-full text-xs">
                      <thead className="text-neutral-500 sticky top-0 bg-neutral-900">
                        <tr>
                          <th className="text-left px-3 py-1.5 font-medium">File</th>
                          <th className="text-right px-3 py-1.5 font-medium">Lat</th>
                          <th className="text-right px-3 py-1.5 font-medium">Lng</th>
                          <th className="text-right px-3 py-1.5 font-medium">Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {previews.slice(0, 50).map((p) => (
                          <tr
                            key={p.image_id}
                            className={`border-t border-neutral-800/50 ${
                              p.confidence < minConfidence ? 'opacity-40' : ''
                            }`}
                          >
                            <td className="px-3 py-1 text-neutral-300 truncate max-w-[200px]" title={p.file_path}>
                              {p.file_path.split(/[\\/]/).pop()}
                            </td>
                            <td className="px-3 py-1 text-right tabular-nums text-neutral-400">{p.inferred_lat.toFixed(4)}</td>
                            <td className="px-3 py-1 text-right tabular-nums text-neutral-400">{p.inferred_lng.toFixed(4)}</td>
                            <td className="px-3 py-1 text-right">
                              <ConfidenceBadge value={p.confidence} />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Step: Applying */}
          {step === 'applying' && (
            <div className="flex flex-col items-center py-8 space-y-3">
              <div className="animate-spin h-8 w-8 border-2 border-blue-500 border-t-transparent rounded-full" />
              <p className="text-sm text-neutral-300">Applying GPS estimates…</p>
            </div>
          )}

          {/* Step: Done */}
          {step === 'done' && result && (
            <div className="space-y-3">
              <div className="bg-emerald-900/20 border border-emerald-800/40 rounded-lg p-4 space-y-2">
                <p className="text-sm font-medium text-emerald-300">✓ GPS gap filling complete</p>
                <div className="text-xs space-y-1 text-neutral-300">
                  <p>Filled: <span className="text-emerald-400 font-medium">{result.filled.toLocaleString()}</span> images</p>
                  {result.skipped_override > 0 && (
                    <p>Skipped (user overrides): {result.skipped_override.toLocaleString()}</p>
                  )}
                  {result.skipped_low_confidence > 0 && (
                    <p>Skipped (low confidence): {result.skipped_low_confidence.toLocaleString()}</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-neutral-800">
          {step === 'preview' && (
            <>
              <button
                onClick={() => { setStep('idle'); setPreviews([]) }}
                className="px-3 py-1.5 text-sm text-neutral-400 hover:text-neutral-200 transition-colors"
              >
                ← Back
              </button>
              <button
                onClick={handleApply}
                className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded-lg transition-colors"
              >
                Apply to ~{estimatedFillable.toLocaleString()} images
              </button>
            </>
          )}
          {step === 'done' && (
            <button
              onClick={handleClose}
              className="px-4 py-1.5 bg-neutral-700 hover:bg-neutral-600 text-white text-sm rounded-lg transition-colors"
            >
              Done
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color = pct >= 80 ? 'text-emerald-400' : pct >= 50 ? 'text-amber-400' : 'text-red-400'
  return <span className={`tabular-nums font-medium ${color}`}>{pct}%</span>
}
