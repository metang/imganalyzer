import React from 'react'
import type { XmpData } from '../global'
import type { AnalysisState } from '../hooks/useAnalysis'

interface SidebarProps {
  imageName: string
  state: AnalysisState
  onReanalyze: () => void
  onCancel: () => void
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-2 py-1 border-b border-neutral-800 last:border-0">
      <span className="text-neutral-500 text-xs w-32 shrink-0">{label}</span>
      <span className="text-neutral-200 text-xs break-words min-w-0">{value}</span>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <h3 className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 mb-1 px-4">{title}</h3>
      <div className="px-4">{children}</div>
    </div>
  )
}

function ScoreBar({ value, max = 10 }: { value: number; max?: number }) {
  const pct = Math.min(100, (value / max) * 100)
  const color = pct >= 70 ? 'bg-green-500' : pct >= 40 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-neutral-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-neutral-300 tabular-nums w-8 text-right">{value.toFixed(1)}</span>
    </div>
  )
}

function TagList({ items }: { items: string[] }) {
  if (!items.length) return <span className="text-neutral-600 text-xs">—</span>
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((item, i) => (
        <span key={i} className="px-1.5 py-0.5 bg-neutral-800 rounded text-[11px] text-neutral-300">
          {item}
        </span>
      ))}
    </div>
  )
}

function XmpResults({ xmp }: { xmp: XmpData }) {
  return (
    <div className="overflow-y-auto">
      {/* AI Section */}
      <Section title="AI Analysis">
        {xmp.description && <Row label="Description" value={xmp.description} />}
        {xmp.sceneType && <Row label="Scene" value={xmp.sceneType} />}
        {xmp.mainSubject && <Row label="Subject" value={xmp.mainSubject} />}
        {xmp.lighting && <Row label="Lighting" value={xmp.lighting} />}
        {xmp.mood && <Row label="Mood" value={xmp.mood} />}
        {xmp.aestheticScore !== undefined && (
          <div className="py-1 border-b border-neutral-800">
            <div className="flex justify-between mb-1">
              <span className="text-neutral-500 text-xs">Aesthetic</span>
              {xmp.aestheticLabel && <span className="text-xs text-neutral-400">{xmp.aestheticLabel}</span>}
            </div>
            <ScoreBar value={xmp.aestheticScore} />
          </div>
        )}
        {xmp.detectedObjects && xmp.detectedObjects.length > 0 && (
          <div className="py-1 border-b border-neutral-800">
            <span className="text-neutral-500 text-xs block mb-1">Objects</span>
            <TagList items={xmp.detectedObjects} />
          </div>
        )}
        {xmp.keywords && xmp.keywords.length > 0 && (
          <div className="py-1 border-b border-neutral-800">
            <span className="text-neutral-500 text-xs block mb-1">Keywords</span>
            <TagList items={xmp.keywords} />
          </div>
        )}
      </Section>

      {/* Faces */}
      {(xmp.faceCount !== undefined || (xmp.faceIdentities && xmp.faceIdentities.length > 0)) && (
        <Section title="Faces">
          {xmp.faceCount !== undefined && <Row label="Count" value={xmp.faceCount} />}
          {xmp.faceIdentities && xmp.faceIdentities.length > 0 && (
            <div className="py-1 border-b border-neutral-800">
              <span className="text-neutral-500 text-xs block mb-1">Identities</span>
              <TagList items={xmp.faceIdentities} />
            </div>
          )}
        </Section>
      )}

      {/* Technical */}
      <Section title="Technical">
        {xmp.sharpnessScore !== undefined && (
          <div className="py-1 border-b border-neutral-800">
            <div className="flex justify-between mb-1">
              <span className="text-neutral-500 text-xs">Sharpness</span>
              {xmp.sharpnessLabel && <span className="text-xs text-neutral-400">{xmp.sharpnessLabel}</span>}
            </div>
            <ScoreBar value={xmp.sharpnessScore} />
          </div>
        )}
        {xmp.exposureEV !== undefined && <Row label="Exposure EV" value={`${xmp.exposureEV > 0 ? '+' : ''}${xmp.exposureEV.toFixed(2)}${xmp.exposureLabel ? ` (${xmp.exposureLabel})` : ''}`} />}
        {xmp.noiseLevel !== undefined && <Row label="Noise" value={`${xmp.noiseLevel.toFixed(2)}${xmp.noiseLabel ? ` (${xmp.noiseLabel})` : ''}`} />}
        {xmp.snrDb !== undefined && <Row label="SNR" value={`${xmp.snrDb.toFixed(1)} dB`} />}
        {xmp.dynamicRangeStops !== undefined && <Row label="Dynamic Range" value={`${xmp.dynamicRangeStops.toFixed(1)} stops`} />}
        {xmp.highlightClippingPct !== undefined && <Row label="Highlight Clip" value={`${xmp.highlightClippingPct.toFixed(2)}%`} />}
        {xmp.shadowClippingPct !== undefined && <Row label="Shadow Clip" value={`${xmp.shadowClippingPct.toFixed(2)}%`} />}
        {xmp.avgSaturation !== undefined && <Row label="Saturation" value={xmp.avgSaturation.toFixed(2)} />}
        {xmp.dominantColors && xmp.dominantColors.length > 0 && (
          <div className="py-1 border-b border-neutral-800">
            <span className="text-neutral-500 text-xs block mb-1">Colors</span>
            <div className="flex gap-1 flex-wrap">
              {xmp.dominantColors.map((c, i) => (
                <div key={i} className="flex items-center gap-1">
                  <div className="w-4 h-4 rounded border border-neutral-600" style={{ backgroundColor: c }} />
                  <span className="text-[10px] text-neutral-400">{c}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Section>

      {/* Camera */}
      {(xmp.cameraMake || xmp.cameraModel || xmp.fNumber || xmp.exposureTime || xmp.focalLength || xmp.iso) && (
        <Section title="Camera">
          {(xmp.cameraMake || xmp.cameraModel) && <Row label="Camera" value={[xmp.cameraMake, xmp.cameraModel].filter(Boolean).join(' ')} />}
          {xmp.lens && <Row label="Lens" value={xmp.lens} />}
          {xmp.fNumber && <Row label="Aperture" value={`f/${xmp.fNumber}`} />}
          {xmp.exposureTime && <Row label="Shutter" value={xmp.exposureTime} />}
          {xmp.focalLength && <Row label="Focal Length" value={`${xmp.focalLength} mm`} />}
          {xmp.iso && <Row label="ISO" value={xmp.iso} />}
          {xmp.createDate && <Row label="Date" value={xmp.createDate} />}
          {(xmp.imageWidth || xmp.imageHeight) && <Row label="Dimensions" value={`${xmp.imageWidth ?? '?'} × ${xmp.imageHeight ?? '?'}`} />}
          {(xmp.gpsLatitude || xmp.gpsLongitude) && <Row label="GPS" value={`${xmp.gpsLatitude}, ${xmp.gpsLongitude}`} />}
          {(xmp.locationCity || xmp.locationCountry) && (
            <Row label="Location" value={[xmp.locationCity, xmp.locationState, xmp.locationCountry].filter(Boolean).join(', ')} />
          )}
        </Section>
      )}
    </div>
  )
}

export function Sidebar({ imageName, state, onReanalyze, onCancel }: SidebarProps) {
  return (
    <div className="w-80 shrink-0 flex flex-col bg-neutral-900 border-l border-neutral-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-neutral-800 flex items-center justify-between gap-2">
        <span className="text-sm font-medium truncate text-neutral-200" title={imageName}>{imageName}</span>
        {(state.status === 'done' || state.status === 'cached') && (
          <button
            onClick={onReanalyze}
            className="shrink-0 text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
          >
            Re-analyze
          </button>
        )}
        {state.status === 'analyzing' && (
          <button
            onClick={onCancel}
            className="shrink-0 text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-red-900 text-neutral-300 transition-colors"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto pt-3">
        {state.status === 'idle' && (
          <div className="px-4 text-neutral-600 text-sm">Select an image to analyze</div>
        )}

        {state.status === 'analyzing' && (
          <div className="px-4 space-y-3">
            <div className="flex items-center gap-2 text-sm text-neutral-300">
              <div className="w-4 h-4 border-2 border-neutral-600 border-t-blue-400 rounded-full animate-spin shrink-0" />
              {state.stage}
            </div>
            <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-500"
                style={{ width: `${state.pct}%` }}
              />
            </div>
            <div className="text-xs text-neutral-600">{state.pct}%</div>
          </div>
        )}

        {state.status === 'error' && (
          <div className="px-4 space-y-3">
            <div className="text-red-400 text-sm font-medium">Analysis failed</div>
            <pre className="text-[11px] text-red-300/70 bg-neutral-900 rounded p-2 overflow-x-auto whitespace-pre-wrap break-all">
              {state.message}
            </pre>
            <button
              onClick={onReanalyze}
              className="text-xs px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {(state.status === 'done' || state.status === 'cached') && (
          <>
            {state.status === 'cached' && (
              <div className="mx-4 mb-2 px-2 py-1 rounded bg-green-900/30 border border-green-800/50 text-[11px] text-green-400">
                Cached XMP sidecar
              </div>
            )}
            <XmpResults xmp={state.xmp} />
          </>
        )}
      </div>
    </div>
  )
}
