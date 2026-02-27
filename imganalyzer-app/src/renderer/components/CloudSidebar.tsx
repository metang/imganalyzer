import React from 'react'
import type { XmpData } from '../global'
import type { CloudAnalysisState } from '../hooks/useCloudAnalysis'

interface CloudSidebarProps {
  imageName: string
  state: CloudAnalysisState
  onAnalyze: () => void
}

// ─── Shared sub-components (mirror of Sidebar.tsx) ────────────────────────

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

function ScoreBar({ value, max = 10 }: { value: number; max?: number }) {
  const pct = Math.max(0, Math.min(1, value / max))
  const color =
    pct >= 0.7 ? 'bg-purple-500' :
    pct >= 0.4 ? 'bg-purple-400' :
                 'bg-purple-700'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct * 100}%` }} />
      </div>
      <span className="text-neutral-400 text-xs w-6 text-right">{value.toFixed(1)}</span>
    </div>
  )
}

// ─── Cloud icon ───────────────────────────────────────────────────────────

function CloudIcon() {
  return (
    <svg className="w-3.5 h-3.5 text-purple-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M2.25 15a4.5 4.5 0 004.5 4.5H18a3.75 3.75 0 001.332-7.257 3 3 0 00-3.758-3.848 5.25 5.25 0 00-10.233 2.33A4.502 4.502 0 002.25 15z" />
    </svg>
  )
}

// ─── Results renderer ─────────────────────────────────────────────────────

function CloudResults({ xmp }: { xmp: XmpData }) {
  return (
    <div className="overflow-y-auto">
      <Section title="AI Analysis · Cloud">
        {xmp.aestheticScore !== undefined && (
          <div className="py-1 border-b border-neutral-800">
            <div className="flex justify-between mb-1">
              <span className="text-neutral-500 text-xs">Aesthetic</span>
              {xmp.aestheticLabel && (
                <span className="text-xs text-purple-400">{xmp.aestheticLabel}</span>
              )}
            </div>
            <ScoreBar value={xmp.aestheticScore} />
          </div>
        )}
        {xmp.description && <Row label="Cloud Description" value={xmp.description} />}
        {xmp.sceneType   && <Row label="Cloud Scene"       value={xmp.sceneType} />}
        {xmp.mainSubject && <Row label="Cloud Subject"     value={xmp.mainSubject} />}
        {xmp.lighting    && <Row label="Cloud Lighting"    value={xmp.lighting} />}
        {xmp.mood        && <Row label="Cloud Mood"        value={xmp.mood} />}
        {xmp.keywords && xmp.keywords.length > 0 && (
          <div className="py-1 border-b border-neutral-800">
            <span className="text-neutral-500 text-xs block mb-1">Cloud Keywords</span>
            <TagList items={xmp.keywords} />
          </div>
        )}
      </Section>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────

export function CloudSidebar({ imageName, state, onAnalyze }: CloudSidebarProps) {
  return (
    <div className="w-80 shrink-0 flex flex-col bg-neutral-900 border-r border-neutral-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-purple-900/60 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <CloudIcon />
          <span className="text-sm font-medium text-purple-300 shrink-0">Cloud AI</span>
          <span className="text-[11px] text-neutral-600 truncate">· gpt-4.1</span>
        </div>
        {(state.status === 'idle' || state.status === 'error') && (
          <button
            onClick={onAnalyze}
            className="shrink-0 text-xs px-2 py-1 rounded bg-purple-900/60 hover:bg-purple-800/80 text-purple-300 border border-purple-800/60 transition-colors"
          >
            Analyze
          </button>
        )}
        {state.status === 'done' && (
          <button
            onClick={onAnalyze}
            className="shrink-0 text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
          >
            Re-analyze
          </button>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto pt-3">
        {state.status === 'idle' && (
          <div className="px-4 space-y-3">
            <p className="text-neutral-600 text-xs leading-relaxed">
              Compare with GitHub Copilot (gpt-4.1) cloud analysis.
            </p>
            <p className="text-neutral-700 text-[11px] truncate" title={imageName}>{imageName}</p>
            <button
              onClick={onAnalyze}
              className="w-full text-xs px-3 py-2 rounded bg-purple-900/50 hover:bg-purple-800/70 text-purple-300 border border-purple-800/50 transition-colors"
            >
              Analyze with Cloud AI
            </button>
          </div>
        )}

        {state.status === 'analyzing' && (
          <div className="px-4 space-y-3">
            <div className="flex items-center gap-2 text-sm text-purple-300">
              <div className="w-4 h-4 border-2 border-purple-900 border-t-purple-400 rounded-full animate-spin shrink-0" />
              Analyzing with gpt-4.1…
            </div>
            <p className="text-neutral-600 text-xs">Sending image to GitHub Copilot cloud…</p>
          </div>
        )}

        {state.status === 'error' && (
          <div className="px-4 space-y-3">
            <div className="text-red-400 text-sm font-medium">Cloud analysis failed</div>
            <pre className="text-[11px] text-red-300/70 bg-neutral-950 rounded p-2 overflow-x-auto whitespace-pre-wrap break-all">
              {state.message}
            </pre>
            <button
              onClick={onAnalyze}
              className="text-xs px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {state.status === 'done' && (
          <CloudResults xmp={state.xmp} />
        )}
      </div>
    </div>
  )
}
