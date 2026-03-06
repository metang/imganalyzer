import { useState, useEffect, useCallback, useRef } from 'react'
import type { XmpData, AnalysisProgress, AnalysisRunResult } from '../global'

export type AnalysisState =
  | { status: 'idle' }
  | { status: 'cached'; xmp: XmpData }
  | { status: 'analyzing'; stage: string; pct: number }
  | { status: 'done'; xmp: XmpData }
  | { status: 'error'; message: string }

function getErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

function applyAnalysisResult(result: AnalysisRunResult): AnalysisState {
  if (result.cancelled) {
    return { status: 'idle' }
  }
  if (result.error) {
    return { status: 'error', message: result.error }
  }
  if (result.xmp) {
    return { status: 'done', xmp: result.xmp }
  }
  return { status: 'error', message: 'Analysis returned no data' }
}

export function useAnalysis(imagePath: string | null) {
  const [state, setState] = useState<AnalysisState>({ status: 'idle' })
  // Ref incremented on every new imagePath / cancel so in-flight awaits can
  // detect they've been superseded without needing a closure over a boolean.
  const epochRef = useRef(0)

  // Subscribe to progress events
  useEffect(() => {
    const unsub = window.api.onAnalysisProgress((p: AnalysisProgress) => {
      if (p.imagePath !== imagePath) return
      setState({ status: 'analyzing', stage: p.stage, pct: p.pct })
    })
    return () => { unsub() }
  }, [imagePath])

  // When a new image is selected: read cached XMP or auto-analyze
  useEffect(() => {
    if (!imagePath) {
      setState({ status: 'idle' })
      return
    }

    // Bump epoch — any in-flight reanalyze from previous image becomes stale
    const epoch = ++epochRef.current

    async function init() {
      try {
        const cached = await window.api.readXmp(imagePath)
        if (epochRef.current !== epoch) return
        if (cached) {
          setState({ status: 'cached', xmp: cached })
        } else {
          setState({ status: 'analyzing', stage: 'Starting…', pct: 0 })
          const result = await window.api.runAnalysis(imagePath, 'local')
          if (epochRef.current !== epoch) return
          setState(applyAnalysisResult(result))
        }
      } catch (err) {
        if (epochRef.current !== epoch) return
        setState({ status: 'error', message: getErrorMessage(err) })
      }
    }

    void init()
    // No cleanup needed — epoch guards stale results
  }, [imagePath])

  const reanalyze = useCallback(async () => {
    if (!imagePath) return
    const epoch = ++epochRef.current
    setState({ status: 'analyzing', stage: 'Starting…', pct: 0 })
    try {
      const result = await window.api.runAnalysis(imagePath, 'local')
      if (epochRef.current !== epoch) return
      setState(applyAnalysisResult(result))
    } catch (err) {
      if (epochRef.current !== epoch) return
      setState({ status: 'error', message: getErrorMessage(err) })
    }
  }, [imagePath])

  const cancel = useCallback(() => {
    if (!imagePath) return
    // Bump epoch so any pending runAnalysis result is ignored when it resolves
    epochRef.current++
    window.api.cancelAnalysis(imagePath).catch(() => {})
    setState({ status: 'idle' })
  }, [imagePath])

  return { state, reanalyze, cancel }
}
