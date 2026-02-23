import { useState, useEffect, useCallback } from 'react'
import type { XmpData, AnalysisProgress } from '../global'

export type AnalysisState =
  | { status: 'idle' }
  | { status: 'cached'; xmp: XmpData }
  | { status: 'analyzing'; stage: string; pct: number }
  | { status: 'done'; xmp: XmpData }
  | { status: 'error'; message: string }

export function useAnalysis(imagePath: string | null) {
  const [state, setState] = useState<AnalysisState>({ status: 'idle' })

  // Subscribe to progress events once
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

    let cancelled = false

    async function init() {
      if (!imagePath) return
      const cached = await window.api.readXmp(imagePath)
      if (cancelled) return
      if (cached) {
        setState({ status: 'cached', xmp: cached })
      } else {
        // Auto-analyze
        setState({ status: 'analyzing', stage: 'Starting…', pct: 0 })
        const result = await window.api.runAnalysis(imagePath, 'local')
        if (cancelled) return
        if (result.error) {
          setState({ status: 'error', message: result.error })
        } else if (result.xmp) {
          setState({ status: 'done', xmp: result.xmp })
        } else {
          setState({ status: 'error', message: 'Analysis returned no data' })
        }
      }
    }

    init()
    return () => { cancelled = true }
  }, [imagePath])

  const reanalyze = useCallback(async () => {
    if (!imagePath) return
    setState({ status: 'analyzing', stage: 'Starting…', pct: 0 })
    const result = await window.api.runAnalysis(imagePath, 'local')
    if (result.error) {
      setState({ status: 'error', message: result.error })
    } else if (result.xmp) {
      setState({ status: 'done', xmp: result.xmp })
    } else {
      setState({ status: 'error', message: 'Analysis returned no data' })
    }
  }, [imagePath])

  const cancel = useCallback(() => {
    if (imagePath) {
      window.api.cancelAnalysis(imagePath)
      setState({ status: 'idle' })
    }
  }, [imagePath])

  return { state, reanalyze, cancel }
}
