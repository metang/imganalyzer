import { useState, useCallback, useEffect } from 'react'
import type { XmpData } from '../global'

export type CloudAnalysisState =
  | { status: 'idle' }
  | { status: 'analyzing' }
  | { status: 'done'; xmp: XmpData }
  | { status: 'error'; message: string }

export function useCloudAnalysis(imagePath: string | null) {
  const [state, setState] = useState<CloudAnalysisState>({ status: 'idle' })

  // Reset to idle whenever the image changes
  useEffect(() => {
    setState({ status: 'idle' })
  }, [imagePath])

  const analyze = useCallback(async () => {
    if (!imagePath) return
    // Capture the path at call time so a stale closure doesn't resolve wrong
    const path = imagePath
    setState({ status: 'analyzing' })
    try {
      const result = await window.api.runCopilotAnalysis(path)
      // Guard: if the user navigated away while the request was in flight, ignore
      if (path !== imagePath) return
      if (result.error) {
        setState({ status: 'error', message: result.error })
      } else if (result.xmp) {
        setState({ status: 'done', xmp: result.xmp })
      } else {
        setState({ status: 'error', message: 'Cloud AI returned no data' })
      }
    } catch (err: unknown) {
      setState({ status: 'error', message: String(err) })
    }
  }, [imagePath])

  const reset = useCallback(() => {
    setState({ status: 'idle' })
  }, [])

  return { state, analyze, reset }
}
