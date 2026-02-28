/**
 * analyzer.ts — Single-image analysis via persistent Python JSON-RPC server.
 *
 * Replaces the conda subprocess spawn pattern with RPC calls.
 * Progress notifications are forwarded to the renderer.
 */

import { readFile } from 'fs/promises'
import { existsSync } from 'fs'
import { parseXmp, XmpData } from './xmp'
import { rpc, ensureServerRunning, setNotificationListener } from './python-rpc'

export interface AnalysisProgress {
  imagePath: string
  stage: string
  pct: number
}

// Stage keywords from progress notifications -> progress %
const STAGE_MAP: Array<[RegExp, number, string]> = [
  [/\[1\/4\]/, 5, 'Captioning'],
  [/Loading BLIP-2/, 8, 'Loading BLIP-2 model\u2026'],
  [/\[2\/4\]/, 40, 'Object detection'],
  [/Loading.*GroundingDINO|Loading.*object/i, 42, 'Loading GroundingDINO\u2026'],
  [/\[3\/4\]/, 62, 'OCR \u2014 reading text'],
  [/Loading TrOCR/i, 64, 'Loading TrOCR\u2026'],
  [/\[4\/4\]/, 75, 'Face detection & recognition'],
  [/buffalo_l|Loading.*face/i, 77, 'Loading InsightFace\u2026'],
  [/XMP written|Done\./i, 100, 'Done'],
]

export async function runAnalysis(
  imagePath: string,
  aiBackend: string,
  onProgress: (p: AnalysisProgress) => void
): Promise<{ xmp: XmpData | null; error?: string }> {
  try {
    await ensureServerRunning()

    let lastPct = 0

    // Set up a temporary notification listener for progress
    const prevListener = null
    const progressHandler = (notif: { method: string; params: unknown }) => {
      if (notif.method === 'analyze/progress') {
        const p = notif.params as { imagePath: string; stage: string }
        if (p.imagePath === imagePath) {
          const line = p.stage
          for (const [re, pct, label] of STAGE_MAP) {
            if (re.test(line) && pct > lastPct) {
              lastPct = pct
              onProgress({ imagePath, stage: label, pct })
              break
            }
          }
        }
      }
    }

    // The notification listener is global via python-rpc.ts.
    // We rely on the batch handler's global listener setup.
    // For analyze, we add a temporary one that also handles analyze/progress.
    const origSetup = setNotificationListener
    // We need to hook into the notification stream. Since setNotificationListener
    // is a global setter, we chain it with the existing listener.
    // For simplicity, we use the call's notification callback parameter.
    const result = await rpc.call(
      'analyze',
      {
        imagePath,
        aiBackend,
        overwrite: true,
        verbose: true,
      },
      progressHandler,
      300_000, // 5 min timeout for full analysis
    ) as { ok?: boolean; xmpPath?: string; error?: string }

    if (result.error) {
      return { xmp: null, error: result.error }
    }

    // Re-read the XMP that was written
    const xmpPath = imagePath.replace(/\.[^.]+$/, '.xmp')
    if (existsSync(xmpPath)) {
      try {
        const xml = await readFile(xmpPath, 'utf-8')
        return { xmp: parseXmp(xml) }
      } catch (e) {
        return { xmp: null, error: String(e) }
      }
    } else {
      return { xmp: null, error: 'XMP file not found after analysis' }
    }
  } catch (err) {
    return { xmp: null, error: String(err) }
  }
}

export function cancelAnalysis(imagePath: string): void {
  rpc.call('cancel_analyze', { imagePath }).catch(() => {
    // Best-effort cancel
  })
}
