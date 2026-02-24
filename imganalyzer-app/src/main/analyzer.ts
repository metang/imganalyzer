import { spawn, ChildProcess } from 'child_process'
import { readFile } from 'fs/promises'
import { existsSync } from 'fs'
import { parseXmp, XmpData } from './xmp'

export interface AnalysisProgress {
  imagePath: string
  stage: string
  pct: number
}

// Map imagePath → running subprocess so we can cancel
const running = new Map<string, ChildProcess>()

// Stage keywords from CLI stdout → progress %
const STAGE_MAP: Array<[RegExp, number, string]> = [
  [/\[1\/3\]/, 5, 'Captioning'],
  [/Loading BLIP-2/, 8, 'Loading BLIP-2 model…'],
  [/\[2\/3\]/, 40, 'Object detection'],
  [/Loading.*GroundingDINO|Loading.*object/i, 42, 'Loading GroundingDINO…'],
  [/\[3\/3\]/, 65, 'Face detection & recognition'],
  [/buffalo_l|Loading.*face/i, 67, 'Loading InsightFace…'],
  [/XMP written|Done\./i, 100, 'Done'],
]

export async function runAnalysis(
  imagePath: string,
  aiBackend: string,
  onProgress: (p: AnalysisProgress) => void
): Promise<{ xmp: XmpData | null; error?: string }> {
  // Resolve imganalyzer package root (two levels up from dist/main/)
  const pkgRoot = process.env.IMGANALYZER_PKG_ROOT || 'D:\\Code\\imganalyzer'

  return new Promise((resolve) => {
    const args = [
      'run', '-n', 'imganalyzer', '--no-capture-output',
      'python', '-m', 'imganalyzer.cli',
      'analyze', imagePath,
      '--ai', aiBackend,
      '--overwrite',
      '--verbose'
    ]

    const child = spawn('conda', args, {
      cwd: pkgRoot,
      env: { ...process.env, HF_HUB_DISABLE_SYMLINKS_WARNING: '1' }
    })

    running.set(imagePath, child)

    let stderr = ''
    let lastPct = 0
    // Line buffers — pipe chunks may split across line boundaries
    let stdoutBuf = ''
    let stderrBuf = ''

    const emitProgress = (line: string) => {
      for (const [re, pct, label] of STAGE_MAP) {
        if (re.test(line) && pct > lastPct) {
          lastPct = pct
          onProgress({ imagePath, stage: label, pct })
          break
        }
      }
    }

    const processLines = (buf: string, chunk: string): string => {
      const combined = buf + chunk
      const lines = combined.split('\n')
      // Last element may be an incomplete line — keep it in the buffer
      const incomplete = lines.pop() ?? ''
      lines.forEach(emitProgress)
      return incomplete
    }

    child.stdout.on('data', (chunk: Buffer) => {
      stdoutBuf = processLines(stdoutBuf, chunk.toString())
    })

    child.stderr.on('data', (chunk: Buffer) => {
      const text = chunk.toString()
      stderr += text
      stderrBuf = processLines(stderrBuf, text)
    })

    child.on('close', async (code) => {
      running.delete(imagePath)

      if (code !== 0) {
        resolve({ xmp: null, error: `Process exited with code ${code}:\n${stderr.slice(-1000)}` })
        return
      }

      // Re-read the XMP that was written
      const xmpPath = imagePath.replace(/\.[^.]+$/, '.xmp')
      if (existsSync(xmpPath)) {
        try {
          const xml = await readFile(xmpPath, 'utf-8')
          resolve({ xmp: parseXmp(xml) })
        } catch (e) {
          resolve({ xmp: null, error: String(e) })
        }
      } else {
        resolve({ xmp: null, error: 'XMP file not found after analysis' })
      }
    })

    child.on('error', (err) => {
      running.delete(imagePath)
      resolve({ xmp: null, error: err.message })
    })
  })
}

export function cancelAnalysis(imagePath: string): void {
  const child = running.get(imagePath)
  if (child) {
    child.kill('SIGTERM')
    running.delete(imagePath)
  }
}
