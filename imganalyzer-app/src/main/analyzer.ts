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
  [/\[1\/4\]/, 5, 'Caption + aesthetic scoring'],
  [/Loading BLIP-2/, 8, 'Loading BLIP-2 model…'],
  [/Loading.*aesthetic/i, 12, 'Loading aesthetic model…'],
  [/\[2\/4\]/, 40, 'Object detection'],
  [/Loading.*GroundingDINO|Loading.*object/i, 42, 'Loading GroundingDINO…'],
  [/\[3\/4\]/, 65, 'Face detection & recognition'],
  [/buffalo_l|Loading.*face/i, 67, 'Loading InsightFace…'],
  [/\[4\/4\]/, 90, 'Merging results'],
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

    const emitProgress = (line: string) => {
      for (const [re, pct, label] of STAGE_MAP) {
        if (re.test(line) && pct > lastPct) {
          lastPct = pct
          onProgress({ imagePath, stage: label, pct })
          break
        }
      }
    }

    child.stdout.on('data', (chunk: Buffer) => {
      const text = chunk.toString()
      text.split('\n').forEach(emitProgress)
    })

    child.stderr.on('data', (chunk: Buffer) => {
      const text = chunk.toString()
      stderr += text
      text.split('\n').forEach(emitProgress)
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
