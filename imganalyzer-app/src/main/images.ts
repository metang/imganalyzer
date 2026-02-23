import { readdir, readFile } from 'fs/promises'
import { join, extname, basename } from 'path'
import { statSync, writeFileSync, existsSync } from 'fs'
import { execFile } from 'child_process'
import { promisify } from 'util'
import { tmpdir } from 'os'

const execFileAsync = promisify(execFile)

export const IMAGE_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp',
  '.heic', '.heif',
  '.arw', '.cr2', '.cr3', '.nef', '.nrw', '.orf', '.raf', '.rw2',
  '.dng', '.pef', '.srw', '.erf', '.kdc', '.mrw', '.3fr', '.fff',
  '.sr2', '.srf', '.x3f', '.iiq', '.mos', '.raw'
])

export const RAW_EXTENSIONS = new Set([
  '.arw', '.cr2', '.cr3', '.nef', '.nrw', '.orf', '.raf', '.rw2',
  '.dng', '.pef', '.srw', '.erf', '.kdc', '.mrw', '.3fr', '.fff',
  '.sr2', '.srf', '.x3f', '.iiq', '.mos', '.raw'
])

export interface ImageFile {
  path: string
  name: string
  ext: string
  isRaw: boolean
  xmpPath: string
  hasXmp: boolean
  size: number
  mtime: number
}

export async function listImages(folderPath: string): Promise<ImageFile[]> {
  const entries = await readdir(folderPath)
  const images: ImageFile[] = []

  for (const entry of entries) {
    const ext = extname(entry).toLowerCase()
    if (!IMAGE_EXTENSIONS.has(ext)) continue
    const fullPath = join(folderPath, entry)
    try {
      const s = statSync(fullPath)
      if (!s.isFile()) continue
      const xmpPath = fullPath.replace(/\.[^.]+$/, '.xmp')
      const hasXmp = (() => { try { statSync(xmpPath); return true } catch { return false } })()
      images.push({
        path: fullPath,
        name: basename(entry),
        ext,
        isRaw: RAW_EXTENSIONS.has(ext),
        xmpPath,
        hasXmp,
        size: s.size,
        mtime: s.mtimeMs
      })
    } catch {
      // skip unreadable files
    }
  }

  return images.sort((a, b) => a.name.localeCompare(b.name))
}

// Cache thumbnails in memory to avoid re-generating on every gallery render
const thumbCache = new Map<string, string>()
// In-flight promises to deduplicate concurrent requests for the same image
const inFlight = new Map<string, Promise<string>>()
// Limit concurrent thumbnail processes to avoid overwhelming the system
const MAX_CONCURRENT = 4
let activeCount = 0
const queue: Array<() => void> = []

function acquireSlot(): Promise<void> {
  if (activeCount < MAX_CONCURRENT) {
    activeCount++
    return Promise.resolve()
  }
  return new Promise((resolve) => queue.push(() => { activeCount++; resolve() }))
}

function releaseSlot(): void {
  activeCount--
  if (queue.length > 0) {
    const next = queue.shift()!
    next()
  }
}

export async function getThumbnail(imagePath: string): Promise<string> {
  if (thumbCache.has(imagePath)) return thumbCache.get(imagePath)!
  if (inFlight.has(imagePath)) return inFlight.get(imagePath)!

  const promise = (async () => {
    await acquireSlot()
    try {
      const jpegBuffer = await pythonThumbnail(imagePath)
      const dataUrl = `data:image/jpeg;base64,${jpegBuffer.toString('base64')}`
      thumbCache.set(imagePath, dataUrl)
      return dataUrl
    } catch (err) {
      console.error('Thumbnail error for', imagePath, err)
      return ''
    } finally {
      releaseSlot()
      inFlight.delete(imagePath)
    }
  })()

  inFlight.set(imagePath, promise)
  return promise
}

// Write the thumbnail script to a temp file once, reuse it for all calls.
// conda run rejects multiline -c arguments, so we must use a file.
let THUMB_SCRIPT_PATH: string | null = null

function getThumbnailScriptPath(): string {
  if (THUMB_SCRIPT_PATH && existsSync(THUMB_SCRIPT_PATH)) return THUMB_SCRIPT_PATH

  const script = [
    'import sys, io',
    'from pathlib import Path',
    '',
    'path = Path(sys.argv[1])',
    'ext = path.suffix.lower()',
    '',
    "if ext in ('.heic', '.heif'):",
    '    from pillow_heif import register_heif_opener',
    '    register_heif_opener()',
    '',
    'from PIL import Image',
    '',
    "if ext in ('.arw','.cr2','.cr3','.nef','.nrw','.orf','.raf','.rw2',",
    "           '.dng','.pef','.srw','.erf','.kdc','.mrw','.3fr','.fff',",
    "           '.sr2','.srf','.x3f','.iiq','.mos','.raw'):",
    '    import rawpy',
    '    with rawpy.imread(str(path)) as raw:',
    '        rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)',
    '    import numpy as np',
    '    img = Image.fromarray(rgb)',
    'else:',
    '    img = Image.open(path)',
    "    img = img.convert('RGB')",
    '',
    'img.thumbnail((400, 300), Image.LANCZOS)',
    'buf = io.BytesIO()',
    "img.save(buf, format='JPEG', quality=80)",
    'sys.stdout.buffer.write(buf.getvalue())',
  ].join('\n')

  const p = join(tmpdir(), 'imganalyzer_thumb.py')
  writeFileSync(p, script, 'utf-8')
  THUMB_SCRIPT_PATH = p
  return p
}

async function pythonThumbnail(imagePath: string): Promise<Buffer> {
  const scriptPath = getThumbnailScriptPath()
  const { stdout } = await execFileAsync(
    'conda',
    ['run', '-n', 'imganalyzer', '--no-capture-output', 'python', scriptPath, imagePath],
    { encoding: 'buffer', maxBuffer: 10 * 1024 * 1024, timeout: 30000 }
  )
  return stdout as unknown as Buffer
}

// ── Full-resolution image for lightbox ───────────────────────────────────────

// MIME types that the renderer's <img> tag can display natively
const NATIVE_MIME: Record<string, string> = {
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png':  'image/png',
  '.webp': 'image/webp',
  '.bmp':  'image/bmp',
  '.gif':  'image/gif',
  '.tif':  'image/tiff',
  '.tiff': 'image/tiff',
}

// Cache for full-res data URLs (one at a time — only the open lightbox image)
const fullResCache = new Map<string, string>()
const fullResInFlight = new Map<string, Promise<string>>()

let FULL_SCRIPT_PATH: string | null = null

function getFullScriptPath(): string {
  if (FULL_SCRIPT_PATH && existsSync(FULL_SCRIPT_PATH)) return FULL_SCRIPT_PATH

  // Full-res version: no thumbnail, high quality JPEG output
  const script = [
    'import sys, io',
    'from pathlib import Path',
    '',
    'path = Path(sys.argv[1])',
    'ext = path.suffix.lower()',
    '',
    "if ext in ('.heic', '.heif'):",
    '    from pillow_heif import register_heif_opener',
    '    register_heif_opener()',
    '',
    'from PIL import Image',
    '',
    "if ext in ('.arw','.cr2','.cr3','.nef','.nrw','.orf','.raf','.rw2',",
    "           '.dng','.pef','.srw','.erf','.kdc','.mrw','.3fr','.fff',",
    "           '.sr2','.srf','.x3f','.iiq','.mos','.raw'):",
    '    import rawpy',
    '    with rawpy.imread(str(path)) as raw:',
    '        rgb = raw.postprocess(use_camera_wb=True, output_bps=8)',
    '    import numpy as np',
    '    img = Image.fromarray(rgb)',
    'else:',
    '    img = Image.open(path)',
    "    img = img.convert('RGB')",
    '',
    '# Limit to a sensible display size (4K) to keep the data URL manageable',
    'MAX_DIM = 3840',
    'if img.width > MAX_DIM or img.height > MAX_DIM:',
    '    img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)',
    '',
    'buf = io.BytesIO()',
    "img.save(buf, format='JPEG', quality=92)",
    'sys.stdout.buffer.write(buf.getvalue())',
  ].join('\n')

  const p = join(tmpdir(), 'imganalyzer_fullres.py')
  writeFileSync(p, script, 'utf-8')
  FULL_SCRIPT_PATH = p
  return p
}

export async function getFullImage(imagePath: string): Promise<string> {
  if (fullResCache.has(imagePath)) return fullResCache.get(imagePath)!
  if (fullResInFlight.has(imagePath)) return fullResInFlight.get(imagePath)!

  const promise = (async () => {
    try {
      const ext = extname(imagePath).toLowerCase()
      let dataUrl: string

      if (NATIVE_MIME[ext]) {
        // Read the original file bytes directly — fastest path, no re-encoding
        const buf = await readFile(imagePath)
        dataUrl = `data:${NATIVE_MIME[ext]};base64,${buf.toString('base64')}`
      } else {
        // RAW / HEIC — decode via Python at full resolution
        const scriptPath = getFullScriptPath()
        const { stdout } = await execFileAsync(
          'conda',
          ['run', '-n', 'imganalyzer', '--no-capture-output', 'python', scriptPath, imagePath],
          { encoding: 'buffer', maxBuffer: 200 * 1024 * 1024, timeout: 120000 }
        )
        const buf = stdout as unknown as Buffer
        dataUrl = `data:image/jpeg;base64,${buf.toString('base64')}`
      }

      // Keep only the last 2 full-res images to cap memory
      if (fullResCache.size >= 2) {
        const oldest = fullResCache.keys().next().value
        if (oldest) fullResCache.delete(oldest)
      }
      fullResCache.set(imagePath, dataUrl)
      return dataUrl
    } catch (err) {
      console.error('Full-res error for', imagePath, err)
      return ''
    } finally {
      fullResInFlight.delete(imagePath)
    }
  })()

  fullResInFlight.set(imagePath, promise)
  return promise
}
