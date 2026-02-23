import { readdir } from 'fs/promises'
import { join, extname, basename } from 'path'
import { statSync } from 'fs'
import { execFile } from 'child_process'
import { promisify } from 'util'

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

async function pythonThumbnail(imagePath: string): Promise<Buffer> {
  const script = `
import sys, io
from pathlib import Path

path = Path(sys.argv[1])
ext = path.suffix.lower()

if ext in ('.heic', '.heif'):
    from pillow_heif import register_heif_opener
    register_heif_opener()

from PIL import Image

if ext in ('.arw','.cr2','.cr3','.nef','.nrw','.orf','.raf','.rw2',
           '.dng','.pef','.srw','.erf','.kdc','.mrw','.3fr','.fff',
           '.sr2','.srf','.x3f','.iiq','.mos','.raw'):
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
    import numpy as np
    img = Image.fromarray(rgb)
else:
    img = Image.open(path)
    img = img.convert('RGB')

img.thumbnail((400, 300), Image.LANCZOS)
buf = io.BytesIO()
img.save(buf, format='JPEG', quality=80)
sys.stdout.buffer.write(buf.getvalue())
`
  const { stdout } = await execFileAsync(
    'conda',
    ['run', '-n', 'imganalyzer', '--no-capture-output', 'python', '-c', script, imagePath],
    { encoding: 'buffer', maxBuffer: 10 * 1024 * 1024, timeout: 30000 }
  )
  return stdout as unknown as Buffer
}
