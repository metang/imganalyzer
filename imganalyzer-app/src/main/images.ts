import { readdir, readFile, stat, access } from 'fs/promises'
import { join, extname, basename } from 'path'
import { rpc } from './python-rpc'

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

  // Use Promise.allSettled to stat all candidate files concurrently instead
  // of blocking the main thread with statSync per file.
  const candidates = entries
    .map((entry) => ({ entry, ext: extname(entry).toLowerCase() }))
    .filter(({ ext }) => IMAGE_EXTENSIONS.has(ext))

  const results = await Promise.allSettled(
    candidates.map(async ({ entry, ext }) => {
      const fullPath = join(folderPath, entry)
      const s = await stat(fullPath)
      if (!s.isFile()) return null
      const xmpPath = fullPath.replace(/\.[^.]+$/, '.xmp')
      const hasXmp = await access(xmpPath).then(() => true, () => false)
      return {
        path: fullPath,
        name: basename(entry),
        ext,
        isRaw: RAW_EXTENSIONS.has(ext),
        xmpPath,
        hasXmp,
        size: s.size,
        mtime: s.mtimeMs
      } as ImageFile
    })
  )

  for (const r of results) {
    if (r.status === 'fulfilled' && r.value) {
      images.push(r.value)
    }
  }

  return images.sort((a, b) => a.name.localeCompare(b.name))
}

// ── Bounded LRU thumbnail cache ──────────────────────────────────────────────
// At 500K images the old unbounded Map would OOM.  This LRU cache evicts the
// least-recently-used entry when the size exceeds MAX_THUMB_CACHE.  Map
// iteration order is insertion order, so a "get" promotes the key by
// re-inserting it.
const MAX_THUMB_CACHE = 1000
const thumbCache = new Map<string, string>()

function thumbCacheGet(key: string): string | undefined {
  const val = thumbCache.get(key)
  if (val !== undefined) {
    // Promote to most-recently-used by re-inserting
    thumbCache.delete(key)
    thumbCache.set(key, val)
  }
  return val
}

function thumbCacheSet(key: string, value: string): void {
  // If key already exists, delete first so the re-insert moves it to the end
  if (thumbCache.has(key)) thumbCache.delete(key)
  thumbCache.set(key, value)
  // Evict oldest entries if over limit
  while (thumbCache.size > MAX_THUMB_CACHE) {
    const oldest = thumbCache.keys().next().value
    if (oldest !== undefined) thumbCache.delete(oldest)
    else break
  }
}
// In-flight promises to deduplicate concurrent requests for the same image
const inFlight = new Map<string, Promise<string>>()
// Limit concurrent thumbnail RPC calls to avoid overwhelming the server
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
  const cached = thumbCacheGet(imagePath)
  if (cached !== undefined) return cached
  if (inFlight.has(imagePath)) return inFlight.get(imagePath)!

  const promise = (async () => {
    await acquireSlot()
    try {
      const result = await rpc.call('thumbnail', { imagePath }) as { data: string }
      const dataUrl = `data:image/jpeg;base64,${result.data}`
      thumbCacheSet(imagePath, dataUrl)
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
        // RAW / HEIC — decode via Python RPC at full resolution
        const result = await rpc.call('fullimage', { imagePath }) as
          { native: true; path: string } | { native: false; data: string }

        if (result.native) {
          // Server said it's native — read directly (shouldn't happen since
          // we already check NATIVE_MIME, but handle it gracefully)
          const buf = await readFile(result.path)
          const mime = NATIVE_MIME[ext] || 'image/jpeg'
          dataUrl = `data:${mime};base64,${buf.toString('base64')}`
        } else {
          dataUrl = `data:image/jpeg;base64,${result.data}`
        }
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
