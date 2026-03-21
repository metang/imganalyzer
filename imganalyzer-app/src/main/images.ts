import { access, mkdir, readdir, readFile, stat, unlink, utimes, writeFile } from 'fs/promises'
import type { Dirent } from 'fs'
import { createHash } from 'crypto'
import { basename, dirname, extname, join } from 'path'
import { rpc } from './python-rpc'
import type { ThumbnailCacheConfig, ThumbnailCacheConfigInput } from './settings'
import { getAppSettings, updateAppSettings } from './settings'

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

function isErrnoCode(err: unknown, code: string): boolean {
  return typeof err === 'object' && err !== null && 'code' in err && (err as { code?: string }).code === code
}

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

export async function getThumbnailCacheConfig(): Promise<ThumbnailCacheConfig> {
  const settings = await getAppSettings()
  await mkdir(settings.thumbnailCache.directory, { recursive: true })
  return settings.thumbnailCache
}

export async function setThumbnailCacheConfig(config: ThumbnailCacheConfigInput): Promise<ThumbnailCacheConfig> {
  const bundle = await updateAppSettings({ thumbnailCache: config })
  const resolved = bundle.settings.thumbnailCache
  await mkdir(resolved.directory, { recursive: true })
  void triggerThumbnailCacheCleanup()
  return resolved
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

// ── Persistent on-disk thumbnail cache ───────────────────────────────────────

interface CachedThumbFile {
  path: string
  size: number
  mtimeMs: number
}

let writesSinceCleanup = 0
let cleanupPromise: Promise<void> | null = null
const CLEANUP_WRITE_INTERVAL = 64

async function thumbnailCacheFilePath(imagePath: string): Promise<string> {
  const sourceStat = await stat(imagePath)
  const cfg = await getThumbnailCacheConfig()
  const material = `${imagePath}|${sourceStat.size}|${sourceStat.mtimeMs}|thumb-v1|400x300|q80`
  const hash = createHash('sha1').update(material).digest('hex')
  return join(cfg.directory, hash.slice(0, 2), `${hash}.jpg`)
}

async function readThumbnailFromDisk(imagePath: string): Promise<string | null> {
  try {
    const cachePath = await thumbnailCacheFilePath(imagePath)
    const buf = await readFile(cachePath)
    const now = new Date()
    void utimes(cachePath, now, now)
    return `data:image/jpeg;base64,${buf.toString('base64')}`
  } catch (err) {
    if (isErrnoCode(err, 'ENOENT') || isErrnoCode(err, 'ENOTDIR')) {
      return null
    }
    console.warn('Disk thumbnail cache read failed for', imagePath, err)
    return null
  }
}

async function writeThumbnailToDisk(imagePath: string, base64Jpeg: string): Promise<void> {
  try {
    const cachePath = await thumbnailCacheFilePath(imagePath)
    await mkdir(dirname(cachePath), { recursive: true })
    await writeFile(cachePath, Buffer.from(base64Jpeg, 'base64'))
    writesSinceCleanup += 1
    if (writesSinceCleanup >= CLEANUP_WRITE_INTERVAL) {
      writesSinceCleanup = 0
      void triggerThumbnailCacheCleanup()
    }
  } catch (err) {
    console.warn('Disk thumbnail cache write failed for', imagePath, err)
  }
}

async function collectCachedThumbFiles(dirPath: string): Promise<CachedThumbFile[]> {
  let entries: Dirent[]
  try {
    entries = await readdir(dirPath, { withFileTypes: true })
  } catch (err) {
    if (isErrnoCode(err, 'ENOENT')) return []
    throw err
  }

  const files: CachedThumbFile[] = []
  for (const entry of entries) {
    const fullPath = join(dirPath, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await collectCachedThumbFiles(fullPath)))
      continue
    }
    if (!entry.isFile() || !entry.name.endsWith('.jpg')) continue
    try {
      const fileStat = await stat(fullPath)
      files.push({ path: fullPath, size: fileStat.size, mtimeMs: fileStat.mtimeMs })
    } catch (err) {
      if (!isErrnoCode(err, 'ENOENT')) throw err
    }
  }
  return files
}

async function runThumbnailCacheCleanup(): Promise<void> {
  const cfg = await getThumbnailCacheConfig()
  const maxBytes = cfg.maxGB * 1024 * 1024 * 1024
  if (!Number.isFinite(maxBytes) || maxBytes <= 0) return

  const files = await collectCachedThumbFiles(cfg.directory)
  let totalBytes = files.reduce((sum, f) => sum + f.size, 0)
  if (totalBytes <= maxBytes) return

  files.sort((a, b) => a.mtimeMs - b.mtimeMs)
  for (const file of files) {
    if (totalBytes <= maxBytes) break
    try {
      await unlink(file.path)
      totalBytes -= file.size
    } catch (err) {
      if (!isErrnoCode(err, 'ENOENT')) throw err
    }
  }
}

export function triggerThumbnailCacheCleanup(): Promise<void> {
  if (cleanupPromise) return cleanupPromise
  cleanupPromise = runThumbnailCacheCleanup()
    .catch((err) => {
      console.warn('Thumbnail cache cleanup failed:', err)
    })
    .finally(() => {
      cleanupPromise = null
    })
  return cleanupPromise
}

export async function getThumbnail(imagePath: string): Promise<string> {
  const cached = thumbCacheGet(imagePath)
  if (cached !== undefined) return cached
  if (inFlight.has(imagePath)) return inFlight.get(imagePath)!

  const promise = (async () => {
    await acquireSlot()
    try {
      const diskCached = await readThumbnailFromDisk(imagePath)
      if (diskCached) {
        thumbCacheSet(imagePath, diskCached)
        return diskCached
      }

      const result = await rpc.call('thumbnail', { imagePath }) as { data: string }
      const dataUrl = `data:image/jpeg;base64,${result.data}`
      thumbCacheSet(imagePath, dataUrl)
      void writeThumbnailToDisk(imagePath, result.data)
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

/**
 * Return a 1024px cached image (from the coordinator's decoded image store)
 * as a data URL, or empty string if not available.
 *
 * This is Tier 2 in the three-tier lightbox:
 * thumbnail (400×300 blurred) → cached (1024px sharp) → full-res (3840px).
 */
export async function getCachedImage(imagePath: string): Promise<string> {
  try {
    const result = await rpc.call('cachedimage', { imagePath }) as {
      available: boolean
      data?: string
      width?: number
      height?: number
    }
    if (result?.available && result.data) {
      return `data:image/jpeg;base64,${result.data}`
    }
    return ''
  } catch {
    return ''
  }
}
