/**
 * thumbnailCache.ts — Shared renderer-side thumbnail cache.
 *
 * Centralises thumbnail storage across FacesView, Thumbnail, VirtualGrid (and
 * any other consumers) so base64 data URIs are not duplicated across multiple
 * module-level caches (see perf bug F3).
 *
 * Key ideas:
 *   • Each cache entry holds a blob URL (~60 bytes) rather than a full base64
 *     data URI (~30–50 KB). The binary payload lives once in a Blob.
 *   • Two LRU caches: one for regular image thumbnails (keyed by file path or
 *     image_id) and one for face crops (keyed by occurrence_id). The RPC
 *     endpoints differ, so they are fetched through separate batchers.
 *   • Requests issued within BATCH_WINDOW_MS are coalesced into a single RPC
 *     call. All callers (Thumbnail.tsx, VirtualGrid.tsx, FacesView.tsx…) share
 *     the same in-flight map, so duplicate requests never race.
 *   • On LRU eviction and on `clearThumbCache()` we revoke the blob URL via
 *     `URL.revokeObjectURL`. The cache is sized generously (3000 per kind) so
 *     entries still visible on screen are never evicted in practice.
 */

type ThumbCallback = (src: string | null) => void

interface LruEntry {
  url: string
}

class BlobUrlLru<K> {
  private readonly map = new Map<K, LruEntry>()
  private readonly max: number

  constructor(max: number) {
    this.max = max
  }

  get(key: K): string | undefined {
    const entry = this.map.get(key)
    if (!entry) return undefined
    // Touch — move to end (MRU).
    this.map.delete(key)
    this.map.set(key, entry)
    return entry.url
  }

  has(key: K): boolean {
    return this.map.has(key)
  }

  set(key: K, url: string): void {
    const existing = this.map.get(key)
    if (existing) {
      if (existing.url === url) {
        // Same URL — just touch (move to MRU end).
        this.map.delete(key)
        this.map.set(key, existing)
        return
      }
      // Different URL for the same key — keep the existing one rather than
      // revoking it, because cells are likely already rendering <img src=…>
      // pointing at the existing URL. Revoking it mid-load would break those
      // images. Drop the new (duplicate) blob URL instead.
      revokeBlobUrl(url)
      this.map.delete(key)
      this.map.set(key, existing)
      return
    }
    this.map.set(key, { url })
    this.evictIfNeeded()
  }

  private evictIfNeeded(): void {
    while (this.map.size > this.max) {
      const oldestKey = this.map.keys().next().value as K | undefined
      if (oldestKey === undefined) break
      const entry = this.map.get(oldestKey)
      this.map.delete(oldestKey)
      if (entry) revokeBlobUrl(entry.url)
    }
  }

  clear(): void {
    for (const entry of this.map.values()) revokeBlobUrl(entry.url)
    this.map.clear()
  }
}

function revokeBlobUrl(url: string): void {
  if (url.startsWith('blob:')) {
    try { URL.revokeObjectURL(url) } catch { /* ignore */ }
  }
}

// ── Base64 → Blob URL conversion ─────────────────────────────────────────────

function base64ToBlob(base64: string, mime = 'image/jpeg'): Blob {
  const binary = atob(base64)
  const len = binary.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i)
  return new Blob([bytes], { type: mime })
}

/**
 * Normalise a value returned by the backend into a blob URL whenever
 * possible. Accepts raw base64, `data:image/…;base64,…`, `blob:`, `http(s):`,
 * `file:` strings. Already-http/blob/file URLs are passed through untouched.
 * Returns null for empty / unparseable inputs.
 */
function toBlobUrl(raw: string | null | undefined): string | null {
  if (!raw) return null
  const trimmed = raw.trim()
  if (!trimmed) return null
  if (
    trimmed.startsWith('blob:')
    || trimmed.startsWith('http://')
    || trimmed.startsWith('https://')
    || trimmed.startsWith('file://')
  ) {
    return trimmed
  }
  let base64: string
  let mime = 'image/jpeg'
  if (trimmed.startsWith('data:')) {
    const commaIdx = trimmed.indexOf(',')
    if (commaIdx < 0) return null
    const header = trimmed.slice(5, commaIdx)
    const semiIdx = header.indexOf(';')
    if (semiIdx > 0) mime = header.slice(0, semiIdx) || mime
    base64 = trimmed.slice(commaIdx + 1)
  } else {
    base64 = trimmed
  }
  try {
    const blob = base64ToBlob(base64, mime)
    return URL.createObjectURL(blob)
  } catch {
    return null
  }
}

// ── Image thumbnail batcher ──────────────────────────────────────────────────

const BATCH_WINDOW_MS = 32
const IMAGE_CHUNK_SIZE = 24
const FACE_CHUNK_SIZE = 50
const IMAGE_CACHE_MAX = 3000
const FACE_CACHE_MAX = 3000

const imageCache = new BlobUrlLru<string>(IMAGE_CACHE_MAX)
const imagePendingCallbacks = new Map<string, ThumbCallback[]>()
const imagePendingItems = new Map<string, { file_path: string; image_id?: number }>()
// Tracks file paths whose batch RPC has been dispatched but not yet resolved.
// Used to dedupe a second `requestImageThumbnail` call (e.g. from a fallback
// timer or a re-mounted cell) into the in-flight request — without this, the
// duplicate request fires a second RPC, whose resolution would overwrite the
// already-displayed blob URL with a fresh one and revoke the original mid-load.
const imageInFlight = new Set<string>()
let imageBatchTimer: ReturnType<typeof setTimeout> | null = null

export function getCachedImageThumb(filePath: string): string | undefined {
  return imageCache.get(filePath)
}

/**
 * Drop pending callbacks and in-flight markers for paths the caller no longer
 * cares about (e.g. after switching folder/year filter). The RPCs themselves
 * can't be cancelled but their results will be ignored, freeing the in-flight
 * dedup so the same path can be requested fresh later, and preventing stale
 * thumbnails from polluting the LRU cache.
 *
 * Pass `null` to drop ALL pending image requests.
 */
export function cancelImageThumbnails(filePaths: Iterable<string> | null): void {
  if (filePaths === null) {
    for (const path of imageInFlight) imagePendingCallbacks.delete(path)
    imageInFlight.clear()
    imagePendingItems.clear()
    return
  }
  for (const path of filePaths) {
    imagePendingCallbacks.delete(path)
    imageInFlight.delete(path)
    imagePendingItems.delete(path)
  }
}

export function requestImageThumbnail(
  filePath: string,
  imageId: number | null | undefined,
  callback: ThumbCallback,
): void {
  const cached = imageCache.get(filePath)
  if (cached) {
    callback(cached)
    return
  }

  const cbs = imagePendingCallbacks.get(filePath) ?? []
  cbs.push(callback)
  imagePendingCallbacks.set(filePath, cbs)

  // Already dispatched and waiting — the callback above will fire when the
  // in-flight RPC resolves. No need to enqueue a duplicate batch entry.
  if (imageInFlight.has(filePath)) return

  imagePendingItems.set(
    filePath,
    imageId != null ? { file_path: filePath, image_id: imageId } : { file_path: filePath },
  )

  if (imageBatchTimer !== null) return
  imageBatchTimer = setTimeout(() => { void flushImageBatch() }, BATCH_WINDOW_MS)
}

async function flushImageBatch(): Promise<void> {
  imageBatchTimer = null
  if (imagePendingItems.size === 0) return

  const items = [...imagePendingItems.values()]
  imagePendingItems.clear()
  for (const item of items) imageInFlight.add(item.file_path)

  const chunks: Array<typeof items> = []
  for (let i = 0; i < items.length; i += IMAGE_CHUNK_SIZE) {
    chunks.push(items.slice(i, i + IMAGE_CHUNK_SIZE))
  }

  await Promise.all(chunks.map(async (chunk) => {
    try {
      const result = await window.api.getThumbnailsBatch(chunk)
      for (const item of chunk) {
        const raw = result[item.file_path]
          ?? (item.image_id != null ? result[String(item.image_id)] : undefined)
        const url = toBlobUrl(raw)
        // If cancelImageThumbnails() ran while we were awaiting, the path was
        // removed from imageInFlight. Drop the result rather than poisoning
        // the LRU cache with an entry no visible cell asked for.
        const cancelled = !imageInFlight.has(item.file_path)
        if (cancelled) {
          if (url) revokeBlobUrl(url)
          continue
        }
        if (url) imageCache.set(item.file_path, url)
        const cbs = imagePendingCallbacks.get(item.file_path) ?? []
        imagePendingCallbacks.delete(item.file_path)
        imageInFlight.delete(item.file_path)
        for (const cb of cbs) cb(url)
      }
    } catch {
      for (const item of chunk) {
        const cbs = imagePendingCallbacks.get(item.file_path) ?? []
        imagePendingCallbacks.delete(item.file_path)
        imageInFlight.delete(item.file_path)
        for (const cb of cbs) cb(null)
      }
    }
  }))
}

// ── Face crop thumbnail batcher ──────────────────────────────────────────────

const faceCache = new BlobUrlLru<number>(FACE_CACHE_MAX)
const facePendingCallbacks = new Map<number, ThumbCallback[]>()
const facePendingIds = new Set<number>()
const faceInFlight = new Set<number>()
let faceBatchTimer: ReturnType<typeof setTimeout> | null = null

export function getCachedFaceThumb(occurrenceId: number): string | undefined {
  return faceCache.get(occurrenceId)
}

export function requestFaceThumbnail(
  occurrenceId: number,
  callback: ThumbCallback,
): void {
  const cached = faceCache.get(occurrenceId)
  if (cached) {
    callback(cached)
    return
  }

  const cbs = facePendingCallbacks.get(occurrenceId) ?? []
  cbs.push(callback)
  facePendingCallbacks.set(occurrenceId, cbs)

  if (faceInFlight.has(occurrenceId)) return

  facePendingIds.add(occurrenceId)

  if (faceBatchTimer !== null) return
  faceBatchTimer = setTimeout(() => { void flushFaceBatch() }, BATCH_WINDOW_MS)
}

/** Seed the face-crop cache with inline base64 payloads returned by the
 *  faces/persons or faces/person-clusters endpoints. Avoids an extra RPC. */
export function primeFaceThumbCache(entries: Array<{ id: number; base64: string }>): void {
  for (const { id, base64 } of entries) {
    if (faceCache.has(id)) continue
    const url = toBlobUrl(base64)
    if (url) faceCache.set(id, url)
  }
}

async function flushFaceBatch(): Promise<void> {
  faceBatchTimer = null
  if (facePendingIds.size === 0) return

  const ids = [...facePendingIds]
  facePendingIds.clear()
  for (const id of ids) faceInFlight.add(id)

  const chunks: number[][] = []
  for (let i = 0; i < ids.length; i += FACE_CHUNK_SIZE) {
    chunks.push(ids.slice(i, i + FACE_CHUNK_SIZE))
  }

  await Promise.all(chunks.map(async (chunk) => {
    try {
      const result = await window.api.getFaceCropBatch(chunk)
      for (const id of chunk) {
        const b64 = result.thumbnails?.[String(id)]
        const url = b64 ? toBlobUrl(b64) : null
        if (url) faceCache.set(id, url)
        const cbs = facePendingCallbacks.get(id) ?? []
        facePendingCallbacks.delete(id)
        faceInFlight.delete(id)
        for (const cb of cbs) cb(url)
      }
    } catch {
      for (const id of chunk) {
        const cbs = facePendingCallbacks.get(id) ?? []
        facePendingCallbacks.delete(id)
        faceInFlight.delete(id)
        for (const cb of cbs) cb(null)
      }
    }
  }))
}

// ── Unified API (spec) ───────────────────────────────────────────────────────

/**
 * Promise-based helper. For `kind === 'face'` the id is an occurrence_id; for
 * `kind === 'image'` the id must be a file path string (typed any to keep the
 * call site terse — use `requestImageThumbnail` / `requestFaceThumbnail` for
 * stricter typing).
 */
export function requestThumb(
  id: number | string,
  kind: 'image' | 'face' = 'image',
): Promise<string | null> {
  return new Promise((resolve) => {
    if (kind === 'face') {
      if (typeof id !== 'number') { resolve(null); return }
      requestFaceThumbnail(id, resolve)
    } else {
      if (typeof id !== 'string') { resolve(null); return }
      requestImageThumbnail(id, undefined, resolve)
    }
  })
}

export function getCachedThumb(id: number | string): string | undefined {
  if (typeof id === 'number') return faceCache.get(id)
  return imageCache.get(id)
}

export function clearThumbCache(): void {
  imageCache.clear()
  faceCache.clear()
  imagePendingCallbacks.clear()
  imagePendingItems.clear()
  imageInFlight.clear()
  facePendingCallbacks.clear()
  facePendingIds.clear()
  faceInFlight.clear()
}
