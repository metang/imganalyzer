const readline = require('node:readline')
const path = require('node:path')

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
})

const fakeImagePath = path.join(process.cwd(), 'e2e', 'fixtures', 'fake-image.jpg')
const onePixelImageBase64 = 'R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=='
const onePixelImageDataUrl = `data:image/gif;base64,${onePixelImageBase64}`

const JANUARY_FOLDER = 'E:/Pic/2013/01'
const CURRENT_FOLDER = 'E:/Pic/2013/02-current'
const SLOW_STALE_FOLDER = 'E:/Pic/2013/03-slow-stale'
const SLOW_FAILURE_FOLDER = 'E:/Pic/2013/04-slow-failure'
const ALL_IMAGES_TOTAL = 474186
const SLOW_DELAY_MS = 450

class RpcError extends Error {
  constructor(code, message) {
    super(message)
    this.code = code
  }
}

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`)
}

function result(id, value) {
  send({ jsonrpc: '2.0', id, result: value })
}

function sendError(id, code, message) {
  send({ jsonrpc: '2.0', id, error: { code, message } })
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function emptyStatus() {
  return {
    total_images: 0,
    modules: {},
    totals: { pending: 0, running: 0, done: 0, failed: 0, skipped: 0 },
    remaining_images: 0,
    nodes: {
      master: {
        id: 'master',
        role: 'master',
        displayName: 'Fake master',
        platform: process.platform,
        runningJobs: 0,
        activeModules: [],
      },
      workers: [],
    },
    recent_results: [],
    pre_decode: { done: 0, failed: 0, total: 0, running: false },
  }
}

function fakeSearchResult(overrides = {}) {
  return {
    image_id: 1,
    file_path: fakeImagePath,
    score: 1,
    width: 100,
    height: 80,
    file_size: 1234,
    camera_make: 'FakeCam',
    camera_model: 'E2E',
    lens_model: null,
    focal_length: null,
    f_number: null,
    exposure_time: null,
    iso: null,
    date_time_original: '2024-01-01T00:00:00Z',
    gps_latitude: null,
    gps_longitude: null,
    location_city: null,
    location_state: null,
    location_country: null,
    sharpness_score: null,
    sharpness_label: null,
    exposure_ev: null,
    exposure_label: null,
    noise_level: null,
    noise_label: null,
    snr_db: null,
    dynamic_range_stops: null,
    highlight_clipping_pct: null,
    shadow_clipping_pct: null,
    avg_saturation: null,
    dominant_colors: [],
    description: 'Result served by the fake JSON-RPC backend',
    scene_type: 'test',
    main_subject: 'hermetic smoke test',
    lighting: null,
    mood: null,
    keywords: ['fake', 'e2e'],
    detected_objects: [],
    face_count: 0,
    face_identities: [],
    face_clusters: [],
    has_people: false,
    ocr_text: null,
    cloud_description: null,
    aesthetic_score: null,
    aesthetic_label: null,
    aesthetic_reason: null,
    perception_iaa: null,
    perception_iaa_label: null,
    perception_iqa: null,
    perception_iqa_label: null,
    perception_ista: null,
    perception_ista_label: null,
    ...overrides,
  }
}

function folder(pathValue, name, imageCount, childCount = 0) {
  return {
    path: pathValue,
    name,
    parent_path: null,
    depth: 0,
    image_count: imageCount,
    child_count: childCount,
  }
}

const largeArchiveFolders = Array.from({ length: 1200 }, (_, index) => {
  const suffix = String(index + 1).padStart(4, '0')
  return folder(`E:/Pic/archive-${suffix}`, `Archive ${suffix}`, (index % 17) + 1)
})

const galleryFolders = [
  folder(JANUARY_FOLDER, '01', 863),
  folder(CURRENT_FOLDER, '02 Current', 2),
  folder(SLOW_STALE_FOLDER, '03 Slow Stale', 99),
  folder(SLOW_FAILURE_FOLDER, '04 Slow Failure', 42),
  ...largeArchiveFolders,
]

function fakeGalleryFolders() {
  return galleryFolders
}

function galleryChunk(items, total) {
  return {
    items,
    nextCursor: null,
    hasMore: false,
    total,
  }
}

async function fakeGalleryChunk(params) {
  const folderPath = params?.folderPath ?? null

  if (folderPath === SLOW_FAILURE_FOLDER) {
    await delay(SLOW_DELAY_MS)
    throw new RpcError(-32000, 'gallery/listImagesChunk timed out for fake slow folder')
  }

  if (folderPath === SLOW_STALE_FOLDER) {
    await delay(SLOW_DELAY_MS)
    return galleryChunk([
      fakeSearchResult({
        image_id: 303,
        description: 'Obsolete slow gallery result that must never replace the current folder',
      }),
    ], 99)
  }

  if (folderPath === CURRENT_FOLDER) {
    return galleryChunk([
      fakeSearchResult({
        image_id: 202,
        description: 'Current fast gallery result',
      }),
    ], 2)
  }

  if (folderPath === JANUARY_FOLDER) {
    return galleryChunk([
      fakeSearchResult({
        image_id: 101,
        description: 'January gallery result',
      }),
    ], 863)
  }

  return galleryChunk([fakeSearchResult()], ALL_IMAGES_TOTAL)
}

function fakeThumbnailBatch(params) {
  const thumbnails = {}
  const items = Array.isArray(params?.items) ? params.items : []
  for (const item of items) {
    const key = item?.file_path || (item?.image_id != null ? String(item.image_id) : null)
    if (key) thumbnails[key] = onePixelImageDataUrl
  }
  return { thumbnails, errors: {} }
}

async function handle(method, params) {
  switch (method) {
    case 'status':
      return emptyStatus()
    case 'workers/list':
      return { workers: [] }
    case 'gallery/listFolders':
      return { folders: fakeGalleryFolders(), totalImages: ALL_IMAGES_TOTAL }
    case 'gallery/listImagesChunk':
      return fakeGalleryChunk(params)
    case 'search':
      send({
        jsonrpc: '2.0',
        method: 'search/progress',
        params: { phase: 'fake', message: 'Fake backend search complete', progress: 1 },
      })
      return { results: [fakeSearchResult()], total: 1, hasMore: false }
    case 'search/warmup':
      return { ok: true }
    case 'search/resolve-face-query':
    case 'search/resolveFaceQuery':
      return { face: null, faces: [], faceMatch: 'all', remainingQuery: params?.query ?? '' }
    case 'image/details':
      return { result: fakeSearchResult() }
    case 'thumbnail':
      return { data: onePixelImageBase64 }
    case 'thumbnails/batch':
      return fakeThumbnailBatch(params)
    case 'cachedimage':
      return { available: true, data: onePixelImageBase64, width: 1, height: 1 }
    case 'fullimage':
      return { native: false, data: onePixelImageBase64 }
    case 'geo/stats':
      return { total_images: 0, geotagged: 0, countries: [], top_cities: [] }
    case 'geo/clusters':
      return { clusters: [], total: 0 }
    case 'geo/nearby':
      return { images: [], total: 0 }
    case 'geo/heatmap':
      return { points: [] }
    case 'geo/cluster-preview':
      return { images: [], total: 0 }
    case 'geo/stats-extended':
      return {
        total_images: 0,
        geotagged: 0,
        gps_sources: [],
        countries: [],
        top_cities: [],
        monthly_activity: [],
        location_diversity: [],
        camera_by_country: [],
        top_locations: [],
        furthest_from_home: null,
      }
    case 'geo/gap-filler-preview':
      return { fillable: 0, total_missing: 0, previews: [] }
    case 'geo/gap-filler-apply':
      return { filled: 0, skipped_override: 0, skipped_low_confidence: 0 }
    case 'geo/trip-detect':
      return { trips: [] }
    case 'geo/trip-timeline':
      return { stops: [], route_points: [], total_images: 0 }
    case 'geo/geocode':
      return { lat: null, lng: null, count: 0 }
    case 'faces/clusters':
      return { clusters: [], has_occurrences: false, total_count: 0, deferred_cluster_ids: [] }
    case 'faces/list':
      return { faces: [] }
    case 'faces/persons':
      return { persons: [] }
    case 'faces/images':
      return { images: [] }
    case 'faces/cluster-images':
      return { occurrences: [] }
    case 'faces/person-clusters':
      return { clusters: [] }
    case 'faces/person-link-suggestions':
    case 'faces/cluster-link-suggestions':
      return { suggestions: [] }
    case 'faces/person-similar-images':
      return { images: [] }
    case 'faces/person-direct-links':
      return { links: [] }
    case 'faces/crop':
      return { data: null }
    case 'faces/crop-batch':
      return { thumbnails: {} }
    case 'albums/list':
      return { albums: [] }
    case 'albums/presets':
      return { presets: {} }
    case 'albums/story':
      return { chapters: [] }
    case 'albums/chapter/moments':
      return { moments: [] }
    case 'albums/moment/images':
      return { images: [] }
    case 'albums/check-new':
      return { added_to_albums: [] }
    case 'albums/refresh':
      return { item_count: 0 }
    case 'albums/story/generate':
    case 'albums/story/generate-narrative':
      return { chapters_updated: 0 }
    case 'albums/export':
      return { path: '' }
    case 'queue_clear':
      return { deleted: 0 }
    case 'ingest':
      return { registered: 0, enqueued: 0, skipped: 0 }
    case 'run':
      send({ jsonrpc: '2.0', method: 'run/done', params: { paused: false } })
      return { ok: true }
    case 'cancel_run':
    case 'rebuild':
    case 'workers/pause':
    case 'workers/resume':
    case 'workers/remove':
      return { ok: true }
    case 'shutdown':
      return { ok: true }
    default:
      throw new RpcError(-32601, `Method not found: ${method}`)
  }
}

send({ jsonrpc: '2.0', method: 'server/ready', params: { fake: true } })

async function handleLine(line) {
  if (!line.trim()) return

  let request
  try {
    request = JSON.parse(line)
  } catch {
    sendError(null, -32700, 'Parse error')
    return
  }

  if (typeof request.id === 'undefined') return

  if (typeof request.method !== 'string') {
    sendError(request.id, -32600, 'Invalid Request')
    return
  }

  try {
    const value = await handle(request.method, request.params ?? {})
    result(request.id, value)
    if (request.method === 'shutdown') {
      setTimeout(() => process.exit(0), 10)
    }
  } catch (err) {
    if (err instanceof RpcError) {
      sendError(request.id, err.code, err.message)
      return
    }
    sendError(request.id, -32603, err instanceof Error ? err.message : String(err))
  }
}

rl.on('line', (line) => {
  void handleLine(line)
})

rl.on('close', () => process.exit(0))
process.on('SIGTERM', () => process.exit(0))
process.on('SIGINT', () => process.exit(0))
