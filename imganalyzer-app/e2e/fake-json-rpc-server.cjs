const readline = require('node:readline')
const path = require('node:path')

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
})

const fakeImagePath = path.join(process.cwd(), 'e2e', 'fixtures', 'fake-image.jpg')

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`)
}

function result(id, value) {
  send({ jsonrpc: '2.0', id, result: value })
}

function error(id, message) {
  send({ jsonrpc: '2.0', id, error: { code: -32603, message } })
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

function fakeSearchResult() {
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
  }
}

function handle(method, params) {
  switch (method) {
    case 'status':
      return emptyStatus()
    case 'workers/list':
      return { workers: [] }
    case 'gallery/listFolders':
      return { folders: [], totalImages: 0 }
    case 'gallery/listImagesChunk':
      return { items: [], nextCursor: null, hasMore: false, total: 0 }
    case 'search':
      send({
        jsonrpc: '2.0',
        method: 'search/progress',
        params: { phase: 'fake', message: 'Fake backend search complete', progress: 1 },
      })
      return { results: [fakeSearchResult()], total: 1, hasMore: false }
    case 'search/resolve-face-query':
    case 'search/resolveFaceQuery':
      return { face: null, faces: [], faceMatch: 'all', remainingQuery: params?.query ?? '' }
    case 'image/details':
      return { result: fakeSearchResult() }
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
      return { ok: true }
  }
}

send({ jsonrpc: '2.0', method: 'server/ready', params: { fake: true } })

rl.on('line', (line) => {
  if (!line.trim()) return

  let request
  try {
    request = JSON.parse(line)
  } catch {
    return
  }

  if (typeof request.id === 'undefined' || typeof request.method !== 'string') {
    return
  }

  try {
    const value = handle(request.method, request.params ?? {})
    result(request.id, value)
    if (request.method === 'shutdown') {
      setTimeout(() => process.exit(0), 10)
    }
  } catch (err) {
    error(request.id, err instanceof Error ? err.message : String(err))
  }
})

rl.on('close', () => process.exit(0))
process.on('SIGTERM', () => process.exit(0))
process.on('SIGINT', () => process.exit(0))
