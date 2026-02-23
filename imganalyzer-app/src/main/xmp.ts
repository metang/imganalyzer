import { XMLParser } from 'fast-xml-parser'

export interface XmpData {
  // AI
  description?: string
  sceneType?: string
  mainSubject?: string
  lighting?: string
  mood?: string
  aestheticScore?: number
  aestheticLabel?: string
  faceCount?: number
  faceIdentities?: string[]
  faceDetails?: string[]
  detectedObjects?: string[]
  keywords?: string[]
  // Technical
  sharpnessScore?: number
  sharpnessLabel?: string
  exposureEV?: number
  exposureLabel?: string
  noiseLevel?: number
  noiseLabel?: string
  snrDb?: number
  dynamicRangeStops?: number
  highlightClippingPct?: number
  shadowClippingPct?: number
  avgSaturation?: number
  warmCoolRatio?: number
  dominantColors?: string[]
  // Camera / metadata
  cameraMake?: string
  cameraModel?: string
  lens?: string
  fNumber?: string
  exposureTime?: string
  focalLength?: string
  iso?: string
  createDate?: string
  imageWidth?: number
  imageHeight?: number
  gpsLatitude?: string
  gpsLongitude?: string
  locationCity?: string
  locationState?: string
  locationCountry?: string
}

function attr(desc: Record<string, unknown>, key: string): string | undefined {
  const v = desc[key]
  return v !== undefined ? String(v) : undefined
}

function bagItems(desc: Record<string, unknown>, key: string): string[] {
  const val = desc[key] as Record<string, unknown> | undefined
  if (!val) return []
  const bag = val['rdf:Bag'] as Record<string, unknown> | undefined
  if (!bag) return []
  const items = bag['rdf:li']
  if (!items) return []
  if (Array.isArray(items)) return items.map(String)
  return [String(items)]
}

function seqItems(desc: Record<string, unknown>, key: string): string[] {
  const val = desc[key] as Record<string, unknown> | undefined
  if (!val) return []
  const seq = val['rdf:Seq'] as Record<string, unknown> | undefined
  if (!seq) return []
  const items = seq['rdf:li']
  if (!items) return []
  if (Array.isArray(items)) return items.map(String)
  return [String(items)]
}

function dcDescription(desc: Record<string, unknown>): string | undefined {
  try {
    const dc = desc['dc:description'] as Record<string, unknown>
    const alt = dc['rdf:Alt'] as Record<string, unknown>
    const li = alt['rdf:li']
    if (Array.isArray(li)) {
      const first = li[0]
      return typeof first === 'object' && first !== null ? (first as Record<string, string>)['#text'] : String(first)
    }
    if (typeof li === 'object' && li !== null) {
      return (li as Record<string, string>)['#text']
    }
    return String(li)
  } catch {
    return undefined
  }
}

export function parseXmp(xml: string): XmpData {
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: '',
    parseAttributeValue: false,
    allowBooleanAttributes: true,
    trimValues: true
  })

  let parsed: Record<string, unknown>
  try {
    parsed = parser.parse(xml)
  } catch {
    return {}
  }

  const rdf = (parsed['x:xmpmeta'] as Record<string, unknown>)?.['rdf:RDF'] as Record<string, unknown>
  if (!rdf) return {}
  const desc = rdf['rdf:Description'] as Record<string, unknown>
  if (!desc) return {}

  const result: XmpData = {}

  // Description
  try { result.description = dcDescription(desc) } catch {}

  // Keywords
  result.keywords = bagItems(desc, 'dc:subject')

  // AI fields
  const sceneType = attr(desc, 'imganalyzer:AISceneType')
  if (sceneType) result.sceneType = sceneType
  const mainSubject = attr(desc, 'imganalyzer:AIMainSubject')
  if (mainSubject) result.mainSubject = mainSubject
  const lighting = attr(desc, 'imganalyzer:AILighting')
  if (lighting) result.lighting = lighting
  const mood = attr(desc, 'imganalyzer:AIMood')
  if (mood) result.mood = mood

  const aScore = attr(desc, 'imganalyzer:AestheticScore')
  if (aScore) result.aestheticScore = parseFloat(aScore)
  const aLabel = attr(desc, 'imganalyzer:AestheticLabel')
  if (aLabel) result.aestheticLabel = aLabel

  const fCount = attr(desc, 'imganalyzer:FaceCount')
  if (fCount) result.faceCount = parseInt(fCount)
  result.faceIdentities = bagItems(desc, 'imganalyzer:FaceIdentities')
  result.faceDetails = bagItems(desc, 'imganalyzer:FaceDetails')
  result.detectedObjects = bagItems(desc, 'imganalyzer:AIDetectedObjects')

  // Technical
  const sh = attr(desc, 'imganalyzer:SharpnessScore')
  if (sh) result.sharpnessScore = parseFloat(sh)
  result.sharpnessLabel = attr(desc, 'imganalyzer:SharpnessLabel')
  const ev = attr(desc, 'imganalyzer:ExposureEV')
  if (ev) result.exposureEV = parseFloat(ev)
  result.exposureLabel = attr(desc, 'imganalyzer:ExposureLabel')
  const nl = attr(desc, 'imganalyzer:NoiseLevel')
  if (nl) result.noiseLevel = parseFloat(nl)
  result.noiseLabel = attr(desc, 'imganalyzer:NoiseLabel')
  const snr = attr(desc, 'imganalyzer:SNR_dB')
  if (snr) result.snrDb = parseFloat(snr)
  const dr = attr(desc, 'imganalyzer:DynamicRangeStops')
  if (dr) result.dynamicRangeStops = parseFloat(dr)
  const hlc = attr(desc, 'imganalyzer:HighlightClippingPct')
  if (hlc) result.highlightClippingPct = parseFloat(hlc)
  const shc = attr(desc, 'imganalyzer:ShadowClippingPct')
  if (shc) result.shadowClippingPct = parseFloat(shc)
  const sat = attr(desc, 'imganalyzer:AvgSaturation')
  if (sat) result.avgSaturation = parseFloat(sat)
  const wcr = attr(desc, 'imganalyzer:WarmCoolRatio')
  if (wcr) result.warmCoolRatio = parseFloat(wcr)
  result.dominantColors = seqItems(desc, 'imganalyzer:DominantColors')

  // Camera metadata
  result.cameraMake = attr(desc, 'tiff:Make')
  result.cameraModel = attr(desc, 'tiff:Model')
  result.lens = attr(desc, 'photoshop:Lens')
  result.fNumber = attr(desc, 'exif:FNumber')
  result.exposureTime = attr(desc, 'exif:ExposureTime')
  result.focalLength = attr(desc, 'exif:FocalLength')
  result.createDate = attr(desc, 'xmp:CreateDate')
  const w = attr(desc, 'tiff:ImageWidth')
  if (w) result.imageWidth = parseInt(w)
  const h = attr(desc, 'tiff:ImageHeight')
  if (h) result.imageHeight = parseInt(h)
  result.gpsLatitude = attr(desc, 'exif:GPSLatitude')
  result.gpsLongitude = attr(desc, 'exif:GPSLongitude')
  result.locationCity = attr(desc, 'Iptc4xmpCore:Location')
  result.locationState = attr(desc, 'Iptc4xmpCore:ProvinceState')
  result.locationCountry = attr(desc, 'Iptc4xmpCore:CountryName')

  // ISO from rdf:Seq
  const isoItems = seqItems(desc, 'exif:ISOSpeedRatings')
  if (isoItems.length > 0) result.iso = isoItems[0]

  return result
}
