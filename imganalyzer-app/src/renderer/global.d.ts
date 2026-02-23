// Shared type declarations for renderer — mirrors main-process interfaces
// without importing across the process boundary.

export interface XmpData {
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

export interface AnalysisProgress {
  imagePath: string
  stage: string
  pct: number
}

declare global {
  interface Window {
    api: {
      openFolder(): Promise<string | null>
      listImages(folderPath: string): Promise<ImageFile[]>
      getThumbnail(imagePath: string): Promise<string>
      readXmp(imagePath: string): Promise<XmpData | null>
      runAnalysis(imagePath: string, aiBackend: string): Promise<{ xmp: XmpData | null; error?: string }>
      cancelAnalysis(imagePath: string): Promise<void>
      onAnalysisProgress(cb: (p: AnalysisProgress) => void): () => void
    }
  }
}
