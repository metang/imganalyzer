export const ALL_MODULE_KEYS = [
  'metadata',
  'technical',
  'caption',
  'objects',
  'faces',
  'perception',
  'embedding',
] as const

export type ModuleKey = (typeof ALL_MODULE_KEYS)[number]

export interface ModuleDisplayMetadata {
  label: string
  progressLabel: string
  resultLabel: string
  passSelectorLabel: string
  passSelectorNote?: string
}

export const MODULE_METADATA = {
  metadata: {
    label: 'Metadata',
    progressLabel: 'Metadata',
    resultLabel: 'metadata',
    passSelectorLabel: 'Metadata (EXIF / GPS / IPTC)',
  },
  technical: {
    label: 'Technical',
    progressLabel: 'Technical',
    resultLabel: 'technical',
    passSelectorLabel: 'Technical (sharpness, exposure, noise)',
  },
  caption: {
    label: 'Caption',
    progressLabel: 'Caption',
    resultLabel: 'caption',
    passSelectorLabel: 'Caption & Keywords (Qwen 3.5)',
  },
  objects: {
    label: 'Objects',
    progressLabel: 'Objects (DINO)',
    resultLabel: 'objects',
    passSelectorLabel: 'Object Detection (GroundingDINO)',
  },
  faces: {
    label: 'Faces',
    progressLabel: 'Faces',
    resultLabel: 'faces',
    passSelectorLabel: 'Face Recognition (InsightFace)',
    passSelectorNote: 'requires objects',
  },
  perception: {
    label: 'Perception',
    progressLabel: 'Perception',
    resultLabel: 'perception',
    passSelectorLabel: 'Perception (UniPercept)',
  },
  embedding: {
    label: 'Embeddings',
    progressLabel: 'Embeddings',
    resultLabel: 'embedding',
    passSelectorLabel: 'Embeddings',
  },
} as const satisfies Record<ModuleKey, ModuleDisplayMetadata>

export const PASS_SELECTOR_MODULE_KEYS = [
  'metadata',
  'technical',
  'objects',
  'faces',
  'caption',
  'perception',
  'embedding',
] as const satisfies readonly ModuleKey[]

export const DEFAULT_PASS_SELECTOR_KEYS = PASS_SELECTOR_MODULE_KEYS

export const WORKER_THREAD_POOL_MODULE_KEYS = [
  'metadata',
  'technical',
] as const satisfies readonly ModuleKey[]

export interface PassSelectorModule {
  uiKey: ModuleKey
  moduleKey: ModuleKey
  label: string
  note?: string
}

export const PASS_SELECTOR_MODULES: readonly PassSelectorModule[] = PASS_SELECTOR_MODULE_KEYS.map(
  (key) => {
    const metadata = MODULE_METADATA[key]
    return {
      uiKey: key,
      moduleKey: key,
      label: metadata.passSelectorLabel,
      ...(metadata.passSelectorNote ? { note: metadata.passSelectorNote } : {}),
    }
  },
)

export const LEGACY_MODULE_KEYS = [
  'blip2',
  'cloud_ai',
  'local_ai',
  'aesthetic',
  'ocr',
] as const

export type LegacyModuleKey = (typeof LEGACY_MODULE_KEYS)[number]

export const LEGACY_RETRY_MODULE_ALIASES: Partial<Record<LegacyModuleKey, ModuleKey>> = {
  blip2: 'caption',
  cloud_ai: 'caption',
  local_ai: 'caption',
  aesthetic: 'perception',
}

export function isModuleKey(module: string): module is ModuleKey {
  return (ALL_MODULE_KEYS as readonly string[]).includes(module)
}

export function getModuleProgressLabel(module: string): string {
  return isModuleKey(module) ? MODULE_METADATA[module].progressLabel : module
}

export function getModuleResultLabel(module: string): string {
  return isModuleKey(module) ? MODULE_METADATA[module].resultLabel : module
}

export function formatModuleResultLabelList(modules: readonly ModuleKey[]): string {
  return modules.map((module) => MODULE_METADATA[module].resultLabel).join(', ')
}

export function resolveRetryModuleKey(module: string): string {
  return LEGACY_RETRY_MODULE_ALIASES[module as LegacyModuleKey] ?? module
}
