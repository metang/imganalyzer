import type { ModuleKey } from '../hooks/useBatchProcess'

// ── Pass definitions ──────────────────────────────────────────────────────────

/**
 * A UI pass row.  `uiKey` is unique per row and used for checkbox state.
 * `moduleKey` is the CLI key it maps to (multiple rows may share one moduleKey).
 */
interface PassDef {
  label: string
  uiKey: string
  moduleKey: ModuleKey
  note?: string
}

const PASSES: PassDef[] = [
  { label: 'Metadata (EXIF / GPS / IPTC)',           uiKey: 'metadata',   moduleKey: 'metadata'  },
  { label: 'Technical (sharpness, exposure, noise)',  uiKey: 'technical',  moduleKey: 'technical' },
  { label: 'Caption & Scene (BLIP-2)',                uiKey: 'caption',    moduleKey: 'local_ai'  },
  { label: 'Object Detection (GroundingDINO)',        uiKey: 'objects',    moduleKey: 'local_ai'  },
  { label: 'OCR / Text (TrOCR)',                      uiKey: 'ocr',        moduleKey: 'local_ai'  },
  { label: 'Face Recognition (InsightFace)',          uiKey: 'faces',      moduleKey: 'local_ai'  },
  { label: 'Cloud AI',                                uiKey: 'cloud_ai',   moduleKey: 'cloud_ai',  note: 'requires local_ai' },
  { label: 'Aesthetic Score',                         uiKey: 'aesthetic',  moduleKey: 'aesthetic', note: 'requires local_ai' },
  { label: 'Embeddings',                              uiKey: 'embedding',  moduleKey: 'embedding' },
]

const CLOUD_PROVIDERS = [
  { value: 'copilot',   label: 'GitHub Copilot' },
  { value: 'openai',    label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
]

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * selectedKeys holds unique UI keys (one per row).
 * Call `resolveModuleKeys(selectedKeys)` to get the deduplicated CLI module keys.
 */
export interface PassSelectorValue {
  selectedKeys: Set<string>
  workers: number
  cloudWorkers: number
  cloudProvider: string
  recursive: boolean
  noHash: boolean
}

/**
 * Convert the UI-level selectedKeys set to the deduplicated CLI module key array.
 * e.g. if any of caption/objects/ocr/faces is selected, 'local_ai' is included once.
 */
export function resolveModuleKeys(selectedKeys: Set<string>): ModuleKey[] {
  const result = new Set<ModuleKey>()
  for (const pass of PASSES) {
    if (selectedKeys.has(pass.uiKey)) {
      result.add(pass.moduleKey)
    }
  }
  return [...result]
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  value: PassSelectorValue
  onChange(value: PassSelectorValue): void
  disabled?: boolean
}

export function PassSelector({ value, onChange, disabled }: Props) {
  const { selectedKeys, workers, cloudWorkers, cloudProvider, recursive, noHash } = value

  const toggleKey = (uiKey: string) => {
    const next = new Set(selectedKeys)
    if (next.has(uiKey)) next.delete(uiKey)
    else next.add(uiKey)
    onChange({ ...value, selectedKeys: next })
  }

  return (
    <div className="flex flex-col gap-5">

      {/* ── Pass checkboxes ─────────────────────────────────────────────────── */}
      <fieldset disabled={disabled} className="flex flex-col gap-1.5">
        <legend className="text-xs font-semibold text-neutral-400 uppercase tracking-wider mb-2">
          Analysis passes
        </legend>
        {PASSES.map((pass) => {
          const checked = selectedKeys.has(pass.uiKey)
          return (
            <label
              key={pass.uiKey}
              className="flex items-center gap-2.5 cursor-pointer group select-none"
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggleKey(pass.uiKey)}
                className="
                  w-4 h-4 rounded accent-blue-500 cursor-pointer
                  disabled:cursor-not-allowed disabled:opacity-50
                "
              />
              <span className="text-sm text-neutral-300 group-hover:text-neutral-100 transition-colors">
                {pass.label}
              </span>
              {pass.note && (
                <span className="text-xs text-neutral-600 italic">({pass.note})</span>
              )}
            </label>
          )
        })}
      </fieldset>

      {/* ── Options ─────────────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-3">
        <p className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Options</p>

        {/* Local workers slider — metadata & technical */}
        <div className="flex items-center gap-3">
          <label className="text-sm text-neutral-300 w-36 shrink-0">
            Local workers
            <span className="block text-xs text-neutral-600 font-normal">metadata, technical</span>
          </label>
          <input
            type="range"
            min={1}
            max={16}
            step={1}
            value={workers}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, workers: Number(e.target.value) })}
            className="flex-1 accent-blue-500 disabled:opacity-50"
          />
          <span className="text-sm text-neutral-300 w-5 text-right">{workers}</span>
        </div>

        {/* Cloud workers slider — cloud_ai & aesthetic */}
        <div className="flex items-center gap-3">
          <label className="text-sm text-neutral-300 w-36 shrink-0">
            Cloud workers
            <span className="block text-xs text-neutral-600 font-normal">cloud_ai, aesthetic</span>
          </label>
          <input
            type="range"
            min={1}
            max={16}
            step={1}
            value={cloudWorkers}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, cloudWorkers: Number(e.target.value) })}
            className="flex-1 accent-blue-500 disabled:opacity-50"
          />
          <span className="text-sm text-neutral-300 w-5 text-right">{cloudWorkers}</span>
        </div>

        {/* Cloud provider dropdown */}
        <div className="flex items-center gap-3">
          <label className="text-sm text-neutral-300 w-36 shrink-0">Cloud provider</label>
          <select
            value={cloudProvider}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, cloudProvider: e.target.value })}
            className="
              flex-1 px-2 py-1 rounded bg-neutral-800 border border-neutral-700
              text-sm text-neutral-200 focus:outline-none focus:border-blue-500
              disabled:opacity-50
            "
          >
            {CLOUD_PROVIDERS.map((p) => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>

        {/* Recursive toggle */}
        <label className="flex items-center gap-2.5 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={recursive}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, recursive: e.target.checked })}
            className="w-4 h-4 rounded accent-blue-500 disabled:opacity-50"
          />
          <span className="text-sm text-neutral-300">Recursive (include subfolders)</span>
        </label>

        {/* Skip hash toggle */}
        <label className="flex items-center gap-2.5 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={noHash}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, noHash: e.target.checked })}
            className="w-4 h-4 rounded accent-blue-500 disabled:opacity-50"
          />
          <span className="text-sm text-neutral-300">Skip hash computation (faster ingest)</span>
        </label>
      </div>
    </div>
  )
}

/** Default value — all passes enabled, 2 local workers, 4 cloud workers. */
export function defaultPassSelectorValue(): PassSelectorValue {
  return {
    selectedKeys: new Set<string>([
      'metadata', 'technical', 'caption', 'objects', 'ocr', 'faces',
      'cloud_ai', 'aesthetic', 'embedding',
    ]),
    workers: 2,
    cloudWorkers: 4,
    cloudProvider: 'copilot',
    recursive: true,
    noHash: false,
  }
}
