import type { ModuleKey } from '../hooks/useBatchProcess'

// ── Pass definitions ──────────────────────────────────────────────────────────

/**
 * A UI pass row.  `uiKey` is unique per row and used for checkbox state.
 * `moduleKey` is the CLI key sent to the backend (one-to-one since the refactor).
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
  { label: 'Object Detection (GroundingDINO)',        uiKey: 'objects',    moduleKey: 'objects'   },
  { label: 'Face Recognition (InsightFace)',          uiKey: 'faces',      moduleKey: 'faces',     note: 'requires objects' },
  { label: 'Caption & Keywords (Qwen 3.5)',           uiKey: 'caption',    moduleKey: 'caption'  },
  { label: 'Perception (UniPercept)',                  uiKey: 'perception', moduleKey: 'perception' },
  { label: 'Embeddings',                              uiKey: 'embedding',  moduleKey: 'embedding' },
]

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * selectedKeys holds unique UI keys (one per row).
 * Call `resolveModuleKeys(selectedKeys)` to get the CLI module keys.
 */
export interface PassSelectorValue {
  selectedKeys: Set<string>
  workers: number
  recursive: boolean
  noHash: boolean
  forceReprocess: boolean
}

/**
 * Convert the UI-level selectedKeys set to the CLI module key array.
 * Each UI pass maps to a distinct module key since the split-pass refactor.
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
  const { selectedKeys, workers, recursive, noHash, forceReprocess } = value

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

        {/* Workers slider — metadata & technical thread pool */}
        <div className="flex items-center gap-3">
          <label className="text-sm text-neutral-300 w-36 shrink-0">
            Workers
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

        <label className="flex items-center gap-2.5 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={forceReprocess}
            disabled={disabled}
            onChange={(e) => onChange({ ...value, forceReprocess: e.target.checked })}
            className="w-4 h-4 rounded accent-blue-500 disabled:opacity-50"
          />
          <span className="text-sm text-neutral-300">
            Force reprocess selected passes (ignore existing analysis)
          </span>
        </label>
      </div>
    </div>
  )
}

/** Default value — all passes enabled, 2 workers. */
export function defaultPassSelectorValue(): PassSelectorValue {
  return {
    selectedKeys: new Set<string>([
      'metadata', 'technical', 'objects', 'faces',
      'caption', 'perception', 'embedding',
    ]),
    workers: 2,
    recursive: true,
    noHash: false,
    forceReprocess: false,
  }
}
