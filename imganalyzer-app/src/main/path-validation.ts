/**
 * Path validation for IPC handlers.
 *
 * Maintains a runtime allowlist of directory roots that the renderer is
 * permitted to access. Every fs:*, shell:openPath, and local-file://
 * request is checked against this list before touching the filesystem.
 *
 * Approved roots are added when:
 *  - The user picks a folder via the native dialog  (dialog:openFolder)
 *  - The thumbnail cache directory is resolved       (auto-registered)
 *  - app.getPath('userData') is registered at startup
 */

import { resolve, normalize, sep } from 'path'

const approvedRoots = new Set<string>()
const approvedPaths = new Set<string>()

const DIRECTORY_PATH_KEYS = new Set(['path', 'parent_path', 'folderPath'])
const FILE_PATH_KEYS = new Set(['file_path', 'cover_file_path', 'filePath', 'imagePath'])

function canonical(p: string): string {
  const abs = normalize(resolve(p))
  return process.platform === 'win32' ? abs.toLowerCase() : abs
}

function looksLikeAbsoluteFilesystemPath(value: string): boolean {
  if (!value || typeof value !== 'string') return false
  if (process.platform === 'win32') {
    return /^[A-Za-z]:[\\/]/.test(value) || value.startsWith('\\\\')
  }
  return value.startsWith('/')
}

/** Register a directory so that all files beneath it are accessible. */
export function registerApprovedDirectory(dir: string): void {
  const norm = canonical(dir)
  if (norm) approvedRoots.add(norm)
}

/** Register one specific file path for later read/open access. */
export function registerApprovedPath(filePath: string): void {
  if (!looksLikeAbsoluteFilesystemPath(filePath)) return
  const norm = canonical(filePath)
  if (norm) approvedPaths.add(norm)
}

/**
 * Walk an RPC payload and register any returned filesystem paths so follow-up
 * thumbnail/full-image requests from the renderer are allowed.
 */
export function registerApprovedPathsFromPayload(payload: unknown): void {
  const seen = new Set<unknown>()

  const visit = (value: unknown): void => {
    if (value == null) return
    if (typeof value !== 'object') return
    if (seen.has(value)) return
    seen.add(value)

    if (Array.isArray(value)) {
      value.forEach(visit)
      return
    }

    for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
      if (typeof nested === 'string' && looksLikeAbsoluteFilesystemPath(nested)) {
        if (DIRECTORY_PATH_KEYS.has(key)) {
          registerApprovedDirectory(nested)
        } else if (FILE_PATH_KEYS.has(key) || key.endsWith('_path')) {
          registerApprovedPath(nested)
        }
      } else {
        visit(nested)
      }
    }
  }

  visit(payload)
}

/**
 * Returns `true` when `filePath` resolves to a location inside one of the
 * approved root directories.  The check is case-insensitive on Windows.
 */
export function validateFilePath(filePath: string): boolean {
  if (!filePath || typeof filePath !== 'string') return false

  const norm = canonical(filePath)
  if (approvedPaths.has(norm)) {
    return true
  }

  for (const root of approvedRoots) {
    // Exact match (e.g. the root directory itself) or child path
    if (norm === root || norm.startsWith(root + sep)) {
      return true
    }
  }

  return false
}

/** Throw a descriptive error when a path is rejected. */
export function assertPathAllowed(filePath: string, context: string): void {
  if (!validateFilePath(filePath)) {
    console.warn(
      `[SEC] Blocked ${context} access to path outside approved directories: ${filePath}`,
    )
    throw new Error(`Access denied: path is outside approved directories`)
  }
}
