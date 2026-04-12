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

function canonical(p: string): string {
  const abs = normalize(resolve(p))
  return process.platform === 'win32' ? abs.toLowerCase() : abs
}

/** Register a directory so that all files beneath it are accessible. */
export function registerApprovedDirectory(dir: string): void {
  const norm = canonical(dir)
  if (norm) approvedRoots.add(norm)
}

/**
 * Returns `true` when `filePath` resolves to a location inside one of the
 * approved root directories.  The check is case-insensitive on Windows.
 */
export function validateFilePath(filePath: string): boolean {
  if (!filePath || typeof filePath !== 'string') return false

  const norm = canonical(filePath)

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
