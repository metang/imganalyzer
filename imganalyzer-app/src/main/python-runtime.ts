import { spawn, execSync } from 'child_process'
import type { ChildProcess, SpawnOptions } from 'child_process'
import { app } from 'electron'
import { dirname } from 'path'

export const PKG_ROOT = process.env.IMGANALYZER_PKG_ROOT || dirname(app.getAppPath())
export const CONDA_ENV = 'imganalyzer'
const PYTHON_MODULE_COMMAND_ENV = 'IMGANALYZER_PYTHON_MODULE_COMMAND'

function getPythonModuleCommandOverride(): { command: string; args: string[] } | null {
  const raw = process.env[PYTHON_MODULE_COMMAND_ENV]?.trim()
  if (!raw) return null

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error(`${PYTHON_MODULE_COMMAND_ENV} must be a JSON array, e.g. ["node","fake-server.cjs"]`)
  }

  if (
    !Array.isArray(parsed) ||
    parsed.length === 0 ||
    !parsed.every((part): part is string => typeof part === 'string' && part.length > 0)
  ) {
    throw new Error(`${PYTHON_MODULE_COMMAND_ENV} must be a non-empty JSON array of strings`)
  }

  const [command, ...args] = parsed
  return { command, args }
}

export function createPythonEnv(extraEnv: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv {
  return {
    ...process.env,
    HF_HUB_DISABLE_SYMLINKS_WARNING: '1',
    PYTHONIOENCODING: 'utf-8',
    PYTHONUTF8: '1',
    ...extraEnv,
  }
}

export function spawnPythonModule(
  moduleName: string,
  moduleArgs: string[] = [],
  options: Omit<SpawnOptions, 'cwd' | 'env'> & { extraEnv?: NodeJS.ProcessEnv } = {},
): ChildProcess {
  const { extraEnv = {}, ...spawnOptions } = options
  const override = getPythonModuleCommandOverride()
  if (override) {
    return spawn(
      override.command,
      override.args,
      {
        cwd: PKG_ROOT,
        env: createPythonEnv(extraEnv),
        stdio: ['pipe', 'pipe', 'pipe'],
        ...spawnOptions,
      },
    )
  }

  return spawn(
    'conda',
    [
      'run', '-n', CONDA_ENV, '--no-capture-output',
      'python', '-m', moduleName,
      ...moduleArgs,
    ],
    {
      cwd: PKG_ROOT,
      env: createPythonEnv(extraEnv),
      stdio: ['pipe', 'pipe', 'pipe'],
      ...spawnOptions,
    },
  )
}

export function killProcessTree(pid: number | undefined): void {
  if (!pid) return
  if (process.platform === 'win32') {
    execSync(`taskkill /T /F /PID ${pid}`, { stdio: 'ignore' })
    return
  }
  process.kill(pid, 'SIGTERM')
}
