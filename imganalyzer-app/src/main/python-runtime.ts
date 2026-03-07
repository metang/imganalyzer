import { spawn, execSync } from 'child_process'
import type { ChildProcess, SpawnOptions } from 'child_process'
import { app } from 'electron'
import { dirname } from 'path'

export const PKG_ROOT = process.env.IMGANALYZER_PKG_ROOT || dirname(app.getAppPath())
export const CONDA_ENV = 'imganalyzer'

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
