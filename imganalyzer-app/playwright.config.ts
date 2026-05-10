import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  outputDir: 'node_modules/.cache/playwright-results',
  use: {
    trace: 'off',
    screenshot: 'off',
    video: 'off',
  },
})
