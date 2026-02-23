import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  main: {
    build: {
      rollupOptions: {
        // Keep @github/copilot-sdk and @github/copilot as external so that
        // Vite does NOT bundle them into the CJS main bundle. They are pure
        // ESM packages that use import.meta.resolve() internally; bundling
        // them breaks that resolution. Electron's Node runtime resolves them
        // natively via the dynamic import() call in copilot-analyzer.ts.
        external: ['@github/copilot-sdk', '@github/copilot']
      }
    }
  },
  preload: {
    build: {
      rollupOptions: {
        external: ['@github/copilot-sdk', '@github/copilot']
      }
    }
  },
  renderer: {
    plugins: [react()]
  }
})
