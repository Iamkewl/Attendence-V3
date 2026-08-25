import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// ATT-053: dedicated Vitest config so the production Vite pipeline in
// vite.config.js stays untouched. Tests run in jsdom; no webcam/backend.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{js,jsx}'],
  },
})
