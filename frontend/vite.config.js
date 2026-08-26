import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },
  preview: {
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },
  // ATT-050: build-time chunk splitting. Without manualChunks the main
  // bundle ships the entire lucide-react icon set and axios inline; on a
  // slow university Wi-Fi classroom the first-load bundle can exceed 1 MiB
  // before the CDN compresses it. Split vendors into stable cacheable
  // chunks so the main app chunk stays small and a returning visitor skips
  // re-downloading the vendor code on subsequent deploys.
  //
  // Vite 8 (rolldown) requires manualChunks as a function, not an object.
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('/react-dom/') || id.includes('/react/')) {
              return 'react-vendor'
            }
            if (id.includes('/react-router-dom/') || id.includes('/react-router/')) {
              return 'router'
            }
            if (id.includes('/axios/')) {
              return 'axios'
            }
            if (id.includes('/lucide-react/')) {
              return 'icons'
            }
          }
          return undefined
        },
      },
    },
  },
})
