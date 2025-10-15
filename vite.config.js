import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  root: 'client',
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Allow external connections
    port: 5173
  },
  build: {
    outDir: '../client/dist',
    emptyOutDir: true
  },
  define: {
    'import.meta.env.GIT_REV': JSON.stringify(process.env.GIT_REV || 'unknown')
  }
})
