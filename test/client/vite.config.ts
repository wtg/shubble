import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5174,
  },
  define: {
    'import.meta.env.VITE_TEST_BACKEND_URL': JSON.stringify(process.env.VITE_TEST_BACKEND_URL || 'http://localhost:4000'),
    'import.meta.env.VITE_TEST_FRONTEND_URL': JSON.stringify(process.env.VITE_TEST_FRONTEND_URL || 'http://localhost:5174')
  }
})
