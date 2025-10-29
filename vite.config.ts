import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  root: 'client',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'robots.txt'],
      manifest: {
        name: "Shubble Web App",
        short_name: "Shubble",
        description: "Making RPI Shuttles Reliable, Predictable, and Accountable with Real Time Data",
        start_url: "/",
        background_color: "#a1c3ff",
        theme_color: "#a1c3ff",
        orientation: "any",
        display: "standalone",
        icons: [
          {
            src: "/shubble192.png",
            sizes: "192x192",
            type: "image/png"
          },
          {
            src: "/shubble512.png",
            sizes: "512x512",
            type: "image/png"
          }
        ]
      }
    })
  ],
  build: {
    outDir: '../client/dist',
    emptyOutDir: true
  },
  define: {
    'import.meta.env.GIT_REV': JSON.stringify(process.env.GIT_REV || 'unknown')
  }
})
