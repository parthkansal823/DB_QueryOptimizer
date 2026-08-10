import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // The components never import React, relying on the automatic JSX runtime.
  // Vitest's transform pipeline does not pick that up from the React plugin
  // and falls back to the classic `React.createElement` form, so every render
  // in a test dies on "React is not defined" while dev and build are fine.
  esbuild: { jsx: 'automatic' },
  test: {
    // jsdom rather than node: these tests assert on what a panel renders, and
    // the bug they exist to catch was a sentence that contradicted the chart
    // beside it -- which is only visible once something is rendered.
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.js',
  },
  build: {
    rollupOptions: {
      output: {
        // Everything landed in one 640 kB chunk, which Vite warned about on
        // every build. Recharts is the bulk of that and it changes only when
        // the dependency is upgraded, so keeping it separate from the
        // dashboard code means editing a panel no longer invalidates the
        // whole bundle in a returning visitor's cache.
        // Rolldown (Vite 8) takes the function form only -- the object form
        // it replaced fails the build rather than being ignored.
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (/node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return 'react'
          // Recharts and the d3 packages it pulls in.
          return 'charts'
        },
      },
    },
  },
})
