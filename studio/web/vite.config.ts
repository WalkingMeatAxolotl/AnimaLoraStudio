/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时把 /api、/samples 反向代理到 FastAPI（默认 127.0.0.1:8765）。
// 构建产物会被 FastAPI 挂在 /studio 路径下。
export default defineConfig({
  plugins: [react()],
  base: '/studio/',
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8765',
      '/samples': 'http://127.0.0.1:8765',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
