import { defineConfig } from "vitest/config";

export default defineConfig({
  build: {
    // TensorFlow.js is intentionally a lazy training-only chunk. The camera UI
    // ships separately at roughly 150 kB minified.
    chunkSizeWarningLimit: 1200,
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
