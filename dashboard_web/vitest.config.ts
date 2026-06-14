import { defineConfig } from "vitest/config";

// Unit tests only (jsdom). The Vite build config lives in vite.config.ts.
// No React plugin needed: the tested modules (hooks, taskMetrics) are plain
// TypeScript with no JSX, and renderHook does not require a JSX transform.
export default defineConfig({
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
