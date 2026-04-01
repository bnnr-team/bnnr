import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    // Wheel serves UI from bnnr.dashboard (see serve._frontend_dist_candidates).
    outDir: "../src/bnnr/dashboard/frontend/dist",
    emptyOutDir: true,
  },
});
