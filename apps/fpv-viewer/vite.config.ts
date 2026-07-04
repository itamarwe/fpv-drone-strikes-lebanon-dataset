import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base so the built app works from any mount point
// (e.g. itamarweiss.com/fpv/ or local file serving).
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    port: 5185,
    proxy: {
      // Local dev: scene data (scene_meta.json + point bins) comes from the
      // python tool server which serves the repo's scenes/ directory.
      "/scenes": { target: "http://127.0.0.1:8766", changeOrigin: true },
      // Local dev: generated WebP thumbnails are served by the fpv-tool Next
      // app (public/thumbnails). Cards fall back to the remote JPG on error.
      "/thumbnails": { target: "http://127.0.0.1:3001", changeOrigin: true },
    },
  },
});
