import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Connect, Plugin } from "vite";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const appRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(appRoot, "../..");

const MIME: Record<string, string> = {
  ".json": "application/json",
  ".bin": "application/octet-stream",
  ".webp": "image/webp",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
};

// Dev/preview only: serve the repo's scene data and the generated thumbnails
// straight from disk, so no other local server (python or Next) is needed.
// In production these same URL prefixes come from CloudFront via
// VITE_SCENE_BASE / VITE_THUMB_BASE.
function serveRepoData(): Plugin {
  const roots: Record<string, string> = {
    "/scenes": path.join(repoRoot, "scenes"),
    "/thumbnails": path.join(repoRoot, "apps/fpv-tool/public/thumbnails"),
  };
  const middleware: Connect.NextHandleFunction = (req, res, next) => {
    const url = (req.url ?? "").split("?")[0];
    const prefix = Object.keys(roots).find((p) => url.startsWith(`${p}/`));
    if (!prefix) return next();
    const rel = decodeURIComponent(url.slice(prefix.length + 1));
    const file = path.join(roots[prefix], rel);
    // stay inside the root
    if (!file.startsWith(roots[prefix] + path.sep)) return next();
    fs.stat(file, (err, stat) => {
      if (err || !stat.isFile()) {
        res.statusCode = 404;
        res.end("not found");
        return;
      }
      res.setHeader("content-type", MIME[path.extname(file)] ?? "application/octet-stream");
      res.setHeader("content-length", String(stat.size));
      fs.createReadStream(file).pipe(res);
    });
  };
  return {
    name: "serve-repo-data",
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}

// Relative base so the built app works from any mount point
// (e.g. itamarweiss.com/fpv/ or local file serving).
export default defineConfig({
  base: "./",
  plugins: [react(), serveRepoData()],
  server: { port: 5185 },
  preview: { port: 5186 },
});
