import path from "node:path";

const appRoot = process.cwd();

export const repoRoot = path.resolve(
  process.env.FPV_REPO_ROOT ?? path.join(appRoot, "../.."),
);

export const scenesDir = path.resolve(
  process.env.FPV_SCENES_DIR ?? path.join(repoRoot, "scenes"),
);

export const viewerIndexPath = path.join(
  repoRoot,
  "tools",
  "scene_viewer",
  "index.html",
);

export const pythonApiUrl = (
  process.env.FPV_PYTHON_API_URL ?? "http://127.0.0.1:8766"
).replace(/\/$/, "");

// Browser-safe URL helpers live in lib/urls; re-export for server callers.
export { basePath, withBasePath, thumbBase } from "@/lib/urls";
