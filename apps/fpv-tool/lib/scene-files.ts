import fs from "node:fs/promises";
import path from "node:path";
import { scenesDir, viewerIndexPath, withBasePath, basePath } from "@/lib/config";
import { sceneDataBase } from "@/lib/asset-urls";
import { mimeFor } from "@/lib/mime";
import { ensureChild } from "@/lib/paths";

const VIEWER_PAGE_RE = /\/viewer(?:\/index\.html)?\/?$/;

function viewerParts(parts: string[]): string[] {
  const trimmed = parts.at(-1) === "index.html" ? parts.slice(0, -1) : parts;
  if (trimmed.at(-1) !== "viewer") return [];
  return trimmed;
}

export async function readFileResponse(filePath: string): Promise<Response> {
  const data = await fs.readFile(filePath);
  return new Response(data, {
    headers: {
      "content-type": mimeFor(filePath),
      "cache-control": "no-store",
    },
  });
}

async function serveGenericViewer(parts: string[]): Promise<Response | null> {
  const viewerPartsPath = viewerParts(parts);
  if (!viewerPartsPath.length) return null;
  const viewerDir = ensureChild(scenesDir, ...viewerPartsPath);
  try {
    await fs.access(path.join(viewerDir, "scene_meta.json"));
  } catch {
    return null;
  }
  const html = await fs.readFile(viewerIndexPath, "utf8");
  const relScene = viewerPartsPath.slice(0, -1).join("/");
  const sceneBase = sceneDataBase(relScene, basePath);
  const body = html
    .replaceAll("__SCENE_BASE__", sceneBase)
    .replaceAll("__APP_BASE__", basePath)
    .replaceAll("__API_BASE__", "");
  return new Response(body, {
    headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" },
  });
}

// Serve the 3D viewer for a scene addressed by id alone (clean /viewer/<id> URL),
// locating which video directory holds it. Scene data still loads from
// /scenes/<stem>/<id>/viewer/ via SCENE_BASE.
export async function serveViewerBySceneId(sceneId: string): Promise<Response | null> {
  let stems: string[];
  try {
    stems = (await fs.readdir(scenesDir, { withFileTypes: true }))
      .filter((d) => d.isDirectory())
      .map((d) => d.name);
  } catch {
    return null;
  }
  for (const stem of stems) {
    // The scene id is prefixed with its video slug, so only inspect matching dirs.
    if (sceneId !== stem && !sceneId.startsWith(`${stem}_`)) continue;
    const viewerDir = ensureChild(scenesDir, stem, sceneId, "viewer");
    try {
      await fs.access(path.join(viewerDir, "scene_meta.json"));
    } catch {
      continue;
    }
    const html = await fs.readFile(viewerIndexPath, "utf8");
    const sceneBase = sceneDataBase(`${stem}/${sceneId}`, basePath);
    const body = html
      .replaceAll("__SCENE_BASE__", sceneBase)
      .replaceAll("__APP_BASE__", basePath)
      .replaceAll("__API_BASE__", "");
    return new Response(body, {
      headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" },
    });
  }
  return null;
}

export async function serveSceneRequest(urlPath: string): Promise<Response | null> {
  const normalized = urlPath.replace(/\/$/, "") || "/scenes";
  if (normalized === "/scenes") {
    return readFileResponse(path.join(process.cwd(), "public", "scenes", "index.html"));
  }
  if (!normalized.startsWith("/scenes/")) return null;

  const parts = normalized.slice("/scenes/".length).split("/").filter(Boolean);
  if (!parts.length) return null;

  if (VIEWER_PAGE_RE.test(normalized)) {
    return serveGenericViewer(parts);
  }

  try {
    const target = ensureChild(scenesDir, ...parts);
    const stat = await fs.stat(target);
    if (!stat.isFile()) return null;
    return readFileResponse(target);
  } catch {
    return null;
  }
}
