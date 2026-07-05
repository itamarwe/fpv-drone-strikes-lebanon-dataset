import fs from "node:fs";
import path from "node:path";
import { repoRoot, scenesDir } from "@/lib/config";

export type VideoRecord = {
  date: string;
  description: string;
  town: string;
  thumbnailUrl: string; // remote (cloudfront) source thumbnail, used as fallback
  thumbWidths: number[] | null; // locally generated responsive WebP widths
  blurDataURL: string | null; // tiny placeholder for a blur-up load
  videoUrl: string;
  videoFile: string; // e.g. 2026-05-26_anti_drone_platform_barashit.mp4
  slug: string; // slugify(videoFile) == scene directory / thumbnail folder name
  sceneId: string | null; // reconstructed scene id, when one exists
};

type RawVideo = {
  date?: string;
  description?: string;
  town?: string;
  thumbnail_url?: string;
  video_url?: string;
};

type ThumbManifest = Record<string, { widths: number[]; blurDataURL: string }>;

function slugify(value: string): string {
  const stem = value.replace(/\.[^./]+$/, "");
  return (
    stem
      .replace(/[^a-zA-Z0-9._-]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 120) || "video"
  );
}

function fileNameFromUrl(url: string): string {
  return url.split("/").pop() ?? "";
}

// The canonical list of every video lives in the annotator's inline VIDEOS array
// (the same list that populates its dropdown). Read it straight from source so the
// gallery and the annotator can never drift apart.
export function readAnnotatorVideos(): RawVideo[] {
  const htmlPath = path.join(repoRoot, "tools", "annotator.html");
  let html: string;
  try {
    html = fs.readFileSync(htmlPath, "utf8");
  } catch {
    return [];
  }
  const start = html.indexOf("const VIDEOS");
  if (start < 0) return [];
  const open = html.indexOf("[", start);
  const close = html.indexOf("\n];", open);
  if (open < 0 || close < 0) return [];
  const literal = `${html.slice(open, close)}]`; // trailing comma is fine in a JS array literal
  try {
    // Trusted, in-repo source; parse the array literal (single-quoted, unquoted keys).
    return Function(`"use strict"; return (${literal});`)() as RawVideo[];
  } catch {
    return [];
  }
}

function loadThumbManifest(): ThumbManifest {
  const manifestPath = path.join(process.cwd(), "public", "thumbnails", "manifest.json");
  try {
    return JSON.parse(fs.readFileSync(manifestPath, "utf8")) as ThumbManifest;
  } catch {
    return {};
  }
}

// Map each video slug -> the first completed scene id under scenes/<slug>/.
function buildSceneIndex(): Map<string, string> {
  const index = new Map<string, string>();
  let stems: fs.Dirent[];
  try {
    stems = fs.readdirSync(scenesDir, { withFileTypes: true });
  } catch {
    return index;
  }
  for (const stem of stems) {
    if (!stem.isDirectory() || index.has(stem.name)) continue;
    let scenes: fs.Dirent[];
    try {
      scenes = fs.readdirSync(path.join(scenesDir, stem.name), { withFileTypes: true });
    } catch {
      continue;
    }
    for (const scene of scenes) {
      if (!scene.isDirectory()) continue;
      const meta = path.join(scenesDir, stem.name, scene.name, "viewer", "scene_meta.json");
      if (fs.existsSync(meta)) {
        index.set(stem.name, scene.name);
        break;
      }
    }
  }
  return index;
}

export function loadVideos(): VideoRecord[] {
  const sceneIndex = buildSceneIndex();
  const thumbs = loadThumbManifest();
  const rows: VideoRecord[] = [];
  const seen = new Set<string>();

  for (const raw of readAnnotatorVideos()) {
    const videoUrl = (raw.video_url ?? "").trim();
    const videoFile = fileNameFromUrl(videoUrl);
    if (!videoFile || seen.has(videoFile)) continue;
    seen.add(videoFile);
    const slug = slugify(videoFile);
    const thumb = thumbs[slug];
    rows.push({
      date: (raw.date ?? "").trim(),
      description: (raw.description ?? "").trim(),
      town: (raw.town ?? "").trim(),
      thumbnailUrl: (raw.thumbnail_url ?? "").trim(),
      thumbWidths: thumb?.widths ?? null,
      blurDataURL: thumb?.blurDataURL ?? null,
      videoUrl,
      videoFile,
      slug,
      sceneId: sceneIndex.get(slug) ?? null,
    });
  }

  // Newest first.
  rows.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
  return rows;
}
