import fs from "node:fs";
import path from "node:path";
import { repoRoot, scenesDir } from "@/lib/config";

export type VideoRecord = {
  date: string;
  description: string;
  town: string;
  thumbnailUrl: string;
  videoUrl: string;
  videoFile: string; // e.g. 2026-05-26_anti_drone_platform_barashit_02.mp4
  slug: string; // slugify(videoFile) == scene directory name
  sceneId: string | null; // reconstructed scene id, when one exists
};

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
  const csvPath = path.join(repoRoot, "geo", "fpv_drone_map_records.csv");
  let text: string;
  try {
    text = fs.readFileSync(csvPath, "utf8");
  } catch {
    return [];
  }

  const sceneIndex = buildSceneIndex();
  const rows: VideoRecord[] = [];
  const seen = new Set<string>();

  for (const line of text.split(/\r?\n/).slice(1)) {
    if (!line.trim()) continue;
    // Columns: id,date,description,town,lat,lng,status,thumbnail_url,video_url.
    // The URL fields are always last and never contain commas, so read the tail
    // from the end and the descriptive fields from the front.
    const cols = line.split(",");
    if (cols.length < 9) continue;
    const videoUrl = cols[cols.length - 1].trim();
    const thumbnailUrl = cols[cols.length - 2].trim();
    const videoFile = fileNameFromUrl(videoUrl);
    if (!videoFile || seen.has(videoFile)) continue;
    seen.add(videoFile);
    const slug = slugify(videoFile);
    rows.push({
      date: cols[1].trim(),
      description: cols[2].trim(),
      town: cols[3].trim(),
      thumbnailUrl,
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
