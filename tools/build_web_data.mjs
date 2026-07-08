#!/usr/bin/env node
/**
 * Bundle the read-only viewer's dataset into public/data/videos.json.
 *
 * Sources (all in this repo):
 *  - tools/annotator.html          -> canonical VIDEOS list (date/description/town/urls)
 *  - annotations/*_annotations.json -> segment markers (manual preferred over auto)
 *  - scenes/<stem>/<sceneId>/viewer/scene_meta.json -> which videos have a 3D scene
 *  - build/thumbnails/manifest.json  -> responsive thumb widths + blur
 *
 * The output is a single static JSON the app fetches at startup, so production
 * needs no backend: videos play from CloudFront, scenes/thumbnails from
 * whatever bases VITE_SCENE_BASE / VITE_THUMB_BASE point at.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const outFile = path.join(repoRoot, "build", "web", "videos.json");

function slugify(value) {
  const stem = value.replace(/\.[^./]+$/, "");
  return stem.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 120) || "video";
}

function readAnnotatorVideos() {
  const html = fs.readFileSync(path.join(repoRoot, "tools", "annotator.html"), "utf8");
  const start = html.indexOf("const VIDEOS");
  const open = html.indexOf("[", start);
  const close = html.indexOf("\n];", open);
  const literal = `${html.slice(open, close)}]`;
  return Function(`"use strict"; return (${literal});`)();
}

// Prefer a manual annotation (no auto_generated flag) over an auto one.
function readAnnotations() {
  const dir = path.join(repoRoot, "annotations");
  const byVideo = new Map();
  for (const name of fs.readdirSync(dir)) {
    if (!name.endsWith("_annotations.json")) continue;
    let data;
    try {
      data = JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"));
    } catch {
      continue;
    }
    const videoFile = data.video_file;
    if (!videoFile) continue;
    const existing = byVideo.get(videoFile);
    const manual = !data.auto_generated;
    if (!existing || (manual && existing.auto_generated)) {
      byVideo.set(videoFile, data);
    }
  }
  return byVideo;
}

// Map each video slug -> ALL completed scene variants (sorted, canonical first).
function buildSceneIndex() {
  const scenesDir = path.join(repoRoot, "scenes");
  const index = new Map();
  if (!fs.existsSync(scenesDir)) return index;
  for (const stem of fs.readdirSync(scenesDir, { withFileTypes: true })) {
    if (!stem.isDirectory()) continue;
    const base = path.join(scenesDir, stem.name);
    const variants = [];
    for (const scene of fs.readdirSync(base, { withFileTypes: true }).sort((a, b) =>
      a.name.localeCompare(b.name),
    )) {
      if (!scene.isDirectory()) continue;
      if (fs.existsSync(path.join(base, scene.name, "viewer", "scene_meta.json"))) {
        variants.push(`${stem.name}/${scene.name}`);
      }
    }
    if (variants.length) index.set(stem.name, variants);
  }
  return index;
}

function readThumbManifest() {
  try {
    return JSON.parse(
      fs.readFileSync(path.join(repoRoot, "build/thumbnails/manifest.json"), "utf8"),
    );
  } catch {
    return {};
  }
}

const annotations = readAnnotations();
const sceneIndex = buildSceneIndex();
const thumbs = readThumbManifest();

const seen = new Set();
const videos = [];
for (const raw of readAnnotatorVideos()) {
  const videoUrl = (raw.video_url ?? "").trim();
  const videoFile = videoUrl.split("/").pop() ?? "";
  if (!videoFile || seen.has(videoFile)) continue;
  seen.add(videoFile);
  const slug = slugify(videoFile);
  const ann = annotations.get(videoFile);
  const thumb = thumbs[slug];
  const scenePaths = sceneIndex.get(slug) ?? null;
  videos.push({
    videoFile,
    slug,
    date: (raw.date ?? "").trim(),
    description: (raw.description ?? "").trim(),
    town: (raw.town ?? "").trim(),
    videoUrl,
    thumbnailUrl: (raw.thumbnail_url ?? "").trim(),
    thumbWidths: thumb?.widths ?? null,
    blur: thumb?.blurDataURL ?? null,
    scenePath: scenePaths?.[0] ?? null,
    scenePaths,
    segments: ann?.segments ?? null,
    annotationAuto: ann ? Boolean(ann.auto_generated) : null,
  });
}

videos.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));

fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, JSON.stringify({ generated_at: new Date().toISOString(), videos }));
const withScenes = videos.filter((v) => v.scenePath).length;
const withAnn = videos.filter((v) => v.segments).length;
console.log(
  `wrote ${path.relative(repoRoot, outFile)}: ${videos.length} videos, ` +
    `${withAnn} annotated, ${withScenes} with 3D scenes`,
);
