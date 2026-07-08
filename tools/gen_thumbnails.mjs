#!/usr/bin/env node
/**
 * Generate responsive, retina-ready thumbnails for the video gallery.
 *
 * Source list is the annotator's VIDEOS array. For each video it takes the
 * published thumbnail JPG, falling back to a local reconstructed scene frame and
 * then to an ffmpeg grab from the video, and emits multiple WebP widths into
 * apps/fpv-tool/public/thumbnails/<slug>/<w>.webp cropped to a uniform 16:9, plus
 * a tiny blur placeholder and a manifest the gallery reads to build <img srcset>.
 * The widths cover mobile 1x/2x through desktop 1x/2x. Runs are incremental
 * (skips already-generated videos); pass --force to regenerate everything.
 *
 *   node scripts/gen-thumbnails.mjs
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const execFileP = promisify(execFile);

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const annotatorPath = path.join(repoRoot, "tools", "annotator.html");
const outDir = path.join(repoRoot, "build", "thumbnails");

const WIDTHS = [320, 480, 640, 960, 1280]; // mobile 1x/2x, tablet, desktop 1x/2x
const QUALITY = 78;
const CONCURRENCY = 3;
const ASPECT = 9 / 16;

function slugify(value) {
  const stem = value.replace(/\.[^./]+$/, "");
  return stem.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 120) || "video";
}

// Read the canonical video list from the annotator's inline VIDEOS array so the
// generated thumbnails match exactly what the gallery renders.
function readVideos() {
  const html = fs.readFileSync(annotatorPath, "utf8");
  const start = html.indexOf("const VIDEOS");
  const open = html.indexOf("[", start);
  const close = html.indexOf("\n];", open);
  const literal = `${html.slice(open, close)}]`;
  const arr = Function(`"use strict"; return (${literal});`)();
  const rows = [];
  const seen = new Set();
  for (const v of arr) {
    const videoUrl = (v.video_url ?? "").trim();
    const thumbUrl = (v.thumbnail_url ?? "").trim();
    const videoFile = videoUrl.split("/").pop() ?? "";
    if (!videoFile || seen.has(videoFile)) continue;
    seen.add(videoFile);
    rows.push({ slug: slugify(videoFile), thumbUrl, videoUrl });
  }
  return rows;
}

async function fetchBuffer(url, attempts = 4) {
  for (let i = 0; i < attempts; i += 1) {
    try {
      const res = await fetch(url);
      if (res.ok) return Buffer.from(await res.arrayBuffer());
      // 403/429 from the CDN are usually transient rate-limits -> back off and retry.
      if (res.status !== 403 && res.status !== 429) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      if (i === attempts - 1) throw err;
    }
    await new Promise((r) => setTimeout(r, 600 * 2 ** i));
  }
  throw new Error("exhausted retries");
}

// Fallback for videos with a reconstructed scene: reuse a local scene frame.
function sceneFrame(slug) {
  const base = path.join(repoRoot, "scenes", slug);
  let scenes;
  try {
    scenes = fs.readdirSync(base, { withFileTypes: true }).filter((d) => d.isDirectory());
  } catch {
    return null;
  }
  for (const scene of scenes) {
    const framesDir = path.join(base, scene.name, "frames");
    let frames;
    try {
      frames = fs.readdirSync(framesDir).filter((f) => /\.(jpe?g|png)$/i.test(f)).sort();
    } catch {
      continue;
    }
    if (frames.length) {
      return fs.readFileSync(path.join(framesDir, frames[Math.floor(frames.length / 2)]));
    }
  }
  return null;
}

// Fallback when a video has no published thumbnail JPG: grab a frame from the
// video itself (ffmpeg seeks over HTTP, no full download).
async function frameFromVideo(videoUrl) {
  if (!videoUrl) return null;
  const tmp = path.join(os.tmpdir(), `fpvthumb_${Math.random().toString(36).slice(2)}.jpg`);
  try {
    await execFileP(
      "ffmpeg",
      ["-y", "-loglevel", "error", "-ss", "1", "-i", videoUrl, "-frames:v", "1", "-q:v", "3", tmp],
      { timeout: 90000 },
    );
    const buf = fs.readFileSync(tmp);
    return buf;
  } catch {
    return null;
  } finally {
    try {
      fs.unlinkSync(tmp);
    } catch {
      /* ignore */
    }
  }
}

async function processOne(video, manifest) {
  const dir = path.join(outDir, video.slug);
  try {
    let buf = null;
    if (video.thumbUrl) {
      buf = await fetchBuffer(video.thumbUrl).catch(() => null);
    }
    if (!buf) buf = sceneFrame(video.slug);
    if (!buf) buf = await frameFromVideo(video.videoUrl);
    if (!buf) {
      process.stdout.write("x");
      return false;
    }
    const meta = await sharp(buf).metadata();
    const srcW = meta.width || WIDTHS.at(-1);
    let widths = WIDTHS.filter((w) => w <= srcW);
    if (widths.length === 0) widths = [srcW];
    fs.mkdirSync(dir, { recursive: true });
    for (const w of widths) {
      await sharp(buf)
        .resize(w, Math.round(w * ASPECT), { fit: "cover", position: "centre" })
        .webp({ quality: QUALITY })
        .toFile(path.join(dir, `${w}.webp`));
    }
    const blur = await sharp(buf).resize(16, 9, { fit: "cover" }).webp({ quality: 40 }).toBuffer();
    manifest[video.slug] = {
      widths,
      blurDataURL: `data:image/webp;base64,${blur.toString("base64")}`,
    };
    process.stdout.write(".");
    return true;
  } catch {
    process.stdout.write("x");
    return false;
  }
}

// Keep manifest entries whose files are still on disk, so re-runs are incremental
// (only missing videos get re-fetched -> gentle on the CDN, resumable).
function loadExistingManifest() {
  try {
    const m = JSON.parse(fs.readFileSync(path.join(outDir, "manifest.json"), "utf8"));
    const kept = {};
    for (const [slug, entry] of Object.entries(m)) {
      const w = entry.widths?.[0];
      if (w && fs.existsSync(path.join(outDir, slug, `${w}.webp`))) kept[slug] = entry;
    }
    return kept;
  } catch {
    return {};
  }
}

async function run() {
  const force = process.argv.includes("--force");
  const videos = readVideos();
  fs.mkdirSync(outDir, { recursive: true });
  const manifest = force ? {} : loadExistingManifest();
  const todo = videos.filter((v) => !manifest[v.slug]);
  console.log(
    `${WIDTHS.join("/")}px WebP thumbnails: ${videos.length} videos, ` +
      `${videos.length - todo.length} cached, ${todo.length} to generate...`,
  );
  let cursor = 0;
  async function worker() {
    while (cursor < todo.length) {
      await processOne(todo[cursor++], manifest);
    }
  }
  await Promise.all(Array.from({ length: CONCURRENCY }, worker));
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest));
  console.log(`\ndone: ${Object.keys(manifest).length}/${videos.length} thumbnails -> ${path.relative(repoRoot, outDir)}`);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
