#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const html = fs.readFileSync(path.join(root, "tools/annotator.html"), "utf8");
const start = html.indexOf("const VIDEOS");
const open = html.indexOf("[", start);
const close = html.indexOf("\n];", open);
const videos = Function(`"use strict"; return (${html.slice(open, close)}]);`)();
const errors = [];
const readme = fs.readFileSync(path.join(root, "README.md"), "utf8");
const readmeVideos = new Set(
  [...readme.matchAll(/\/videos\/([a-z0-9_-]+\.mp4)/g)].map((match) => match[1]),
);
const catalogVideos = new Set(videos.map((video) => video.video_url.split("/").pop()));

for (const videoFile of readmeVideos) {
  if (!catalogVideos.has(videoFile)) errors.push(`${videoFile}: README entry missing from catalog`);
}
for (const videoFile of catalogVideos) {
  if (!readmeVideos.has(videoFile)) errors.push(`${videoFile}: catalog entry missing from README`);
}

for (const video of videos) {
  const videoFile = video.video_url.split("/").pop();
  const stem = videoFile.replace(/\.mp4$/, "");
  const annotationPath = path.join(root, "annotations", `${stem}_annotations.json`);
  if (!fs.existsSync(annotationPath)) {
    errors.push(`${videoFile}: missing canonical annotation`);
    continue;
  }
  const annotation = JSON.parse(fs.readFileSync(annotationPath, "utf8"));
  for (const [key, expected] of Object.entries({
    video_file: videoFile,
    video_url: video.video_url,
    description: video.description,
    date: video.date,
    town: video.town,
  })) {
    if (annotation[key] !== expected) errors.push(`${videoFile}: annotation ${key} differs from catalog`);
  }
  if (!video.thumbnail_url.endsWith(`/thumbnails/${stem}.jpg`)) {
    errors.push(`${videoFile}: thumbnail stem differs from video stem`);
  }
}

const annotationFiles = fs.readdirSync(path.join(root, "annotations")).filter((name) => name.endsWith(".json"));
if (annotationFiles.length !== videos.length) {
  errors.push(`catalog has ${videos.length} videos but annotations has ${annotationFiles.length} JSON files`);
}
for (const name of annotationFiles) {
  if (!/^\d{4}-\d{2}-\d{2}_[a-z0-9_]+_annotations\.json$/.test(name)) {
    errors.push(`${name}: non-canonical annotation filename`);
  }
}

if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log(`catalog audit passed: ${videos.length} videos and ${annotationFiles.length} canonical annotations`);
