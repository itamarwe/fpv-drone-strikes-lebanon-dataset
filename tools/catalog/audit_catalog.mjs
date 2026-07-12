#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  publicVideo,
  readJson,
  readSceneManifests,
  sortedRecords,
  validateCatalog,
  validateRedirects,
} from "./catalog_lib.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const catalog = readJson(path.join(root, "data/catalog.json"));
const redirects = readJson(path.join(root, "data/redirects.json"));
const errors = [
  ...validateCatalog(catalog),
  ...validateRedirects(redirects, catalog),
];
const records = sortedRecords(catalog);
const byId = new Map(records.map((record) => [record.id, record]));

const generatedVideos = readJson(path.join(root, "tools/apps/annotator/catalog-videos.json"));
const expectedVideos = records.map(publicVideo);
if (JSON.stringify(generatedVideos) !== JSON.stringify(expectedVideos)) {
  errors.push("tools/apps/annotator/catalog-videos.json is not generated from data/catalog.json");
}

const readme = fs.readFileSync(path.join(root, "README.md"), "utf8");
const readmeIds = [...readme.matchAll(/\/videos\/([a-z0-9_-]+)\.mp4/g)].map((match) => match[1]);
if (JSON.stringify(readmeIds) !== JSON.stringify(records.map((record) => record.id))) {
  errors.push("README video rows do not match catalog order and IDs");
}

const annotationDir = path.join(root, "annotations");
for (const name of fs.readdirSync(annotationDir).filter((item) => item.endsWith(".json"))) {
  if (!/^\d{4}-\d{2}-\d{2}_[a-z0-9_]+_annotations\.json$/.test(name)) {
    errors.push(`${name}: non-canonical annotation filename`);
    continue;
  }
  const id = name.replace(/_annotations\.json$/, "");
  const record = byId.get(id);
  if (!record) {
    errors.push(`${name}: annotation has no catalog record`);
    continue;
  }
  const annotation = readJson(path.join(annotationDir, name));
  for (const [key, expected] of Object.entries({
    video_file: `${id}.mp4`,
    video_url: `https://d2fioemadmrru3.cloudfront.net/${record.media.video_key}`,
    description: record.description,
    date: record.date,
    town: record.town,
  })) {
    if (annotation[key] !== expected) errors.push(`${name}: ${key} differs from catalog`);
  }
}

for (const [videoId, scenes] of readSceneManifests(root)) {
  if (!byId.has(videoId)) errors.push(`${videoId}: scene directory has no catalog record`);
  for (const scene of scenes) {
    if (scene.schema_version !== 1) errors.push(`${scene.path}: schema_version must be 1`);
    if (scene.video_id !== videoId) errors.push(`${scene.path}: video_id differs from directory`);
    if (scene.path !== `${scene.video_id}/${scene.scene_id}`) {
      errors.push(`${scene.path}: path must be <video_id>/<scene_id>`);
    }
    if (scene.viewer_key !== `scenes/${scene.path}/viewer/scene_meta.json`) {
      errors.push(`${scene.path}: viewer_key differs from path`);
    }
  }
}

if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
const annotationCount = fs.readdirSync(annotationDir).filter((name) => name.endsWith(".json")).length;
const sceneCount = [...readSceneManifests(root).values()].reduce((sum, scenes) => sum + scenes.length, 0);
console.log(`catalog audit passed: ${records.length} videos, ${annotationCount} annotations, ${sceneCount} scenes`);
