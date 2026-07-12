import fs from "node:fs";
import path from "node:path";

export const CDN_BASE = "https://d2fioemadmrru3.cloudfront.net";
export const ID_PATTERN = /^\d{4}-\d{2}-\d{2}_[a-z0-9_]+$/;

export function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

export function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

export function videoFileFor(record) {
  return path.posix.basename(record.media.video_key);
}

export function thumbnailFileFor(record) {
  return path.posix.basename(record.media.thumbnail_key);
}

export function publicVideo(record) {
  return {
    date: record.date,
    description: record.description,
    town: record.town,
    thumbnail_url: `${CDN_BASE}/${record.media.thumbnail_key}`,
    video_url: `${CDN_BASE}/${record.media.video_key}`,
  };
}

export function sortedRecords(catalog) {
  return [...catalog.videos].sort((a, b) => b.date.localeCompare(a.date));
}

export function validateCatalog(catalog) {
  const errors = [];
  if (catalog.schema_version !== 1) errors.push("catalog.schema_version must be 1");
  if (!Array.isArray(catalog.videos)) errors.push("catalog.videos must be an array");
  const ids = new Set();
  const sourceIds = new Set();
  for (const [index, record] of (catalog.videos ?? []).entries()) {
    const at = `videos[${index}]`;
    if (!ID_PATTERN.test(record.id ?? "")) errors.push(`${at}.id is not canonical`);
    if (ids.has(record.id)) errors.push(`${at}.id duplicates ${record.id}`);
    ids.add(record.id);
    if (record.date !== record.id?.slice(0, 10)) errors.push(`${at}.date differs from id`);
    if (!record.description?.trim()) errors.push(`${at}.description is required`);
    if (!record.town?.trim()) errors.push(`${at}.town is required`);
    if (record.media?.video_key !== `videos/${record.id}.mp4`) {
      errors.push(`${at}.media.video_key differs from id`);
    }
    if (record.media?.thumbnail_key !== `thumbnails/${record.id}.jpg`) {
      errors.push(`${at}.media.thumbnail_key differs from id`);
    }
    const provenance = record.provenance;
    if (provenance && typeof provenance !== "object") errors.push(`${at}.provenance must be an object`);
    if (provenance?.source !== "legacy_import") {
      if (!provenance?.source_url) errors.push(`${at}.provenance.source_url is required`);
      if (!provenance?.channel) errors.push(`${at}.provenance.channel is required`);
      if (!provenance?.message_id) errors.push(`${at}.provenance.message_id is required`);
      if (!provenance?.discovered_at) errors.push(`${at}.provenance.discovered_at is required`);
      if (provenance?.channel && provenance?.message_id) {
        const sourceId = `${provenance.channel}:${provenance.message_id}`;
        if (sourceIds.has(sourceId)) errors.push(`${at}.provenance duplicates ${sourceId}`);
        sourceIds.add(sourceId);
      }
    }
  }
  return errors;
}

export function validateRedirects(payload, catalog) {
  const errors = [];
  if (payload.schema_version !== 1) errors.push("redirects.schema_version must be 1");
  const currentIds = new Set(catalog.videos.map((record) => record.id));
  const fromIds = new Set();
  for (const [index, redirect] of (payload.redirects ?? []).entries()) {
    const at = `redirects[${index}]`;
    if (!ID_PATTERN.test(redirect.from ?? "")) errors.push(`${at}.from is not canonical`);
    if (!ID_PATTERN.test(redirect.to ?? "")) errors.push(`${at}.to is not canonical`);
    if (redirect.from === redirect.to) errors.push(`${at} redirects to itself`);
    if (fromIds.has(redirect.from)) errors.push(`${at}.from duplicates ${redirect.from}`);
    fromIds.add(redirect.from);
    if (!currentIds.has(redirect.to)) errors.push(`${at}.to is not a current catalog id`);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(redirect.created_at ?? "")) {
      errors.push(`${at}.created_at must be YYYY-MM-DD`);
    }
    if (redirect.review_after && !/^\d{4}-\d{2}-\d{2}$/.test(redirect.review_after)) {
      errors.push(`${at}.review_after must be YYYY-MM-DD`);
    }
  }
  for (const redirect of payload.redirects ?? []) {
    let cursor = redirect.to;
    const visited = new Set([redirect.from]);
    while (fromIds.has(cursor)) {
      if (visited.has(cursor)) {
        errors.push(`redirect cycle contains ${cursor}`);
        break;
      }
      visited.add(cursor);
      cursor = payload.redirects.find((item) => item.from === cursor).to;
    }
  }
  return errors;
}

export function readSceneManifests(root) {
  const sceneRoot = path.join(root, "scene-manifests");
  const byVideo = new Map();
  if (!fs.existsSync(sceneRoot)) return byVideo;
  for (const videoId of fs.readdirSync(sceneRoot)) {
    const videoDir = path.join(sceneRoot, videoId);
    if (!fs.statSync(videoDir).isDirectory()) continue;
    const scenes = fs
      .readdirSync(videoDir)
      .filter((name) => name.endsWith(".json"))
      .map((name) => readJson(path.join(videoDir, name)));
    byVideo.set(videoId, scenes);
  }
  return byVideo;
}
