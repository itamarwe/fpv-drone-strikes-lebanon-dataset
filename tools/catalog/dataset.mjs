#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  CDN_BASE,
  readJson,
  validateCatalog,
  validateRedirects,
  writeJson,
} from "./catalog_lib.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const bucket = process.env.FPV_BUCKET ?? "s3://fpv-drone-strikes-lebanon-dataset";
const [command, ...rawArgs] = process.argv.slice(2);

function options(args) {
  const result = { _: [] };
  for (let i = 0; i < args.length; i += 1) {
    const value = args[i];
    if (!value.startsWith("--")) result._.push(value);
    else if (i + 1 < args.length && !args[i + 1].startsWith("--")) result[value.slice(2)] = args[++i];
    else result[value.slice(2)] = true;
  }
  return result;
}

function required(opts, key) {
  if (!opts[key]) throw new Error(`--${key} is required`);
  return opts[key];
}

function run(program, args, runOptions = {}) {
  const result = spawnSync(program, args, { cwd: root, stdio: "inherit", ...runOptions });
  if (result.status !== 0) throw new Error(`${program} exited ${result.status}`);
}

function aws(args) {
  run("aws", args);
}

async function invalidate(distributionId, paths) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const result = spawnSync(
      "aws",
      ["cloudfront", "create-invalidation", "--distribution-id", distributionId, "--paths", ...paths],
      { cwd: root, encoding: "utf8" },
    );
    if (result.status === 0) return JSON.parse(result.stdout).Invalidation.Id;

    const message = result.stderr || "cloudfront invalidation failed";
    if (!message.includes("TooManyInvalidationsInProgress") || attempt === 2) throw new Error(message);

    const active = spawnSync(
      "aws",
      ["cloudfront", "list-invalidations", "--distribution-id", distributionId, "--output", "json"],
      { cwd: root, encoding: "utf8" },
    );
    if (active.status !== 0) throw new Error(active.stderr || "could not list CloudFront invalidations");
    const inProgress = JSON.parse(active.stdout).InvalidationList?.Items?.filter((item) => item.Status === "InProgress") ?? [];
    if (!inProgress.length) {
      await new Promise((resolve) => setTimeout(resolve, 5000));
      continue;
    }
    console.log(`CloudFront invalidation capacity is full; waiting for ${inProgress.length} in-progress invalidation(s)`);
    for (const invalidation of inProgress) {
      aws(["cloudfront", "wait", "invalidation-completed", "--distribution-id", distributionId, "--id", invalidation.Id]);
    }
  }
}

async function verifyUrl(url, attempts = 5) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const response = await fetch(`${url}${url.includes("?") ? "&" : "?"}verify=${Date.now()}`, {
      method: "HEAD",
      redirect: "follow",
    });
    if (response.ok) return;
    if (attempt === attempts - 1) throw new Error(`${url} returned HTTP ${response.status}`);
    await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt));
  }
}

function sceneIds(opts) {
  const value = opts.ids ?? opts.id;
  if (!value || typeof value !== "string") throw new Error("--id or --ids is required");
  const ids = [...new Set(value.split(",").map((id) => id.trim()).filter(Boolean))];
  if (!ids.length) throw new Error("at least one scene ID is required");
  return ids;
}

function loadState() {
  return {
    catalog: readJson(path.join(root, "data/catalog.json")),
    redirects: readJson(path.join(root, "data/redirects.json")),
  };
}

function validateState(state) {
  const errors = [
    ...validateCatalog(state.catalog),
    ...validateRedirects(state.redirects, state.catalog),
  ];
  if (errors.length) throw new Error(errors.join("\n"));
}

function saveState(state) {
  validateState(state);
  writeJson(path.join(root, "data/catalog.json"), state.catalog);
  writeJson(path.join(root, "data/redirects.json"), state.redirects);
  run("node", ["tools/catalog/generate_views.mjs"]);
  run("node", ["tools/catalog/generate_redirect_function.mjs"]);
}

async function add(opts) {
  const metadata = readJson(path.resolve(required(opts, "metadata")));
  const video = path.resolve(required(opts, "video"));
  const thumbnail = path.resolve(required(opts, "thumbnail"));
  if (!fs.existsSync(video) || !fs.existsSync(thumbnail)) throw new Error("video or thumbnail file is missing");
  const state = loadState();
  if (state.catalog.videos.some((record) => record.id === metadata.id)) {
    throw new Error(`catalog already contains ${metadata.id}`);
  }
  const record = {
    ...metadata,
    media: {
      video_key: `videos/${metadata.id}.mp4`,
      thumbnail_key: `thumbnails/${metadata.id}.jpg`,
    },
  };
  const candidate = { ...state, catalog: { ...state.catalog, videos: [record, ...state.catalog.videos] } };
  validateState(candidate);
  if (!opts["no-upload"]) {
    aws(["s3", "cp", video, `${bucket}/${record.media.video_key}`, "--content-type", "video/mp4"]);
    aws(["s3", "cp", thumbnail, `${bucket}/${record.media.thumbnail_key}`, "--content-type", "image/jpeg"]);
    await verifyUrl(`${CDN_BASE}/${record.media.video_key}`);
    await verifyUrl(`${CDN_BASE}/${record.media.thumbnail_key}`);
  }
  saveState(candidate);
  console.log(`added catalog record ${record.id}; open a PR before publishing the manifest`);
}

function annotate(opts) {
  const id = required(opts, "id");
  const source = path.resolve(required(opts, "annotation"));
  const state = loadState();
  const record = state.catalog.videos.find((item) => item.id === id);
  if (!record) throw new Error(`unknown catalog id ${id}`);
  const annotation = readJson(source);
  Object.assign(annotation, {
    video_file: `${id}.mp4`,
    video_url: `${CDN_BASE}/${record.media.video_key}`,
    description: record.description,
    date: record.date,
    town: record.town,
  });
  writeJson(path.join(root, "annotations", `${id}_annotations.json`), annotation);
  run("npm", ["run", "audit"]);
  console.log(`attached annotation to ${id}`);
}

function addScene(opts) {
  const id = required(opts, "id");
  const source = path.resolve(required(opts, "scene"));
  const state = loadState();
  if (!state.catalog.videos.some((item) => item.id === id)) throw new Error(`unknown catalog id ${id}`);
  const sceneId = path.basename(source);
  const viewer = path.join(source, "viewer", "scene_meta.json");
  if (!fs.existsSync(viewer)) throw new Error(`${viewer} is missing`);
  const destination = path.join(root, "scenes", id, sceneId);
  if (path.resolve(source) !== path.resolve(destination)) fs.cpSync(source, destination, { recursive: true });
  writeJson(path.join(root, "scene-manifests", id, `${sceneId}.json`), {
    schema_version: 1,
    video_id: id,
    scene_id: sceneId,
    path: `${id}/${sceneId}`,
    viewer_key: `scenes/${id}/${sceneId}/viewer/scene_meta.json`,
  });
  console.log(`registered scene ${sceneId}; run dataset:publish with this local scenes directory present`);
}

async function unpublishScenes(opts) {
  const ids = sceneIds(opts);
  const state = loadState();
  const catalogIds = new Set(state.catalog.videos.map((video) => video.id));
  const unknown = ids.filter((id) => !catalogIds.has(id));
  if (unknown.length) throw new Error(`unknown catalog IDs: ${unknown.join(", ")}`);

  const manifestUrl = `${CDN_BASE}/data/videos.json?unpublish=${Date.now()}`;
  const response = await fetch(manifestUrl);
  if (!response.ok) throw new Error(`could not fetch current web manifest: HTTP ${response.status}`);
  const current = await response.json();
  const selected = new Set(ids);
  const videos = (current.videos ?? []).map((video) => (
    selected.has(video.slug) ? { ...video, scenePath: null, sceneQuality: null } : video
  ));
  const removedReferences = videos.filter((video) => selected.has(video.slug) && video.scenePath === null).length;
  if (removedReferences !== ids.length) throw new Error("current web manifest is missing one or more requested videos");

  const localManifests = ids.filter((id) => fs.existsSync(path.join(root, "scene-manifests", id)));
  console.log(`will unpublish ${ids.length} scene(s); remove ${localManifests.length} tracked scene-manifest director${localManifests.length === 1 ? "y" : "ies"}`);
  if (!opts.execute) {
    console.log("dry run only; rerun with --execute to update CloudFront and delete scene assets");
    return;
  }

  for (const id of localManifests) fs.rmSync(path.join(root, "scene-manifests", id), { recursive: true });
  fs.mkdirSync(path.join(root, "build", "web"), { recursive: true });
  writeJson(path.join(root, "build", "web", "current-videos.json"), { ...current, videos });
  run("node", ["tools/publishing/build_web_data.mjs"]);

  const published = readJson(path.join(root, "build", "web", "videos.json"));
  const publishedById = new Map(published.videos.map((video) => [video.slug, video]));
  for (const id of ids) {
    const video = publishedById.get(id);
    if (!video || video.scenePath !== null) throw new Error(`${id}: generated manifest still references a scene`);
  }

  aws(["s3", "cp", "build/web/videos.json", `${bucket}/data/videos.json`, "--content-type", "application/json", "--cache-control", "public,max-age=300"]);
  const distributionId = opts["distribution-id"] ?? "E1FTYLW4OET6KU";
  const manifestInvalidation = await invalidate(distributionId, ["/data/videos.json"]);
  aws(["cloudfront", "wait", "invalidation-completed", "--distribution-id", distributionId, "--id", manifestInvalidation]);
  await verifyUrl(`${CDN_BASE}/data/videos.json`);

  for (const id of ids) aws(["s3", "rm", `${bucket}/scenes/${id}/`, "--recursive"]);
  const scenePaths = ids.map((id) => `/scenes/${id}/*`);
  for (let offset = 0; offset < scenePaths.length; offset += 15) {
    const sceneInvalidation = await invalidate(distributionId, scenePaths.slice(offset, offset + 15));
    aws(["cloudfront", "wait", "invalidation-completed", "--distribution-id", distributionId, "--id", sceneInvalidation]);
  }
  console.log(`unpublished ${ids.length} scene(s); videos and annotations were not changed`);
}

function replaceStrings(value, from, to) {
  if (typeof value === "string") return value.split(from).join(to);
  if (Array.isArray(value)) return value.map((item) => replaceStrings(item, from, to));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, replaceStrings(item, from, to)]));
  }
  return value;
}

async function rename(opts) {
  const from = required(opts, "from");
  const to = required(opts, "to");
  const reason = required(opts, "reason");
  if (!opts.execute) throw new Error("rename copies public assets; rerun with --execute after reviewing from/to");
  const state = loadState();
  const record = state.catalog.videos.find((item) => item.id === from);
  if (!record) throw new Error(`unknown catalog id ${from}`);
  if (state.catalog.videos.some((item) => item.id === to)) throw new Error(`catalog already contains ${to}`);

  aws(["s3", "cp", `${bucket}/videos/${from}.mp4`, `${bucket}/videos/${to}.mp4`]);
  aws(["s3", "cp", `${bucket}/thumbnails/${from}.jpg`, `${bucket}/thumbnails/${to}.jpg`]);
  aws(["s3", "cp", `${bucket}/thumbnails/${from}/`, `${bucket}/thumbnails/${to}/`, "--recursive"]);

  const oldSceneDir = path.join(root, "scene-manifests", from);
  const scenes = fs.existsSync(oldSceneDir)
    ? fs.readdirSync(oldSceneDir).filter((name) => name.endsWith(".json")).map((name) => readJson(path.join(oldSceneDir, name)))
    : [];
  for (const scene of scenes) {
    const newSceneId = scene.scene_id.split(from).join(to);
    aws([
      "s3", "cp",
      `${bucket}/scenes/${scene.video_id}/${scene.scene_id}/viewer/`,
      `${bucket}/scenes/${to}/${newSceneId}/viewer/`,
      "--recursive",
    ]);
    const tmp = path.join(os.tmpdir(), `${newSceneId}-scene-meta.json`);
    aws(["s3", "cp", `${bucket}/scenes/${to}/${newSceneId}/viewer/scene_meta.json`, tmp]);
    fs.writeFileSync(tmp, fs.readFileSync(tmp, "utf8").split(from).join(to));
    aws(["s3", "cp", tmp, `${bucket}/scenes/${to}/${newSceneId}/viewer/scene_meta.json`, "--content-type", "application/json", "--cache-control", "public,max-age=300"]);
  }
  await verifyUrl(`${CDN_BASE}/videos/${to}.mp4`);
  await verifyUrl(`${CDN_BASE}/thumbnails/${to}.jpg`);

  Object.assign(record, {
    id: to,
    date: to.slice(0, 10),
    media: { video_key: `videos/${to}.mp4`, thumbnail_key: `thumbnails/${to}.jpg` },
  });
  const annotationFrom = path.join(root, "annotations", `${from}_annotations.json`);
  if (fs.existsSync(annotationFrom)) {
    const annotation = replaceStrings(readJson(annotationFrom), from, to);
    fs.rmSync(annotationFrom);
    writeJson(path.join(root, "annotations", `${to}_annotations.json`), annotation);
  }
  if (fs.existsSync(oldSceneDir)) {
    for (const scene of scenes) {
      const updated = replaceStrings(scene, from, to);
      writeJson(path.join(root, "scene-manifests", to, `${updated.scene_id}.json`), updated);
    }
    fs.rmSync(oldSceneDir, { recursive: true });
  }
  state.redirects.redirects.push({
    from,
    to,
    created_at: new Date().toISOString().slice(0, 10),
    review_after: opts["review-after"],
    reason,
  });
  saveState(state);
  console.log(`renamed ${from} to ${to}; old objects remain until post-publish cleanup`);
}

function help() {
  console.log(`Usage:
  npm run dataset:add -- --video FILE --thumbnail FILE --metadata JSON [--no-upload]
  npm run dataset:annotate -- --id ID --annotation JSON
  npm run dataset:add-scene -- --id ID --scene DIR
  npm run dataset:unpublish-scenes -- --ids ID,ID [--execute]
  npm run dataset:rename -- --from ID --to ID --reason TEXT [--review-after DATE] --execute
  npm run dataset:publish
  npm run dataset:verify`);
}

const opts = options(rawArgs);
if (command === "add") await add(opts);
else if (command === "annotate") annotate(opts);
else if (command === "add-scene") addScene(opts);
else if (command === "unpublish-scenes") await unpublishScenes(opts);
else if (command === "rename") await rename(opts);
else if (command === "publish") run("bash", ["tools/publishing/publish_web.sh", ...(opts["skip-scenes"] ? ["--skip-scenes"] : [])]);
else if (command === "verify") {
  run("npm", ["run", "catalog:check"]);
  run("npm", ["run", "check-public"]);
} else help();
