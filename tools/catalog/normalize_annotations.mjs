#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const annotationsDir = path.join(repoRoot, "annotations");
const annotatorPath = path.join(repoRoot, "tools", "annotator.html");
const apply = process.argv.includes("--apply");

function readCatalog() {
  const html = fs.readFileSync(annotatorPath, "utf8");
  const start = html.indexOf("const VIDEOS");
  const open = html.indexOf("[", start);
  const close = html.indexOf("\n];", open);
  return Function(`"use strict"; return (${html.slice(open, close)}]);`)();
}

function canonicalAnnotationName(name) {
  return name.replace(/^(\d{4})_(\d{2})_(\d{2})_/, "$1-$2-$3_");
}

function annotationRank(entry) {
  return [entry.data.auto_generated ? 0 : 1, entry.data.exclusion_masks?.length ?? 0];
}

function preferredAnnotation(entries) {
  return [...entries].sort((a, b) => {
    const left = annotationRank(a);
    const right = annotationRank(b);
    return right[0] - left[0] || right[1] - left[1] || a.name.localeCompare(b.name);
  })[0];
}

const catalog = readCatalog();
const catalogByVideo = new Map(
  catalog.map((video) => [video.video_url.split("/").pop(), video]),
);
const grouped = new Map();

for (const name of fs.readdirSync(annotationsDir).filter((name) => name.endsWith(".json"))) {
  const entry = {
    name,
    data: JSON.parse(fs.readFileSync(path.join(annotationsDir, name), "utf8")),
  };
  const canonical = canonicalAnnotationName(name);
  const entries = grouped.get(canonical) ?? [];
  entries.push(entry);
  grouped.set(canonical, entries);
}

const actions = [];
for (const [canonicalName, entries] of [...grouped].sort(([a], [b]) => a.localeCompare(b))) {
  const winner = preferredAnnotation(entries);
  const video = catalogByVideo.get(winner.data.video_file);
  if (!video) throw new Error(`No catalog entry for ${winner.data.video_file} (${winner.name})`);

  const data = {
    ...winner.data,
    video_file: winner.data.video_file,
    video_url: video.video_url,
    description: video.description,
    date: video.date,
    town: video.town,
  };

  for (const entry of entries) {
    if (entry.name !== canonicalName) actions.push({ type: "remove", path: entry.name });
  }
  actions.push({
    type: "write",
    path: canonicalName,
    source: winner.name,
    data,
  });
}

if (apply) {
  for (const action of actions.filter((action) => action.type === "remove")) {
    fs.rmSync(path.join(annotationsDir, action.path));
  }
  for (const action of actions.filter((action) => action.type === "write")) {
    fs.writeFileSync(path.join(annotationsDir, action.path), `${JSON.stringify(action.data, null, 2)}\n`);
  }
}

const removals = actions.filter((action) => action.type === "remove").length;
const writes = actions.filter((action) => action.type === "write").length;
console.log(`${apply ? "applied" : "would apply"}: ${writes} canonical annotations, ${removals} obsolete files removed`);
