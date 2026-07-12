#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  publicVideo,
  readJson,
  sortedRecords,
  validateCatalog,
} from "./catalog_lib.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const check = process.argv.includes("--check");
const catalog = readJson(path.join(root, "data/catalog.json"));
const errors = validateCatalog(catalog);
if (errors.length) throw new Error(errors.join("\n"));

function commitGenerated(file, content) {
  const existing = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
  if (existing === content) return false;
  if (check) throw new Error(`${path.relative(root, file)} is not generated from data/catalog.json`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
  return true;
}

const records = sortedRecords(catalog);
const annotatorJson = `${JSON.stringify(records.map(publicVideo), null, 2)}\n`;
commitGenerated(path.join(root, "tools/catalog-videos.json"), annotatorJson);

const readmePath = path.join(root, "README.md");
const readme = fs.readFileSync(readmePath, "utf8");
const marker = "## Videos\n";
const markerIndex = readme.indexOf(marker);
if (markerIndex < 0) throw new Error("README.md is missing the Videos section");
const header = [
  "## Videos",
  "",
  "| Date | Image | Description | Original title (Arabic) | Town | Link |",
  "| --- | --- | --- | --- | --- | --- |",
];
const rows = records.map((record) => {
  const alt = record.description.replaceAll('"', "'").replaceAll("|", "-");
  const description = record.description.replaceAll("|", "-");
  const originalTitle = (record.original_title ?? "").replaceAll("|", "-");
  const town = record.town.replaceAll("|", "-");
  const thumbnail = `https://d2fioemadmrru3.cloudfront.net/${record.media.thumbnail_key}`;
  const video = `https://d2fioemadmrru3.cloudfront.net/${record.media.video_key}`;
  return `| ${record.date} | <img src="${thumbnail}" alt="${alt}" width="180"> | ${description} | ${originalTitle} | ${town} | [Download](${video}) |`;
});
const generatedReadme = `${readme.slice(0, markerIndex)}${[...header, ...rows, ""].join("\n")}`;
commitGenerated(readmePath, generatedReadme);
console.log(`generated catalog views for ${records.length} videos${check ? " (check)" : ""}`);
