#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(appRoot, "../../..");
const publicDir = path.join(appRoot, "..", "public");

function copyDir(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const from = path.join(src, entry.name);
    const to = path.join(dest, entry.name);
    if (entry.isDirectory()) copyDir(from, to);
    else fs.copyFileSync(from, to);
  }
}

function copyFile(src, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
}

copyDir(path.join(repoRoot, "tools", "scene_viewer"), path.join(publicDir, "tools", "scene_viewer"));
copyFile(
  path.join(repoRoot, "tools", "annotator.html"),
  path.join(publicDir, "annotate", "index.html"),
);
copyFile(
  path.join(repoRoot, "tools", "scene_browser.html"),
  path.join(publicDir, "scenes", "index.html"),
);

console.log("synced web assets into apps/fpv-tool/public");
