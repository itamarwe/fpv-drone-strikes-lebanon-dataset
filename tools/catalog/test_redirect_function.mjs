#!/usr/bin/env node

import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
execFileSync("node", ["tools/catalog/generate_redirect_function.mjs"], { cwd: root });
const context = {};
vm.runInNewContext(fs.readFileSync(path.join(root, "ops/cloudfront-uri-redirects.js"), "utf8"), context);

const renamed = context.handler({
  request: {
    uri: "/videos/2026-05-26_anti_drone_platform_barashit.mp4",
    querystring: { download: { value: "yes%20please" }, tag: { multiValue: [{ value: "a" }, { value: "b" }] } },
  },
});
assert.equal(renamed.statusCode, 308);
assert.equal(
  renamed.headers.location.value,
  "/videos/2026-05-26_anti_drone_platform_biranit.mp4?download=yes%20please&tag=a&tag=b",
);

const dated = context.handler({
  request: { uri: "/annotations/2026_05_26_example_annotations.json", querystring: {} },
});
assert.equal(dated.headers.location.value, "/annotations/2026-05-26_example_annotations.json");

const unchanged = { uri: "/videos/current.mp4", querystring: {} };
assert.equal(context.handler({ request: unchanged }), unchanged);
console.log("redirect function tests passed");
