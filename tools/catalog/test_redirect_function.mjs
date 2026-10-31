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

// A redirect source that is a prefix of another live ID must not rewrite it.
const location = (uri) => context.handler({ request: { uri, querystring: {} } }).headers?.location.value;
const longer = { uri: "/videos/2026-05-09_strike_on_soldiers_near_deir_mimas.mp4", querystring: {} };
assert.equal(context.handler({ request: longer }), longer);
assert.equal(
  location("/videos/2026-05-09_strike_on_soldiers.mp4"),
  "/videos/2026-05-09_soldiers_gathering_jal_al_alam_mmirleb_15988.mp4",
);
assert.equal(
  location("/videos/2026-05-09_strike_on_soldiers_near_shlomi.mp4"),
  "/videos/2026-05-09_soldiers_gathering_shlomi_mmirleb_15970.mp4",
);
assert.equal(
  location("/annotations/2026-05-09_strike_on_soldiers_annotations.json"),
  "/annotations/2026-05-09_soldiers_gathering_jal_al_alam_mmirleb_15988_annotations.json",
);
assert.equal(
  location("/thumbnails/2026-05-09_strike_on_soldiers/320.webp"),
  "/thumbnails/2026-05-09_soldiers_gathering_jal_al_alam_mmirleb_15988/320.webp",
);
assert.equal(
  location("/scenes/2026-05-26_anti_drone_platform_barashit/2026-05-26_anti_drone_platform_barashit_seg01/viewer/scene_meta.json"),
  "/scenes/2026-05-26_anti_drone_platform_biranit/2026-05-26_anti_drone_platform_biranit_seg01/viewer/scene_meta.json",
);

const unchanged = { uri: "/videos/current.mp4", querystring: {} };
assert.equal(context.handler({ request: unchanged }), unchanged);
console.log("redirect function tests passed");
