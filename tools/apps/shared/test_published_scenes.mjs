import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("./published-scenes.js", import.meta.url), "utf8");
const sandbox = {
  Date: { now: () => 123 },
  encodeURIComponent,
  window: {},
  fetch: async () => ({
    ok: true,
    json: async () => ({
      videos: [
        {
          videoFile: "flight.mp4",
          date: "2026-07-13",
          description: "Test flight",
          town: "Test town",
          scenePath: "flight/flight_seg01_seg02",
        },
        { videoFile: "no-scene.mp4", scenePath: null },
        { videoFile: "invalid.mp4", scenePath: "flight/../invalid" },
      ],
    }),
  }),
};
vm.runInNewContext(source, sandbox);
const scenes = await sandbox.window.loadPublishedScenes();

assert.equal(scenes.length, 1);
assert.deepEqual(JSON.parse(JSON.stringify(scenes[0])), {
  scene_id: "flight_seg01_seg02",
  path: "flight/flight_seg01_seg02",
  viewer_url: "/scenes/flight/flight_seg01_seg02/viewer/index.html",
  exists: true,
  video_file: "flight.mp4",
  description: "Test flight",
  date: "2026-07-13",
  town: "Test town",
  segment_ids: ["seg01", "seg02"],
});
console.log("published scene catalog client: ok");
