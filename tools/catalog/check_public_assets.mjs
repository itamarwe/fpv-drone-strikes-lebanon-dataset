#!/usr/bin/env node

const base = (process.env.FPV_CDN_BASE ?? "https://d2fioemadmrru3.cloudfront.net").replace(/\/$/, "");
const manifestResponse = await fetch(`${base}/data/videos.json?integrity=${Date.now()}`);
if (!manifestResponse.ok) throw new Error(`manifest HTTP ${manifestResponse.status}`);
const manifest = await manifestResponse.json();
const urls = [];

for (const video of manifest.videos) {
  urls.push(video.videoUrl, video.thumbnailUrl, `${base}/annotations/${video.slug}_annotations.json`);
  if (video.scenePath) urls.push(`${base}/scenes/${video.scenePath}/viewer/scene_meta.json`);
}

let index = 0;
const failures = [];
async function worker() {
  while (index < urls.length) {
    const url = urls[index++];
    try {
      const response = await fetch(url, { method: "HEAD", redirect: "follow" });
      if (!response.ok) failures.push(`${response.status} ${url}`);
    } catch (error) {
      failures.push(`${error.message} ${url}`);
    }
  }
}

await Promise.all(Array.from({ length: 12 }, worker));
console.log(
  JSON.stringify(
    {
      videos: manifest.videos.length,
      scenes: manifest.videos.filter((video) => video.scenePath).length,
      urlsChecked: urls.length,
      failures,
    },
    null,
    2,
  ),
);
if (failures.length) process.exit(1);
