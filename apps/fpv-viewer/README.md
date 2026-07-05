# FPV Viewer (read-only)

A standalone, read-only web app for the FPV strike dataset: a video gallery, a
video page with the flight annotations, and a 3D viewer for reconstructed
scenes. No annotating, no saving — viewing only.

Built with **Vite + React + three.js**, the same stack as the embedded apps on
[itamarweiss.com](https://itamarweiss.com) (`apps/photo-geolocation`), and
styled after the site's Geist-dark design (black background, `#ededed` text,
`#3291ff` links, Doto site title). Hash routing (`#/video/...`, `#/scene/...`)
so deep links work from any static mount point without server rewrites.

## Development

```bash
cd apps/fpv-viewer
npm install
npm run dev            # http://localhost:5185
```

`npm run dev`/`build` first runs `scripts/build-data.mjs`, which bundles
`public/data/videos.json` from the repo:

- `tools/annotator.html` — canonical video list (date/description/town/urls)
- `annotations/*_annotations.json` — segment markers (manual preferred over auto)
- `scenes/<stem>/<sceneId>/viewer/` — which videos have a 3D scene
- `apps/fpv-tool/public/thumbnails/manifest.json` — responsive thumb widths + blur

In dev, the Vite server itself serves `/scenes` (from the repo's `scenes/`
directory) and `/thumbnails` (from `apps/fpv-tool/public/thumbnails/`) straight
from disk — **no python or Next server is needed**. Thumbnails fall back to the
remote CloudFront JPG when a local WebP is missing. In production those two URL
prefixes point at CloudFront via `VITE_SCENE_BASE` / `VITE_THUMB_BASE`, and the
videos (the playback clock for the 3D scene view) stream from CloudFront
directly, so nothing dynamic runs anywhere.

## Production build

```bash
VITE_SCENE_BASE=https://<cdn>/scenes \
VITE_THUMB_BASE=https://<cdn>/thumbnails \
npm run build          # -> dist/
```

`dist/` is fully static: videos stream from CloudFront, scene data and
thumbnails from whatever the two env bases point at.

## Production TODOs (deploying to itamarweiss.com)

1. **Generate all thumbnails** (some may be pending due to CDN rate limits):
   `cd apps/fpv-tool && npm run gen-thumbnails` until it reports 152/152.
2. **Upload thumbnails to the CDN bucket** (paths become
   `thumbnails/<slug>/<width>.webp`):

   ```bash
   aws s3 sync apps/fpv-tool/public/thumbnails/ s3://<BUCKET>/thumbnails/ \
     --exclude "manifest.json" \
     --content-type image/webp \
     --cache-control "public,max-age=31536000,immutable"
   ```

3. **Upload scene viewer data** — `scene_meta.json`, the two point bins, and
   the per-frame `camera_view_assets/` (the scene view's corner panel shows the
   actual/render/overlay VGGT frames from there). ~15 MB of bins + ~7 MB of
   frames per scene, ~1.5 GB total:

   ```bash
   cd <dataset-repo>
   for d in scenes/*/*/viewer; do
     aws s3 cp "$d/scene_meta.json"        "s3://<BUCKET>/$d/" ;
     aws s3 cp "$d/points_positions.bin"   "s3://<BUCKET>/$d/" \
       --content-type application/octet-stream --cache-control "public,max-age=31536000" ;
     aws s3 cp "$d/points_colors.bin"      "s3://<BUCKET>/$d/" \
       --content-type application/octet-stream --cache-control "public,max-age=31536000" ;
     aws s3 sync "$d/camera_view_assets/"  "s3://<BUCKET>/$d/camera_view_assets/" \
       --content-type image/jpeg --cache-control "public,max-age=31536000,immutable" ;
   done
   ```

   Future pipeline improvement: have scene generation also emit a compact
   animated artifact (e.g. an animated WebP per view mode) so the frame panel
   is one request instead of ~125.

   Also: `2026-05-26_anti_drone_platform_barashit_02.mp4` is missing from the
   CDN (403) — upload it so its video page can play.

4. **CORS**: the app is served from `itamarweiss.com` but fetches scene bins /
   thumbnails from the CDN — add a CORS policy on the bucket/distribution
   allowing `GET` from `https://itamarweiss.com` (or `*`).
5. **Build with the CDN bases** (step above) and copy `dist/` into the personal
   site as a pre-built embedded app (the `solar-system` pattern):

   ```bash
   cp -R dist/ <itamarwe.github.io>/public/fpv/
   ```

6. **Add rewrites** in the personal site's `next.config.ts`:

   ```ts
   { source: "/fpv",  destination: "/fpv/index.html" },
   { source: "/fpv/", destination: "/fpv/index.html" },
   ```

7. Optional: link it from the site nav/about page, and re-run steps 1–5
   whenever new scenes are reconstructed (`npm run build-data` picks them up).
