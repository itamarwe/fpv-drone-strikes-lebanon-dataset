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

In dev, the Vite server proxies `/scenes` to the python tool server
(`127.0.0.1:8766`) and `/thumbnails` to the fpv-tool Next app (`127.0.0.1:3001`).
Thumbnails fall back to the remote CloudFront JPG when the local WebP is missing.

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

3. **Upload scene viewer data** — only `scene_meta.json` + the two point bins
   are needed (~15 MB per scene, ~1 GB total; skip the heavy per-frame
   `camera_view_assets/`):

   ```bash
   cd <dataset-repo>
   for d in scenes/*/*/viewer; do
     aws s3 cp "$d/scene_meta.json"        "s3://<BUCKET>/$d/" ;
     aws s3 cp "$d/points_positions.bin"   "s3://<BUCKET>/$d/" \
       --content-type application/octet-stream --cache-control "public,max-age=31536000" ;
     aws s3 cp "$d/points_colors.bin"      "s3://<BUCKET>/$d/" \
       --content-type application/octet-stream --cache-control "public,max-age=31536000" ;
   done
   ```

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
