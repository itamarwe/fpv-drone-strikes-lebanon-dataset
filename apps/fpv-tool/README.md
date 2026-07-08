# FPV Tool — Next.js front end

Next.js App Router shell for the FPV annotation and 3D scene viewer. It is
structured to drop into [itamarwe.github.io](https://github.com/itamarwe/itamarwe.github.io)
the same way `apps/photo-geolocation/` is embedded today.

## Architecture

```text
Browser
  └─ Next.js (apps/fpv-tool)          port 3001
       ├─ /annotate/                  legacy annotator UI (static, synced from tools/)
       ├─ /scenes/                    scene browser
       ├─ /scenes/.../viewer/         generic viewer + per-scene data
       ├─ /tools/scene_viewer/        shared viewer JS/CSS/assets
       └─ /api/*                       proxied ──► Python fpv_tool_server  port 8766
```

| Layer | Responsibility |
|-------|----------------|
| **Next.js** | UI routes, scene static files, API proxy |
| **Python** (`tools/fpv_tool_server.py`) | VGGT jobs, ffmpeg, annotations FS, reconstruction |
| **Scene data** (`scenes/<video>/<scene_id>/viewer/`) | `scene_meta.json`, point clouds, camera frames |
| **Generic viewer** (`tools/scene_viewer/`) | One Three.js app for all scenes |

## Local development

Terminal 1 — Python backend (VGGT / annotations):

```bash
python tools/fpv_tool_server.py --host 127.0.0.1 --port 8766
```

Terminal 2 — Next.js:

```bash
cd apps/fpv-tool
cp .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3001/

## Integrating into itamarwe.github.io

Target mount path (suggested): `/research/fpv-drone-strikes/`

1. Copy `apps/fpv-tool/` into the blog repo under `apps/fpv-tool/`.
2. Add to root `package.json`:
   ```json
   "build:fpv-tool": "cd apps/fpv-tool && npm ci --no-audit --no-fund && npm run build",
   "build": "npm run build:photo-geolocation && npm run build:fpv-tool && next build"
   ```
3. Set `NEXT_PUBLIC_BASE_PATH=/research/fpv-drone-strikes` when building the embedded app.
4. Add rewrites in `next.config.ts` for the sub-path (same pattern as photo-geolocation).
5. Host the Python API separately (not on Vercel). Set `FPV_PYTHON_API_URL` in Vercel env
   to that host for `/api/*` proxying.

**Read-only mode on Vercel:** ship pre-built `scenes/` under `public/` and omit the Python
proxy — viewers work, reconstruction does not.

## CloudFront asset root

FPV videos in the annotator already use full CloudFront URLs (`d2fioemadmrru3.cloudfront.net`).
Scene binaries use the same pattern via `FPV_ASSET_ROOT`:

| Env | Default | Production |
|-----|---------|------------|
| `FPV_ASSET_ROOT` | *(empty — local `/scenes/...`)* | `https://d2fioemadmrru3.cloudfront.net` |

`scene_meta.json` keeps **relative** paths (`points_positions.bin`, `camera_view_assets/...`).
The viewer shell injects `SCENE_BASE` at serve time:

- Local: `/scenes/<video>/<scene_id>/viewer/`
- CDN: `https://d2fioemadmrru3.cloudfront.net/scenes/<video>/<scene_id>/viewer/`

Upload layout on S3 (mirror repo paths):

```text
s3://fpv-drone-strikes-lebanon-dataset/scenes/<video>/<scene_id>/viewer/
  scene_meta.json
  points_positions.bin
  points_colors.bin
  camera_view_assets/...
```

The viewer page itself stays on your site; only heavy scene data is fetched from CloudFront.

## Environment

See `.env.example`.

## Migration path

1. **Now** — legacy HTML UIs + generic viewer, API proxied to Python
2. **Next** — React pages replace annotator / scene browser incrementally
3. **Later** — TypeScript API routes replace Python proxy where Vercel-compatible;
   keep Python worker for VGGT only
