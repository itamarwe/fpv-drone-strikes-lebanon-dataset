#!/usr/bin/env bash
# Publish everything the public web viewer (itamarweiss.com/fpv) consumes to
# S3/CloudFront. The git repo stays the source of truth for annotations and
# tooling; S3 is the home of all runtime data:
#
#   thumbnails/<slug>/<w>.webp        responsive gallery thumbnails
#   scenes/<stem>/<id>/viewer/...     3D scene data (meta + point bins + frames)
#   annotations/<name>.json           published copy of the annotations
#   data/videos.json                  the app manifest (list + annotations + scene index)
#
# Requires AWS credentials with write access to the bucket. Run from anywhere:
#   npm run publish-web                  # full publish (thumbnails, scenes, data)
#   npm run publish-web:fast             # fast: annotations + data + thumbnails only
set -euo pipefail

BUCKET="${FPV_BUCKET:-s3://fpv-drone-strikes-lebanon-dataset}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SKIP_SCENES=0
[ "${1:-}" = "--skip-scenes" ] && SKIP_SCENES=1

cd "$REPO"

mkdir -p build/web
curl -fsSL "${FPV_CDN_BASE:-https://d2fioemadmrru3.cloudfront.net}/data/videos.json" \
  -o build/web/current-videos.json

echo "== 1/5 generate thumbnails (incremental) =="
node tools/publishing/gen_thumbnails.mjs

echo "== 2/5 bake calibration + build web data manifest =="
node tools/publishing/apply_calibration.mjs
node tools/publishing/build_web_data.mjs

echo "== 3/5 upload thumbnails =="
aws s3 sync build/thumbnails/ "$BUCKET/thumbnails/" \
  --exclude "manifest.json" \
  --content-type image/webp \
  --cache-control "public,max-age=31536000,immutable"

if [ "$SKIP_SCENES" = "0" ]; then
  echo "== 4/5 upload scene viewer data (incremental, base only) =="
  for d in scenes/*/*/viewer; do
    [ -f "$d/scene_meta.json" ] || continue
    # Production ships base reconstructions only; skip density variants.
    case "$d" in *__*) continue;; esac
    # sync skips unchanged files (size+mtime), so re-publishing is cheap
    aws s3 sync "$d/" "$BUCKET/$d/" \
      --exclude "*" --include "scene_meta.json" \
      --content-type application/json --cache-control "public,max-age=300"
    aws s3 sync "$d/" "$BUCKET/$d/" \
      --exclude "*" --include "*.bin" \
      --content-type application/octet-stream \
      --cache-control "public,max-age=31536000"
    [ -d "$d/camera_view_assets" ] && aws s3 sync "$d/camera_view_assets/" \
      "$BUCKET/$d/camera_view_assets/" \
      --content-type image/jpeg --cache-control "public,max-age=31536000,immutable"
  done
else
  echo "== 4/5 (skipped scenes) =="
fi

echo "== 5/5 upload annotations + redirects + data manifest =="
aws s3 sync annotations/ "$BUCKET/annotations/" \
  --exclude "*" --include "*.json" \
  --content-type application/json --cache-control "public,max-age=300"
aws s3 cp data/redirects.json "$BUCKET/data/redirects.json" \
  --content-type application/json --cache-control "public,max-age=300"

if [ "${FPV_DEPLOY_REDIRECTS:-0}" = "1" ]; then
  ops/deploy_redirects.sh
fi

# The manifest is the public commit point: assets and redirects are available
# before readers can observe the new catalog state.
aws s3 cp build/web/videos.json "$BUCKET/data/videos.json" \
  --content-type application/json --cache-control "public,max-age=300"

echo "done: published to $BUCKET"
