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
#   tools/publish_web.sh                 # full publish (thumbnails, scenes, data)
#   tools/publish_web.sh --skip-scenes   # fast: annotations + data + thumbnails only
set -euo pipefail

BUCKET="${FPV_BUCKET:-s3://fpv-drone-strikes-lebanon-dataset}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SKIP_SCENES=0
[ "${1:-}" = "--skip-scenes" ] && SKIP_SCENES=1

cd "$REPO"

echo "== 1/5 generate thumbnails (incremental) =="
node tools/gen_thumbnails.mjs

echo "== 2/5 build web data manifest =="
node tools/build_web_data.mjs

echo "== 3/5 upload thumbnails =="
aws s3 sync build/thumbnails/ "$BUCKET/thumbnails/" \
  --exclude "manifest.json" \
  --content-type image/webp \
  --cache-control "public,max-age=31536000,immutable"

if [ "$SKIP_SCENES" = "0" ]; then
  echo "== 4/5 upload scene viewer data =="
  for d in scenes/*/*/viewer; do
    [ -f "$d/scene_meta.json" ] || continue
    aws s3 cp "$d/scene_meta.json" "$BUCKET/$d/" \
      --content-type application/json --cache-control "public,max-age=300"
    for bin in points_positions.bin points_colors.bin; do
      [ -f "$d/$bin" ] && aws s3 cp "$d/$bin" "$BUCKET/$d/" \
        --content-type application/octet-stream \
        --cache-control "public,max-age=31536000"
    done
    [ -d "$d/camera_view_assets" ] && aws s3 sync "$d/camera_view_assets/" \
      "$BUCKET/$d/camera_view_assets/" \
      --content-type image/jpeg --cache-control "public,max-age=31536000,immutable"
  done
else
  echo "== 4/5 (skipped scenes) =="
fi

echo "== 5/5 upload annotations + data manifest =="
aws s3 sync annotations/ "$BUCKET/annotations/" \
  --exclude "*" --include "*.json" \
  --content-type application/json --cache-control "public,max-age=300"
# Short cache on the manifest so updates show up within minutes, no
# CloudFront invalidation needed.
aws s3 cp build/web/videos.json "$BUCKET/data/videos.json" \
  --content-type application/json --cache-control "public,max-age=300"

echo "done: published to $BUCKET"
