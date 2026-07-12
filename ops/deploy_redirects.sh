#!/usr/bin/env bash
set -euo pipefail

NAME="${FPV_REDIRECT_FUNCTION:-fpv-filename-redirects}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO"
node tools/catalog/generate_redirect_function.mjs
ETAG="$(aws cloudfront describe-function --name "$NAME" --stage DEVELOPMENT --query ETag --output text)"
ETAG="$(aws cloudfront update-function \
  --name "$NAME" \
  --if-match "$ETAG" \
  --function-config "Comment=Generated from data/redirects.json,Runtime=cloudfront-js-2.0" \
  --function-code fileb://ops/cloudfront-uri-redirects.js \
  --query ETag --output text)"
aws cloudfront publish-function --name "$NAME" --if-match "$ETAG" >/dev/null
echo "published CloudFront redirect function: $NAME"
