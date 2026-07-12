# Contributing — adding videos to the dataset

This dataset's media lives in a **private** S3 bucket (`fpv-drone-strikes-lebanon-dataset`)
and is served publicly through CloudFront. Public links must use the CloudFront domain
`d2fioemadmrru3.cloudfront.net` — direct S3 URLs are blocked (HTTP 403) by design.

## Naming

Use `YYYY-MM-DD_short_description` (lowercase, underscores). Use the **same base name**
for the video and its thumbnail.

## Add a video

You need AWS credentials with write access to the bucket (already configured on the
upload machine).

```bash
npm install
npm run dataset:add -- \
  --video /path/to/video.mp4 \
  --thumbnail /path/to/thumbnail.jpg \
  --metadata /path/to/catalog-record.json
```

This uploads and verifies media before updating `data/catalog.json`, then
regenerates README and annotator views. Do not edit those generated files by
hand. See `docs/ingest_instructions.md` for the metadata contract.

## 3. Commit & push

Run `npm run catalog:check`, then commit the catalog and generated changes and
open a pull request. Publishing runs after merge when AWS OIDC is configured.

## Important

- Always use the `d2fioemadmrru3.cloudfront.net` domain in links.
- **Do not** make the bucket public or add public-read policies. Abuse protection
  (per-IP rate limiting + a cost alert) depends on the bucket staying private behind
  CloudFront.
