# OpenClaw FPV ingest handoff

Repository: `itamarwe/fpv-drone-strikes-lebanon-dataset`

Public asset base: `https://d2fioemadmrru3.cloudfront.net`

Never commit credentials, local environment files, downloaded media, or AWS
configuration. Never make the S3 bucket public.

## Current workflow

Until `data/catalog.json` and the `dataset:*` commands land:

1. Derive one canonical stem: `YYYY-MM-DD_short_description`, lowercase with
   underscores.
2. Check README, the annotator list, and S3 for the stem and Telegram message ID
   before uploading. Stop on a likely duplicate.
3. Upload the MP4 to `videos/<stem>.mp4` and JPG to
   `thumbnails/<stem>.jpg` in the private dataset bucket.
4. Verify both CloudFront URLs return HTTP 200.
5. Add the video to both `README.md` and `tools/annotator.html`, using identical
   date, description, town, thumbnail URL, and video URL.
6. Open a pull request. Do not edit the `itamarwe.github.io` repository: the
   website reads the CloudFront manifest.
7. Run `npm run publish-web:fast` after the repository change is accepted so
   responsive thumbnails and `data/videos.json` are updated.
8. Run `npm run check-public` after publishing and confirm the new card, video,
   and thumbnail resolve. The current `npm run audit` requires an annotation, so
   run it after the annotation stage until the catalog-aware validator replaces
   it.

Upload assets before publishing references. Publish `data/videos.json` last.

## Target workflow

After the catalog migration, replace steps 3-8 with:

```bash
npm run dataset:add -- \
  --video /path/to/video.mp4 \
  --thumbnail /path/to/thumbnail.jpg \
  --metadata /path/to/catalog-record.json
```

Then open a PR containing the catalog change. OpenClaw must not edit generated
README rows, annotator lists, redirects, or the website repository directly.
The post-merge publisher will generate and publish those views.

## Required metadata

Provide:

- canonical `id`/stem and date
- concise English description
- original Arabic title when available
- town/location, with `approx` when uncertain
- Telegram source URL, channel, message ID, and discovery timestamp
- source video and thumbnail paths

The Telegram channel + message ID is the ingestion idempotency key.

## Annotation and scene stages

OpenClaw only performs discovery and initial media ingestion. Later tools attach
data using the same video ID:

```bash
npm run dataset:annotate -- --id <video-id> --annotation <file>
npm run dataset:add-scene -- --id <video-id> --scene <directory>
```

Neither stage creates a second catalog entry.

## Renames

Never rename S3 keys or Git files ad hoc. Use:

```bash
npm run dataset:rename -- --from <old-id> --to <new-id> --reason "..."
```

This must copy assets, update catalog/annotation/scene references, add redirect
records, publish the new manifest, verify old and new URLs, and only then remove
old objects.

## Failure behavior

- If upload succeeds but Git/validation fails, leave the object unreferenced and
  report its key; do not publish a partial manifest.
- If CloudFront verification fails, do not add or merge the catalog record.
- If metadata is uncertain, mark it for review instead of guessing a filename.
- If a possible duplicate exists, stop and report both records.
