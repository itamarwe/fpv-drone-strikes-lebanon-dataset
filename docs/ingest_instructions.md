# FPV ingest instructions

Repository: `itamarwe/fpv-drone-strikes-lebanon-dataset`

Public asset base: `https://d2fioemadmrru3.cloudfront.net`

Never commit credentials, local environment files, downloaded media, or AWS
configuration. Never make the S3 bucket public.

## Workflow

1. Derive one canonical stem: `YYYY-MM-DD_short_description`, lowercase with
   underscores.
2. Check `data/catalog.json` for the stem and Telegram message ID. Stop on a
   likely duplicate.
3. Create a metadata JSON file using the fields below.
4. Run:

```bash
npm run dataset:add -- \
  --video /path/to/video.mp4 \
  --thumbnail /path/to/thumbnail.jpg \
  --metadata /path/to/catalog-record.json
```

5. Run `npm run catalog:check` and open a PR containing the catalog and generated
   changes.

The command uploads media first, verifies both CloudFront URLs, then updates the
catalog. Ingest producers must not edit generated README rows, annotator lists,
redirects, or the website repository directly. The website reads the published
manifest and needs no per-video commit.

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

The ingest producer only performs discovery and initial media ingestion. Later tools attach
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
