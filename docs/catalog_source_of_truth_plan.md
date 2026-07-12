# Catalog source-of-truth implementation plan

## End state

The FPV repository owns all durable dataset state:

- `data/catalog.json` identifies every video and its descriptive metadata.
- `annotations/<video-id>_annotations.json` owns annotation content.
- `scene-manifests/<video-id>/<scene-id>.json` records lightweight scene metadata.
- `data/redirects.json` records old-to-current identities and review dates.

S3/CloudFront contains media and generated runtime artifacts. `README.md`, the
annotator video list, and `data/videos.json` are generated projections. The
`itamarweiss.com/fpv` application only consumes `data/videos.json`; it does not
own a second catalog.

## Catalog record

```json
{
  "id": "2026-05-26_anti_drone_platform_biranit",
  "date": "2026-05-26",
  "description": "Iron Dome platform, Biranit barracks...",
  "original_title": "...",
  "town": "Biranit barracks (Israel), approx",
  "provenance": {
    "source": "telegram",
    "source_url": "https://t.me/...",
    "discovered_at": "2026-07-12T00:00:00Z"
  },
  "media": {
    "video_key": "videos/2026-05-26_anti_drone_platform_biranit.mp4",
    "thumbnail_key": "thumbnails/2026-05-26_anti_drone_platform_biranit.jpg"
  }
}
```

The stable `id` is the filename stem. Optional annotations and scenes are joined
by this ID rather than copied into the catalog. A missing annotation or scene is
a valid lifecycle state, not a validation failure. Derived status values are:
`media_published`, `annotated`, and `scene_published`.

## Redirect record

```json
{
  "from": "2026-05-26_anti_drone_platform_barashit",
  "to": "2026-05-26_anti_drone_platform_biranit",
  "created_at": "2026-07-12",
  "review_after": "2026-10-15",
  "reason": "Correct place-name spelling"
}
```

One redirect record generates both CloudFront asset redirects and website route
redirects. Systematic migrations, such as underscore dates, remain compact
rules. Semantic renames use explicit records.

## Commands

The final operator interface should be:

```bash
npm run dataset:add -- --video /path/video.mp4 --thumbnail /path/thumb.jpg --metadata /path/record.json
npm run dataset:annotate -- --id <video-id> --annotation /path/annotation.json
npm run dataset:add-scene -- --id <video-id> --scene /path/scene
npm run dataset:rename -- --from <old-id> --to <new-id> --reason "..."
npm run dataset:publish
npm run dataset:verify
```

Each mutation validates locally and changes Git-owned files. Publishing is a
separate, idempotent step.

## Transaction order

Every publish follows the same order:

1. Validate schema, unique IDs, names, references, and redirect chains.
2. Generate README and the annotator/web projections in a temporary directory.
3. Upload new video, thumbnail, annotation, and scene objects.
4. Verify every newly referenced CloudFront URL returns success.
5. Deploy redirect data/rules.
6. Publish `data/videos.json` last, using an atomic S3 object replacement.
7. Run the complete public integrity check.

If any step before 6 fails, the public manifest remains unchanged. Old objects
are removed only after redirects and the new manifest have been verified.

## Implementation phases

### 1. Introduce the catalog without changing production

- Add JSON Schemas for `catalog.json`, redirects, and scene manifests.
- Import the current 149 README-backed records.
- Add a validator that compares catalog, README, annotator, annotations, and S3
  while allowing catalog records that do not yet have annotations or scenes.
- Require generated output to match the existing checked-in output exactly.

Acceptance: catalog validation passes and the generated manifest has no semantic
diff from production.

### 2. Generate duplicated views

- Generate the README video table from the catalog.
- Generate annotator input JSON from the catalog; remove embedded `const VIDEOS`.
- Make `build_web_data.mjs` consume catalog + annotations + scene manifests.
- Add `git diff --exit-code` checks for generated files in CI.

Acceptance: editing the catalog is the only way to add or rename a video.

### 3. Add transactional publishing

- Implement the six `dataset:*` commands above.
- Publish changed assets first and `videos.json` last.
- Preserve scene references when publishing from a clean checkout.
- Record a release report containing uploaded keys and verification results.

Acceptance: a failed upload cannot expose a broken card on the website.

### 4. Automate after merge

- Run validation on pull requests.
- Publish on merge to `main` through GitHub Actions and AWS OIDC.
- Keep credentials out of GitHub secrets where OIDC roles can be used.
- Prevent concurrent publishers with one workflow concurrency group.

Acceptance: merging a valid dataset PR updates CloudFront and the website without
a separate website commit or manual deployment.

### 5. Switch ingest producers to the catalog contract

- The Telegram crawler uploads media, then opens a PR containing one catalog record.
- It never edits README or the website repository directly.
- It records Telegram provenance and an idempotency key/message ID.
- Reprocessing the same Telegram post must update or skip the existing record,
  never create a duplicate.

Acceptance: the crawler can ingest the same message twice without duplicate media,
catalog rows, or README entries.

## Pull-request split

1. Catalog schema, importer, validator, and generated-output comparison.
2. README/annotator/manifest generators switched to catalog input.
3. Transactional publisher, redirects generator, and integrity report.
4. GitHub Actions OIDC publishing.
5. Crawler integration against the stable ingest command contract.

This keeps each behavioral change reviewable and preserves the current working
pipeline throughout the migration.
