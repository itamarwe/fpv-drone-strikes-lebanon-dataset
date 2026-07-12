# Ops

Infrastructure config for the dataset's S3 bucket + CloudFront distribution
(`s3://fpv-drone-strikes-lebanon-dataset` → `d2fioemadmrru3.cloudfront.net`,
distribution `E1FTYLW4OET6KU`).

## WAF rate limiting (`waf-rules.json`)

The distribution is protected by the WAFv2 WebACL `fpv-dataset-ratelimit`
(scope `CLOUDFRONT`, region `us-east-1`, id `29b603ed-8495-4c3e-a74f-5601cbba7e4c`),
default action **Allow**, with two per-IP rate rules:

| Rule | Scope | Limit (per IP / 5 min) | Purpose |
| --- | --- | --- | --- |
| `RateLimitVideos` | path starts with `/videos/` | 300 | throttle bulk MP4 scraping / hotlinking |
| `RateLimitStatic` | all requests | 3000 | generous ceiling for the web viewer (thumbnails, scenes, frames, `data/videos.json`); still stops asset-library scrapers |

The video request counts toward both, so the 300 cap on `/videos/` always holds.
`RateLimitStatic` was sized for the viewer: a gallery load (~150 thumbnails) plus
several 3D scenes (~30 requests each, since the scene view windows its frame
preloading) stays well under 3000.

`waf-rules.json` uses base64 for the byte-match string (`L3ZpZGVvcy8=` = `/videos/`),
as the AWS CLI expects for that blob field.

### Apply

```bash
export AWS_PROFILE=admin   # or a profile with wafv2 write access
LOCK=$(aws wafv2 get-web-acl --scope CLOUDFRONT --region us-east-1 \
  --name fpv-dataset-ratelimit --id 29b603ed-8495-4c3e-a74f-5601cbba7e4c \
  --query LockToken --output text)
aws wafv2 update-web-acl --scope CLOUDFRONT --region us-east-1 \
  --name fpv-dataset-ratelimit --id 29b603ed-8495-4c3e-a74f-5601cbba7e4c \
  --lock-token "$LOCK" --default-action Allow={} \
  --visibility-config SampledRequestsEnabled=true,CloudWatchMetricsEnabled=true,MetricName=fpv-dataset-ratelimit \
  --rules file://ops/waf-rules.json
```

Tune limits by editing `waf-rules.json` and re-applying.

## Temporary filename redirects

`cloudfront-uri-redirects.js` is associated with the distribution's default
cache behavior as the `fpv-filename-redirects` viewer-request function. It sends
HTTP 308 responses for legacy underscore-date annotation URLs and for the
`barashit` to `biranit` correction.

Retire the function after **2026-10-15**, once old links have had a three-month
migration window. Before removing it, check CloudFront logs for requests that
still match either legacy pattern. The migration record is in
`filename-migration.json`.
