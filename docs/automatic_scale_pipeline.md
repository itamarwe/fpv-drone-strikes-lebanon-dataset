# Automatic Scale Pipeline

Date: 2026-06-30

Goal: estimate metric scale for VGGT FPV reconstructions without relying on
median drone speed. The path should remain automatic, but the result must carry
confidence and reject weak cues that disagree.

## Core Idea

VGGT gives a relative camera path in arbitrary scene units. To convert it to
meters, we need one or more metric references. Instead of asking for one manual
reference, we can generate many weak automatic scale hypotheses:

```text
meters_per_vggt_unit = metric_distance_to_target / relative_vggt_distance_to_terminal
```

Each detection produces a vote. A robust majority vote chooses the densest
agreement cluster in log scale.

## Automatic Cues

### 1. Known Object Size

Detect objects whose dimensions are approximately known:

- Merkava / tank
- Humvee / HMMWV
- Namer / APC
- D9 / bulldozer
- soldier / person
- door
- window
- building story
- tree

For each detection:

```text
distance_m ~= known_object_size_m * focal_pixels / observed_object_size_pixels
scale_m_per_unit = distance_m / distance_vggt_units(frame_camera, terminal_camera)
```

This requires camera FOV. It is still noisy because object pose, cropping, and
partial visibility matter. Vehicle-width and vehicle-height votes should not be
trusted individually; they become useful only when many frames agree.

### 2. Metric Depth

Use a metric monocular depth model such as Depth Anything V2 Metric to estimate
object or ground distance in meters:

```text
scale_m_per_unit = median_depth_m_inside_detection / relative_vggt_distance_to_terminal
```

Depth is a separate model and can drift by scene/domain, but it is independent
of object-size priors. Agreement between metric-depth votes and object-size
votes is valuable.

### 3. Building Repetition

For buildings, count repeated windows/stories:

```text
known_height_m ~= story_count * 3.0
```

This is weaker than vehicle dimensions, but often available when vehicles are
small or occluded.

### 4. Map Anchors

When map-visible features exist, match reconstructed scene points to:

- road corners
- building corners
- wall ends
- field/road boundaries
- target coordinate and a second anchor

Two anchors provide scale; three or more anchors can solve a full similarity
transform into local ENU meters. This is stronger than object-size-only scale.

### 5. Terrain / DSM

After georeferencing, align reconstructed low/ground points to DEM/DSM. This
can improve vertical offset and AGL. It does not solve scale alone unless the XY
path is already georegistered enough to sample terrain.

## Voting

`tools/auto_scale_hypotheses.py` emits `auto_scale_votes.csv` where each row is
one scale vote:

- object label
- detection confidence
- prior class
- method
- metric distance estimate
- relative VGGT distance
- `scale_m_per_unit`
- weight

Then it votes in log scale:

1. Convert all scale votes to `log(scale)`.
2. Find the densest weighted window, default factor `1.7`.
3. Use the weighted median of that inlier cluster.
4. Report inlier count, outlier count, inlier weight fraction, methods, and
   object priors represented in the winning cluster.

This avoids taking an average across incompatible cues, for example a partially
visible tree and a full vehicle width.

## FPV HUD / Overlay Filtering

Before creating scale votes, the estimator now rejects common FPV overlay
artifacts:

- top-band propeller/HUD strokes
- central reticle horizontal strokes
- small central reticle glyph boxes
- fixed lower-left colored HUD marker boxes

This is important because open-vocabulary detectors can label overlay stripes as
objects such as `namer apc`. The filter is enabled by default for `estimate` and
`visualize`. It can be disabled with `--disable-artifact-filter`, or tuned with
`--hud-min-aspect` and `--hud-top-fraction`.

## Commands

Write object priors and a detection template:

```bash
python3 tools/auto_scale_hypotheses.py write-priors \
  --out /tmp/fpv-flight-paths/auto_scale_object_priors.json

python3 tools/auto_scale_hypotheses.py write-detection-template \
  --out /tmp/fpv-flight-paths/detections_template.csv
```

Run optional zero-shot detection, if `transformers` is installed:

```bash
python3 tools/auto_scale_hypotheses.py detect-zeroshot \
  --recon-dir /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap \
  --out /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/auto_detections.csv \
  --model google/owlv2-base-patch16-ensemble \
  --threshold 0.12 \
  --frame-step 2
```

Estimate scale from detections:

```bash
python3 tools/auto_scale_hypotheses.py estimate \
  --recon-dir /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap \
  --detections /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/auto_detections.csv \
  --hfov-deg 90
```

Outputs:

- `auto_scale_votes.csv`
- `auto_scale_summary.json`

Visualize the object measurements and vote cluster:

```bash
python3 tools/auto_scale_hypotheses.py visualize \
  --recon-dir /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap \
  --detections /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/owlv2_detections.csv \
  --hfov-deg 90
```

This writes:

- `scale_visuals/scale_detection_examples.png`
- `scale_visuals/scale_vote_histogram.png`
- `scale_visuals/scale_vote_cluster.png`

## Current Status

The automatic scale-voting engine is implemented and a first OWLv2 zero-shot
test was run on the Sholef attack-overlap reconstruction:

```bash
/tmp/fpv-model-benchmark/venv/bin/python tools/auto_scale_hypotheses.py detect-zeroshot \
  --recon-dir /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap \
  --out /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/owlv2_detections.csv \
  --threshold 0.08 \
  --frame-step 12 \
  --device cpu

python3 tools/auto_scale_hypotheses.py estimate \
  --recon-dir /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap \
  --detections /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/owlv2_detections.csv \
  --hfov-deg 90
```

Result:

- Detections: 202 across 4 sampled frames
- HUD/overlay detections rejected before voting: 26
- Scale votes after label/size-prior/HUD filtering: 169
- Winning cluster: 37 votes
- Inlier weight fraction: 0.34
- Quality: `weak`
- Output files:
  - `/tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/owlv2_detections.csv`
  - `/tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/auto_scale_votes.csv`
  - `/tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/auto_scale_summary.json`

This is not yet strong enough to promote to measured scale. It is useful as the
first vote cloud and shows that the quality gate is doing the right thing:
noisy detector outputs create hypotheses, but the system marks them weak unless
there is enough agreement.

The strongest next step is to add a metric-depth cue and a faster/more
domain-appropriate detector. Agreement between object-size votes and metric
depth votes should be required before accepting automatic scale.

## Why This Is Better Than Median Speed

Median speed is a behavioral prior. It can make any path look plausible by
choosing a convenient speed. Object/map/depth votes are scene measurements.
They can fail, but they fail in ways we can inspect and reject. Majority voting
lets the pipeline say:

```text
three vehicle-size cues + two depth cues agree -> plausible scale
window/tree cues disagree -> outliers
no agreement -> scale unresolved
```
