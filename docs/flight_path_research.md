# FPV Flight Segment Extraction and Scale Research

Date: 2026-06-30

This note summarizes two practical questions for deriving 3D drone flight paths
from the edited FPV videos:

1. Automatically find flight sections inside edited source videos.
2. Convert relative 3D reconstruction units into meters for AGL height and
   ground speed.

## 1. Flight segment extraction

### Annotation interpretation

The repo currently has 25 annotation JSON files under `annotations/`.
I treated each `flight_start` or `new_flight_start` as the start of a flight
interval and the next non-flight marker as its end. Under that interpretation:

- Annotated flight intervals: 69
- Boundary targets evaluated: 138 starts/ends
- Median flight interval duration: 6.2 s
- Mean flight interval duration: 9.0 s
- End-marker types: 25 `replay_start`, 22 `pause_start`, 22 `other`

The hardest labels are the `other` cuts because they are often visually similar
FPV-to-FPV cuts rather than obvious black frames, pauses, banners, or replays.

### Methods worth considering

#### Classical scene/change detection

FFmpeg's `scene` score and PySceneDetect's content/adaptive detectors are good
baselines for hard cuts, black frames, and sudden visual changes. They are easy
to run at scale and do not need model weights. For this dataset they should be
tuned for high recall, because extra candidate marks are acceptable and missing
flight boundaries is expensive.

Useful references:

- FFmpeg filters documentation: https://ffmpeg.org/ffmpeg-filters.html
- PySceneDetect detectors documentation:
  https://www.scenedetect.com/docs/latest/api/detectors.html

#### Learned shot-boundary detection

TransNetV2 is a stronger model for shot/transition detection, especially where
the transition is not just a single-frame hard cut. It should be tested after
the lightweight baseline, not before, because the baseline already performs well
and is simpler to deploy on all 161 videos.

Reference:

- TransNetV2: https://github.com/soCzech/TransNetV2

#### Dataset-specific interval classification

Boundary detection alone does not answer "is this interval flight?" The next
stage should classify each interval between candidate marks into:

- intro/banner
- flight
- pause/freeze/black
- replay/duplicate
- other overlay/edit

Start with feature rules plus a small learned classifier trained on the current
annotations:

- motion features from low-resolution frame differences
- black/freeze duration
- logo/banner layout features
- OCR or template matching for fixed title cards if needed
- frame-embedding similarity to detect replayed segments

The validation split should be by video, not by segment, so near-duplicate clips
do not leak into train and validation together.

#### Concatenating overlapping flight snippets

After candidate flight intervals are extracted, concatenate consecutive
intervals only when the end of one and the start of the next are visually and
geometrically compatible. A robust sequence is:

1. Compare pHash/DINO/CLIP-like embeddings over the last 1-2 s of segment A and
   the first 1-2 s of segment B.
2. If similarity is high, verify with local feature matches or VGGT/COLMAP pose
   overlap.
3. If a segment is a replay, identify it by sequence similarity against earlier
   flight frames and drop it from the reconstruction input.
4. Otherwise, save segments separately.

### Baseline implemented in this repo

I added `tools/pipeline/evaluate_flight_boundaries.py`.

It streams each annotated CloudFront MP4 with FFmpeg, samples frames, computes
lightweight visual features, detects candidate boundaries, and evaluates whether
each annotated flight start/end has a nearby candidate.

The script caches compact sampled-frame features under `/tmp/fpv-flight-boundaries`
by default, not in the repo.

Commands used:

```bash
python3 tools/pipeline/evaluate_flight_boundaries.py \
  --json-out /tmp/fpv-flight-boundaries/results_baseline.json

python3 tools/pipeline/evaluate_flight_boundaries.py \
  --cut-score 0.25 \
  --peak-quantile 0.80 \
  --merge-sec 0.35 \
  --json-out /tmp/fpv-flight-boundaries/results_high_recall.json

python3 tools/pipeline/evaluate_flight_boundaries.py \
  --fps 10 \
  --cut-score 0.25 \
  --peak-quantile 0.80 \
  --merge-sec 0.30 \
  --json-out /tmp/fpv-flight-boundaries/results_high_recall_fps10.json
```

Results:

| Setting | FPS | Candidates | Median candidates/video | Recall @ 0.5 s | Recall @ 1.0 s | Median timing error |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 5 | 424 | 16 | 86.2% | 94.9% | 0.176 s |
| high recall | 5 | 601 | 23 | 95.7% | 100.0% | 0.100 s |
| high recall | 10 | 651 | 25 | 97.1% | 100.0% | 0.133 s |

Recommendation: use the 5 fps high-recall setting for bulk processing. It hit
all annotated flight starts/ends within 1 second and only cost about 25 candidate
marks per minute of source video. Use 10 fps only when tighter manual review is
needed.

Important limitation: this is only boundary recall. It does not yet decide which
candidate-to-candidate intervals are flight intervals, nor does it remove
replays.

## 2. Scale: pixels/reconstruction units to meters

Implementation/run note: `tools/pipeline/flight_path_pipeline.py` now parses the 25
annotation files into 69 flight intervals, marks the last interval per video as
the attack segment, extracts VGGT-ready frames, parses VGGT `.glb` outputs into
relative camera paths, and writes scale/plot artifacts. See
`docs/flight_path_scale_run.md` for the first run and current quota blocker.

### Camera specs help, but do not solve scale alone

Camera intrinsics matter a lot:

- focal length / field of view
- principal point
- lens distortion
- actual edited-frame crop and aspect ratio

For an image width `W` and horizontal FOV `hfov`, an approximate pixel focal
length is:

```text
fx = (W / 2) / tan(hfov / 2)
```

This improves VGGT/COLMAP geometry and object measurements. But monocular video
still has a scale ambiguity: camera poses and depths can be geometrically
consistent while all distances are multiplied by an unknown constant.

References:

- VGGT: https://github.com/facebookresearch/vggt
- COLMAP FAQ: https://colmap.github.io/faq.html
- OpenCV camera calibration:
  https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

### Best scale strategy

Use the relative 3D path from VGGT/COLMAP, then estimate one global scale factor
from external metric cues. Do not trust a single cue when avoidable.

Recommended cues, strongest first:

1. **Georeferenced scene anchors.** If the target area is known, match road
   corners, building corners, walls, field boundaries, or the target point to
   satellite/orthophoto coordinates. This can solve a similarity transform from
   reconstruction units to local ENU meters.
2. **Known object dimensions.** Tanks, Humvees, Namer APCs, D9 bulldozers,
   windows, doors, shipping containers, road lane widths, and utility poles can
   all provide scale. Use multiple objects/frames and robust fitting, because
   variant, occlusion, and perspective errors are large.
3. **Terrain/DSM/DEM alignment.** Once approximate geolocation is known, sampled
   ground points from the reconstruction should lie near the terrain/DSM surface.
   This is useful for scale and vertical offset, especially over hilly terrain.
4. **Impact/end constraints.** If the final frame is close to a known object or
   ground contact, it can constrain scale, but it is often noisy because impact
   may be on a vehicle, wall, tree, or roof rather than bare ground.
5. **Speed priors.** Typical FPV speed can sanity-check results, but should not
   be the primary scale source.

Copernicus DEM GLO-30 is a useful global starting point, but it is DSM-like and
30 m resolution, so it is not enough for low-altitude urban/vehicle-scale AGL by
itself.

Reference:

- Copernicus DEM:
  https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM

### Practical scale pipeline

1. Extract flight-only segments and remove replays/pauses.
2. Run VGGT/COLMAP on each continuous flight segment.
3. Estimate intrinsics from camera specs or calibrate the same camera/lens class.
4. Pick metric constraints:
   - known vehicle/object dimensions in frames
   - map/orthophoto feature distances
   - terrain/DSM ground-contact constraints
5. Fit one scale factor, and if georeferenced anchors exist, a full similarity
   transform:

```text
metric_point = s * R * reconstruction_point + t
```

6. Compute height above ground:

```text
AGL(t) = camera_z_metric(t) - terrain_or_surface_elevation(x(t), y(t))
```

7. Compute ground speed from smoothed metric camera centers:

```text
ground_speed(t) = horizontal_distance(camera_center[t], camera_center[t-1]) / dt
```

8. Report uncertainty bands, not just a single path:
   - one scale from object dimensions
   - one scale from map anchors
   - one scale from terrain alignment
   - final fused estimate with residuals

### Questions that would materially improve scale

- What exact camera/lens/FOV is used, and is the uploaded video cropped or
  stabilized relative to the original?
- Are the videos original files or social-media re-encodes?
- For each target, do we have a confirmed coordinate or only town-level
  geolocation?
- Which object classes should be treated as known-size priors in this dataset
  (Merkava, Humvee, Namer, D9, buildings, doors/windows, road lanes)?
- Can we manually mark a few object-size constraints in the same annotator UI?
