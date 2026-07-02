# Flight Path Scale Run

Date: 2026-06-30

This note records the first scale/path pipeline run over the 25 annotated FPV
videos.

## What Was Built

The reusable pipeline is `tools/flight_path_pipeline.py`.

It supports:

1. Parsing annotation JSON files into flight intervals.
2. Marking the last flight interval in each video as `is_attack=true`.
3. Extracting annotated flight frames into one frame sequence per video.
4. Scoring which earlier flight segments overlap the attack segment.
5. Building `attack_reconstructions/` groups that include the attack segment
   plus only overlapping earlier flight segments.
6. Running VGGT on those frame sequences through `facebook/vggt-omega`.
7. Parsing VGGT `.glb` files into per-frame relative camera paths.
8. Generating scale reports and plots for relative path, speed, height proxy,
   and scene point cloud plus path.

Heavy artifacts are written outside the repo by default:

```text
/tmp/fpv-flight-paths/
```

## Current Run

Commands used:

```bash
python3 tools/flight_path_pipeline.py \
  --out-dir /tmp/fpv-flight-paths \
  manifest

python3 tools/flight_path_pipeline.py \
  --out-dir /tmp/fpv-flight-paths \
  extract-frames \
  --video-cache-dir /tmp/fpv-model-benchmark/videos \
  --sample-fps 2 \
  --width 960

/Users/itamarwe/Documents/code/itamarwe.github.io/research/fpv-drone-strikes/vggt/venv/bin/python \
  tools/flight_path_pipeline.py \
  --out-dir /tmp/fpv-flight-paths \
  run-vggt \
  --max-frames 180 \
  --vggt-timeout 900

/Users/itamarwe/Documents/code/itamarwe.github.io/research/fpv-drone-strikes/vggt/venv/bin/python \
  tools/flight_path_pipeline.py \
  --out-dir /tmp/fpv-flight-paths \
  extract-vggt

python3 tools/flight_path_pipeline.py \
  --out-dir /tmp/fpv-flight-paths \
  scale-report

python3 tools/flight_path_pipeline.py \
  --out-dir /tmp/fpv-flight-paths \
  visualize

python3 tools/flight_path_pipeline.py \
  --out-dir /tmp/fpv-flight-paths \
  analyze-overlap \
  --threshold 0.72 \
  --samples-per-segment 8

python3 tools/flight_path_pipeline.py \
  --out-dir /tmp/fpv-flight-paths \
  extract-group-frames \
  --recon-subdir attack_reconstructions \
  --refresh
```

Artifacts produced:

- `/tmp/fpv-flight-paths/flight_segments.csv`
- `/tmp/fpv-flight-paths/flight_segments.json`
- `/tmp/fpv-flight-paths/videos.json`
- `/tmp/fpv-flight-paths/reconstructions/<video_id>/frames/`
- `/tmp/fpv-flight-paths/reconstructions/<video_id>/frames.csv`
- `/tmp/fpv-flight-paths/reports/reconstructions/scale_report.md`
- `/tmp/fpv-flight-paths/reports/reconstructions/scale_report.csv`
- `/tmp/fpv-flight-paths/reports/attack_reconstructions/scale_report.md`
- `/tmp/fpv-flight-paths/reports/attack_reconstructions/scale_report.csv`
- `/tmp/fpv-flight-paths/plots/flight_segment_durations.png`
- `/tmp/fpv-flight-paths/overlap_report.csv`
- `/tmp/fpv-flight-paths/reconstruction_groups.json`
- `/tmp/fpv-flight-paths/attack_reconstructions/<group_id>/frames.csv`
- `/tmp/fpv-flight-paths/plots/<recon_subdir>/<video_id>_scene_path.png`

## Coverage

Annotation parsing:

- Videos: 25
- Flight intervals: 69
- Attack intervals: 25

Frame extraction:

- Videos with sampled flight frames: 25
- Total sampled frames at 2 fps: 1,245
- Per-video frame count range: 26 to 119

VGGT reconstruction:

- Completed `.glb` files: 2
- Parsed relative paths: 2
- Remaining videos: 23

Overlap grouping:

- Non-attack flight intervals tested: 44
- Included by attack-overlap appearance gate: 31
- Excluded by gate: 13
- Reconstruction groups: 25

Important consequence: `2026-06-03_merkava_tank_zawtar_al_sharqiyah` was
already reconstructed in the older all-segment mode, but the overlap gate keeps
only its attack segment. Treat that older all-segment VGGT output as a diagnostic
artifact, not a valid combined attack-scene reconstruction.

`2026-06-06_sholef_howitzer_adaissah` passed the gate for all three segments, so
its completed VGGT output was safely reused under:

```text
/tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/
```

The batch stopped because the public Hugging Face ZeroGPU Space returned:

```text
You have exceeded your ZeroGPU quota (60s requested vs. 0s left).
Authenticate with a Hugging Face token for more quota.
```

The pipeline is resumable: rerunning `run-vggt` skips existing `.glb` files.
If a Hugging Face token is available, set `HF_TOKEN` before rerunning:

```bash
export HF_TOKEN=...
/Users/itamarwe/Documents/code/itamarwe.github.io/research/fpv-drone-strikes/vggt/venv/bin/python \
  tools/flight_path_pipeline.py \
  --out-dir /tmp/fpv-flight-paths \
  run-vggt \
  --max-frames 180 \
  --vggt-timeout 900
```

## Scale Status

The current scale report is intentionally conservative. VGGT and monocular video
recover camera path shape in arbitrary scene units. Metric scale requires at
least one external metric constraint.

Available now:

- Relative VGGT camera path.
- Relative speed and height-proxy plots.
- Scene point-cloud renders with the path drawn through the reconstructed scene.
- Manual scale constraint template:
  `/tmp/fpv-flight-paths/scale_constraints_template.csv`

Not available yet as a real measurement:

- Known object dimension constraints.
- Satellite/map anchor distances.
- DEM/DSM terrain alignment.

Speed priors are no longer used as the default scale factor. They can be enabled
as optional sanity checks, but they are not measurement-based scale.

An automatic scale-voting prototype now lives in
`tools/auto_scale_hypotheses.py`; see `docs/automatic_scale_pipeline.md`.
It generates scale hypotheses from object/depth detections and accepts a metric
scale only when multiple cues agree.

Preferred scale methods, in order:

1. **Known object dimensions in the VGGT reconstruction.** Pick two reconstructed
   scene/object points spanning a known dimension, then add
   `observed_units` and `known_meters` to `scale_constraints.csv`.
2. **Map or satellite anchor distances.** Match two or more reconstructed scene
   points to map-visible features and use their real distance.
3. **Full georeferencing.** With three or more map anchors, solve a similarity
   transform from VGGT units into local ENU meters.
4. **Terrain/DSM alignment.** After georeferencing, align reconstructed ground
   points to DEM/DSM to improve vertical offset and AGL estimates.
5. **Terminal impact constraint.** Useful for vertical offset if final pose is
   near ground/target, but it does not by itself solve global scale.

## First Two VGGT Results

The current default report is relative-only because no metric constraints have
been added yet.

| Reconstruction | Frames | Flight Segments | Relative Path Units | Rejected Pose Jumps |
| --- | ---: | ---: | ---: | ---: |
| `2026-06-03_merkava_tank_zawtar_al_sharqiyah` legacy all-segment | 42 | 4 | 0.1094 | 0 |
| `2026-06-06_sholef_howitzer_adaissah` legacy all-segment | 48 | 3 | 0.2388 | 1 |
| `2026-06-06_sholef_howitzer_adaissah_attack_overlap` grouped | 48 | 3 | 0.2388 | 1 |

No row above should be cited as meters until a metric scale constraint is added.

## Best Next Step

To turn scale from a prior into a measurement, add manual metric constraints to
the annotator:

- object type
- frame timestamp
- two image points spanning a known dimension
- known dimension in meters
- optional confidence

Good first object priors for this dataset:

- Merkava hull length/width
- Humvee length/width/height
- Namer length/width/height
- D9 bulldozer length/width/height
- road/lane widths
- building doors/windows when visible

With those constraints, the pipeline can fit `meters_per_vggt_unit` per
reconstruction and report residuals across multiple object/map/terrain scale
cues.
