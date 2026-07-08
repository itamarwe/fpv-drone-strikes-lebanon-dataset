# AMB3R Experiment

Date: 2026-06-30

Goal: test AMB3R as a metric-scale reconstruction backend for the FPV flight
segments, and compare it with the current VGGT path.

## Why AMB3R Is Relevant

AMB3R is directly aligned with the scale problem: it is a feed-forward
metric-scale 3D reconstruction model with VO and SfM backends. Unlike the
VGGT+MoGe-2 plan, where VGGT gives relative multi-view geometry and MoGe-2
supplies metric scale, AMB3R tries to produce metric-scale multi-view point maps
and camera poses in one pipeline.

This makes it worth testing as:

1. a replacement reconstruction path for selected attack segments, and
2. an independent metric-scale vote to compare against VGGT+MoGe-2,
   object-size votes, and map anchors.

## What I Checked

Repository:

- <https://github.com/HengyiWang/amb3r>

Local clone used for inspection:

- `/tmp/amb3r`

Important repo facts from the README and scripts:

- AMB3R-SfM entrypoint: `python sfm/run.py --data_path <image-folder>`
- AMB3R-VO entrypoint: `python slam/run.py --data_path <image-folder>`
- output `.npz` contains `pts`, `conf`, `pose`, `images`, `sky_mask`, and
  keyframe information
- output `.ply` contains a colored point cloud
- official install expects CUDA 11.8 packages:
  - `torch==2.5.0` CUDA 11.8
  - `torch-scatter`
  - `pytorch3d`
  - `flash-attn`
  - `spconv-cu118`
- official checkpoint must be downloaded from Google Drive and placed at
  `./checkpoints/amb3r.pt`

## Prepared FPV Input

I prepared an 8-frame smoke-test folder from the existing Sholef
attack-overlap VGGT reconstruction:

```bash
python3 tools/prepare_amb3r_experiment.py --copy
```

Outputs:

- `/tmp/fpv-flight-paths/amb3r_inputs/sholef_smoke/`
- `/tmp/fpv-flight-paths/amb3r_inputs/sholef_smoke/manifest.csv`

The folder contains simple sequential JPEG names, which AMB3R's demo dataset
loader can consume directly.

## Local Run Attempt

Command attempted:

```bash
/tmp/fpv-model-benchmark/venv/bin/python /tmp/amb3r/sfm/run.py \
  --data_path /tmp/fpv-flight-paths/amb3r_inputs/sholef_smoke \
  --demo_name sholef_smoke \
  --results_path /tmp/fpv-flight-paths/amb3r_outputs/sholef_smoke \
  --target_point_count 200000 \
  --save_res True
```

Immediate local failure:

```text
ModuleNotFoundError: No module named 'open3d'
```

The more important blocker is not just `open3d`. In the existing local venv,
AMB3R import also fails on CUDA/backend dependencies:

```text
missing open3d
missing xformers
missing spconv
missing pytorch3d
missing flash_attn
missing torch_scatter
missing omegaconf
missing evo
missing timm
AMB3R import fail: No module named 'torch_scatter'
```

This machine also reports:

```text
cuda False
mps False
```

So AMB3R is not runnable in this local environment without moving to a CUDA
machine or heavily porting the code/dependencies.

## GPU Run Command

On a CUDA machine with the official environment and checkpoint:

```bash
cd /tmp/amb3r

python sfm/run.py \
  --data_path /tmp/fpv-flight-paths/amb3r_inputs/sholef_smoke \
  --demo_name sholef_smoke \
  --results_path /tmp/fpv-flight-paths/amb3r_outputs/sholef_smoke \
  --target_point_count 200000 \
  --save_res True
```

Expected outputs:

- `/tmp/fpv-flight-paths/amb3r_outputs/sholef_smoke/scene_sholef_smoke_results.npz`
- `/tmp/fpv-flight-paths/amb3r_outputs/sholef_smoke/scene_sholef_smoke_points.ply`

## How To Use AMB3R Outputs

If AMB3R runs successfully, compare it against VGGT as follows:

1. Load AMB3R `pose` camera centers in metric units.
2. Load VGGT `relative_path.csv` camera centers in arbitrary units.
3. Match AMB3R frames to VGGT frame indices using the prepared manifest.
4. Fit a 7-DoF similarity transform from VGGT camera centers to AMB3R camera
   centers.
5. Extract the scale factor as `meters_per_vggt_unit`.
6. Report residuals:
   - per-camera alignment error
   - path length difference
   - attack segment endpoint mismatch
   - whether AMB3R pose trajectory is stable or jumps
7. Treat the result as one scale vote family. Accept only if residuals are low
   and it agrees with independent cues.

## Recommendation

AMB3R is worth testing, but not on this Mac/CPU environment. It should be run on
a CUDA machine first. If it produces stable metric poses on the Sholef smoke
set, the next repo step is a parser:

```text
AMB3R .npz -> amb3r_path.csv -> similarity fit against VGGT -> scale vote
```

If AMB3R fails on this domain, VGGT+MoGe-2 remains the more modular path because
MoGe-2 can be run frame-by-frame and used only as a metric scale cue.
