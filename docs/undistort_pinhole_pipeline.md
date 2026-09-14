# Lens undistortion before VGGT-Omega

Revised 2026-09-14. This documents the preprocessing phase added to the 3D
pipeline after the synthetic ground-truth study and the two real-scene tests,
and the batch tooling that re-runs the site's starred scenes with it.

## Why

FPV feeds are fisheye. VGGT-Omega assumes a pinhole camera, so on distorted
input it reads barrel-compressed edges as a longer lens and pushes edge
content to the wrong depth. On the synthetic Blender flyover with exact ground
truth this was the dominant error; undistorting the frames first removed it
almost entirely. On the two real scenes it halved the camera-path disagreement
with an independent SfM reconstruction and cut local scale drift by 40-45%.

| Input | Camera path error (synthetic, vs truth) | Depth error (synthetic) | Path disagreement (real Scene A / B) | Local scale std (real A / B) |
|---|---:|---:|---:|---:|
| raw fisheye frames | 2.2% of path (max 18 m) | 43% | 1.15% / 1.25% | 0.061 / 0.083 |
| undistorted to pinhole | 0.24% (max 1.9 m) | 2% | 0.75% / 0.67% | 0.036 / 0.047 |

Details: `scenes/synthetic_zofa_al_bayada/synthetic_zofa_fpv_roof_flyover_120f/ERROR_SOURCES_AND_TRAINING.md`
and `scenes/real_two_scene_lens_test/FINDINGS.md`. Before/after media for posts:
`reports/lens_undistortion_before_after/`.

## The phase

1. **Self-calibrate the lens per video.** The published frames are a zoomed,
   re-cropped centre of the lens (660x280-ish "clean crop"), so a manufacturer
   model does not apply; every video is calibrated from its own frames.
   COLMAP SIFT + exhaustive matching, then GLOMAP's global mapper with a single
   `OPENCV_FISHEYE` camera (focal and k1..k4 free, principal point at the crop
   centre). 2-3 minutes on CPU for 125 frames. Script: `tools/pipeline/starred_calibrate_remote.sh`.
2. **Remap every frame to a pinhole** with the calibrated focal length and the
   principal point at the centre (`tools/pipeline/undistort_real_scene.py`).
   Barrel distortion means nothing is cropped (valid fraction 1.0).
3. **Run VGGT-Omega on the remapped frames** exactly as before
   (`tools/run_vggt_omega_direct_on_runpod.py`, now with `--slim-predictions`
   to drop the derivable 1.2 GB of world points and image tensors before download).
4. **Score without ground truth**: Sim(3) residual of the VGGT camera centres
   against the GLOMAP path (fraction of path length), local scale drift over
   20-frame windows, and VGGT's own focal estimate over the calibrated pinhole
   focal (1.0 means the model now sees the lens it assumes). Any known length
   (the 3 m and 7 m anchors) must be re-measured on the new reconstruction;
   scale calibrations do not transfer between runs.

## Starred-scene batch

`benchmarks/starred_undistort/starred_scenes.json` lists the 26 scenes that
were starred on the site on 2026-09-14 (3,265 published frames). One command
runs everything unattended and is safe to re-run after any interruption:

```bash
python3 tools/pipeline/starred_prepare.py          # download published frames, meta, point buffers (no GPU)
tools/pipeline/run_starred_night.sh                # create an A100 pod, calibrate + undistort + VGGT-Omega every scene, post-process, delete the pod
POD_ID=<id> tools/pipeline/run_starred_night.sh    # same on an existing pod
```

`starred_run_batch.py` stages: pod, setup (VGGT-Omega venv + HF token),
push (frames to `/workspace/starred/`), calib (one background CPU job on the
pod covering every scene, overlapping the GPU work), scenes (per scene: wait
for its calibration, fetch `cameras.txt`, undistort locally, run VGGT-Omega,
fetch slim artifacts, build the standard viewer), post, pod-down. State lives
in `benchmarks/starred_undistort/batch_status.json`; the pod is created with
`--stop-after`/`--terminate-after` so the cloud enforces the budget even if
the driver dies. Expected cost: about 7-9 minutes of A100 time per scene,
roughly 4 hours and $7-8 for all 26 at the secure A100 rate.

`starred_postprocess.py` produces, per scene, `metrics.json`, an overlay
viewer (`scenes/starred_undistort/<video_id>/overlay/index.html`, both runs in
the published viewer frame, served by `tools/local_scene_viewer_server.py`)
and a before/after transition video with stills under
`reports/starred_undistort/<video_id>/`, plus `reports/starred_undistort/index.html`
and `SUMMARY.md`.

Disk: each scene keeps about 300 MB locally (slim predictions, 3M-point GLB
and viewer buffers); the full 26-scene run needs about 9 GB.
