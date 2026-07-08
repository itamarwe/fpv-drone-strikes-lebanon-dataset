# Metric Geometry Models for FPV Scale

Date: 2026-06-30

Goal: evaluate whether MoGe-2 or similar monocular metric geometry models can
provide the metric scale for VGGT FPV reconstructions.

## Summary

MoGe-2 is worth testing as a first-class scale cue. Unlike original MoGe, which
predicts affine-invariant point maps without true global scale, MoGe-2 claims a
metric-scale 3D point map from a single image. The official repository exposes
pretrained Hugging Face models and returns `points`, `depth`, `mask`, optional
`normal`, and estimated `intrinsics`. That output can become an automatic scale
vote by aligning VGGT's relative point/camera geometry to MoGe-2's per-frame
metric point/depth map.

This should not be treated as ground truth. Monocular metric scale is learned
from priors and training data. FPV drone footage has domain shifts: wide/FOV
unknown lenses, compression, HUD overlays, blurred/censored regions, oblique
aerial views, explosions, and long-range rural scenes. The right use is as one
independent vote family in the scale ensemble, not as the sole source of meters.

## Models To Test

### 0. AMB3R

Most direct multi-view candidate.

- Paper: <https://arxiv.org/abs/2511.20343>
- Code: <https://github.com/HengyiWang/amb3r>

Useful properties:

- targets metric-scale multi-view 3D reconstruction
- provides VO and SfM backends
- outputs point maps, confidence, and camera poses
- supports VGGT-Omega and other foundation-model backends in its benchmark stack

Main risk: the current public installation is CUDA/Linux-oriented. The official
environment requires CUDA 11.8 packages including `torch-scatter`, `pytorch3d`,
`flash-attn`, and `spconv-cu118`, plus a Google Drive checkpoint. It is a strong
candidate for a CUDA box, but not a clean local Mac/CPU test. See
`docs/amb3r_experiment.md`.

### 1. MoGe-2

Primary candidate.

- Project: <https://wangrc.site/MoGe2Page/>
- Code: <https://github.com/microsoft/MoGe>
- HF models:
  - `Ruicheng/moge-2-vitl`
  - `Ruicheng/moge-2-vitl-normal`
  - `Ruicheng/moge-2-vitb-normal`
  - `Ruicheng/moge-2-vits-normal`

Useful properties:

- predicts metric-scale point maps in camera coordinates
- predicts depth maps
- predicts normalized camera intrinsics
- can accept known horizontal FOV via CLI (`--fov_x`)
- has `moge infer` CLI that can write maps, GLB, and PLY outputs
- model sizes include ViT-S/B/L, so we can start with `vits`/`vitb` for batch
  throughput and rerun selected frames with `vitl-normal`

Main risk: its metric scale may be biased by the unusual FPV domain. It can
still be useful if its scale agrees with object-size votes and/or map anchors.

### 2. UniDepthV2

Good comparison model.

- Paper: <https://arxiv.org/abs/2502.20110>
- Code: <https://github.com/lpiccinelli-eth/UniDepth>
- HF models listed in the repo include `unidepth-v2-vits14` and
  `unidepth-v2-vitl14`.

Useful properties:

- predicts metric depth and 3D points from RGB only
- predicts intrinsics
- supports known intrinsics when available
- outputs confidence/uncertainty in V2

Main risk: heavier dependency stack, and performance can still degrade under
domain shift.

### 3. Depth Pro

Good fast metric-depth baseline.

- Paper: <https://arxiv.org/abs/2410.02073>
- Code: <https://github.com/apple/ml-depth-pro>

Useful properties:

- metric depth in meters
- estimates focal length in pixels
- fast single-image inference
- sharp object boundaries

Main risk: outputs depth, not a full point-map-first geometry representation.
We can still unproject using its predicted focal length and compare to VGGT.

### 4. Metric3D / Metric3D v2

Useful baseline, especially if camera intrinsics/FOV are known.

- Metric3D: <https://arxiv.org/abs/2307.10984>
- Metric3Dv2: <https://arxiv.org/abs/2404.15506>

Useful properties:

- designed for zero-shot metric depth
- explicitly addresses metric ambiguity across camera models
- strong established baseline

Main risk: more sensitive to camera model assumptions than MoGe-2/UniDepthV2
unless we supply credible FPV camera FOV.

### 5. Metric Anything

Promising newer direction.

- Paper/project: <https://arxiv.org/abs/2601.22054>

Useful properties:

- targets metric depth from noisy heterogeneous sources
- reports prompt-free monocular depth plus camera intrinsics recovery
- specifically frames camera-dependent metric ambiguity as a training problem

Main risk: newer project, integration and weight availability need verification
before it can be used in the pipeline.

## How To Use These For VGGT Scale

The intended design is not "MoGe-2 instead of VGGT". It is:

```text
VGGT: recover temporally consistent camera path + scene geometry in arbitrary units.
MoGe-2: estimate per-frame metric geometry in camera coordinates.
Alignment: fit one global scale that maps VGGT units into MoGe-2 meters.
```

This keeps VGGT responsible for multi-view consistency and path shape. MoGe-2 is
only a metric reference. The scale estimate is accepted only if the per-frame
MoGe-2 votes agree with each other and with other independent cues.

For each reconstructed flight group:

1. Select frames used by VGGT, excluding HUD/overlay/censored regions where
   possible.
2. Run MoGe-2 on each selected frame and save:
   - metric depth map
   - metric point map
   - valid mask
   - predicted intrinsics / FOV
3. Project VGGT scene points or camera-near scene rays into the same frame.
4. Match robust samples between VGGT relative geometry and model metric geometry:
   - static background pixels, not moving drone HUD
   - road/terrain/building masks if available
   - high-confidence model mask
   - avoid sky, blur, smoke, explosion, and target impact artifacts
5. Estimate one scale per frame:

```text
scale_m_per_vggt_unit =
  robust_median( metric_depth_or_point_distance / vggt_relative_distance )
```

6. Cluster frame-scale votes in log space, same as object-size votes.
7. Accept only if:
   - MoGe-2 frame votes agree with each other
   - MoGe-2 agrees with at least one independent cue family
     (object dimensions, map anchors, or terrain/DSM)
   - estimated FOV/intrinsics are plausible for the FPV camera

### Strong Alignment Version

The strongest version needs VGGT per-frame camera pose and either projected
VGGT scene points or VGGT per-frame depth/point maps.

For frame `i`:

```text
P_vggt_cam_units = T_world_to_cam_i * P_vggt_world_units
P_moge_cam_m     = MoGe2(frame_i).points

scale_i = robust_fit_scale(P_vggt_cam_units, P_moge_cam_m)
```

Use only pixels/samples that are likely static scene:

- valid in both models
- non-sky
- outside HUD/reticle/prop/censor masks
- not impact smoke/explosion/blur
- preferably road, ground, buildings, walls, or parked target objects

The fit can start as a one-parameter robust scale:

```text
s_i = median(||P_moge_cam_m|| / ||P_vggt_cam_units||)
```

Then improve to a robust similarity fit if VGGT/MoGe point correspondences are
stable enough. The final reconstruction scale is the log-space consensus across
frames:

```text
s = consensus({s_i})
path_meters = s * path_vggt_units
```

### Fallback With Only VGGT Camera Centers

If the exported VGGT artifact only gives camera centers and not per-frame
camera poses/projections, MoGe-2 can still provide weaker votes:

```text
scale_i =
  MoGe2 metric distance to target/ground/object in frame_i
  / VGGT relative distance from camera_i to terminal camera or target proxy
```

This is useful as a diagnostic but weaker, because the denominator is a path
distance/proxy rather than a true point correspondence. The pipeline should
prefer the strong alignment version when VGGT poses and projected scene points
are available.

## Practical First Experiment

Run only on the current Sholef attack-overlap reconstruction first.

Inputs:

- `/tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/frames`
- `/tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/relative_path.csv`
- `/tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/point_cloud.npz`

Suggested command to test once dependencies are installed:

```bash
moge infer \
  -i /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/frames \
  --o /tmp/fpv-flight-paths/attack_reconstructions/2026-06-06_sholef_howitzer_adaissah_attack_overlap/moge2 \
  --version v2 \
  --pretrained Ruicheng/moge-2-vits-normal \
  --maps \
  --device cpu
```

If the FPV camera horizontal FOV becomes known, rerun with:

```bash
--fov_x <degrees>
```

Expected outputs to integrate:

- metric depth maps
- point maps
- masks
- estimated FOV/intrinsics

## Evaluation KPIs

For each model and frame:

- predicted FOV stability across frames
- per-frame scale estimate
- log-scale cluster spread
- agreement with object-size votes
- agreement with map/terrain anchors if available
- temporal stability of ground plane / road / building depths
- qualitative overlay: metric depth on original frame, VGGT projected samples,
  accepted/rejected sample mask

The model should fail closed. If MoGe-2 and object-size votes disagree, the
report should say scale unresolved rather than averaging them.

## Recommendation

Implement MoGe-2 as the next automatic scale cue. The output is closer to what
we need than plain metric-depth maps because it gives point maps and camera
intrinsics, not only depth. Use `moge-2-vits-normal` or `moge-2-vitb-normal` for
batch exploration, then `moge-2-vitl-normal` for high-value attack segments.

Depth Pro and UniDepthV2 should be run on the same frames as control models. If
all three metric-geometry models cluster around the same scale and agree with
object/map cues, that is strong evidence. If only one model agrees, keep it as a
weak vote.
