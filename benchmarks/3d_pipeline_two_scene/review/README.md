# Two-scene reconstruction review workflow

This directory is the evidence layer for the full-quality reconstruction
comparison. It starts empty of wins: the report lists missing runs and only
shows a positive claim after a human marks it `confirmed` and provides files
that exist.

## Result locations and contracts

GPU outputs use immutable attempt directories:

```text
results/<scene>/<profile>/<method>/attempt-<UTC timestamp>/<mode>/
```

The current Surflo CLI adds an extra directory named after the image folder
(normally `images`). The inventory searches recursively so this does not hide
its outputs. The inventory retains every nested `run.json`, including failed
attempts, and ignores copied `latest/` aliases to prevent double-counting. A
runner may additionally write `run_manifest.json` following
[`result_contract.schema.json`](result_contract.schema.json). Complete metadata
records the method revision, exact input-frame manifest/hashes,
preprocessing/crop, coordinate and unit convention, cameras, native artifacts,
runtime/memory/cost, and limitations. Failed and OOM runs also get a manifest;
they are not deleted from the matrix.

Image metrics have a separate, stricter contract in
[`image_pairs.schema.json`](image_pairs.schema.json). Every reference/render
pair names its frame and camera record. The evaluator accepts only identity
pixel mappings at identical dimensions. Crop, resize and intrinsics adjustment
must happen upstream and be recorded. Evaluation views are rejected if their
target RGB was used for photometric camera fitting.

## Inventory and report

These commands are safe before GPU results exist; absent runs remain `not_run`:

```bash
python tools/pipeline/quality_result_inventory.py \
  --results-root /workspace/fpv-3d-benchmark/results \
  --bundle-root /workspace/fpv-3d-benchmark/bundle \
  --output benchmarks/3d_pipeline_two_scene/review/inventory.json

python tools/pipeline/quality_review_report.py \
  --inventory benchmarks/3d_pipeline_two_scene/review/inventory.json \
  --claims benchmarks/3d_pipeline_two_scene/review/confirmed_claims.json \
  --output-dir benchmarks/3d_pipeline_two_scene/review/site \
  --incomplete-closure
```

`--incomplete-closure` is required for this stopped run. It adds a prominent
incomplete/no-recommendation banner and relabels the evidence section as partial
findings without changing or discarding the underlying artifacts.

The HTML report links native files and makes an aspect-preserving contact sheet
from actual render/depth/normal images. It never inserts mock reconstruction
images. Preprocessed source inputs are not included in the output contact sheet.

## QuerySplat: honest native scoring

The pinned inference script writes `input_frames/`, `rendered/`,
`predicted_input_cameras.{json,npz}`, `gaussians.ply`, `pointcloud.ply`,
`vggt_depth_pointcloud.ply`, VGGT input depths and timing. Its `rendered/`
images correspond to predicted **input** cameras. Generate an explicit input
view manifest and score it as such:

```bash
run_dir=/workspace/fpv-3d-benchmark/results/scene_a/common16/querysplat/attempt-<UTC>/feed_forward

python tools/pipeline/quality_make_querysplat_pairs.py \
  --run-dir "$run_dir" --scene scene_a --mode feed_forward \
  --output "$run_dir/image_pairs.input_views.json"

python tools/pipeline/evaluate_3d_visualization.py \
  --pairs-manifest "$run_dir/image_pairs.input_views.json" \
  --output "$run_dir/input_view_metrics.json" \
  --lpips-device off

python tools/pipeline/quality_pair_contact_sheet.py \
  --pairs-manifest "$run_dir/image_pairs.input_views.json" \
  --output "$run_dir/input_render_contact_sheet.jpg"
```

Repeat with `--mode tto` for the TTO directory. These are useful fidelity and
artifact diagnostics, but not unseen-view scores. The pinned release has a
renderer API for in-memory Gaussians plus caller-supplied cameras; it does not
ship a standalone saved-PLY novel-view command. A held-out QuerySplat score
therefore needs a renderer adapter and separately derived evaluation cameras.

The provided bounded proxy route runs only VGGT-Omega's aggregator/camera head
on Train16 plus the evaluation images, aligns that camera frame to the frozen
QuerySplat run using Train16 correspondences, reloads the frozen Gaussian PLY,
and renders evaluation views. Evaluation images never enter QuerySplat scene
encoding or TTO. Run it only after the native diagnostics succeed:

The reconstruction run must include `--gaussian_save_opacity_threshold 0`.
QuerySplat's default exported PLY removes Gaussians below opacity 0.05 even
though its native render uses the full in-memory set. If both `0` and `0.05`
are requested, select `gaussians_opacity0.ply` below. The adapter verifies the
threshold in `inference_timing.json` and refuses a lossy PLY.

```bash
python tools/pipeline/quality_render_querysplat_evaluation.py \
  --querysplat-repo /workspace/repos/QuerySplat \
  --config /workspace/repos/QuerySplat/checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --gaussians-ply "$run_dir/gaussians_opacity0.ply" \
  --frozen-cameras "$run_dir/predicted_input_cameras.json" \
  --train-images /workspace/fpv-3d-benchmark/bundle/scenes/scene_a/profiles/common16/images \
  --train-frames-csv /workspace/fpv-3d-benchmark/bundle/scenes/scene_a/profiles/common16/frames.csv \
  --evaluation-images /workspace/fpv-3d-benchmark/bundle/scenes/scene_a/profiles/heldout/images \
  --evaluation-frames-csv /workspace/fpv-3d-benchmark/bundle/scenes/scene_a/profiles/heldout/frames.csv \
  --scene scene_a --method-run-id scene_a.querysplat.feed_forward \
  --output-dir "$run_dir/evaluation_proxy"

python tools/pipeline/evaluate_3d_visualization.py \
  --pairs-manifest "$run_dir/evaluation_proxy/image_pairs.evaluation_proxy.json" \
  --output "$run_dir/evaluation_proxy/metrics.json" --lpips-device cuda
```

The adapter fails if Train16 alignment exceeds frozen residual thresholds and
records every transform, camera and alpha mask. This remains a proxy held-out
appearance test: evaluation RGB participates in camera estimation, joint
attention can change Train16 pose estimates, and the short flight path can make
frame alignment ill-conditioned. Report its centre/orientation residuals and
pose uncertainty. It is not independent survey truth. The example points at
the prepared legacy `heldout` directory; call it a final benchmark evaluation
only if the revised near-duplicate and temporal-buffer split QA has passed.

## Surflo: full-quality geometry and appearance

Pinned plain inference produces `initial.ply` and `final.ply`. Guided inference
produces `point_cloud_normals.ply`, normally `point_cloud_rgb.ply`, `mesh.ply`,
and (with the default texture block) `mesh_textured.ply`. UV texture mode can
instead export `mesh_textured.glb` when its optional dependencies are installed.

Use the equal-input outputs first. For a geometry turntable, render the actual
mesh with fixed deterministic exploratory cameras:

```bash
surflo_run=/workspace/fpv-3d-benchmark/results/scene_a/common16/surflo/attempt-<UTC>/guided_default/images

blender -b --python tools/pipeline/quality_blender_turntable.py -- \
  --input "$surflo_run/mesh.ply" \
  --output-dir "$surflo_run/review_neutral" --mode neutral \
  --engine cycles_cpu --samples 32 --threads 4 \
  --frames 48 --width 1600 --height 1200
```

For appearance, prefer the UV-textured GLB and use `--mode native`. Native PLY
vertex colours are explicitly connected to Principled Base Color rather than
relying on Blender importer defaults. CPU denoising is off by default for
minimal Blender builds; add `--denoise` only when OpenImageDenoiser is present. The helper
rejects point-only PLYs instead of producing blank frames. Review Surflo plain
and other point clouds in CloudCompare or MeshLab and save explicitly labelled
screenshots alongside the native artifact. Turntables are marked `exploratory`
in `turntable_manifest.json`; do not feed them to the photometric evaluator.

The pinned Surflo CLI does not persist its refined cameras or camera-matched
renders. Consequently the mesh/point review can proceed now, while a matched
photometric comparison requires an upstream adapter that exports the camera
state used during inference.

## Baseline review

The prepared published VGGT baseline contains raw position/colour buffers but
not a camera-linked render set. Keep it as the current-product geometry in the
existing viewer and export exact view metadata before using it in image scores.

Pinned Scal3R exports `mat.txt` (c2w rows), `intri.yml`, `extri.yml`, optional
depth EXRs, per-frame masks, block PLYs and `points/whole.ply`. Preserve the
frames-CSV row mapping in `run_manifest.json`; numbered camera rows alone are
not a durable frame identity. Its included Viser viewer provides a useful RGB
point-cloud and camera-path inspection:

```bash
cd /workspace/repos/Scal3R
python scripts/visualize/viser_viewer.py \
  --ply_path /workspace/fpv-3d-benchmark/results/scene_a/dense_train/scal3r_zju/points/whole.ply \
  --pose_path /workspace/fpv-3d-benchmark/results/scene_a/dense_train/scal3r_zju/mat.txt \
  --intri_path /workspace/fpv-3d-benchmark/results/scene_a/dense_train/scal3r_zju/intri.yml \
  --host 0.0.0.0 --port 8080
```

Scal3R and GLOMAP outputs are geometry/scaffold comparisons, not direct peers
to a textured mesh or Gaussian appearance render. GLOMAP's sparse model can be
converted to PLY with `colmap model_converter`, but that does not create a dense
or photorealistic result. For a fair presentation, show RGB point/sparse modes
alongside neutral geometry views, and reserve PSNR/SSIM/LPIPS for an actual
camera-linked renderer.

## Presenting positive results

Add only confirmed claims to
[`confirmed_claims.json`](confirmed_claims.json). Each claim needs at least one
existing evidence file. The report fails if a claimed evidence file is missing.
Good evidence packages combine a neutral geometry view, an RGB/texture view,
a synchronized camera-matched comparison where available, the native artifact,
and the relevant metric JSON. Keep the complete two-scene matrix immediately
below the highlighted wins so scene-specific gains and failures remain visible.

## Source audit

[`source_audit.md`](source_audit.md) records the inspected revisions, exact
native exports and current offline-render limitations.
