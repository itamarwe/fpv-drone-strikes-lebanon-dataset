# Two-scene 3D pipeline benchmark

**Revised 2026-09-08:** [REVISED_PLAN.md](REVISED_PLAN.md) is the current experiment
and presentation plan, using the confirmed [scene measurements](calibration_references.json)
and evaluating full-quality 3D reconstruction as a primary goal. The package documented below implements the earlier
draft. Its inputs are reusable, but the scoring corrections, revised splits,
and renderer integration listed in that plan are required before benchmarking.

This benchmark isolates two goals:

1. metric scale and reduced metric distortion; and
2. improved final visualization.

The published VGGT scenes are references, not independent ground truth. The
manual 3 m and 7 m calibrations are used only for scale-agreement reporting.

## Locked inputs

`benchmark.json` selects the canonical published scene for Scene A and
Scene B. `prepare_3d_pipeline_benchmark.py` downloads the already-cleaned
viewer images, baseline camera metadata, and baseline point buffers. It creates:

- `full`: every published path frame in chronological order;
- `common16`: 16 segment-stratified, uniformly sampled frames; and
- `heldout`: temporal midpoints excluded from `common16`.

The generated bundle is ignored by Git and lives under `work/`. A portable
archive is written under `transfer/`.

```bash
python tools/pipeline/prepare_3d_pipeline_benchmark.py \
  --archive benchmarks/3d_pipeline_two_scene/transfer/fpv-3d-benchmark.tar.gz

python tools/pipeline/check_3d_pipeline_benchmark.py \
  --bundle benchmarks/3d_pipeline_two_scene/work/bundle
```

## Findings so far (2026-09-08 common16 run and 2026-09-09 session 2; see [EXECUTION_STATUS.md](EXECUTION_STATUS.md))

- [METRIC_SCALE_FINDINGS.md](results/METRIC_SCALE_FINDINGS.md): MoGe-3 automatic scale rejected on both scenes.
- [GLOMAP_COMMON16_FINDINGS.md](results/GLOMAP_COMMON16_FINDINGS.md): coherent sparse paths, 13/16 and 15/16 cameras.
- [SCAL3R_COMMON16_FINDINGS.md](results/SCAL3R_COMMON16_FINDINGS.md): full coverage; Scene A late drift.
- [VGGT_OMEGA_MATCHED_CONTROL_FINDINGS.md](results/VGGT_OMEGA_MATCHED_CONTROL_FINDINGS.md): matched-input control and reference-scale transfer gates.
- [SURFLO_COMMON16_FINDINGS.md](results/SURFLO_COMMON16_FINDINGS.md): negative equal-input result.
- [QUERYSPLAT_COMMON16_FINDINGS.md](results/QUERYSPLAT_COMMON16_FINDINGS.md): best appearance result; Scene B holds up on held-out proxy views, Scene A does not.
- [DENSE_TRAIN_FINDINGS.md](results/DENSE_TRAIN_FINDINGS.md): 82/101-view runs (session 2): GLOMAP and the VGGT-Omega control scale well, Scal3R degrades, dense QuerySplat is the Scene B recommendation.

## Scene viewers and side-by-side comparison

Every retrieved reconstruction can be browsed in the standard scene viewer.
The exporter aligns each candidate into the published VGGT frame (Umeyama
Sim(3) on camera centres, the same fit the trajectory evaluator uses), writes
the viewer buffers under `scenes/benchmark_3d_two_scene/`, and records the
alignment, a visual far-point crop and the benchmark metrics in each
`scene_meta.json`. Surflo is excluded because its CLI exports no cameras.

```bash
python tools/pipeline/build_benchmark_scene_viewers.py
python tools/local_scene_viewer_server.py --port 8766
# then open
#   http://127.0.0.1:8766/benchmarks/3d_pipeline_two_scene/review/scene_comparison.html
```

The comparison page shows two viewers side by side with a per-scene matrix
(views, points, registered cameras, path Sim(3) RMSE, held-out PSNR/SSIM/LPIPS)
and deep-links through `?scene=&left=&right=`. The inherited metre-per-unit
scale is a visual convenience, not an independent calibration.

## Method matrix

| Method | Stage | Profiles | Prepared status |
|---|---|---|---|
| Published VGGT | reference | full/common16 | available in bundle |
| Scal3R-ZJU | reconstruction | full/common16 | runnable |
| SSMB + LightGlue | reconstruction | full/common16 | blocked by unreleased SSMB code/weights |
| MoGe-3 + GLOMAP | reconstruction/scale | full/common16 | runnable with documented integration limitation |
| Surflo plain + guided | representation | common16 | runnable |
| QuerySplat feed-forward + TTO | representation | common16 | runnable; gated VGGT-Ω access verified locally |

The MoGe-3/GLOMAP test uses GLOMAP for SfM, samples MoGe-3 metric depths at
registered GLOMAP observations, robustly estimates a single global metric
scale, and reports per-frame scale dispersion. GLOMAP does not expose a
documented native dense-depth-prior bundle-adjustment API, so this first test
does not claim that MoGe depth changes GLOMAP's camera optimization.

## RunPod handoff

No pod is created by this benchmark package. After credit is available:

1. provision one 80 GB NVIDIA GPU pod with at least 150 GB persistent volume;
2. set both automatic stop and automatic termination deadlines;
3. transfer and extract `fpv-3d-benchmark.tar.gz` under `/workspace`;
4. supply a Hugging Face read token through RunPod secrets or log in
   interactively inside the pod—never store it in this bundle;
5. run the bundled preflight before any installation; and
6. set up, download, and execute one method at a time.

Inside the extracted `bundle/` directory:

```bash
./orchestration/runpod_3d_pipeline_benchmark.sh preflight
./orchestration/runpod_3d_pipeline_benchmark.sh setup-base
./orchestration/runpod_3d_pipeline_benchmark.sh checkpoint-preflight

./orchestration/runpod_3d_pipeline_benchmark.sh setup scal3r_zju
./orchestration/runpod_3d_pipeline_benchmark.sh download scal3r_zju
./orchestration/runpod_3d_pipeline_benchmark.sh run scal3r_zju scene_a common16
```

Repeat `setup`, `download`, and `run` for `moge3_glomap`, `surflo`, and
`querysplat`. Reconstruction methods run on both `common16` and `full`;
representation methods are locked to `common16`. The script deliberately
rejects `ssmb_lightglue` until the missing upstream implementation exists.

All method results are written under `/workspace/fpv-3d-benchmark/results`.
Retrieve that directory before terminating the pod.

## Evaluation contract

Geometry methods report registration coverage, trajectory continuity,
Sim(3)-aligned agreement with the published VGGT path, global metric scale,
per-frame log-scale MAD, and reference-scale error. Agreement with VGGT is not
ground-truth pose accuracy.

Visualization methods should render the locked `heldout` cameras before image
metrics are computed. The evaluator reports PSNR and SSIM. LPIPS is reserved
for the GPU method environment. Manual review records holes, floaters, edge
quality, temporal shimmer, load time, asset bytes, browser memory, and FPS.

## Remaining scientific gap

One known length per scene determines only a global scale. It cannot establish
that depth-dependent or spatially varying metric distortion is correct.
Several independently measured distances at different depths would turn the
distortion score into an objective metric rather than a consistency proxy.
