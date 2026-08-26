# FixAnything Integration Research

Date: 2026-08-26 (updated same day: code and checkpoints are released)

Question: could FixAnything (<https://fix-anything.github.io/>) be integrated
into the repo's 3D scene reconstruction workflow?

Short answer: yes, as a *rendering refinement* stage on top of the existing
VGGT scenes — not as a reconstruction backend. Inference code and checkpoints
are public (Apache 2.0), so a zero-shot experiment is runnable now on a rented
GPU. Hard caveat: the output is generative and must never feed the measurement
path.

## What FixAnything Is

- Paper: "FixAnything: 3D-Consistent Rendering Refinement via Video Generative
  Priors", Khiem Vuong, Deva Ramanan, Srinivasa Narasimhan (CMU), ECCV 2026,
  arXiv 2608.23549 (posted 2026-08-24).
- It is not a reconstruction model. It takes a *rendered video* from an
  existing 3D representation — 3DGS, NeRF, mesh, or a sparse point cloud —
  along a camera path, and treats artifact cleanup as video-to-video
  translation with a pretrained video diffusion model.
- Adaptation is lightweight: LoRA finetuning on the order of ~20 paired videos
  (noisy render + clean reference at the same poses).
- To keep outputs 3D-consistent rather than merely pretty, it uses
  SfM-recovered camera-pose accuracy on the refined video as a DPO reward, so
  refined videos are explicitly optimized to support downstream reconstruction.
- The sparse point cloud case is called out directly: even a COLMAP-sparse
  cloud rendered along a path gives the video model enough camera control.

## What the Released Code Provides

Repository: <https://github.com/kvuong2711/fix-anything> (inspected from a
local clone at `/home/user/kvuong2711/fix-anything`). Checkpoints:
<https://huggingface.co/kvuong2711/fix-anything>. Code and weights are
Apache 2.0 (the Wan2.1 license). Facts that matter for integration:

- **Inference only.** No training/finetuning code is released, so own-domain
  LoRA finetuning on our paired renders/frames is not currently actionable —
  zero-shot is what we can test.
- **Base model.** Wan2.1-I2V-14B-480P (~60 GB download) plus a
  `fixanything_lora.safetensors` LoRA from HF.
  `scripts/download_models.py --model_dir checkpoints` fetches both.
- **Entrypoint.** `scripts/run_inference.py --input <video-or-frame-folder>
  --output_dir <dir>`; writes `generated.mp4`, resized `input.mp4`, and
  `side_by_side.mp4`. The whole API is two functions (`load_pipeline()`,
  `fix_video()`), easy to wrap from `tools/pipeline/`.
- **Fixed input contract.** Exactly 61 frames, resized/cropped to 832×480
  (extra frames are dropped; the last frame is internally repeated 4×).
  Longer flight segments must be chunked into 61-frame windows.
- **`--clean_frame_indices`** marks input frames the model keeps as-is
  (default: first and last). For our loop this is the anchor mechanism: when
  rendering the point cloud along the recovered camera path, real video frames
  can be injected at their poses and pinned as clean anchors.
- **Runtime characteristics.** bf16, 10 diffusion steps, tiled VAE, and
  DiffSynth VRAM offloading to CPU are enabled by default, so a single large
  GPU (e.g. RunPod A6000/A100 class) should suffice; FlashAttention-2 is
  optional but recommended. Deps are pinned (torch 2.6/cu126, a fixed
  DiffSynth-Studio commit, transformers<5).
- **Bonus path.** `scripts/run_mapanything.py` reconstructs a few photos with
  MapAnything and renders a 61-frame path for refinement — a possible
  quick-look path for videos too short or damaged for VGGT.

## Why It Maps Well onto This Repo

The pipeline already produces exactly the inputs FixAnything consumes and the
paired data its finetuning needs:

- Each scene has a sparse colored point cloud (`point_cloud.npz`, viewer
  `points_positions.bin`/`points_colors.bin`) plus recovered per-frame camera
  poses (`relative_path.csv`).
- `camera_views/f_*_vggt_render.jpg` are point-cloud renders at the recovered
  poses; `frames/f_*.jpg` are the real frames at the same poses. That is a
  ready-made (noisy render, clean ground truth) pair per frame — the
  finetuning data recipe, generated for free by
  `render_vggt_reprojection_samples.py` across ~50 scenes.

Plausible uses, in increasing ambition:

1. **Presentation renders.** Refine novel-view flythroughs of the sparse
   VGGT clouds into watchable video for the scene browser. Lowest risk,
   clearly labelable as synthetic.
2. **Reconstruction densification loop.** Render the sparse cloud along the
   recovered (or a slightly perturbed) camera path, refine with FixAnything,
   then re-run VGGT/AMB3R/COLMAP on the refined video to get a denser, cleaner
   scene. This is the loop the paper's DPO objective is designed for.
3. **Rescuing weak scenes.** Short, blurry, or smoke-heavy attack segments
   that currently reconstruct poorly might benefit most from (2).

## Blockers and Caveats

- **Generative provenance hazard.** This dataset documents real strikes and
  the viewer offers measurements (scale, AGL, speed). A video diffusion model
  hallucinates plausible detail — vehicles, damage, terrain that were never
  observed. Refined renders and any geometry re-reconstructed from them must
  be stored and labeled as synthetic derivatives, kept out of
  `scene_state.json` scale calibration and measurement workflows, and flagged
  in scene metadata (e.g. a `generative_refinement` block in
  `scene_manifest.json.model_config`).
- **Domain gap.** FPV combat footage (heavy compression, motion blur, smoke,
  low light) is far from typical training footage. The released LoRA was
  trained on DL3DV-style scenes; since no finetuning code is released, the
  zero-shot experiment below is also the test of whether the domain gap is
  fatal. Our paired renders/frames remain the right finetuning data if
  training code appears later.
- **Resolution mismatch.** Output is fixed at 832×480; our source frames and
  point-cloud renders should be prepared at (or cropped to) that aspect so the
  crop-and-resize does not silently cut off scene content.
- **Hardware.** ~60 GB of weights plus a large-VRAM GPU; the existing RunPod
  flow (`runpod_amb3r_setup.sh`) is the natural home, same as the AMB3R
  experiment. VRAM offloading is built in, so a single-GPU pod works.

## Suggested Next Step

A zero-shot RunPod experiment, before any deeper integration:

1. Pick one good scene (e.g. the Sholef attack overlap) and export a 61-frame
   window: the point-cloud renders along the recovered camera path
   (`camera_views/f_*_vggt_render.jpg` order per `frames.csv`), reusing
   `prepare_amb3r_experiment.py`-style tooling.
2. On a RunPod GPU: install per the repo README, run
   `scripts/download_models.py`, then `scripts/run_inference.py --input
   <frames-folder> --output_dir <out>`; try both the default clean anchors
   (real first/last frames swapped in at indices 0 and 60) and
   `--clean_frame_indices ""`.
3. Judge the refined video visually against the real frames, then re-run VGGT
   on `generated.mp4` frames and compare against the original reconstruction
   with `compare_reconstructions.py` (density, pose agreement with
   `relative_path.csv`).
4. Only if (3) shows a real win, design the pipeline stage: a
   `tools/pipeline/` wrapper, 61-frame windowing for longer segments, and a
   `generative_refinement` provenance block in scene metadata.
