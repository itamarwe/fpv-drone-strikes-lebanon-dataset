# FixAnything Integration Research

Date: 2026-08-26

Question: could FixAnything (<https://fix-anything.github.io/>) be integrated
into the repo's 3D scene reconstruction workflow?

Short answer: not as a reconstruction backend, and not yet in practice (no
public code found as of this writing), but it is a genuinely good fit as a
future *rendering refinement* stage on top of the existing VGGT scenes — with a
hard caveat that its output is generative and must never feed the measurement
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

Note: direct fetches of the project page and arXiv were blocked by the session
network policy, so the above is compiled from search summaries of the abstract
and project page. Re-verify details (base video model, GPU needs, exact data
recipe) against the paper before building anything.

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

- **No public code or checkpoints found yet.** The paper is two days old;
  searches surface no GitHub repo. Nothing to integrate until CMU releases it.
  Worth watching the project page.
- **Generative provenance hazard.** This dataset documents real strikes and
  the viewer offers measurements (scale, AGL, speed). A video diffusion model
  hallucinates plausible detail — vehicles, damage, terrain that were never
  observed. Refined renders and any geometry re-reconstructed from them must
  be stored and labeled as synthetic derivatives, kept out of
  `scene_state.json` scale calibration and measurement workflows, and flagged
  in scene metadata (e.g. a `generative_refinement` block in
  `scene_manifest.json.model_config`).
- **Domain gap.** FPV combat footage (heavy compression, motion blur, smoke,
  low light) is far from typical training footage; own-domain LoRA finetuning
  on the repo's paired renders/frames would likely be required — feasible, per
  the point above, but real work.
- **Hardware.** Video diffusion finetuning + inference needs a serious GPU;
  the existing RunPod flow (`runpod_amb3r_setup.sh`) is the natural home, same
  as the AMB3R experiment.

## Suggested Next Step

Do nothing in-repo until code drops. When it does: reuse
`prepare_amb3r_experiment.py`-style tooling to export one scene's paired
renders/frames, run the released checkpoint zero-shot on a rendered flythrough
of a good scene (e.g. the Sholef attack overlap), and compare a VGGT re-run on
refined vs. raw renders with `compare_reconstructions.py` before considering
any finetuning.
