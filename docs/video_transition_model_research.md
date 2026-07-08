# Video Transition Tagging Models and Papers

Date: 2026-06-30

Goal: find public papers/models that can help automatically tag edited FPV
videos into usable flight sections and non-flight material such as banners,
pauses, replays, fades, dissolves, and cuts.

The linked VGGT post makes the operational target clear: before reconstruction,
keep only genuine FPV flight and drop title cards, freeze/highlight pauses, and
replays. In that post's example, the useful footage was two non-contiguous
chunks, roughly seconds 10-20 and 25-37.

## Executive take

Best candidates to test first, based on the paper/model survey:

1. **OmniShotCut** for public model-based transition tagging. It detects shot
   changes and predicts transition labels, not just cut/no-cut.
2. **TransNetV2** as a robust, older, fast shot-boundary baseline.
3. **PySceneDetect / FFmpeg scene score** as lightweight deterministic
   baselines and fallback signals.
4. **DINOv2/CLIP/VideoMAE embeddings plus a small local classifier** for the
   dataset-specific tags that public shot-boundary models do not know:
   `banner`, `flight`, `pause`, `replay`.

There does not appear to be a public model that directly predicts this repo's
semantic labels out of the box. Public SBD models solve "where is an edit?" and
sometimes "what type of edit?" The FPV-specific work is classifying the
candidate intervals after those edit points.

Follow-up benchmark note: `docs/transition_model_benchmark.md` tests the public
models against the repo annotations. Empirically, TransNetV2 with a low `0.2`
threshold is the strongest public-model boundary source so far. OmniShotCut is
still useful as a transition-label feature source, but it should not be the
primary cutter for these FPV clips.

## Model and dataset survey

### OmniShotCut

Links:

- GitHub: https://github.com/UVA-Computer-Vision-Lab/OmniShotCut
- Hugging Face weights: https://huggingface.co/uva-cv-lab/OmniShotCut
- arXiv: https://arxiv.org/abs/2604.24762

What it provides:

- Public checkpoint: `OmniShotCut_ckpt.pth`
- HF model repo last modified: 2026-06-01
- Install path in README: `pip install git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git`
- Inference modes:
  - `clean_shot`: clean/general shot cuts only
  - `default`: outputs intra/inter labels, including transition classes

Label space in the public code:

- Intra labels: `General`, `Dissolve`, `Wipes`, `Push`, `Slide`, `Zoom`,
  `Fade`, `Doorway`, `Padding`
- Inter labels: `New_Start`, `Hard_Cut`, `Transition_Source`, `Transition`,
  `Sudden_Jump`, `Padding`

Fit for this project:

- Strongest public fit I found because it tags transitions, not only hard cuts.
- The `Sudden_Jump` and `Hard_Cut` labels may catch FPV-to-FPV edits that look
  visually similar.
- The transition labels can help separate branded/outro edits and stylized
  replay transitions.
- It still will not know `flight_start`, `pause_start`, or `replay_start`
  semantically. Treat its output as candidate boundaries and transition-type
  features.

Recommended test:

Run OmniShotCut `mode="default"` on all 25 annotated clips, then evaluate:

- recall near annotated flight starts/ends
- whether missed `other` cuts are recovered
- whether `Fade`/`Transition` labels correlate with replay/outro material
- candidate count per minute versus the existing lightweight detector

### TransNetV2

Links:

- GitHub: https://github.com/soCzech/TransNetV2
- Hugging Face PyTorch port: https://huggingface.co/magnusdtd/TransNetV2
- Hugging Face weights mirror: https://huggingface.co/Sn4kehead/TransNetV2
- Paper: https://arxiv.org/abs/2008.04838

What it provides:

- Established public shot-boundary detector.
- Pretrained weights are available in the original repo flow and mirrored on
  Hugging Face in PyTorch ports.
- Designed for fast shot transition detection, including hard and gradual shot
  boundaries.

Fit for this project:

- Good second test after OmniShotCut.
- Likely better than raw histogram/scene-score rules on subtle shot changes.
- Does not output the detailed transition taxonomy needed to distinguish fade,
  wipe, replay, pause, or banner by itself.

Recommended test:

Use TransNetV2 to generate high-recall shot boundaries, then combine with the
existing handcrafted features:

- black/freeze periods
- logo/banner template similarity
- replay duplicate matching
- frame-embedding similarity

### PySceneDetect and FFmpeg scene score

Links:

- PySceneDetect detectors:
  https://www.scenedetect.com/docs/latest/api/detectors.html
- FFmpeg filters: https://ffmpeg.org/ffmpeg-filters.html

What they provide:

- Deterministic, dependency-light shot/change detection.
- Useful for content changes, adaptive thresholding, black frames, and stable
  baseline comparisons.

Fit for this project:

- Already enough to reach high recall in the local baseline when tuned
  permissively.
- Weak on visually similar FPV-to-FPV cuts.
- Useful as a feature source even if OmniShotCut/TransNetV2 become primary.

### ClipShots

Links:

- GitHub: https://github.com/Tangshitao/ClipShots
- Paper: https://arxiv.org/abs/1808.04234

What it provides:

- Large-scale shot-boundary dataset from YouTube/Weibo.
- Contains train/test sets and an `only_gradual` set for gradual transition
  annotation.
- Baseline code exists in the companion `ClipShots_basline` repo.

Fit for this project:

- Useful background dataset/paper for hard vs gradual transitions.
- Less directly useful as an off-the-shelf model than OmniShotCut or TransNetV2.
- Could be useful if we fine-tune an SBD model, but the domain mismatch is large.

### AutoShot

Links:

- GitHub: https://github.com/wentaozhu/AutoShot
- Paper PDF in repo:
  https://github.com/wentaozhu/AutoShot/blob/main/CVPR23_AutoShot.pdf

What it provides:

- Short-video shot-boundary dataset and baseline from CVPR NAS 2023.
- Domain is social/short-form video, closer to edited propaganda clips than
  older movie-only datasets.
- Model/checkpoint links exist, but are less convenient than Hugging Face
  weights.

Fit for this project:

- Worth reading for dataset design and evaluation.
- Less convenient as the first runnable model because model access appears to
  rely on Baidu/Drive-style artifacts, not a simple HF checkpoint.

### Hugging Face shot-boundary dataset

Link:

- https://huggingface.co/datasets/it-just-works/shot-boundary-detection

What it provides:

- A HF dataset organized for shot-boundary detection.
- Potentially useful for quick experiments or sanity checks if building a local
  classifier.

Fit for this project:

- Secondary. Use the repo's own annotations first, because the FPV edit grammar
  is domain-specific.

### Shot2Story

Links:

- Hugging Face: https://huggingface.co/ByteDance/shot2story
- arXiv: https://arxiv.org/abs/2312.10300

What it provides:

- Public weights for shot-level and summarization tasks.
- More about multi-shot video understanding/story summarization than exact
  boundary detection.

Fit for this project:

- Not a first-line transition detector.
- Potentially useful as a pretrained shot/clip embedding model if we want
  interval-level semantic classification later.

### VideoMAE / DINOv2 / CLIP style embeddings

Links:

- VideoMAE HF model: https://huggingface.co/MCG-NJU/videomae-base
- DINOv2 HF model: https://huggingface.co/facebook/dinov2-base
- CLIP HF model: https://huggingface.co/openai/clip-vit-base-patch32

What they provide:

- General visual/video embeddings.
- Good feature backbones for small classifiers when label data is limited.

Fit for this project:

- Best for semantic interval labels, not raw boundary detection.
- For replays: embed short frame windows and run sequence matching/DTW against
  earlier windows.
- For banners/outros: image embeddings plus template/OCR signals should work
  well.
- For flight/non-flight: train a simple classifier on windows between proposed
  boundaries using the current annotations.

## Proposed experiment order

1. Keep the existing lightweight detector as the control.
2. Run OmniShotCut `mode="default"` on the annotated 25 videos.
3. Run TransNetV2 on the same clips.
4. Compare all three on:
   - boundary recall at 0.25 s, 0.5 s, 1.0 s
   - candidate count/minute
   - misses by annotation type: `pause_start`, `replay_start`, `other`
   - whether each detector recovers the subtle 2026-06-03 `other` cuts
5. Build interval features from the union of candidate boundaries:
   - duration
   - black/freeze ratio
   - motion energy
   - OmniShotCut transition labels
   - TransNetV2 probability peak
   - DINOv2/CLIP window embeddings
   - similarity to intro/banner/replay templates
6. Train a tiny classifier or ruleset for:
   - `flight`
   - `banner/outro`
   - `pause/freeze`
   - `replay/duplicate`
   - `other`

## Recommendation

Do not make OmniShotCut or TransNetV2 directly decide "flight segment." Use them
to propose and label transitions. Then learn this dataset's semantic edit
grammar on top of those proposals.

The first real benchmark should be:

```text
existing high-recall baseline
vs OmniShotCut default
vs TransNetV2
vs union(existing + OmniShotCut + TransNetV2)
```

Success criterion for this stage should stay recall-first:

- 100% recall within 1.0 s on annotated flight starts/ends
- lower candidate count than the naive high-recall union if possible
- recover the subtle `other` cuts without missing replay/pause boundaries
