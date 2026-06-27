# Segment Detection Baseline

This repository includes manual segment annotations under `annotations/`.
The baseline detector in `tools/segment_detector.py` uses those examples to
predict segment boundary events automatically.

## Split

The requested 15-video split is stored in:

`splits/segment_detection_15_seed_20260627.json`

It contains 10 train videos and 5 test videos. The annotations branch currently
contains more than 15 annotation files, so the split file explicitly fixes the
15-video subset used for this experiment.

## Mechanism

The detector:

1. Downloads each annotated CloudFront video into `.cache/segment-detector/`.
2. Samples frames with `ffmpeg`.
3. Extracts visual change features from sampled frames:
   - frame difference score
   - luminance
   - saturation
   - edge energy
   - normalized timestamp
4. Finds candidate segment boundaries from visual-change peaks.
5. Trains a small random-forest classifier on the train annotations.
6. Classifies candidate boundaries as `flight_start`, `pause_start`,
   `replay_start`, `other`, or background.

`banner_start` is rule-based at `0.0s`.

## Usage

Evaluate the default 10/5 split:

```bash
python3 tools/segment_detector.py evaluate \
  --split splits/segment_detection_15_seed_20260627.json \
  --output reports/segment_detection_predictions.json \
  --csv reports/segment_detection_summary.csv
```

Predict one video:

```bash
python3 tools/segment_detector.py detect \
  2026-06-12_merkava_tank_yahmor_al_shqif_annotations.json
```

Create another deterministic split:

```bash
python3 tools/segment_detector.py make-split --seed 20260627 --total 15 --train 10
```

## Notes

This is a first baseline. It is intentionally small and transparent so the
annotation set can grow before moving to heavier models. The main expected
failure mode is semantic labeling of visually similar cuts; adding more
annotations should improve the classifier.

On the initial 10/5 split, the baseline runs end-to-end and detects
`flight_start` / `replay_start` reasonably, but `pause_start` and `other`
remain weak with the current visual-change-only features. The next useful
upgrade is to add richer frame features around candidate boundaries, such as
OCR/text-presence cues, optical-flow style motion changes, or clip embeddings.
