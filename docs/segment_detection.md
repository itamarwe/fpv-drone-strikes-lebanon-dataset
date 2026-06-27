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
3. Extracts visual and audio features:
   - pixel/frame-difference score
   - RGB histogram-change score
   - luminance-change score
   - edge-energy-change score
   - combined visual score
   - audio RMS and audio-onset score
   - normalized timestamp
4. Finds candidate segment boundaries with selectable strategies:
   - `pixel`
   - `histogram`
   - `luminance`
   - `edge`
   - `visual`
   - `audio`
   - `combined`
   - `visual_or_audio`
   - `dense`
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

Compare candidate-generation techniques:

```bash
python3 tools/segment_detector.py benchmark \
  --split splits/segment_detection_15_seed_20260627.json \
  --thresholds 2 4 6 8 \
  --output reports/segment_detection_benchmark.json
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

On the initial 10/5 split, the benchmark runs end-to-end. The best tested
configuration was `pixel` candidates at threshold `2.0`, with 0.423 event-label
accuracy on the 5-video test split. `visual`, `combined`, and `visual_or_audio`
clustered around 0.417. Raw `audio` candidates alone performed poorly on this
small split, but audio-onset features are still available to the classifier and
remain worth revisiting as the annotation set grows.

The current baseline detects `flight_start` / `replay_start` best, while
`pause_start` and `other` remain weak. The next useful upgrade is richer
features around candidate boundaries, such as OCR/text-presence cues,
optical-flow style motion changes, or clip embeddings.
