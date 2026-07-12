# Transition Model Benchmark

Date: 2026-06-30

This note records the first comparison of automatic video-transition detectors
against the repository annotations. The goal is not yet to produce final flight
clips. It is to answer: can a detector place candidate marks close enough to
the annotated flight starts and flight ends that a later interval classifier can
decide which spans are real FPV flight?

## Evaluation Target

The benchmark used the 25 annotation JSON files in `annotations/`.

- Annotated flight intervals: 69
- Boundary targets: 138, one start plus one end per annotated flight interval
- Target start types: `flight_start`, `new_flight_start`
- Target end time: the next non-flight annotation after a flight start, or the
  end of the video if no later non-flight marker exists
- Metrics: recall within 0.25 s, 0.5 s, and 1.0 s of each target boundary
- Candidate count is intentionally tracked because extra marks are acceptable
  for now, but a useful detector should not explode into hundreds of marks per
  clip

This scoring matches the current project priority: high recall on the real
flight starts and flight ends is more important than avoiding extra marks
outside the annotated flight sections.

## Models Tested

### Local High-Recall Baseline

Script: `tools/pipeline/evaluate_flight_boundaries.py`

This is a deterministic FFmpeg + NumPy baseline. It samples frames, computes
visual differences, histogram differences, simple perceptual-hash differences,
black-frame periods, and freeze periods.

Run used:

```bash
python3 tools/pipeline/evaluate_flight_boundaries.py \
  --cut-score 0.25 \
  --peak-quantile 0.80 \
  --merge-sec 0.35 \
  --json-out /tmp/fpv-flight-boundaries/results_high_recall.json
```

### TransNetV2

Public model:

- Hugging Face PyTorch port: `magnusdtd/TransNetV2`
- Original project: `soCzech/TransNetV2`

Two thresholds were tested:

- `0.5`, the conservative/default-style threshold
- `0.2`, a high-recall threshold better aligned with this project's goal

### OmniShotCut

Public model:

- GitHub: `UVA-Computer-Vision-Lab/OmniShotCut`
- Hugging Face weights: `uva-cv-lab/OmniShotCut`

This was tested in `default` mode so it emitted transition labels such as
`General/New_Start`, `General/Hard_Cut`, `Fade/Transition_Source`, and
`Zoom/New_Start`.

On the local Mac, OmniShotCut needed a temporary scratch patch because its
published code assumes CUDA/decord. The patched copy ran from
`/tmp/fpv-model-benchmark/OmniShotCut` using MPS when available and predecoded
NumPy frames from FFmpeg.

## Benchmark Command

The reusable benchmark harness is:

```bash
/tmp/fpv-model-benchmark/venv/bin/python tools/pipeline/benchmark_transition_models.py \
  --models baseline transnet \
  --transnet-dir /tmp/fpv-model-benchmark/TransNetV2 \
  --output /tmp/fpv-flight-boundaries/benchmark_baseline_transnet.json

/tmp/fpv-model-benchmark/venv/bin/python tools/pipeline/benchmark_transition_models.py \
  --models transnet \
  --transnet-dir /tmp/fpv-model-benchmark/TransNetV2 \
  --transnet-thresholds 0.2 \
  --output /tmp/fpv-flight-boundaries/benchmark_transnet_02.json

/tmp/fpv-model-benchmark/venv/bin/python tools/pipeline/benchmark_transition_models.py \
  --models omnishotcut \
  --omnishotcut-dir /tmp/fpv-model-benchmark/OmniShotCut \
  --output /tmp/fpv-flight-boundaries/benchmark_omnishotcut.json
```

The video cache used for the run was `/tmp/fpv-model-benchmark/videos`.

## Results

| Method | Candidates | Median / Video | Candidates / Min | Recall @ 0.25 s | Recall @ 0.5 s | Recall @ 1.0 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local baseline, high recall | 601 | 23.0 | 24.9 | 0.739 | 0.957 | 1.000 |
| TransNetV2, threshold 0.5 | 260 | 10.0 | 10.8 | 0.536 | 0.768 | 0.812 |
| OmniShotCut default | 308 | 13.0 | 12.8 | 0.580 | 0.587 | 0.594 |
| TransNetV2, threshold 0.2 | 359 | 14.0 | 14.9 | 0.710 | 0.978 | 1.000 |

## Interpretation

TransNetV2 at threshold `0.2` is the best public-model candidate from this first
run. It produced 359 candidates instead of the baseline's 601, while improving
0.5 s recall from 95.7% to 97.8% and preserving 100% recall within 1.0 s.

TransNetV2 at threshold `0.5` is too conservative for this dataset. It misses
too many pause and replay boundaries, which are exactly the edited sections
that matter for extracting clean flight footage.

OmniShotCut was promising in the literature survey because it emits transition
labels, but its boundary recall was poor on these annotations. It does catch
many replay starts, but it misses most pause/freeze boundaries and many
flight-start boundaries. For this dataset, it should be treated as a feature
source, not as the main boundary detector.

The local baseline remains useful because freeze/black-frame logic is very
strong on pause and replay edits. It is noisy, but that noise is acceptable in
the current recall-first stage.

## Recall by Annotation Source

| Method | `flight_start` @ 0.5 s | `other` @ 0.5 s | `pause_start` @ 0.5 s | `replay_start` @ 0.5 s |
| --- | ---: | ---: | ---: | ---: |
| Local baseline | 0.940 | 0.909 | 1.000 | 1.000 |
| TransNetV2, threshold 0.5 | 0.806 | 0.955 | 0.364 | 0.840 |
| OmniShotCut default | 0.552 | 0.727 | 0.136 | 0.960 |
| TransNetV2, threshold 0.2 | 1.000 | 0.955 | 0.955 | 0.960 |

This table explains the main outcome: thresholded TransNetV2 is good at real
shot boundaries, while the local handcrafted detector is strongest on freeze and
black-frame transitions.

## Remaining Misses at 0.5 s

TransNetV2 at threshold `0.2` missed only three targets at 0.5 s:

- `2026-06-06_merkava_tank_blat_position.mp4`, `pause_start` end at 65.486 s,
  nearest candidate 66.240 s
- `2026-06-06_merkava_tank_blat_position.mp4`, `replay_start` end at 74.653 s,
  nearest candidate 75.160 s
- `2026-06-13_command_center_beaufort_castle_mmirleb_17580.mp4`, `other` end at
  12.100 s, nearest candidate 12.640 s

The local baseline missed six targets at 0.5 s. All were still recovered within
1.0 s.

## Union Experiments

Naive unions were also tested by merging detector candidates within the same
0.35 s window.

| Union | Candidates | Median / Video | Candidates / Min | Recall @ 0.25 s | Recall @ 0.5 s | Recall @ 1.0 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TransNetV2 0.2 + OmniShotCut | 436 | 17.0 | 18.1 | 0.848 | 0.949 | 1.000 |
| Baseline + TransNetV2 0.2 | 607 | 23.0 | 25.2 | 0.667 | 0.971 | 1.000 |
| Baseline + OmniShotCut | 647 | 25.0 | 26.9 | 0.891 | 0.957 | 1.000 |
| All three | 640 | 24.0 | 26.6 | 0.761 | 0.913 | 1.000 |

These numbers should be read carefully. The union implementation picks one
representative timestamp per merged group, so adding detectors can shift a good
candidate just outside the tight 0.25 s or 0.5 s window. The union approach is
still useful, but the next version should preserve detector-specific peaks or
choose a representative based on source priority.

## Recommendation

Use TransNetV2 threshold `0.2` as the first public-model boundary source. Keep
the local freeze/black-frame baseline as a complementary source, especially for
pause and replay edits.

Do not use OmniShotCut as the primary cutter on this data. Keep its labels as
features for a later interval classifier or for explaining transition type.

The next practical detector should be:

1. Generate candidate boundaries from TransNetV2 `0.2`.
2. Add local freeze and black-frame start/end candidates.
3. Preserve source-specific timestamps instead of collapsing everything too
   early.
4. Build interval-level features between candidate boundaries.
5. Train or tune a small classifier for `flight`, `banner/outro`,
   `pause/freeze`, `replay`, and `other`.

At this stage, the best single-model setting already satisfies the most
important criterion: every annotated flight start/end is detected within 1.0 s,
with substantially fewer candidate marks than the purely handcrafted baseline.
