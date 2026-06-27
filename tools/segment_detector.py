#!/usr/bin/env python3
"""Train and evaluate a small automatic FPV segment boundary detector.

The detector is intentionally lightweight:
- ffmpeg samples frames from each video.
- visual change features produce boundary candidates.
- annotation examples train a classifier that labels candidate boundaries.

This is a baseline mechanism, not a final production model. It is designed so
new annotations can be added without changing the code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = REPO_ROOT / "splits" / "segment_detection_15_seed_20260627.json"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "segment-detector"
EVENT_LABELS = {
    "flight_start",
    "new_flight_start",
    "pause_start",
    "replay_start",
    "other",
}
CANDIDATE_STRATEGIES = (
    "dense",
    "pixel",
    "histogram",
    "luminance",
    "edge",
    "visual",
    "audio",
    "combined",
    "visual_or_audio",
)
FEATURE_CACHE: dict[tuple[str, float], tuple[dict[str, Any], float, FrameFeatures]] = {}


@dataclass
class FrameFeatures:
    times: np.ndarray
    score: np.ndarray
    visual_score: np.ndarray
    pixel_score: np.ndarray
    hist_score: np.ndarray
    luminance_delta_score: np.ndarray
    edge_delta_score: np.ndarray
    audio_score: np.ndarray
    audio_rms: np.ndarray
    luminance: np.ndarray
    saturation: np.ndarray
    edge_energy: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def annotation_path(name: str) -> Path:
    path = REPO_ROOT / "annotations" / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def slug_for_url(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    suffix = Path(url.split("?", 1)[0]).suffix or ".mp4"
    return f"{digest}{suffix}"


def ensure_video(annotation: dict[str, Any], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = annotation["video_url"]
    filename = annotation.get("video_file") or slug_for_url(url)
    local = cache_dir / filename
    if local.exists() and local.stat().st_size > 0:
        return local
    tmp = local.with_suffix(local.suffix + ".part")
    print(f"download {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(local)
    return local


def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def sample_frames(video_path: Path, fps: float, width: int, height: int) -> np.ndarray:
    vf = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "format=rgb24"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    frame_size = width * height * 3
    if len(raw) % frame_size:
        raise RuntimeError(f"unexpected raw frame byte count for {video_path}: {len(raw)}")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape((-1, height, width, 3))
    if len(frames) < 2:
        raise RuntimeError(f"too few sampled frames for {video_path}")
    return frames


def sample_audio(video_path: Path, sample_rate: int = 8000) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    try:
        raw = subprocess.check_output(cmd)
    except subprocess.CalledProcessError:
        return np.zeros(0, dtype=np.float32)
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def robust_score(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) or 1e-6
    score = (values - median) / mad
    return np.clip(score, 0, None).astype(np.float32)


def audio_features_at_times(
    audio: np.ndarray,
    times: np.ndarray,
    sample_rate: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) == 0:
        zeros = np.zeros(len(times), dtype=np.float32)
        return zeros, zeros

    hop_seconds = float(np.median(np.diff(times))) if len(times) > 1 else 0.25
    hop = max(1, int(round(sample_rate * hop_seconds)))
    window = max(hop, int(sample_rate * 0.18))
    rms = np.zeros(len(times), dtype=np.float32)
    for i, t in enumerate(times):
        center = int(round(float(t) * sample_rate))
        start = max(0, center - window // 2)
        end = min(len(audio), center + window // 2)
        if end <= start:
            continue
        chunk = audio[start:end]
        rms[i] = float(np.sqrt(np.mean(chunk * chunk)))

    delta = np.zeros(len(times), dtype=np.float32)
    delta[1:] = np.abs(np.diff(rms))
    return rms, robust_score(delta)


def extract_features(video_path: Path, fps: float = 4.0, width: int = 96, height: int = 54) -> FrameFeatures:
    frames = sample_frames(video_path, fps=fps, width=width, height=height).astype(np.float32)
    duration = ffprobe_duration(video_path)
    times = np.arange(len(frames), dtype=np.float32) / fps
    times = np.minimum(times, duration)

    gray = frames.mean(axis=3)
    luminance = gray.mean(axis=(1, 2)) / 255.0
    maxc = frames.max(axis=3)
    minc = frames.min(axis=3)
    saturation = np.divide(maxc - minc, maxc + 1.0).mean(axis=(1, 2))

    gy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    gx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))
    edge_energy = (gx + gy) / 255.0

    pix_diff = np.zeros(len(frames), dtype=np.float32)
    pix_diff[1:] = np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2, 3)) / 255.0

    hist_diff = np.zeros(len(frames), dtype=np.float32)
    bins = np.linspace(0, 256, 17)
    prev_hist = None
    for i, frame in enumerate(frames):
        hist = []
        for channel in range(3):
            h, _ = np.histogram(frame[:, :, channel], bins=bins, density=True)
            hist.append(h)
        cur = np.concatenate(hist)
        if prev_hist is not None:
            hist_diff[i] = np.abs(cur - prev_hist).sum()
        prev_hist = cur

    luminance_delta = np.zeros(len(frames), dtype=np.float32)
    luminance_delta[1:] = np.abs(np.diff(luminance))
    edge_delta = np.zeros(len(frames), dtype=np.float32)
    edge_delta[1:] = np.abs(np.diff(edge_energy))
    audio_rms, audio_score = audio_features_at_times(sample_audio(video_path), times)

    pixel_score = robust_score(pix_diff)
    hist_score = robust_score(hist_diff)
    luminance_delta_score = robust_score(luminance_delta)
    edge_delta_score = robust_score(edge_delta)
    visual_score = np.maximum.reduce(
        [
            pixel_score,
            0.65 * hist_score,
            0.75 * luminance_delta_score,
            0.65 * edge_delta_score,
        ]
    )
    score = np.maximum(visual_score, 0.9 * audio_score)
    return FrameFeatures(
        times,
        score,
        visual_score,
        pixel_score,
        hist_score,
        luminance_delta_score,
        edge_delta_score,
        audio_score,
        audio_rms,
        luminance,
        saturation,
        edge_energy,
    )


def nearest_index(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - t)))


def vector_at(features: FrameFeatures, index: int, video_duration: float) -> list[float]:
    t = float(features.times[index])
    prev_score = float(features.score[max(index - 1, 0)])
    next_score = float(features.score[min(index + 1, len(features.score) - 1)])
    prev_audio = float(features.audio_score[max(index - 1, 0)])
    next_audio = float(features.audio_score[min(index + 1, len(features.audio_score) - 1)])
    return [
        t,
        t / max(video_duration, 1e-6),
        float(features.score[index]),
        float(features.visual_score[index]),
        float(features.pixel_score[index]),
        float(features.hist_score[index]),
        float(features.luminance_delta_score[index]),
        float(features.edge_delta_score[index]),
        float(features.audio_score[index]),
        prev_audio,
        next_audio,
        float(features.audio_rms[index]),
        prev_score,
        next_score,
        float(features.luminance[index]),
        float(features.saturation[index]),
        float(features.edge_energy[index]),
    ]


def candidate_scores(features: FrameFeatures, strategy: str) -> np.ndarray:
    if strategy == "pixel":
        return features.pixel_score
    if strategy == "histogram":
        return features.hist_score
    if strategy == "luminance":
        return features.luminance_delta_score
    if strategy == "edge":
        return features.edge_delta_score
    if strategy == "visual":
        return features.visual_score
    if strategy == "audio":
        return features.audio_score
    if strategy == "combined":
        return features.score
    if strategy == "visual_or_audio":
        return np.maximum(features.visual_score, features.audio_score)
    raise ValueError(f"unknown candidate strategy: {strategy}")


def candidate_indices(
    features: FrameFeatures,
    min_gap: float = 0.75,
    threshold: float = 6.0,
    strategy: str = "visual_or_audio",
) -> list[int]:
    if strategy == "dense":
        step = max(1, int(round(min_gap / max(float(np.median(np.diff(features.times))), 1e-6))))
        return list(range(1, len(features.times) - 1, step))
    scores = candidate_scores(features, strategy)
    candidates: list[int] = []
    last_time = -999.0
    for i in range(1, len(scores) - 1):
        if scores[i] < threshold:
            continue
        if scores[i] < scores[i - 1] or scores[i] < scores[i + 1]:
            continue
        t = float(features.times[i])
        if t - last_time < min_gap:
            if candidates and scores[i] > scores[candidates[-1]]:
                candidates[-1] = i
                last_time = t
            continue
        candidates.append(i)
        last_time = t
    return candidates


def canonical_type(segment_type: str) -> str:
    return "flight_start" if segment_type == "new_flight_start" else segment_type


def annotation_events(annotation: dict[str, Any]) -> list[tuple[float, str]]:
    events = []
    for segment in annotation["segments"]:
        label = canonical_type(segment["type"])
        if label in EVENT_LABELS:
            events.append((float(segment["time"]), label))
    return events


def load_features(annotation_name: str, cache_dir: Path, fps: float) -> tuple[dict[str, Any], float, FrameFeatures]:
    key = (annotation_name, fps)
    if key in FEATURE_CACHE:
        return FEATURE_CACHE[key]
    ann = load_json(annotation_path(annotation_name))
    video = ensure_video(ann, cache_dir)
    duration = ffprobe_duration(video)
    features = extract_features(video, fps=fps)
    FEATURE_CACHE[key] = (ann, duration, features)
    return ann, duration, features


def build_training_rows(
    names: list[str],
    cache_dir: Path,
    fps: float,
    candidate_threshold: float,
    candidate_strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[str] = []
    for name in names:
        ann, duration, features = load_features(name, cache_dir, fps)
        positives: set[int] = set()
        for t, label in annotation_events(ann):
            if t <= 0.2:
                continue
            idx = nearest_index(features.times, t)
            positives.add(idx)
            rows.append(vector_at(features, idx, duration))
            labels.append(label)

        # Hard visual changes that are not near an annotation teach the model
        # which cuts to ignore.
        gold_times = [t for t, _label in annotation_events(ann)]
        for idx in candidate_indices(
            features,
            threshold=candidate_threshold,
            strategy=candidate_strategy,
        ):
            t = float(features.times[idx])
            if any(abs(t - gt) <= 0.6 for gt in gold_times):
                continue
            rows.append(vector_at(features, idx, duration))
            labels.append("background")

    return np.asarray(rows, dtype=np.float32), np.asarray(labels)


def train_model(
    train_names: list[str],
    cache_dir: Path,
    fps: float,
    candidate_threshold: float,
    candidate_strategy: str,
):
    x, y = build_training_rows(
        train_names,
        cache_dir=cache_dir,
        fps=fps,
        candidate_threshold=candidate_threshold,
        candidate_strategy=candidate_strategy,
    )
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=250,
            random_state=20260627,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
    )
    model.fit(x, y)
    return model


def detect_events(
    annotation_name: str,
    model,
    cache_dir: Path,
    fps: float,
    candidate_threshold: float,
    candidate_strategy: str,
    min_confidence: float,
) -> list[dict[str, Any]]:
    ann, duration, features = load_features(annotation_name, cache_dir, fps)
    events: list[dict[str, Any]] = [
        {"time": 0.0, "type": "banner_start", "score": 1.0, "source": "rule"}
    ]
    for idx in candidate_indices(
        features,
        threshold=candidate_threshold,
        strategy=candidate_strategy,
    ):
        vec = np.asarray([vector_at(features, idx, duration)], dtype=np.float32)
        label = str(model.predict(vec)[0])
        if label == "background":
            continue
        proba = getattr(model, "predict_proba")(vec)[0]
        classes = list(getattr(model, "classes_", []))
        confidence = float(proba[classes.index(label)]) if label in classes else 0.0
        if confidence < min_confidence:
            continue
        events.append(
            {
                "time": round(float(features.times[idx]), 3),
                "type": label,
                "score": round(confidence, 3),
                "source": "model",
            }
        )

    # Adjacent candidates can describe the same boundary at low sample rates.
    deduped: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda e: (e["time"], e["type"])):
        if deduped and event["type"] == deduped[-1]["type"] and event["time"] - deduped[-1]["time"] < 0.8:
            if event.get("score", 0) > deduped[-1].get("score", 0):
                deduped[-1] = event
            continue
        deduped.append(event)
    return deduped


def match_events(gold: list[tuple[float, str]], pred: list[dict[str, Any]], tolerance: float) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    used: set[int] = set()
    for gt, label in gold:
        if gt <= 0.2:
            continue
        best_i = None
        best_dt = math.inf
        for i, event in enumerate(pred):
            if i in used or event["type"] == "banner_start":
                continue
            dt = abs(float(event["time"]) - gt)
            if dt < best_dt:
                best_i = i
                best_dt = dt
        y_true.append(label)
        if best_i is not None and best_dt <= tolerance:
            used.add(best_i)
            y_pred.append(str(pred[best_i]["type"]))
        else:
            y_pred.append("missing")
    for i, event in enumerate(pred):
        if i in used or event["type"] == "banner_start":
            continue
        y_true.append("background")
        y_pred.append(str(event["type"]))
    return y_true, y_pred


def cmd_split(args: argparse.Namespace) -> int:
    files = sorted(p.name for p in (REPO_ROOT / "annotations").glob("*.json"))
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(files, size=args.total, replace=False).tolist())
    shuffled = chosen[:]
    rng.shuffle(shuffled)
    train = sorted(shuffled[: args.train])
    test = sorted(shuffled[args.train :])
    data = {
        "name": args.name,
        "seed": args.seed,
        "source": "annotations/*.json",
        "train": train,
        "test": test,
    }
    print(json.dumps(data, indent=2))
    return 0


def evaluate_config(
    train: list[str],
    test: list[str],
    cache_dir: Path,
    fps: float,
    candidate_threshold: float,
    candidate_strategy: str,
    min_confidence: float,
    tolerance: float,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    model = train_model(
        train,
        cache_dir=cache_dir,
        fps=fps,
        candidate_threshold=candidate_threshold,
        candidate_strategy=candidate_strategy,
    )

    all_true: list[str] = []
    all_pred: list[str] = []
    rows: list[dict[str, Any]] = []
    for name in test:
        ann = load_json(annotation_path(name))
        pred = detect_events(
            name,
            model,
            cache_dir=cache_dir,
            fps=fps,
            candidate_threshold=candidate_threshold,
            candidate_strategy=candidate_strategy,
            min_confidence=min_confidence,
        )
        true_labels, pred_labels = match_events(annotation_events(ann), pred, tolerance=tolerance)
        all_true.extend(true_labels)
        all_pred.extend(pred_labels)
        rows.append(
            {
                "annotation": name,
                "gold_events": len([e for e in annotation_events(ann) if e[0] > 0.2]),
                "predicted_events": len([e for e in pred if e["type"] != "banner_start"]),
                "predictions": pred,
            }
        )
    return all_true, all_pred, rows


def score_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    return {
        "events": len(y_true),
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "missing": sum(1 for label in y_pred if label == "missing"),
        "false_positive": sum(1 for label in y_true if label == "background"),
    }


def cmd_evaluate(args: argparse.Namespace) -> int:
    split = load_json(Path(args.split))
    all_true, all_pred, rows = evaluate_config(
        split["train"],
        split["test"],
        cache_dir=Path(args.cache_dir),
        fps=args.fps,
        candidate_threshold=args.candidate_threshold,
        candidate_strategy=args.candidate_strategy,
        min_confidence=args.min_confidence,
        tolerance=args.tolerance,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(classification_report(all_true, all_pred, zero_division=0))
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["annotation", "gold_events", "predicted_events"])
            writer.writeheader()
            writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    split = load_json(Path(args.split))
    rows: list[dict[str, Any]] = []
    strategies = args.strategies or list(CANDIDATE_STRATEGIES)
    for strategy in strategies:
        for threshold in args.thresholds:
            y_true, y_pred, details = evaluate_config(
                split["train"],
                split["test"],
                cache_dir=Path(args.cache_dir),
                fps=args.fps,
                candidate_threshold=threshold,
                candidate_strategy=strategy,
                min_confidence=args.min_confidence,
                tolerance=args.tolerance,
            )
            score = score_predictions(y_true, y_pred)
            rows.append(
                {
                    "strategy": strategy,
                    "threshold": threshold,
                    "accuracy": round(score["accuracy"], 4),
                    "events": score["events"],
                    "missing": score["missing"],
                    "false_positive": score["false_positive"],
                    "predicted_events": sum(r["predicted_events"] for r in details),
                }
            )

    rows.sort(key=lambda r: (r["accuracy"], -r["missing"], -r["false_positive"]), reverse=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "strategy",
            "threshold",
            "accuracy",
            "events",
            "missing",
            "false_positive",
            "predicted_events",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    split = load_json(Path(args.split))
    model = train_model(
        split["train"],
        cache_dir=Path(args.cache_dir),
        fps=args.fps,
        candidate_threshold=args.candidate_threshold,
        candidate_strategy=args.candidate_strategy,
    )
    events = detect_events(
        args.annotation,
        model,
        cache_dir=Path(args.cache_dir),
        fps=args.fps,
        candidate_threshold=args.candidate_threshold,
        candidate_strategy=args.candidate_strategy,
        min_confidence=args.min_confidence,
    )
    print(json.dumps({"annotation": args.annotation, "events": events}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("make-split", help="Create a deterministic train/test split JSON on stdout.")
    split.add_argument("--seed", type=int, default=20260627)
    split.add_argument("--total", type=int, default=15)
    split.add_argument("--train", type=int, default=10)
    split.add_argument("--name", default="segment_detection_custom")
    split.set_defaults(func=cmd_split)

    evaluate = sub.add_parser("evaluate", help="Train on split train videos and evaluate on split test videos.")
    evaluate.add_argument("--split", default=str(DEFAULT_SPLIT))
    evaluate.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    evaluate.add_argument("--fps", type=float, default=4.0)
    evaluate.add_argument("--candidate-threshold", type=float, default=2.0)
    evaluate.add_argument("--candidate-strategy", choices=CANDIDATE_STRATEGIES, default="pixel")
    evaluate.add_argument("--min-confidence", type=float, default=0.20)
    evaluate.add_argument("--tolerance", type=float, default=1.0)
    evaluate.add_argument("--output")
    evaluate.add_argument("--csv")
    evaluate.set_defaults(func=cmd_evaluate)

    benchmark = sub.add_parser("benchmark", help="Compare candidate strategies and thresholds.")
    benchmark.add_argument("--split", default=str(DEFAULT_SPLIT))
    benchmark.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    benchmark.add_argument("--fps", type=float, default=4.0)
    benchmark.add_argument("--min-confidence", type=float, default=0.20)
    benchmark.add_argument("--tolerance", type=float, default=1.0)
    benchmark.add_argument("--thresholds", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    benchmark.add_argument("--strategies", choices=CANDIDATE_STRATEGIES, nargs="+")
    benchmark.add_argument("--output")
    benchmark.set_defaults(func=cmd_benchmark)

    detect = sub.add_parser("detect", help="Predict segment boundary events for one annotation/video.")
    detect.add_argument("annotation", help="Annotation JSON filename under annotations/.")
    detect.add_argument("--split", default=str(DEFAULT_SPLIT))
    detect.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    detect.add_argument("--fps", type=float, default=4.0)
    detect.add_argument("--candidate-threshold", type=float, default=2.0)
    detect.add_argument("--candidate-strategy", choices=CANDIDATE_STRATEGIES, default="pixel")
    detect.add_argument("--min-confidence", type=float, default=0.20)
    detect.set_defaults(func=cmd_detect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
