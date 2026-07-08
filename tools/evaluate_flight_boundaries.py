#!/usr/bin/env python3
"""Evaluate simple video boundary detectors against flight annotations.

The annotations mark semantic points (flight_start, pause_start, replay_start,
etc.). This script turns them into target boundaries and checks whether a visual
change detector produces a candidate near each target.

The detector is intentionally lightweight: FFmpeg samples frames, then Python
computes frame-difference, histogram, pHash-like, black-frame, and freeze
features. It is meant as a baseline before trying heavier shot-boundary models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = ROOT / "annotations"
DEFAULT_CACHE_DIR = Path("/tmp/fpv-flight-boundaries")
FLIGHT_TYPES = {"flight_start", "new_flight_start"}


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    duration: float
    fps_text: str


@dataclass(frozen=True)
class Candidate:
    time: float
    score: float
    reason: str


@dataclass(frozen=True)
class Target:
    time: float
    kind: str
    source_type: str


def run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def ffprobe(url: str) -> VideoMeta:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            url,
        ]
    )
    stream = data["streams"][0]
    fmt = data["format"]
    return VideoMeta(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration=float(fmt["duration"]),
        fps_text=stream.get("avg_frame_rate", ""),
    )


def scaled_size(meta: VideoMeta, width: int) -> tuple[int, int]:
    height = round(meta.height * width / meta.width)
    if height % 2:
        height += 1
    return width, height


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def feature_cache_path(cache_dir: Path, video_file: str, fps: float, width: int) -> Path:
    return cache_dir / f"{slug(video_file)}.fps{fps:g}.w{width}.npz"


def sample_frames(url: str, meta: VideoMeta, fps: float, width: int) -> np.ndarray:
    out_w, out_h = scaled_size(meta, width)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        url,
        "-vf",
        f"fps={fps:g},scale={out_w}:{out_h},format=rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    frame_bytes = out_w * out_h * 3
    if len(proc.stdout) % frame_bytes != 0:
        raise RuntimeError(
            f"Unexpected rawvideo byte count for {url}: "
            f"{len(proc.stdout)} is not divisible by {frame_bytes}"
        )
    frame_count = len(proc.stdout) // frame_bytes
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    return frames.reshape(frame_count, out_h, out_w, 3)


def block_mean_hash(gray: np.ndarray, size: int = 8) -> np.ndarray:
    """Return an average-hash style bit array for frames shaped N,H,W."""
    n, h, w = gray.shape
    h_crop = (h // size) * size
    w_crop = (w // size) * size
    cropped = gray[:, :h_crop, :w_crop]
    blocks = cropped.reshape(n, size, h_crop // size, size, w_crop // size)
    small = blocks.mean(axis=(2, 4))
    med = np.median(small, axis=(1, 2), keepdims=True)
    return small > med


def compute_features(frames: np.ndarray, fps: float) -> dict[str, np.ndarray | float]:
    rgb = frames.astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    n, h, w = gray.shape
    times = np.arange(n, dtype=np.float32) / fps

    y0, y1 = int(h * 0.08), int(h * 0.92)
    x0, x1 = int(w * 0.06), int(w * 0.94)
    inner = gray[:, y0:y1, x0:x1]

    diff = np.zeros(n, dtype=np.float32)
    diff_inner = np.zeros(n, dtype=np.float32)
    if n > 1:
        diff[1:] = np.mean(np.abs(gray[1:] - gray[:-1]), axis=(1, 2))
        diff_inner[1:] = np.mean(np.abs(inner[1:] - inner[:-1]), axis=(1, 2))

    hist = np.zeros((n, 32), dtype=np.float32)
    for i, frame in enumerate(inner):
        hist[i], _ = np.histogram(frame, bins=32, range=(0.0, 1.0), density=True)
    hist_diff = np.zeros(n, dtype=np.float32)
    if n > 1:
        hist_diff[1:] = np.mean(np.abs(hist[1:] - hist[:-1]), axis=1)

    ahash = block_mean_hash(gray)
    hash_diff = np.zeros(n, dtype=np.float32)
    if n > 1:
        hash_diff[1:] = np.mean(ahash[1:] != ahash[:-1], axis=(1, 2))

    return {
        "times": times,
        "diff": diff,
        "diff_inner": diff_inner,
        "hist_diff": hist_diff,
        "hash_diff": hash_diff,
        "brightness": np.mean(gray, axis=(1, 2)).astype(np.float32),
        "black_fraction": np.mean(gray < 0.06, axis=(1, 2)).astype(np.float32),
        "fps": float(fps),
    }


def load_or_compute_features(
    annotation: dict,
    cache_dir: Path,
    fps: float,
    width: int,
    refresh: bool,
) -> tuple[VideoMeta, dict[str, np.ndarray | float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = feature_cache_path(cache_dir, annotation["video_file"], fps, width)
    meta_path = cache_path.with_suffix(".meta.json")

    if cache_path.exists() and meta_path.exists() and not refresh:
        meta_data = json.loads(meta_path.read_text())
        meta = VideoMeta(**meta_data)
        with np.load(cache_path) as data:
            return meta, {k: data[k] for k in data.files}

    meta = ffprobe(annotation["video_url"])
    frames = sample_frames(annotation["video_url"], meta, fps=fps, width=width)
    features = compute_features(frames, fps=fps)
    np.savez_compressed(cache_path, **features)
    meta_path.write_text(json.dumps(meta.__dict__, indent=2))
    return meta, features


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-6:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def normalize(values: np.ndarray) -> np.ndarray:
    q05, q95 = np.quantile(values, [0.05, 0.95])
    if q95 <= q05 + 1e-6:
        return np.zeros_like(values)
    return np.clip((values - q05) / (q95 - q05), 0.0, 1.0)


def local_maxima(score: np.ndarray, min_score: float, radius: int) -> Iterable[int]:
    for i in range(1, len(score) - 1):
        if score[i] < min_score:
            continue
        lo = max(0, i - radius)
        hi = min(len(score), i + radius + 1)
        if score[i] >= np.max(score[lo:hi]):
            yield i


def add_state_edges(
    candidates: list[Candidate],
    times: np.ndarray,
    active: np.ndarray,
    reason: str,
    min_duration_sec: float,
    fps: float,
) -> None:
    min_len = max(1, int(round(min_duration_sec * fps)))
    padded = np.concatenate([[False], active, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(changes[0::2], changes[1::2]):
        if end - start < min_len:
            continue
        candidates.append(Candidate(float(times[start]), 1.0, f"{reason}_start"))
        if end < len(times):
            candidates.append(Candidate(float(times[end]), 1.0, f"{reason}_end"))


def detect_candidates(
    features: dict[str, np.ndarray | float],
    cut_score: float,
    peak_quantile: float,
    merge_sec: float,
    freeze_diff: float,
    freeze_sec: float,
) -> list[Candidate]:
    times = features["times"]  # type: ignore[assignment]
    fps = float(features["fps"])
    diff_inner = features["diff_inner"]  # type: ignore[assignment]
    hist_diff = features["hist_diff"]  # type: ignore[assignment]
    hash_diff = features["hash_diff"]  # type: ignore[assignment]
    black_fraction = features["black_fraction"]  # type: ignore[assignment]

    score = (
        0.52 * normalize(diff_inner)
        + 0.30 * normalize(hist_diff)
        + 0.18 * normalize(hash_diff)
    )
    z = robust_z(score)
    threshold = max(cut_score, float(np.quantile(score, peak_quantile)), float(np.median(score) + 1.5 * np.median(np.abs(score - np.median(score)))))
    threshold = min(threshold, 0.92)
    radius = max(1, int(round(0.45 * fps)))

    candidates: list[Candidate] = [
        Candidate(float(times[i]), float(score[i]), f"visual_peak_z{z[i]:.1f}")
        for i in local_maxima(score, threshold, radius)
    ]

    add_state_edges(
        candidates,
        times,
        black_fraction > 0.68,
        "black",
        min_duration_sec=0.25,
        fps=fps,
    )
    add_state_edges(
        candidates,
        times,
        diff_inner < freeze_diff,
        "freeze",
        min_duration_sec=freeze_sec,
        fps=fps,
    )

    return merge_candidates(candidates, merge_sec=merge_sec)


def merge_candidates(candidates: list[Candidate], merge_sec: float) -> list[Candidate]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: (c.time, -c.score))
    groups: list[list[Candidate]] = []
    for cand in candidates:
        if not groups or cand.time - groups[-1][-1].time > merge_sec:
            groups.append([cand])
        else:
            groups[-1].append(cand)
    merged = []
    for group in groups:
        best = max(group, key=lambda c: c.score)
        reasons = ",".join(sorted({c.reason.split("_z")[0] for c in group}))
        merged.append(Candidate(best.time, best.score, reasons))
    return merged


def load_annotations(paths: Iterable[Path]) -> list[dict]:
    annotations = []
    for path in paths:
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        annotations.append(data)
    return annotations


def targets_from_annotation(annotation: dict, duration: float) -> list[Target]:
    segments = sorted(annotation.get("segments", []), key=lambda s: s["time"])
    targets: list[Target] = []
    for idx, segment in enumerate(segments):
        if segment["type"] not in FLIGHT_TYPES:
            continue
        targets.append(Target(float(segment["time"]), "start", segment["type"]))
        end_time = duration
        end_type = "video_end"
        for nxt in segments[idx + 1 :]:
            if nxt["type"] in FLIGHT_TYPES:
                continue
            end_time = float(nxt["time"])
            end_type = nxt["type"]
            break
        targets.append(Target(end_time, "end", end_type))
    return targets


def nearest(candidate_times: list[float], target_time: float) -> tuple[float | None, float | None]:
    if not candidate_times:
        return None, None
    best = min(candidate_times, key=lambda t: abs(t - target_time))
    return best, abs(best - target_time)


def evaluate_video(
    annotation: dict,
    meta: VideoMeta,
    candidates: list[Candidate],
    tolerances: list[float],
) -> dict:
    targets = targets_from_annotation(annotation, duration=meta.duration)
    candidate_times = [c.time for c in candidates]
    rows = []
    for target in targets:
        best, err = nearest(candidate_times, target.time)
        rows.append(
            {
                "kind": target.kind,
                "source_type": target.source_type,
                "time": target.time,
                "nearest": best,
                "error": err,
                **{f"hit_{tol:g}s": err is not None and err <= tol for tol in tolerances},
            }
        )
    return {
        "video_file": annotation["video_file"],
        "duration": meta.duration,
        "target_count": len(targets),
        "candidate_count": len(candidates),
        "targets": rows,
        "candidates": [c.__dict__ for c in candidates],
    }


def print_summary(results: list[dict], tolerances: list[float]) -> None:
    all_targets = [row for result in results for row in result["targets"]]
    print(f"videos\t{len(results)}")
    print(f"targets\t{len(all_targets)}")
    print(f"candidates\t{sum(r['candidate_count'] for r in results)}")
    for tol in tolerances:
        key = f"hit_{tol:g}s"
        hits = sum(1 for row in all_targets if row[key])
        print(f"recall@{tol:g}s\t{hits}/{len(all_targets)}\t{hits / len(all_targets):.3f}")
    for kind in ["start", "end"]:
        subset = [row for row in all_targets if row["kind"] == kind]
        if not subset:
            continue
        print(f"{kind}_targets\t{len(subset)}")
        for tol in tolerances:
            key = f"hit_{tol:g}s"
            hits = sum(1 for row in subset if row[key])
            print(f"{kind}_recall@{tol:g}s\t{hits}/{len(subset)}\t{hits / len(subset):.3f}")


def print_misses(results: list[dict], tolerance: float) -> None:
    key = f"hit_{tolerance:g}s"
    print(f"\nmisses@{tolerance:g}s")
    for result in results:
        misses = [row for row in result["targets"] if not row[key]]
        if not misses:
            continue
        print(f"\n{result['video_file']}  candidates={result['candidate_count']}")
        for row in misses:
            nearest_text = "none" if row["nearest"] is None else f"{row['nearest']:.3f}"
            err_text = "inf" if row["error"] is None else f"{row['error']:.3f}"
            print(
                f"  {row['kind']:5s} {row['time']:7.3f} "
                f"({row['source_type']}) nearest={nearest_text} err={err_text}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="*", type=Path, default=sorted(ANNOTATION_DIR.glob("*.json")))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--cut-score", type=float, default=0.58)
    parser.add_argument("--peak-quantile", type=float, default=0.88)
    parser.add_argument("--merge-sec", type=float, default=0.55)
    parser.add_argument("--freeze-diff", type=float, default=0.011)
    parser.add_argument("--freeze-sec", type=float, default=0.8)
    parser.add_argument("--tolerances", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--miss-tolerance", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    results = []
    for idx, annotation in enumerate(annotations, start=1):
        print(f"[{idx}/{len(annotations)}] {annotation['video_file']}", file=sys.stderr)
        try:
            meta, features = load_or_compute_features(
                annotation,
                cache_dir=args.cache_dir,
                fps=args.fps,
                width=args.width,
                refresh=args.refresh,
            )
            candidates = detect_candidates(
                features,
                cut_score=args.cut_score,
                peak_quantile=args.peak_quantile,
                merge_sec=args.merge_sec,
                freeze_diff=args.freeze_diff,
                freeze_sec=args.freeze_sec,
            )
            results.append(evaluate_video(annotation, meta, candidates, args.tolerances))
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: command failed for {annotation['video_file']}: {exc}", file=sys.stderr)
            if exc.stderr:
                print(exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr, file=sys.stderr)
            return 2

    print_summary(results, args.tolerances)
    print_misses(results, args.miss_tolerance)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
