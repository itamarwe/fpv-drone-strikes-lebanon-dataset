#!/usr/bin/env python3
"""Benchmark public transition detectors against the FPV annotations.

This is intentionally dependency-light at import time. The public model runners
are enabled only when requested and expect their code/checkpoints to live in a
scratch directory such as /tmp/fpv-model-benchmark.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import evaluate_flight_boundaries as baseline


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = ROOT / "annotations"
DEFAULT_CACHE_DIR = Path("/tmp/fpv-flight-boundaries")
DEFAULT_MODEL_DIR = Path("/tmp/fpv-model-benchmark")
DEFAULT_VIDEO_CACHE_DIR = DEFAULT_MODEL_DIR / "videos"
FLIGHT_TYPES = {"flight_start", "new_flight_start"}


@dataclass(frozen=True)
class Candidate:
    time: float
    score: float
    reason: str


def load_annotations(paths: Iterable[Path]) -> list[dict]:
    annotations = []
    for path in paths:
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        annotations.append(data)
    return annotations


def fps_from_text(text: str) -> float:
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def ffprobe(path_or_url: str | Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,width,height,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path_or_url),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    return {
        "fps": fps_from_text(stream["avg_frame_rate"]),
        "duration": float(data["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
    }


def download_video(annotation: dict, cache_dir: Path, timeout: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / annotation["video_file"]
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "--connect-timeout",
            "20",
            "--max-time",
            str(timeout),
            "-o",
            str(tmp_path),
            annotation["video_url"],
        ],
        check=True,
    )
    tmp_path.replace(out_path)
    return out_path


def decode_rgb_frames(path_or_url: str | Path, width: int, height: int, timeout: int) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path_or_url),
            "-vf",
            f"scale={width}:{height},format=rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=timeout,
    )
    frame_bytes = width * height * 3
    if len(proc.stdout) % frame_bytes:
        raise RuntimeError(f"Unexpected rawvideo byte count: {len(proc.stdout)}")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def targets_from_annotation(annotation: dict, duration: float) -> list[dict]:
    segments = sorted(annotation.get("segments", []), key=lambda s: s["time"])
    targets = []
    for idx, segment in enumerate(segments):
        if segment["type"] not in FLIGHT_TYPES:
            continue
        targets.append({"time": float(segment["time"]), "kind": "start", "source_type": segment["type"]})
        end_time = duration
        end_type = "video_end"
        for nxt in segments[idx + 1 :]:
            if nxt["type"] in FLIGHT_TYPES:
                continue
            end_time = float(nxt["time"])
            end_type = nxt["type"]
            break
        targets.append({"time": end_time, "kind": "end", "source_type": end_type})
    return targets


def merge_candidates(candidates: list[Candidate], merge_sec: float = 0.35) -> list[Candidate]:
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
        reasons = ",".join(sorted({c.reason for c in group}))
        merged.append(Candidate(best.time, best.score, reasons))
    return merged


def candidates_from_probs(probs: np.ndarray, fps: float, threshold: float, reason: str) -> list[Candidate]:
    active = probs >= threshold
    padded = np.concatenate([[False], active, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    candidates = []
    for start, end in zip(changes[0::2], changes[1::2]):
        if end <= start:
            continue
        local = int(np.argmax(probs[start:end]))
        idx = start + local
        candidates.append(Candidate(float(idx / fps), float(probs[idx]), reason))
    return candidates


def nearest(candidate_times: list[float], target_time: float) -> tuple[float | None, float | None]:
    if not candidate_times:
        return None, None
    best = min(candidate_times, key=lambda t: abs(t - target_time))
    return best, abs(best - target_time)


def evaluate_predictions(
    method: str,
    annotations: list[dict],
    per_video: dict[str, dict],
    tolerances: list[float],
) -> dict:
    rows = []
    videos = []
    for annotation in annotations:
        video_file = annotation["video_file"]
        video = per_video[video_file]
        candidates = video["candidates"]
        candidate_times = [c["time"] for c in candidates]
        targets = targets_from_annotation(annotation, duration=video["duration"])
        video_rows = []
        for target in targets:
            best, err = nearest(candidate_times, target["time"])
            row = {
                **target,
                "nearest": best,
                "error": err,
                **{f"hit_{tol:g}s": err is not None and err <= tol for tol in tolerances},
            }
            rows.append(row)
            video_rows.append(row)
        videos.append(
            {
                "video_file": video_file,
                "duration": video["duration"],
                "candidate_count": len(candidates),
                "target_count": len(targets),
                "targets": video_rows,
                "candidates": candidates,
            }
        )

    summary = {
        "method": method,
        "video_count": len(videos),
        "target_count": len(rows),
        "candidate_count": sum(v["candidate_count"] for v in videos),
        "candidate_count_median": float(np.median([v["candidate_count"] for v in videos])) if videos else 0,
        "duration_minutes": sum(v["duration"] for v in videos) / 60.0,
        "recall": {},
        "recall_by_kind": {},
        "recall_by_source_type": {},
        "videos": videos,
    }
    for tol in tolerances:
        key = f"hit_{tol:g}s"
        summary["recall"][f"{tol:g}s"] = sum(1 for row in rows if row[key]) / len(rows) if rows else 0
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        summary["recall_by_kind"][kind] = {
            f"{tol:g}s": sum(1 for row in subset if row[f"hit_{tol:g}s"]) / len(subset)
            for tol in tolerances
        }
    for source_type in sorted({row["source_type"] for row in rows}):
        subset = [row for row in rows if row["source_type"] == source_type]
        summary["recall_by_source_type"][source_type] = {
            f"{tol:g}s": sum(1 for row in subset if row[f"hit_{tol:g}s"]) / len(subset)
            for tol in tolerances
        }
    return summary


def run_baseline(annotations: list[dict], args: argparse.Namespace) -> dict[str, dict]:
    per_video = {}
    for idx, annotation in enumerate(annotations, start=1):
        print(f"[baseline {idx}/{len(annotations)}] {annotation['video_file']}", file=sys.stderr)
        meta, features = baseline.load_or_compute_features(
            annotation,
            cache_dir=args.cache_dir,
            fps=args.baseline_fps,
            width=args.baseline_width,
            refresh=False,
        )
        candidates = baseline.detect_candidates(
            features,
            cut_score=args.baseline_cut_score,
            peak_quantile=args.baseline_peak_quantile,
            merge_sec=args.merge_sec,
            freeze_diff=args.baseline_freeze_diff,
            freeze_sec=args.baseline_freeze_sec,
        )
        per_video[annotation["video_file"]] = {
            "duration": meta.duration,
            "candidates": [c.__dict__ for c in candidates],
        }
    return per_video


def run_transnet(annotations: list[dict], args: argparse.Namespace) -> dict[str, dict]:
    sys.path.insert(0, str(args.transnet_dir))
    from main import TransNetV2Torch  # type: ignore

    model = TransNetV2Torch(str(args.transnet_dir / "transnetv2-pytorch-weights.pth"))
    per_video = {}
    for idx, annotation in enumerate(annotations, start=1):
        print(f"[transnet {idx}/{len(annotations)}] {annotation['video_file']}", file=sys.stderr)
        video_path = download_video(annotation, args.video_cache_dir, timeout=args.download_timeout)
        meta = ffprobe(video_path)
        frames = decode_rgb_frames(video_path, width=48, height=27, timeout=args.ffmpeg_timeout)
        single, many = model.predict_frames(frames)
        max_prob = np.maximum(single, many)
        candidates = []
        for threshold in args.transnet_thresholds:
            candidates.extend(candidates_from_probs(max_prob, meta["fps"], threshold, f"transnet_max@{threshold:g}"))
        per_video[annotation["video_file"]] = {
            "duration": meta["duration"],
            "candidates": [c.__dict__ for c in merge_candidates(candidates, merge_sec=args.merge_sec)],
            "raw": {
                "single_max": float(single.max()) if len(single) else 0,
                "many_max": float(many.max()) if len(many) else 0,
            },
        }
    return per_video


def run_omnishotcut(annotations: list[dict], args: argparse.Namespace) -> dict[str, dict]:
    sys.path.insert(0, str(args.omnishotcut_dir))
    import omnishotcut  # type: ignore

    model = omnishotcut.load("uva-cv-lab/OmniShotCut", filename="OmniShotCut_ckpt.pth")
    model_args = model._model_args
    per_video = {}
    for idx, annotation in enumerate(annotations, start=1):
        print(f"[omnishotcut {idx}/{len(annotations)}] {annotation['video_file']}", file=sys.stderr)
        video_path = download_video(annotation, args.video_cache_dir, timeout=args.download_timeout)
        meta = ffprobe(video_path)
        frames = decode_rgb_frames(
            video_path,
            width=int(model_args.process_width),
            height=int(model_args.process_height),
            timeout=args.ffmpeg_timeout,
        )
        ranges, intra_labels, inter_labels = model.inference(frames, mode="default", overlap=args.omnishotcut_overlap)
        candidates = []
        for frame_range, intra, inter in zip(ranges, intra_labels, inter_labels):
            end_frame = int(frame_range[1])
            time = end_frame / meta["fps"]
            if time >= meta["duration"] - 0.05:
                continue
            candidates.append(Candidate(time, 1.0, f"omnishotcut:{intra}/{inter}"))
        per_video[annotation["video_file"]] = {
            "duration": meta["duration"],
            "candidates": [c.__dict__ for c in merge_candidates(candidates, merge_sec=args.merge_sec)],
        }
    return per_video


def print_table(results: list[dict], tolerances: list[float]) -> None:
    header = ["method", "candidates", "median/video", "cand/min", *[f"recall@{tol:g}s" for tol in tolerances]]
    print("\t".join(header))
    for result in results:
        cand_min = result["candidate_count"] / result["duration_minutes"] if result["duration_minutes"] else 0
        row = [
            result["method"],
            str(result["candidate_count"]),
            f"{result['candidate_count_median']:.1f}",
            f"{cand_min:.1f}",
            *[f"{result['recall'][f'{tol:g}s']:.3f}" for tol in tolerances],
        ]
        print("\t".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="*", type=Path, default=sorted(ANNOTATION_DIR.glob("*.json")))
    parser.add_argument("--models", nargs="+", choices=["baseline", "transnet", "omnishotcut"], default=["baseline"])
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE_DIR / "public_model_benchmark.json")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--video-cache-dir", type=Path, default=DEFAULT_VIDEO_CACHE_DIR)
    parser.add_argument("--download-timeout", type=int, default=240)
    parser.add_argument("--ffmpeg-timeout", type=int, default=180)
    parser.add_argument("--transnet-dir", type=Path, default=DEFAULT_MODEL_DIR / "TransNetV2")
    parser.add_argument("--omnishotcut-dir", type=Path, default=DEFAULT_MODEL_DIR / "OmniShotCut")
    parser.add_argument("--tolerances", nargs="*", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--merge-sec", type=float, default=0.35)
    parser.add_argument("--baseline-fps", type=float, default=5.0)
    parser.add_argument("--baseline-width", type=int, default=160)
    parser.add_argument("--baseline-cut-score", type=float, default=0.25)
    parser.add_argument("--baseline-peak-quantile", type=float, default=0.80)
    parser.add_argument("--baseline-freeze-diff", type=float, default=0.011)
    parser.add_argument("--baseline-freeze-sec", type=float, default=0.8)
    parser.add_argument("--transnet-thresholds", nargs="*", type=float, default=[0.5])
    parser.add_argument("--omnishotcut-overlap", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    results = []
    if "baseline" in args.models:
        results.append(evaluate_predictions("baseline_high_recall", annotations, run_baseline(annotations, args), args.tolerances))
    if "transnet" in args.models:
        thresholds = ",".join(f"{t:g}" for t in args.transnet_thresholds)
        results.append(evaluate_predictions(f"transnet_max@{thresholds}", annotations, run_transnet(annotations, args), args.tolerances))
    if "omnishotcut" in args.models:
        results.append(evaluate_predictions("omnishotcut_default", annotations, run_omnishotcut(annotations, args), args.tolerances))

    print_table(results, args.tolerances)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
