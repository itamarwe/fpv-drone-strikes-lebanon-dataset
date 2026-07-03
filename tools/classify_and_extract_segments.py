#!/usr/bin/env python3
"""Classify transition candidates and extract flight-like segments.

This script consumes candidate boundaries from `benchmark_transition_models.py`
and adds domain-specific evidence:

- blur valley around the candidate
- local motion/freeze/black-frame state
- ORB/RANSAC geometric continuity across the candidate

It then keeps likely edit boundaries, suppresses active-flight perspective
changes, and exports intervals that look like active flight.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = ROOT / "annotations"
DEFAULT_CANDIDATE_JSON = Path("/tmp/fpv-flight-boundaries/benchmark_transnet_02_continuity_raw.json")
DEFAULT_VIDEO_CACHE_DIR = Path("/tmp/fpv-model-benchmark/videos")
DEFAULT_OUTPUT_DIR = Path("/tmp/fpv-flight-boundaries/extracted_segments")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def load_annotations(paths: Iterable[Path]) -> list[dict]:
    annotations = []
    for path in paths:
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        annotations.append(data)
    return annotations


def load_candidate_videos(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    videos = {}
    for result in data.get("results", []):
        for video in result.get("videos", []):
            videos[video["video_file"]] = video
    return videos


def ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    fps_text = stream["avg_frame_rate"]
    if "/" in fps_text:
        num, den = fps_text.split("/", 1)
        fps = float(num) / float(den)
    else:
        fps = float(fps_text)
    return {
        "duration": float(data["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
    }


def resize_gray(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def frame_at(path: Path, time_sec: float, width: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return resize_gray(frame, width)


def read_window(path: Path, center: float, radius: float, width: int) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    start = max(0.0, center - radius)
    end = center + radius
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = []
    times = []
    while True:
        pos_before = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos_before > end:
            break
        ok, frame = cap.read()
        if not ok:
            break
        pos_after = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frames.append(resize_gray(frame, width))
        times.append(pos_after - center)
    cap.release()
    if not frames:
        return np.empty((0, 1, 1), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(frames), np.array(times, dtype=np.float32)


def sharpness(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(lap.var())


def candidate_temporal_features(path: Path, time_sec: float, width: int, radius: float) -> dict:
    frames, rel_times = read_window(path, time_sec, radius=radius, width=width)
    if len(frames) < 4:
        return {
            "blur_valley": 0.0,
            "blur_center_ratio": 1.0,
            "blur_low_fraction": 0.0,
            "motion_before": 0.0,
            "motion_after": 0.0,
            "freeze_before_fraction": 0.0,
            "freeze_after_fraction": 0.0,
            "black_before_fraction": 0.0,
            "black_after_fraction": 0.0,
        }

    sharp = np.array([sharpness(frame) for frame in frames], dtype=np.float32)
    outer = (np.abs(rel_times) >= 0.55) & (np.abs(rel_times) <= radius)
    center = np.abs(rel_times) <= 0.35
    near = np.abs(rel_times) <= 0.60
    outer_med = float(np.median(sharp[outer])) if outer.any() else float(np.median(sharp))
    center_min = float(np.min(sharp[center])) if center.any() else float(np.min(sharp))
    blur_center_ratio = center_min / max(outer_med, 1e-9)
    blur_valley = 1.0 - blur_center_ratio
    blur_low_fraction = float(np.mean(sharp[near] < outer_med * 0.55)) if near.any() else 0.0

    diffs = np.zeros(len(frames), dtype=np.float32)
    diffs[1:] = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2))
    before = (rel_times >= -0.8) & (rel_times <= -0.2)
    after = (rel_times >= 0.2) & (rel_times <= 0.8)
    before_diffs = diffs[before]
    after_diffs = diffs[after]
    before_frames = frames[before]
    after_frames = frames[after]

    return {
        "blur_valley": float(blur_valley),
        "blur_center_ratio": float(blur_center_ratio),
        "blur_low_fraction": blur_low_fraction,
        "motion_before": float(np.median(before_diffs)) if len(before_diffs) else 0.0,
        "motion_after": float(np.median(after_diffs)) if len(after_diffs) else 0.0,
        "freeze_before_fraction": float(np.mean(before_diffs < 0.01)) if len(before_diffs) else 0.0,
        "freeze_after_fraction": float(np.mean(after_diffs < 0.01)) if len(after_diffs) else 0.0,
        "black_before_fraction": float(np.mean(before_frames < 0.06)) if len(before_frames) else 0.0,
        "black_after_fraction": float(np.mean(after_frames < 0.06)) if len(after_frames) else 0.0,
    }


def geometry_features(path: Path, time_sec: float, dt: float, width: int) -> dict:
    before = frame_at(path, time_sec - dt, width)
    after = frame_at(path, time_sec + dt, width)
    if before is None or after is None:
        return {"geom_good_matches": 0, "geom_inliers": 0, "geom_inlier_ratio": 0.0, "geom_warp_diff": None}

    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=7, edgeThreshold=15)
    kp1, des1 = orb.detectAndCompute((before * 255).astype(np.uint8), None)
    kp2, des2 = orb.detectAndCompute((after * 255).astype(np.uint8), None)
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return {"geom_good_matches": 0, "geom_inliers": 0, "geom_inlier_ratio": 0.0, "geom_warp_diff": None}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)
    if len(good) < 8:
        return {
            "geom_good_matches": len(good),
            "geom_inliers": 0,
            "geom_inlier_ratio": 0.0,
            "geom_warp_diff": None,
        }

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if homography is None or mask is None:
        return {
            "geom_good_matches": len(good),
            "geom_inliers": 0,
            "geom_inlier_ratio": 0.0,
            "geom_warp_diff": None,
        }

    inliers = int(mask.ravel().sum())
    warped = cv2.warpPerspective(before, homography, (after.shape[1], after.shape[0]))
    valid = cv2.warpPerspective(np.ones_like(before, dtype=np.uint8) * 255, homography, (after.shape[1], after.shape[0])) > 0
    warp_diff = None
    if valid.mean() > 0.2:
        warp_diff = float(np.mean(np.abs(warped[valid].astype(np.float32) - after[valid].astype(np.float32))))

    return {
        "geom_good_matches": len(good),
        "geom_inliers": inliers,
        "geom_inlier_ratio": float(inliers / len(good)),
        "geom_warp_diff": warp_diff,
    }


def classify_candidate(candidate: dict, features: dict, args: argparse.Namespace) -> tuple[str, bool]:
    blur = (
        features["blur_valley"] >= args.blur_valley
        and features["blur_center_ratio"] <= args.blur_center_ratio
        and features["blur_low_fraction"] >= args.blur_low_fraction
    )
    freeze_or_black = (
        max(features["freeze_before_fraction"], features["freeze_after_fraction"]) >= args.freeze_fraction
        or max(features["black_before_fraction"], features["black_after_fraction"]) >= args.black_fraction
    )
    active_both_sides = (
        features["motion_before"] >= args.active_motion
        and features["motion_after"] >= args.active_motion
        and features["freeze_before_fraction"] <= args.active_max_freeze
        and features["freeze_after_fraction"] <= args.active_max_freeze
    )
    strong_geometry = (
        features["geom_inliers"] >= args.geom_min_inliers
        and features["geom_inlier_ratio"] >= args.geom_min_inlier_ratio
        and (features["geom_warp_diff"] is None or features["geom_warp_diff"] <= args.geom_max_warp_diff)
    )

    if blur:
        return "keep_blur_transition", True
    if freeze_or_black:
        return "keep_freeze_or_black_transition", True
    if strong_geometry and active_both_sides:
        return "suppress_active_geometric_continuity", False
    if candidate.get("score", 0.0) >= args.strong_score:
        return "keep_strong_transnet_score", True
    if (
        candidate.get("boundary_diff") is not None
        and candidate.get("jump_ratio") is not None
        and candidate["boundary_diff"] >= args.min_boundary_diff
        and candidate["jump_ratio"] >= args.min_jump_ratio
    ):
        return "keep_discontinuous_jump", True
    if candidate.get("appearance_jump") is not None and candidate["appearance_jump"] >= args.min_appearance_jump:
        return "keep_appearance_jump", True
    return "suppress_smooth_low_score", False


def interval_features(path: Path, start: float, end: float, width: int) -> dict:
    radius = max(0.05, (end - start) / 2.0)
    center = (start + end) / 2.0
    frames, _ = read_window(path, center, radius=radius, width=width)
    if len(frames) < 4:
        return {
            "motion_median": 0.0,
            "freeze_fraction": 1.0,
            "black_fraction": 1.0,
            "sharpness_median": 0.0,
        }
    diffs = np.zeros(len(frames), dtype=np.float32)
    diffs[1:] = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2))
    sharp = np.array([sharpness(frame) for frame in frames], dtype=np.float32)
    return {
        "motion_median": float(np.median(diffs[1:])) if len(diffs) > 1 else 0.0,
        "freeze_fraction": float(np.mean(diffs[1:] < 0.01)) if len(diffs) > 1 else 1.0,
        "black_fraction": float(np.mean(frames < 0.06)),
        "sharpness_median": float(np.median(sharp)),
    }


def is_flight_like(features: dict, duration: float, args: argparse.Namespace) -> bool:
    return (
        duration >= args.min_segment_sec
        and features["motion_median"] >= args.interval_min_motion
        and features["freeze_fraction"] <= args.interval_max_freeze
        and features["black_fraction"] <= args.interval_max_black
    )


def replay_like_blur_boundary(boundary: dict | None, args: argparse.Namespace) -> bool:
    if boundary is None:
        return False
    return (
        boundary.get("decision") == "keep_blur_transition"
        and max(boundary.get("black_before_fraction", 0.0), boundary.get("black_after_fraction", 0.0))
        >= args.replay_black_fraction
    )


def merge_close_times(times: list[float], merge_sec: float) -> list[float]:
    if not times:
        return []
    times = sorted(times)
    merged = [times[0]]
    for time in times[1:]:
        if time - merged[-1] <= merge_sec:
            merged[-1] = (merged[-1] + time) / 2.0
        else:
            merged.append(time)
    return merged


def extract_clip(input_path: Path, output_path: Path, start: float, end: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end - start)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def process_video(annotation: dict, candidate_video: dict, video_path: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    meta = ffprobe(video_path)
    classified = []
    for candidate in sorted(candidate_video.get("candidates", []), key=lambda item: item["time"]):
        time_sec = float(candidate["time"])
        if time_sec <= args.edge_guard_sec or time_sec >= meta["duration"] - args.edge_guard_sec:
            continue
        temporal = candidate_temporal_features(video_path, time_sec, width=args.feature_width, radius=args.feature_radius)
        geom = geometry_features(video_path, time_sec, dt=args.geom_dt, width=args.geom_width)
        features = {**temporal, **geom}
        decision, keep = classify_candidate(candidate, features, args)
        classified.append(
            {
                **candidate,
                **features,
                "decision": decision,
                "keep_boundary": keep,
            }
        )

    cut_times = merge_close_times(
        [row["time"] for row in classified if row["keep_boundary"]],
        merge_sec=args.merge_sec,
    )
    kept_boundaries = [row for row in classified if row["keep_boundary"]]

    def nearest_kept_boundary(time_sec: float) -> dict | None:
        if not kept_boundaries:
            return None
        nearest = min(kept_boundaries, key=lambda row: abs(row["time"] - time_sec))
        if abs(nearest["time"] - time_sec) <= args.merge_sec:
            return nearest
        return None

    cut_times = [0.0, *cut_times, meta["duration"]]

    intervals = []
    video_out_dir = output_dir / slug(annotation["video_file"]).removesuffix(".mp4")
    for idx, (start, end) in enumerate(zip(cut_times[:-1], cut_times[1:]), start=1):
        duration = end - start
        features = interval_features(video_path, start, end, width=args.interval_width)
        previous_boundary = nearest_kept_boundary(start)
        next_boundary = nearest_kept_boundary(end)
        replay_like = replay_like_blur_boundary(previous_boundary, args)
        flight_like = is_flight_like(features, duration, args) and not replay_like
        row = {
            "index": idx,
            "start": start,
            "end": end,
            "duration": duration,
            "flight_like": flight_like,
            "reject_reason": "post_blur_black_replay_like" if replay_like else None,
            "previous_boundary": previous_boundary,
            "next_boundary": next_boundary,
            "features": features,
        }
        if flight_like and args.extract:
            output_path = video_out_dir / f"segment_{idx:02d}_{start:.3f}_{end:.3f}.mp4"
            extract_clip(video_path, output_path, start, end)
            row["output_path"] = str(output_path)
        intervals.append(row)

    return {
        "video_file": annotation["video_file"],
        "video_path": str(video_path),
        "duration": meta["duration"],
        "candidate_count": len(candidate_video.get("candidates", [])),
        "classified_candidate_count": len(classified),
        "kept_boundary_count": sum(1 for row in classified if row["keep_boundary"]),
        "suppressed_boundary_count": sum(1 for row in classified if not row["keep_boundary"]),
        "classified_candidates": classified,
        "intervals": intervals,
        "extracted_count": sum(1 for row in intervals if row.get("output_path")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="*", type=Path, default=[])
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE_JSON)
    parser.add_argument("--video-cache-dir", type=Path, default=DEFAULT_VIDEO_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_OUTPUT_DIR / "classification_report.json")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--feature-width", type=int, default=320)
    parser.add_argument("--feature-radius", type=float, default=1.2)
    parser.add_argument("--interval-width", type=int, default=240)
    parser.add_argument("--geom-width", type=int, default=640)
    parser.add_argument("--geom-dt", type=float, default=0.16)
    parser.add_argument("--edge-guard-sec", type=float, default=0.05)
    parser.add_argument("--merge-sec", type=float, default=0.35)
    parser.add_argument("--blur-valley", type=float, default=0.65)
    parser.add_argument("--blur-center-ratio", type=float, default=0.35)
    parser.add_argument("--blur-low-fraction", type=float, default=0.25)
    parser.add_argument("--freeze-fraction", type=float, default=0.35)
    parser.add_argument("--black-fraction", type=float, default=0.35)
    parser.add_argument("--active-motion", type=float, default=0.02)
    parser.add_argument("--active-max-freeze", type=float, default=0.25)
    parser.add_argument("--geom-min-inliers", type=int, default=80)
    parser.add_argument("--geom-min-inlier-ratio", type=float, default=0.60)
    parser.add_argument("--geom-max-warp-diff", type=float, default=0.18)
    parser.add_argument("--strong-score", type=float, default=0.245)
    parser.add_argument("--min-boundary-diff", type=float, default=0.04)
    parser.add_argument("--min-jump-ratio", type=float, default=1.1)
    parser.add_argument("--min-appearance-jump", type=float, default=0.18)
    parser.add_argument("--min-segment-sec", type=float, default=1.0)
    parser.add_argument("--interval-min-motion", type=float, default=0.012)
    parser.add_argument("--interval-max-freeze", type=float, default=0.55)
    parser.add_argument("--interval-max-black", type=float, default=0.45)
    parser.add_argument("--replay-black-fraction", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = load_annotations(args.annotations or sorted(ANNOTATION_DIR.glob("*.json")))
    candidate_videos = load_candidate_videos(args.candidate_json)
    reports = []
    for annotation in annotations:
        video_file = annotation["video_file"]
        candidate_video = candidate_videos.get(video_file)
        if candidate_video is None:
            print(f"warning: no candidates for {video_file}", file=sys.stderr)
            continue
        video_path = args.video_cache_dir / video_file
        if not video_path.exists():
            print(f"warning: no cached video for {video_file}: {video_path}", file=sys.stderr)
            continue
        print(f"[extract] {video_file}", file=sys.stderr)
        reports.append(process_video(annotation, candidate_video, video_path, args.output_dir, args))

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps({"videos": reports}, indent=2))

    print("video_file\tkept\tsuppressed\textracted")
    for report in reports:
        print(
            f"{report['video_file']}\t"
            f"{report['kept_boundary_count']}\t"
            f"{report['suppressed_boundary_count']}\t"
            f"{report['extracted_count']}"
        )
        for interval in report["intervals"]:
            if interval.get("output_path"):
                print(f"  {interval['start']:.3f}-{interval['end']:.3f}\t{interval['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
