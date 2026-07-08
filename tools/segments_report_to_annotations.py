#!/usr/bin/env python3
"""Convert a classify_and_extract_segments report into annotator marker JSON.

`classify_and_extract_segments.py` emits a report with, per video, a list of
`intervals` split at kept transition boundaries and flagged `flight_like`. The
annotator web app (`tools/annotator.html`) instead speaks a marker timeline:
each entry is a `{time, type, ...}` where a flight segment runs from a
`flight_start`/`new_flight_start` marker until the next non-flight marker.

This script turns each interval boundary into a marker so the auto-detected
flight segments show up in the annotate view for human review. Output files are
written next to the hand annotations as `<slug>_annotations.json` and carry
`"auto_generated": true` (plus a per-marker `"source": "auto"`) so they are
clearly distinguishable from manual work.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = ROOT / "annotations"

GENERATOR = "classify_and_extract_segments+segments_report_to_annotations"


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes}:{rest:04.1f}"


def marker_type_for_interval(interval: dict, is_first: bool) -> str:
    """Best-guess annotator marker type for the start of an interval.

    The human reviewer refines these; the goal is a sensible default.
    """
    if interval.get("flight_like"):
        return "flight_start"
    # A non-flight opening interval is almost always the title/banner card.
    if is_first:
        return "banner_start"
    if interval.get("reject_reason") == "post_blur_black_replay_like":
        return "replay_start"
    features = interval.get("features") or {}
    freeze = float(features.get("freeze_fraction", 0.0))
    black = float(features.get("black_fraction", 0.0))
    if freeze >= 0.35 or black >= 0.35:
        return "pause_start"
    return "other"


def markers_for_video(video: dict) -> list[dict]:
    intervals = sorted(video.get("intervals", []), key=lambda row: row["start"])
    markers = []
    for position, interval in enumerate(intervals):
        start = float(interval["start"])
        mtype = marker_type_for_interval(interval, is_first=position == 0)
        boundary = interval.get("previous_boundary") or {}
        decision = boundary.get("decision")
        comment_bits = []
        if decision:
            comment_bits.append(f"cut:{decision}")
        if interval.get("reject_reason"):
            comment_bits.append(interval["reject_reason"])
        markers.append(
            {
                "time": round(start, 3),
                "time_formatted": fmt_time(start),
                "type": mtype,
                "source": "auto",
                "flight_like": bool(interval.get("flight_like")),
                "comment": "; ".join(comment_bits),
            }
        )
    return markers


def build_annotation(video: dict, meta: dict | None) -> dict:
    video_file = video["video_file"]
    meta = meta or {}
    return {
        "video_file": video_file,
        "video_url": meta.get("video_url", ""),
        "description": meta.get("description", ""),
        "date": meta.get("date", ""),
        "town": meta.get("town", ""),
        "auto_generated": True,
        "generator": GENERATOR,
        "segments": markers_for_video(video),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--video-meta", type=Path, help="file -> {video_url,description,date,town} map")
    parser.add_argument("--output-dir", type=Path, default=ANNOTATION_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing annotation files (default: skip any existing file, auto or manual)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report_json.read_text())
    meta_map = json.loads(args.video_meta.read_text()) if args.video_meta and args.video_meta.exists() else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    for video in report.get("videos", []):
        video_file = video["video_file"]
        stem = video_file[:-4] if video_file.endswith(".mp4") else video_file
        out_path = args.output_dir / f"{slugify(stem)}_annotations.json"
        if out_path.exists() and not args.overwrite:
            print(f"skip (exists): {out_path.name}")
            skipped += 1
            continue
        annotation = build_annotation(video, meta_map.get(video_file))
        out_path.write_text(json.dumps(annotation, indent=2))
        flights = sum(1 for s in annotation["segments"] if s["flight_like"])
        print(f"wrote {out_path.name}\t{len(annotation['segments'])} markers, {flights} flight")
        written += 1

    print(f"\ntotal: wrote {written}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
