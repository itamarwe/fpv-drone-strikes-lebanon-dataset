#!/usr/bin/env python3
"""Create a reconstruction frame subset with a preserved frame manifest."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Source reconstruction directory with frames.csv.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output reconstruction directory to create.")
    parser.add_argument("--segment-id", help="Keep only this segment id.")
    parser.add_argument("--attack-only", action="store_true", help="Keep only rows marked is_attack=true.")
    parser.add_argument("--target-frames", type=int, default=0, help="Evenly sample to this many frames.")
    parser.add_argument("--copy", action="store_true", help="Copy image files instead of symlinking them.")
    parser.add_argument("--refresh", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def evenly_sample(rows: list[dict[str, str]], target: int) -> list[dict[str, str]]:
    if target <= 0 or len(rows) <= target:
        return rows
    indexes = np.linspace(0, len(rows) - 1, target).round().astype(int)
    return [rows[int(i)] for i in indexes]


def main() -> int:
    args = parse_args()
    source_frames = args.source_dir / "frames"
    rows_path = args.source_dir / "frames.csv"
    if not source_frames.exists():
        raise FileNotFoundError(source_frames)
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    if args.out_dir.exists() and args.refresh:
        shutil.rmtree(args.out_dir)
    frames_dir = args.out_dir / "frames"
    if frames_dir.exists() and any(frames_dir.glob("*.jpg")) and not args.refresh:
        raise SystemExit(f"Output already exists; pass --refresh: {args.out_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(rows_path)
    if args.segment_id:
        rows = [row for row in rows if row["segment_id"] == args.segment_id]
    if args.attack_only:
        rows = [row for row in rows if row["is_attack"].lower() == "true"]
    rows = evenly_sample(rows, args.target_frames)
    if not rows:
        raise SystemExit("No rows selected.")

    out_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        src = source_frames / row["file"]
        dst = frames_dir / f"f_{idx:06d}.jpg"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)
        out_rows.append({**row, "frame_index": str(idx), "file": dst.name})

    with (args.out_dir / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    source_meta = {}
    metadata_path = args.source_dir / "metadata.json"
    if metadata_path.exists():
        source_meta = json.loads(metadata_path.read_text())
    metadata = {
        "subset_id": args.out_dir.name,
        "source_dir": str(args.source_dir),
        "source_video_file": source_meta.get("video_file", out_rows[0].get("video_file", "")),
        "source_video_id": source_meta.get("video_id", Path(out_rows[0].get("video_file", "")).stem),
        "source_sample_fps": source_meta.get("sample_fps", ""),
        "source_width": source_meta.get("width", ""),
        "segment_id": args.segment_id or "",
        "attack_only": bool(args.attack_only),
        "target_frames": args.target_frames,
        "frames": len(out_rows),
        "start_video_time_s": float(out_rows[0]["video_time_s"]),
        "end_video_time_s": float(out_rows[-1]["video_time_s"]),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[subset] selected {len(out_rows)} frames -> {args.out_dir}")
    print(f"[subset] time {metadata['start_video_time_s']:.3f}s to {metadata['end_video_time_s']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
