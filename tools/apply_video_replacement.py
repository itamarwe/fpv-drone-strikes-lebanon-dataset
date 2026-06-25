#!/usr/bin/env python3
"""Apply one downloaded high-quality video replacement to repo metadata."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CSV = ROOT / "geo" / "fpv_drone_map_records.csv"
MANIFEST = ROOT / "2026-06-21_fpv_renamed_from_first_frames_manifest.tsv"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def replace_text_file(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"{old!r} not found in {path}")
    path.write_text(text.replace(old, new))


def update_csv(old_stem: str, new_stem: str) -> None:
    rows: list[dict[str, str]] = []
    with CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit(f"{CSV} has no header")
        for row in reader:
            if old_stem in row["video_url"]:
                row["video_url"] = row["video_url"].replace(old_stem, new_stem)
                row["thumbnail_url"] = row["thumbnail_url"].replace(old_stem, new_stem)
            rows.append(row)
    with CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_manifest(old_stem: str, new_stem: str, high_post: str) -> None:
    rows: list[dict[str, str]] = []
    with MANIFEST.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit(f"{MANIFEST} has no header")
        for row in reader:
            if row["target_stem"] == old_stem:
                row["current_stem"] = f"telegram_mmirleb_{high_post}"
                row["target_stem"] = new_stem
                row["confidence"] = "high"
                row["notes"] = (
                    f"official @mmirleb high-quality Telegram Web download post {high_post}, "
                    f"replacing lower-quality public/direct post; {row['notes']}"
                )
            rows.append(row)
    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloaded-video", required=True, type=Path)
    parser.add_argument("--old-stem", required=True)
    parser.add_argument("--new-stem", required=True)
    parser.add_argument("--high-post", required=True)
    parser.add_argument("--output-dir", default=Path("/Users/buddy/.openclaw/workspace/fpv attacks renamed"), type=Path)
    args = parser.parse_args()

    source = args.downloaded_video
    if not source.exists():
        raise SystemExit(f"Downloaded video does not exist: {source}")

    video_path = args.output_dir / f"{args.new_stem}.mp4"
    thumb_path = args.output_dir / f"{args.new_stem}.jpg"
    shutil.copy2(source, video_path)
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(thumb_path),
    ])

    replace_text_file(README, args.old_stem, args.new_stem)
    update_csv(args.old_stem, args.new_stem)
    update_manifest(args.old_stem, args.new_stem, args.high_post)

    print(video_path)
    print(thumb_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
