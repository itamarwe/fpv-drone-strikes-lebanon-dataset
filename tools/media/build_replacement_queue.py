#!/usr/bin/env python3
"""Build the Telegram high-quality replacement queue from the manifest."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "2026-06-21_fpv_renamed_from_first_frames_manifest.tsv"
README = ROOT / "README.md"
CSV = ROOT / "geo" / "fpv_drone_map_records.csv"


def read_readme_urls() -> dict[str, tuple[str, str]]:
    urls: dict[str, tuple[str, str]] = {}
    for line in README.read_text().splitlines():
        if "d2fioemadmrru3.cloudfront.net/videos/" not in line:
            continue
        video_match = re.search(r"https://d2fioemadmrru3\.cloudfront\.net/videos/([^)\"]+)", line)
        thumb_match = re.search(r"https://d2fioemadmrru3\.cloudfront\.net/thumbnails/([^\"\s]+)", line)
        if not video_match:
            continue
        stem = Path(video_match.group(1)).stem
        urls[stem] = (
            f"https://d2fioemadmrru3.cloudfront.net/videos/{video_match.group(1)}",
            f"https://d2fioemadmrru3.cloudfront.net/thumbnails/{thumb_match.group(1)}" if thumb_match else "",
        )
    return urls


def main() -> int:
    readme_urls = read_readme_urls()
    rows = list(csv.DictReader(MANIFEST.open(), delimiter="\t"))

    print("\t".join([
        "target_stem",
        "current_post",
        "high_post",
        "current_video_url",
        "current_thumbnail_url",
        "new_stem",
        "confidence",
        "notes",
    ]))
    for row in rows:
        current = row["current_stem"]
        direct = re.search(r"direct_(\d+)", current)
        candidate = re.search(r"matched high-quality candidate post (\d+)", row["notes"])
        if not direct or not candidate:
            continue
        current_post = direct.group(1)
        high_post = candidate.group(1)
        if current_post == high_post:
            continue
        target = row["target_stem"]
        current_video, current_thumb = readme_urls.get(target, ("", ""))
        new_stem = re.sub(r"_mmirleb_\d+$", f"_mmirleb_{high_post}", target)
        if new_stem == target:
            new_stem = f"{target}_mmirleb_{high_post}"
        print("\t".join([
            target,
            current_post,
            high_post,
            current_video,
            current_thumb,
            new_stem,
            row["confidence"],
            row["notes"].replace("\t", " ").replace("\n", " "),
        ]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
