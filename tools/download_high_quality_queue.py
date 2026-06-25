#!/usr/bin/env python3
"""Download high-quality Telegram videos from a replacement queue.

This wraps Telegram Web's authenticated downloader helper and records progress
after every row so long runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_QUEUE = Path("/Users/buddy/.openclaw/workspace/reports/2026-06-25_fpv_high_quality_replacement_queue.tsv")
DEFAULT_HELPER = Path("/Users/buddy/.openclaw/workspace/tmp/telegram_high_quality_test/telegram_web_download_post.js")
DEFAULT_OUT_DIR = Path("/Users/buddy/.openclaw/workspace/tmp/fpv_high_quality_downloads")
DEFAULT_STATUS = Path("/Users/buddy/.openclaw/workspace/reports/2026-06-25_fpv_high_quality_download_status.tsv")

FIELDS = [
    "timestamp_utc",
    "target_stem",
    "high_post",
    "new_stem",
    "status",
    "downloaded_path",
    "byte_size",
    "width",
    "height",
    "duration",
    "error",
]


def read_done(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    done: set[str] = set()
    with status_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status") == "ok" and row.get("high_post"):
                done.add(row["high_post"])
    return done


def append_status(status_path: Path, row: dict[str, str]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    exists = status_path.exists()
    with status_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def ffprobe(path: Path) -> tuple[str, str, str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    duration = data.get("format", {}).get("duration", "")
    return str(stream.get("width", "")), str(stream.get("height", "")), str(duration)


def parse_download_path(stdout: str) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("downloaded: "):
            return Path(line.removeprefix("downloaded: ").strip())
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--helper", type=Path, default=DEFAULT_HELPER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = read_done(args.status) if args.resume else set()
    count = 0

    with args.queue.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for queue_row in reader:
            high_post = queue_row["high_post"]
            if high_post in done:
                continue
            if args.limit and count >= args.limit:
                break

            base = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "target_stem": queue_row["target_stem"],
                "high_post": high_post,
                "new_stem": queue_row["new_stem"],
                "status": "",
                "downloaded_path": "",
                "byte_size": "",
                "width": "",
                "height": "",
                "duration": "",
                "error": "",
            }

            cmd = ["node", str(args.helper), "mmirleb", high_post, str(args.out_dir)]
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=240)
                downloaded = parse_download_path(proc.stdout)
                if downloaded is None or not downloaded.exists():
                    raise RuntimeError(f"download completed but path not found; stdout={proc.stdout!r}")
                width, height, duration = ffprobe(downloaded)
                base.update(
                    {
                        "status": "ok",
                        "downloaded_path": str(downloaded),
                        "byte_size": str(downloaded.stat().st_size),
                        "width": width,
                        "height": height,
                        "duration": duration,
                    }
                )
            except Exception as exc:
                base.update({"status": "error", "error": str(exc).replace("\n", " ")[:1000]})

            append_status(args.status, base)
            print(f"{base['status']}\t{high_post}\t{base['downloaded_path']}", flush=True)
            count += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
