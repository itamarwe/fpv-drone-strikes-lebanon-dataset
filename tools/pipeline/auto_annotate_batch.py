#!/usr/bin/env python3
"""Auto-annotate un-annotated FPV videos in resumable, checkpointed chunks.

For each chunk of videos this driver runs the full pipeline:

  1. write stub annotation files (video_file + video_url, empty segments)
  2. TransNetV2 transition detection      -> candidate json
     (tools/pipeline/benchmark_transition_models.py)
  3. classify flight-like segments        -> report json
     (tools/pipeline/classify_and_extract_segments.py, no clip extraction)
  4. convert report to annotator markers  -> annotations/<slug>_annotations.json
     (tools/pipeline/segments_report_to_annotations.py, tagged auto_generated)

Videos that already have an annotation file are skipped up front, so a run is
fully resumable: re-running only processes what is still missing.

The video list + metadata come from the annotator app (tools/annotator.html),
which registers each video with a description/date/town and a cloudfront
video_url. Videos that already have a hand or auto annotation are left untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
ANNOT_DIR = ROOT / "annotations"
ANNOTATOR_HTML = ROOT / "tools" / "annotator.html"
DEFAULT_VIDEO_CACHE = Path("/tmp/fpv-model-benchmark/videos")
DEFAULT_WORK_DIR = Path("/tmp/fpv-auto-annotate")

VIDEO_ENTRY_RE = re.compile(
    r"\{date:'([^']*)',description:'([^']*)',town:'([^']*)',"
    r"thumbnail_url:'[^']*',video_url:'(https://[^']+/videos/([^']+\.mp4))'\}"
)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")


def load_video_meta() -> dict[str, dict]:
    html = ANNOTATOR_HTML.read_text()
    meta: dict[str, dict] = {}
    for date, desc, town, url, video_file in VIDEO_ENTRY_RE.findall(html):
        meta[video_file] = {"date": date, "description": desc, "town": town, "video_url": url}
    return meta


def annotated_video_files() -> set[str]:
    files = set()
    for path in ANNOT_DIR.glob("*_annotations.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        video_file = data.get("video_file")
        if video_file:
            files.add(video_file)
    return files


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def process_chunk(chunk: list[str], meta: dict[str, dict], chunk_idx: int, args: argparse.Namespace) -> None:
    stub_dir = args.work_dir / "stubs" / f"chunk_{chunk_idx:03d}"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub_paths = []
    for video_file in chunk:
        m = meta.get(video_file, {})
        stub = {
            "video_file": video_file,
            "video_url": m.get("video_url", ""),
            "description": m.get("description", ""),
            "date": m.get("date", ""),
            "town": m.get("town", ""),
            "segments": [],
        }
        p = stub_dir / f"{slugify(video_file[:-4])}_annotations.json"
        p.write_text(json.dumps(stub, indent=2))
        stub_paths.append(p)

    cand_json = args.work_dir / "candidates" / f"chunk_{chunk_idx:03d}.json"
    cand_json.parent.mkdir(parents=True, exist_ok=True)
    run([
        args.python, str(TOOLS / "benchmark_transition_models.py"),
        "--models", "transnet",
        "--transnet-thresholds", f"{args.transnet_threshold:g}",
        "--annotations", *[str(p) for p in stub_paths],
        "--video-cache-dir", str(args.video_cache_dir),
        "--output", str(cand_json),
    ])

    report_json = args.work_dir / "reports" / f"chunk_{chunk_idx:03d}.json"
    report_json.parent.mkdir(parents=True, exist_ok=True)
    run([
        args.python, str(TOOLS / "classify_and_extract_segments.py"),
        "--annotations", *[str(p) for p in stub_paths],
        "--candidate-json", str(cand_json),
        "--video-cache-dir", str(args.video_cache_dir),
        "--output-dir", str(args.work_dir / "segments"),
        "--report-json", str(report_json),
    ])

    # Persist a video-meta map so the converter can fill url/description/etc.
    meta_path = args.work_dir / "video_meta.json"
    meta_path.write_text(json.dumps(meta, indent=1))
    run([
        args.python, str(TOOLS / "segments_report_to_annotations.py"),
        "--report-json", str(report_json),
        "--video-meta", str(meta_path),
        "--output-dir", str(ANNOT_DIR),
    ])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="*", help="specific video_file names; default = all un-annotated")
    ap.add_argument("--limit", type=int, help="cap number of videos processed")
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--transnet-threshold", type=float, default=0.2)
    ap.add_argument("--video-cache-dir", type=Path, default=DEFAULT_VIDEO_CACHE)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--python", default=sys.executable, help="interpreter for the sub-tools (needs torch + cv2)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    meta = load_video_meta()

    todo = list(args.videos) if args.videos else list(meta.keys())
    done = annotated_video_files()
    todo = [v for v in todo if v not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"videos to process: {len(todo)}", file=sys.stderr)
    if not todo:
        return 0

    chunks = [todo[i : i + args.chunk_size] for i in range(0, len(todo), args.chunk_size)]
    for idx, chunk in enumerate(chunks):
        print(f"\n===== chunk {idx + 1}/{len(chunks)} ({len(chunk)} videos) =====", file=sys.stderr)
        try:
            process_chunk(chunk, meta, idx, args)
        except subprocess.CalledProcessError as exc:
            # Leave failed videos un-annotated; they are retried on the next run.
            print(f"chunk {idx} failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
