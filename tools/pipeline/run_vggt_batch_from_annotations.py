#!/usr/bin/env python3
"""Sequentially run VGGT scene jobs from annotation JSON files."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = ROOT / "annotations"
FLIGHT_TYPES = {"flight_start", "new_flight_start"}


def slugify(value: str) -> str:
    value = Path(value).stem if value else "video"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return value[:120] or "video"


def build_flight_intervals(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    markers = sorted(annotation.get("segments", []), key=lambda row: float(row["time"]))
    flight_marker_indexes = [idx for idx, row in enumerate(markers) if row.get("type") in FLIGHT_TYPES]
    intervals: list[dict[str, Any]] = []
    for local_index, marker_index in enumerate(flight_marker_indexes, start=1):
        marker = markers[marker_index]
        end_marker: dict[str, Any] | None = None
        for candidate in markers[marker_index + 1 :]:
            if candidate.get("type") not in FLIGHT_TYPES:
                end_marker = candidate
                break
        if end_marker is None:
            continue
        start_s = float(marker["time"])
        end_s = float(end_marker["time"])
        if end_s <= start_s:
            continue
        intervals.append(
            {
                "segment_id": f"seg{local_index:02d}",
                "label": f"seg{local_index:02d}",
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(end_s - start_s, 3),
                "start_type": marker.get("type", ""),
                "end_type": end_marker.get("type", "video_end"),
                "is_attack": local_index == len(flight_marker_indexes),
            }
        )
    return intervals


def select_attack_pause_chain(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not intervals:
        return []
    selected_indexes = [len(intervals) - 1]
    index = len(intervals) - 2
    while index >= 0 and intervals[index].get("end_type") == "pause_start":
        selected_indexes.insert(0, index)
        index -= 1
    return [intervals[index] for index in selected_indexes]


def trim_to_tail_seconds(selected: list[dict[str, Any]], tail_seconds: float) -> list[dict[str, Any]]:
    """Keep only the last `tail_seconds` of the concatenated flight sequence.

    Walks back from the final (attack) segment; the earliest included segment is
    shortened (start_s moved forward) so the total is exactly tail_seconds. Segment
    structure/ids are preserved for whatever survives the window.
    """
    if tail_seconds <= 0:
        return selected
    total = sum(float(s["duration_s"]) for s in selected)
    if total <= tail_seconds:
        return selected
    remaining = tail_seconds
    out: list[dict[str, Any]] = []
    for seg in reversed(selected):
        dur = float(seg["duration_s"])
        if dur <= remaining:
            out.insert(0, seg)
            remaining -= dur
        else:
            trimmed = dict(seg)
            trimmed["start_s"] = round(float(seg["end_s"]) - remaining, 3)
            trimmed["duration_s"] = round(remaining, 3)
            out.insert(0, trimmed)
            remaining = 0.0
            break
    return out


def post_json(base_url: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url, path),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(base_url: str, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(urllib.parse.urljoin(base_url, path), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def request_body(annotation: dict[str, Any], selected: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "video_file": annotation["video_file"],
        "video_url": annotation["video_url"],
        "description": annotation.get("description", ""),
        "date": annotation.get("date", ""),
        "town": annotation.get("town", ""),
        "annotations": [
            {
                "time": round(float(row["time"]), 3),
                "type": row.get("type", ""),
                "comment": row.get("comment", ""),
            }
            for row in annotation.get("segments", [])
        ],
        "segments": selected,
        "sample_fps": args.fps,
        "target_frames": 0,
        "default_scale_m_per_unit": args.scale,
        "crop_preset": args.crop_preset,
        "width": args.width,
        "focal_px": args.focal_px,
        "max_vggt_frames": args.max_vggt_frames,
        "frame_window": args.frame_window,
        "render_camera_views": not args.skip_camera_views,
        "refresh_vggt": not args.no_refresh_vggt,
        "vggt_space": args.vggt_space,
        "vggt_backend": args.vggt_backend,
        "vggt_upload_mode": args.vggt_upload_mode,
        "vggt_timeout": args.vggt_timeout,
        "vggt_conf_thres": args.vggt_conf_thres,
        "vggt_max_points_k": args.vggt_max_points_k,
        "vggt_mask_sky": args.vggt_mask_sky,
        "clahe": {"enabled": True, "clip_limit": args.clahe_clip} if args.clahe else None,
        "adaptive_fps": (
            {
                "enabled": True,
                "base_fps": args.adaptive_base_fps,
                "target_frames": args.adaptive_target,
                "tail_dense_s": args.adaptive_tail_dense_s,
            }
            if args.adaptive_fps
            else None
        ),
        "exclusion_masks": None if args.no_masks else (annotation.get("exclusion_masks") or None),
        "client_sky_seg": args.client_sky_seg,
    }


def scene_dir_for(out_dir: Path, annotation: dict[str, Any], scene_id: str) -> Path:
    return out_dir / "scenes" / slugify(annotation["video_file"]) / scene_id


def scene_is_complete(scene_dir: Path) -> bool:
    return (
        (scene_dir / "vggt_scene.glb").exists()
        and (scene_dir / "relative_path.csv").exists()
        and (scene_dir / "viewer" / "scene_meta.json").exists()
    )


def result_from_scene(
    out_dir: Path,
    path: Path,
    annotation: dict[str, Any],
    scene_id: str,
    selected: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    scene_dir = scene_dir_for(out_dir, annotation, scene_id)
    manifest_path = scene_dir / "scene_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    return {
        "annotation": str(path),
        "video_file": annotation["video_file"],
        "scene_id": scene_id,
        "job_id": manifest.get("job_id", ""),
        "status": status,
        "step": manifest.get("job_step", ""),
        "error": manifest.get("error", ""),
        "viewer_url": manifest.get("viewer_url", f"/scenes/{slugify(annotation['video_file'])}/{scene_id}/viewer/index.html"),
        "segments": [row["segment_id"] for row in selected],
        "model_config": manifest.get("model_config", {}),
    }


def run_batch(args: argparse.Namespace) -> int:
    if args.annotations:
        paths = [Path(p) for p in args.annotations]
    else:
        paths = sorted(ANNOTATION_DIR.glob("*_annotations.json"))[args.offset : args.offset + args.limit]
    if not paths:
        raise SystemExit("No annotation files selected")

    print(f"[batch] selected {len(paths)} annotation file(s)", flush=True)
    results: list[dict[str, Any]] = []
    for batch_index, path in enumerate(paths, start=1):
        annotation = json.loads(path.read_text())
        if args.manual_only and annotation.get("auto_generated"):
            if not args.quiet:
                print(f"[{batch_index}/{len(paths)}] skip auto-generated {path.name}", flush=True)
            continue
        intervals = build_flight_intervals(annotation)
        selected = select_attack_pause_chain(intervals)
        if not selected:
            print(f"[{batch_index}/{len(paths)}] skip {path.name}: no flight intervals", flush=True)
            continue
        selected = trim_to_tail_seconds(selected, args.tail_seconds)

        total_s = sum(float(row["duration_s"]) for row in selected)
        body = request_body(annotation, selected, args)
        segment_ids = ",".join(row["segment_id"] for row in selected)
        scene_id = f"{slugify(annotation['video_file'])}_{'_'.join(row['segment_id'] for row in selected)}"
        if args.skip_existing and scene_is_complete(scene_dir_for(args.out_dir, annotation, scene_id)):
            if not args.quiet:
                print(f"[{batch_index}/{len(paths)}] skip complete {scene_id}", flush=True)
            results.append(result_from_scene(args.out_dir, path, annotation, scene_id, selected, "skipped_complete"))
            continue
        print(f"[{batch_index}/{len(paths)}] {annotation['video_file']} -> {segment_ids} ({total_s:.3f}s @ {args.fps:g}fps)", flush=True)
        response = post_json(args.server, "/api/reconstruct", body)
        job_id = response["job_id"]
        scene_id = response["scene_id"]
        if not args.quiet:
            print(f"[job] {job_id} scene={scene_id}", flush=True)

        last_line = ""
        while True:
            job = get_json(args.server, f"/api/jobs/{job_id}")
            logs = job.get("logs") or []
            line = f"[poll] {scene_id}: {job.get('status')} | {job.get('step')}"
            if logs:
                line += f" | {logs[-1]}"
            if line != last_line:
                if not args.quiet:
                    print(line, flush=True)
                last_line = line
            if job.get("status") in {"done", "error"}:
                scene_manifest = scene_dir_for(args.out_dir, annotation, scene_id) / "scene_manifest.json"
                model_config: dict[str, Any] = {}
                if scene_manifest.exists():
                    model_config = json.loads(scene_manifest.read_text()).get("model_config", {})
                results.append(
                    {
                        "annotation": str(path),
                        "video_file": annotation["video_file"],
                        "scene_id": scene_id,
                        "job_id": job_id,
                        "status": job.get("status"),
                        "step": job.get("step"),
                        "error": job.get("error", ""),
                        "viewer_url": job.get("viewer_url", ""),
                        "segments": [row["segment_id"] for row in selected],
                        "model_config": model_config,
                    }
                )
                print(f"[{batch_index}/{len(paths)}] {scene_id}: {job.get('status')}", flush=True)
                if job.get("status") == "error" and not args.continue_on_error:
                    print(f"[batch] stopping after error in {scene_id}", flush=True)
                    args.results_file.write_text(json.dumps(results, indent=2))
                    return 1
                break
            time.sleep(args.poll_seconds)

    out_path = args.results_file
    out_path.write_text(json.dumps(results, indent=2))
    done = sum(1 for row in results if row["status"] == "done")
    errors = sum(1 for row in results if row["status"] == "error")
    print(f"[batch] complete: {done} done, {errors} error(s); wrote {out_path}", flush=True)
    return 0 if errors == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8766")
    parser.add_argument("--out-dir", type=Path, default=ROOT, help="scene output root configured on the FPV tool server")
    parser.add_argument("--annotations", nargs="*", help="explicit annotation JSON paths (overrides glob/offset/limit)")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--tail-seconds", type=float, default=0.0, help="keep only the last N seconds of the flight sequence (0 = whole chain)")
    parser.add_argument("--no-masks", action="store_true", help="ignore the annotation's exclusion_masks (no black boxes, no mask_black_bg)")
    parser.add_argument("--client-sky-seg", action="store_true", help="run skyseg on CLEAN frames client-side, then paint sky (and boxes) black (sky_then_boxes)")
    parser.add_argument("--width", type=int, default=660)
    parser.add_argument("--crop-preset", default="central_clean")
    parser.add_argument("--scale", type=float, default=117.6)
    parser.add_argument("--focal-px", type=float, default=812.0)
    parser.add_argument("--max-vggt-frames", type=int, default=0)
    parser.add_argument("--frame-window", choices=["all", "first", "last", "even"], default="all")
    parser.add_argument("--vggt-timeout", type=int, default=1800)
    parser.add_argument("--vggt-space", default="")
    parser.add_argument("--vggt-backend", choices=["omega", "classic"], default="omega")
    parser.add_argument("--vggt-upload-mode", choices=["auto", "images", "video"], default="video")
    parser.add_argument("--vggt-conf-thres", type=float, default=50.0)
    parser.add_argument("--vggt-max-points-k", type=float, default=1000.0)
    parser.add_argument("--vggt-mask-sky", action="store_true")
    parser.add_argument("--clahe", action="store_true", help="sequence-uniform CLAHE contrast enhancement (for dark clips)")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE contrast-limit (higher = stronger)")
    parser.add_argument("--adaptive-fps", action="store_true", help="motion-aware keyframing (more frames where motion is high, sharpest-in-window)")
    parser.add_argument("--adaptive-base-fps", type=float, default=24.0, help="dense sampling rate before adaptive keyframing")
    parser.add_argument("--adaptive-target", type=int, default=125, help="number of keyframes to keep when --adaptive-fps")
    parser.add_argument("--adaptive-tail-dense-s", type=float, default=0.0,
                        help="additionally keep every sampled frame in the last N seconds of flight (on top of --adaptive-target)")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--skip-camera-views", action="store_true")
    parser.add_argument("--no-refresh-vggt", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--manual-only", action="store_true",
                        help="skip annotations flagged auto_generated (process only hand-tagged videos)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--results-file", type=Path, default=ROOT / "scenes" / ".batch_results.json")
    args = parser.parse_args()
    if not args.results_file.is_absolute():
        args.results_file = ROOT / args.results_file
    args.out_dir = args.out_dir.resolve()
    return args


def main() -> int:
    try:
        return run_batch(parse_args())
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach local FPV tool server: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
