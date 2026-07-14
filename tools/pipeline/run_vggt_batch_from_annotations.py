#!/usr/bin/env python3
"""Sequentially run VGGT scene jobs from annotation JSON files."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
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


def trim_to_tail_window(
    selected: list[dict[str, Any]],
    tail_seconds: float,
    exclude_tail_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Clip a concatenated flight sequence to ``[-tail-exclude, -exclude]``.

    Segment IDs and source-video timestamps are preserved, including when the
    requested window crosses a pause-linked segment boundary.
    """
    if not selected or (tail_seconds <= 0 and exclude_tail_seconds <= 0):
        return selected
    total = sum(float(s["duration_s"]) for s in selected)
    window_end = max(0.0, total - max(0.0, exclude_tail_seconds))
    window_start = 0.0 if tail_seconds <= 0 else max(0.0, window_end - tail_seconds)
    if window_end <= window_start:
        return []

    out: list[dict[str, Any]] = []
    sequence_start = 0.0
    for seg in selected:
        dur = float(seg["duration_s"])
        sequence_end = sequence_start + dur
        overlap_start = max(sequence_start, window_start)
        overlap_end = min(sequence_end, window_end)
        if overlap_end > overlap_start:
            trimmed = dict(seg)
            source_start = float(seg["start_s"]) + (overlap_start - sequence_start)
            source_end = float(seg["start_s"]) + (overlap_end - sequence_start)
            trimmed["start_s"] = round(source_start, 3)
            trimmed["end_s"] = round(source_end, 3)
            trimmed["duration_s"] = round(source_end - source_start, 3)
            out.append(trimmed)
        sequence_start = sequence_end
    return out


def trim_to_tail_seconds(selected: list[dict[str, Any]], tail_seconds: float) -> list[dict[str, Any]]:
    """Backward-compatible tail trim with no excluded final interval."""
    return trim_to_tail_window(selected, tail_seconds)


def request_json(request: urllib.request.Request, *, retries: int, timeout: int) -> dict[str, Any]:
    """Retry only transient local-server failures.

    A reconstruction request is idempotent by scene id while it is active, so a
    retry after a dropped response returns the same job instead of starting a
    duplicate job.
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # The server can briefly return 5xx while it is creating its job
            # directory. Do not retry client errors because they are input bugs.
            if exc.code < 500:
                raise
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(10, 2**attempt))
    assert last_error is not None
    raise last_error


def post_json(base_url: str, path: str, body: dict[str, Any], *, retries: int) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url, path),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return request_json(request, retries=retries, timeout=60)


def get_json(base_url: str, path: str, *, retries: int) -> dict[str, Any]:
    request = urllib.request.Request(urllib.parse.urljoin(base_url, path), method="GET")
    return request_json(request, retries=retries, timeout=60)


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
        "target_frames": args.adaptive_target if args.adaptive_fps else 0,
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
        "omega_pod_id": args.omega_pod_id,
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


def result_from_scene(out_dir: Path, path: Path, annotation: dict[str, Any], scene_id: str, selected: list[dict[str, Any]], status: str) -> dict[str, Any]:
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
        "quality": manifest.get("quality"),
    }


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    try:
        rows = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(rows, list):
        return {}
    return {str(row.get("scene_id")): row for row in rows if isinstance(row, dict) and row.get("scene_id")}


def write_results(path: Path, results: dict[str, dict[str, Any]]) -> None:
    """Write a durable checkpoint after every resolved scene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(list(results.values()), indent=2))
    tmp.replace(path)


def cleanup_omega_uploads(args: argparse.Namespace) -> bool:
    """Remove completed Gradio upload directories between sequential jobs."""
    cmd = [sys.executable, str(ROOT / "tools" / "pipeline" / "omega_pod.py"), "cleanup", "--ready-timeout", "60"]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode:
        print(
            "[batch] warning: Omega upload-cache cleanup failed; continuing with the durable checkpoint intact: "
            + (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"),
            file=sys.stderr,
            flush=True,
        )
        return False
    if not args.quiet:
        output = result.stdout.strip()
        print(f"[batch] Omega upload cache cleared{': ' + output if output else ''}", flush=True)
    return True


def run_batch(args: argparse.Namespace) -> int:
    if args.annotations:
        paths = [Path(p) for p in args.annotations]
    else:
        paths = sorted(ANNOTATION_DIR.glob("*_annotations.json"))[args.offset : args.offset + args.limit]
    if not paths:
        raise SystemExit("No annotation files selected")

    print(f"[batch] selected {len(paths)} annotation file(s)", flush=True)
    results = {} if args.reset_results else load_results(args.results_file)
    resolved_jobs = 0
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
        selected = trim_to_tail_window(selected, args.tail_seconds, args.exclude_tail_seconds)
        if not selected:
            print(f"[{batch_index}/{len(paths)}] skip {path.name}: tail window is empty", flush=True)
            continue

        total_s = sum(float(row["duration_s"]) for row in selected)
        body = request_body(annotation, selected, args)
        segment_ids = ",".join(row["segment_id"] for row in selected)
        scene_id = f"{slugify(annotation['video_file'])}_{'_'.join(row['segment_id'] for row in selected)}"
        if args.skip_existing and scene_is_complete(scene_dir_for(args.out_dir, annotation, scene_id)):
            if not args.quiet:
                print(f"[{batch_index}/{len(paths)}] skip complete {scene_id}", flush=True)
            results[scene_id] = result_from_scene(args.out_dir, path, annotation, scene_id, selected, "skipped_complete")
            write_results(args.results_file, results)
            continue
        print(f"[{batch_index}/{len(paths)}] {annotation['video_file']} -> {segment_ids} ({total_s:.3f}s @ {args.fps:g}fps)", flush=True)
        response = post_json(args.server, "/api/reconstruct", body, retries=args.server_retries)
        job_id = response["job_id"]
        scene_id = response["scene_id"]
        if not args.quiet:
            print(f"[job] {job_id} scene={scene_id}", flush=True)

        last_line = ""
        while True:
            job = get_json(args.server, f"/api/jobs/{job_id}", retries=args.server_retries)
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
                results[scene_id] = {
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
                write_results(args.results_file, results)
                resolved_jobs += 1
                if args.omega_cleanup_every and resolved_jobs % args.omega_cleanup_every == 0:
                    cleanup_omega_uploads(args)
                print(f"[{batch_index}/{len(paths)}] {scene_id}: {job.get('status')}", flush=True)
                if job.get("status") == "error" and not args.continue_on_error:
                    print(f"[batch] stopping after error in {scene_id}", flush=True)
                    return 1
                break
            time.sleep(args.poll_seconds)

    write_results(args.results_file, results)
    done = sum(1 for row in results.values() if row["status"] == "done")
    errors = sum(1 for row in results.values() if row["status"] == "error")
    print(f"[batch] complete: {done} done, {errors} error(s); wrote {args.results_file}", flush=True)
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
    parser.add_argument(
        "--exclude-tail-seconds",
        type=float,
        default=0.0,
        help="exclude the final N seconds before applying --tail-seconds",
    )
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
    parser.add_argument("--vggt-upload-mode", choices=["auto", "images", "video"], default="images")
    parser.add_argument("--vggt-conf-thres", type=float, default=50.0)
    parser.add_argument("--vggt-max-points-k", type=float, default=1000.0)
    parser.add_argument("--vggt-mask-sky", action="store_true")
    parser.add_argument("--omega-pod-id", default=os.environ.get("RUNPOD_POD_ID", "7i0jtqk99phk2j"))
    parser.add_argument("--clahe", action="store_true", help="sequence-uniform CLAHE contrast enhancement (for dark clips)")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE contrast-limit (higher = stronger)")
    parser.add_argument("--adaptive-fps", action="store_true", help="motion-aware keyframing (more frames where motion is high, sharpest-in-window)")
    parser.add_argument("--adaptive-base-fps", type=float, default=24.0, help="dense sampling rate before adaptive keyframing")
    parser.add_argument("--adaptive-target", type=int, default=125, help="number of keyframes to keep when --adaptive-fps")
    parser.add_argument("--adaptive-tail-dense-s", type=float, default=0.0,
                        help="additionally keep every sampled frame in the last N seconds of flight (on top of --adaptive-target)")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--server-retries", type=int, default=3, help="retry transient local-server failures this many times")
    parser.add_argument(
        "--omega-cleanup-every",
        type=int,
        default=0,
        help="clear completed Omega Gradio uploads every N resolved jobs (0 disables)",
    )
    parser.add_argument("--skip-camera-views", action="store_true")
    parser.add_argument("--no-refresh-vggt", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--manual-only", action="store_true",
                        help="skip annotations flagged auto_generated (process only hand-tagged videos)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("scenes/.batch_results.json"),
        help="checkpoint ledger, relative to --out-dir unless absolute",
    )
    parser.add_argument("--reset-results", action="store_true", help="discard prior checkpoints in --results-file")
    args = parser.parse_args()
    if args.omega_cleanup_every < 0:
        parser.error("--omega-cleanup-every must be zero or greater")
    if args.tail_seconds < 0 or args.exclude_tail_seconds < 0:
        parser.error("tail durations must be zero or greater")
    if not args.results_file.is_absolute():
        args.results_file = args.out_dir / args.results_file
    return args


def main() -> int:
    try:
        return run_batch(parse_args())
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach local FPV tool server: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
