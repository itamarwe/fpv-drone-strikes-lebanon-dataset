#!/usr/bin/env python3
"""Build FPV flight-segment path and scale artifacts.

The pipeline is deliberately split into small stages:

1. `manifest` parses annotation JSON files into flight intervals. The last
   flight interval in each video is marked as the attack segment.
2. `extract-frames` samples annotated flight intervals into one image sequence
   per source video, preserving a frame->segment mapping.
3. `analyze-overlap` compares earlier flight segments with the attack segment
   and builds attack-overlap reconstruction groups, so unrelated scenes are not
   forced into one VGGT world.
4. `run-vggt` sends sampled frames to the public VGGT Hugging Face Space and
   stores a .glb per reconstruction group.
5. `extract-vggt` parses VGGT camera frustums into relative camera paths.
6. `scale-report` computes path metrics and only reports meters when metric
   constraints are available. Speed priors are optional sanity checks, not scale.
7. `visualize` renders scene point-cloud/path, speed, and height-proxy plots.

Heavy generated outputs default to /tmp/fpv-flight-paths so repo history stays
small. The scripts themselves are the durable artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = ROOT / "annotations"
GEO_CSV = ROOT / "geo" / "fpv_drone_map_records.csv"
DEFAULT_OUT_DIR = Path("/tmp/fpv-flight-paths")
DEFAULT_VIDEO_CACHE = Path("/tmp/fpv-model-benchmark/videos")
DEFAULT_RECON_SUBDIR = "reconstructions"
ATTACK_RECON_SUBDIR = "attack_reconstructions"
FLIGHT_TYPES = {"flight_start", "new_flight_start"}


@dataclass(frozen=True)
class FlightSegment:
    video_file: str
    video_url: str
    description: str
    date: str
    town: str
    lat: str
    lon: str
    segment_index: int
    segment_count: int
    segment_id: str
    start_s: float
    end_s: float
    duration_s: float
    start_type: str
    end_type: str
    is_attack: bool


def slug_from_video(video_file: str) -> str:
    return Path(video_file).stem


def run(cmd: list[str], *, timeout: int | None = None) -> None:
    subprocess.run(cmd, check=True, timeout=timeout)


def load_geo() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with GEO_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[Path(row["video_url"]).name] = row
    return rows


def load_segments(annotation_paths: Iterable[Path]) -> list[FlightSegment]:
    geo = load_geo()
    out: list[FlightSegment] = []
    for path in sorted(annotation_paths):
        data = json.loads(path.read_text())
        video_file = data["video_file"]
        geo_row = geo.get(video_file, {})
        markers = sorted(data.get("segments", []), key=lambda s: float(s["time"]))
        flight_marker_indexes = [i for i, s in enumerate(markers) if s["type"] in FLIGHT_TYPES]
        segment_count = len(flight_marker_indexes)
        for local_idx, marker_idx in enumerate(flight_marker_indexes, start=1):
            marker = markers[marker_idx]
            end_s = None
            end_type = "video_end"
            for nxt in markers[marker_idx + 1 :]:
                if nxt["type"] in FLIGHT_TYPES:
                    continue
                end_s = float(nxt["time"])
                end_type = nxt["type"]
                break
            if end_s is None:
                # Current annotations all have explicit non-flight end markers.
                # Keeping this skipped avoids silently ffprobe'ing every URL.
                print(f"[warn] skipping open-ended segment in {video_file}", file=sys.stderr)
                continue
            start_s = float(marker["time"])
            if end_s <= start_s:
                print(f"[warn] skipping non-positive segment in {video_file}: {start_s}->{end_s}", file=sys.stderr)
                continue
            video_slug = slug_from_video(video_file)
            out.append(
                FlightSegment(
                    video_file=video_file,
                    video_url=data["video_url"],
                    description=data.get("description", geo_row.get("description", "")),
                    date=data.get("date", geo_row.get("date", "")),
                    town=data.get("town", geo_row.get("town", "")),
                    lat=geo_row.get("lat", ""),
                    lon=geo_row.get("lon", ""),
                    segment_index=local_idx,
                    segment_count=segment_count,
                    segment_id=f"{video_slug}_seg{local_idx:02d}",
                    start_s=start_s,
                    end_s=end_s,
                    duration_s=end_s - start_s,
                    start_type=marker["type"],
                    end_type=end_type,
                    is_attack=(local_idx == segment_count),
                )
            )
    return out


def write_manifest(segments: list[FlightSegment], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "flight_segments.csv"
    json_path = out_dir / "flight_segments.json"
    fieldnames = list(FlightSegment.__dataclass_fields__)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seg in segments:
            writer.writerow(seg.__dict__)
    json_path.write_text(json.dumps([seg.__dict__ for seg in segments], indent=2))

    by_video: dict[str, list[FlightSegment]] = {}
    for seg in segments:
        by_video.setdefault(seg.video_file, []).append(seg)
    videos_path = out_dir / "videos.json"
    videos = []
    for video_file, items in sorted(by_video.items()):
        videos.append(
            {
                "video_file": video_file,
                "video_id": slug_from_video(video_file),
                "video_url": items[0].video_url,
                "description": items[0].description,
                "date": items[0].date,
                "town": items[0].town,
                "lat": items[0].lat,
                "lon": items[0].lon,
                "flight_segments": len(items),
                "attack_segment_id": next((s.segment_id for s in items if s.is_attack), ""),
                "flight_duration_s": sum(s.duration_s for s in items),
            }
        )
    videos_path.write_text(json.dumps(videos, indent=2))
    print(f"[manifest] {len(segments)} segments across {len(videos)} videos")
    print(f"[manifest] wrote {csv_path}")


def read_manifest(out_dir: Path) -> list[FlightSegment]:
    path = out_dir / "flight_segments.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}. Run `manifest` first.")
    return [FlightSegment(**row) for row in json.loads(path.read_text())]


def download_video(seg: FlightSegment, cache_dir: Path, timeout: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / seg.video_file
    if out.exists() and out.stat().st_size > 0:
        return out
    tmp = out.with_suffix(out.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    run(
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
            str(tmp),
            seg.video_url,
        ]
    )
    tmp.replace(out)
    return out


def video_groups(segments: list[FlightSegment]) -> dict[str, list[FlightSegment]]:
    groups: dict[str, list[FlightSegment]] = {}
    for seg in segments:
        groups.setdefault(seg.video_file, []).append(seg)
    return {k: sorted(v, key=lambda s: s.segment_index) for k, v in sorted(groups.items())}


def recon_root(out_dir: Path, recon_subdir: str) -> Path:
    return out_dir / recon_subdir


def extract_frames(args: argparse.Namespace) -> None:
    segments = read_manifest(args.out_dir)
    groups = video_groups(segments)
    selected = list(groups.items())
    if args.video_file:
        selected = [(k, v) for k, v in selected if k == args.video_file or slug_from_video(k) == args.video_file]
    if args.limit:
        selected = selected[: args.limit]
    frames_root = recon_root(args.out_dir, args.recon_subdir)
    frames_root.mkdir(parents=True, exist_ok=True)

    for video_idx, (video_file, items) in enumerate(selected, start=1):
        video_id = slug_from_video(video_file)
        video_dir = frames_root / video_id
        frames_dir = video_dir / "frames"
        frame_map_path = video_dir / "frames.csv"
        if frames_dir.exists() and any(frames_dir.glob("*.jpg")) and frame_map_path.exists() and not args.refresh:
            print(f"[frames {video_idx}/{len(selected)}] skip existing {video_id}")
            continue
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_path = download_video(items[0], args.video_cache_dir, timeout=args.download_timeout)
        frame_rows = []
        next_frame = 1
        print(f"[frames {video_idx}/{len(selected)}] {video_file}: {len(items)} segments")
        for seg in items:
            tmp_dir = video_dir / f"_tmp_{seg.segment_index:02d}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            vf = f"fps={args.sample_fps:g}"
            if args.width:
                vf += f",scale={args.width}:-2"
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{seg.start_s:.3f}",
                "-to",
                f"{seg.end_s:.3f}",
                "-i",
                str(video_path),
                "-vf",
                vf,
                "-q:v",
                str(args.jpeg_quality),
                str(tmp_dir / "f_%06d.jpg"),
            ]
            run(cmd, timeout=args.ffmpeg_timeout)
            files = sorted(tmp_dir.glob("f_*.jpg"))
            for local_i, src in enumerate(files):
                dst = frames_dir / f"f_{next_frame:06d}.jpg"
                src.rename(dst)
                frame_rows.append(
                    {
                        "frame_index": next_frame,
                        "file": dst.name,
                        "video_file": video_file,
                        "segment_id": seg.segment_id,
                        "segment_index": seg.segment_index,
                        "is_attack": str(seg.is_attack).lower(),
                        "video_time_s": seg.start_s + local_i / args.sample_fps,
                        "segment_time_s": local_i / args.sample_fps,
                    }
                )
                next_frame += 1
            shutil.rmtree(tmp_dir)
        with frame_map_path.open("w", newline="") as f:
            fieldnames = [
                "frame_index",
                "file",
                "video_file",
                "segment_id",
                "segment_index",
                "is_attack",
                "video_time_s",
                "segment_time_s",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frame_rows)
        meta = {
            "video_file": video_file,
            "video_id": video_id,
            "sample_fps": args.sample_fps,
            "width": args.width,
            "frames": len(frame_rows),
            "segments": [s.__dict__ for s in items],
        }
        (video_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"[frames] wrote {len(frame_rows)} frames -> {frames_dir}")


def load_frame_rows(video_dir: Path) -> list[dict[str, str]]:
    return list(csv.DictReader((video_dir / "frames.csv").open()))


def evenly_sample(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if len(rows) <= limit:
        return rows
    idx = np.linspace(0, len(rows) - 1, limit).round().astype(int)
    return [rows[int(i)] for i in idx]


def image_descriptor(path: Path) -> dict[str, np.ndarray]:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    small_rgb = img.resize((64, 36))
    rgb = np.asarray(small_rgb, dtype=np.float32) / 255.0
    gray_img = img.convert("L").resize((32, 32))
    gray = np.asarray(gray_img, dtype=np.float32) / 255.0
    vec = gray.reshape(-1)
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    ahash_img = img.convert("L").resize((8, 8))
    ahash_gray = np.asarray(ahash_img, dtype=np.float32)
    ahash = ahash_gray > ahash_gray.mean()
    hist = np.histogramdd(
        rgb.reshape(-1, 3),
        bins=(4, 4, 4),
        range=((0, 1), (0, 1), (0, 1)),
    )[0].astype(np.float32)
    hist = hist.reshape(-1)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum
    return {"vec": vec, "ahash": ahash.reshape(-1), "hist": hist}


def descriptor_similarity(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    # This is a conservative appearance-overlap score. It is not a replacement
    # for geometric verification, but it prevents blindly mixing unrelated scenes.
    corr = float(np.dot(a["vec"], b["vec"]))
    corr_score = np.clip((corr + 1.0) / 2.0, 0.0, 1.0)
    hash_score = 1.0 - float(np.mean(a["ahash"] != b["ahash"]))
    hist_score = float(np.minimum(a["hist"], b["hist"]).sum())
    return 0.45 * corr_score + 0.25 * hash_score + 0.30 * hist_score


def analyze_overlap(args: argparse.Namespace) -> None:
    source_root = recon_root(args.out_dir, args.source_recon_subdir)
    groups_out = args.out_dir / "reconstruction_groups.json"
    report_out = args.out_dir / "overlap_report.csv"
    all_groups = []
    report_rows = []
    video_dirs = sorted([p for p in source_root.iterdir() if (p / "frames.csv").exists()])
    if args.video_id:
        video_dirs = [p for p in video_dirs if p.name == args.video_id]
    if args.limit:
        video_dirs = video_dirs[: args.limit]

    for video_idx, video_dir in enumerate(video_dirs, start=1):
        frame_rows = load_frame_rows(video_dir)
        by_segment: dict[str, list[dict[str, str]]] = {}
        for row in frame_rows:
            by_segment.setdefault(row["segment_id"], []).append(row)
        attack_ids = [sid for sid, rows in by_segment.items() if any(r["is_attack"] == "true" for r in rows)]
        if not attack_ids:
            print(f"[overlap] no attack segment in {video_dir.name}", file=sys.stderr)
            continue
        attack_id = attack_ids[-1]
        attack_samples = evenly_sample(by_segment[attack_id], args.samples_per_segment)
        attack_descs = [
            (row, image_descriptor(video_dir / "frames" / row["file"]))
            for row in attack_samples
        ]

        included = [attack_id]
        excluded = []
        print(f"[overlap {video_idx}/{len(video_dirs)}] {video_dir.name} attack={attack_id}")
        for segment_id, rows in sorted(by_segment.items(), key=lambda item: int(item[1][0]["segment_index"])):
            is_attack = segment_id == attack_id
            best = 1.0 if is_attack else -1.0
            best_pair = ("", "")
            if not is_attack:
                seg_samples = evenly_sample(rows, args.samples_per_segment)
                for seg_row in seg_samples:
                    seg_desc = image_descriptor(video_dir / "frames" / seg_row["file"])
                    for attack_row, attack_desc in attack_descs:
                        score = descriptor_similarity(seg_desc, attack_desc)
                        if score > best:
                            best = score
                            best_pair = (seg_row["file"], attack_row["file"])
                if best >= args.threshold:
                    included.append(segment_id)
                else:
                    excluded.append(segment_id)
            report_rows.append(
                {
                    "video_id": video_dir.name,
                    "video_file": rows[0]["video_file"],
                    "segment_id": segment_id,
                    "segment_index": rows[0]["segment_index"],
                    "is_attack": str(is_attack).lower(),
                    "best_overlap_score": f"{best:.4f}",
                    "best_segment_frame": best_pair[0],
                    "best_attack_frame": best_pair[1],
                    "included_in_attack_reconstruction": str(is_attack or best >= args.threshold).lower(),
                    "threshold": args.threshold,
                }
            )
        included_sorted = sorted(set(included), key=lambda sid: int(by_segment[sid][0]["segment_index"]))
        excluded_sorted = sorted(set(excluded), key=lambda sid: int(by_segment[sid][0]["segment_index"]))
        all_groups.append(
            {
                "group_id": f"{video_dir.name}_attack_overlap",
                "video_id": video_dir.name,
                "video_file": frame_rows[0]["video_file"],
                "source_recon_subdir": args.source_recon_subdir,
                "attack_segment_id": attack_id,
                "included_segment_ids": included_sorted,
                "excluded_segment_ids": excluded_sorted,
                "threshold": args.threshold,
                "score_type": "appearance_overlap_v1",
            }
        )

    groups_out.write_text(json.dumps(all_groups, indent=2))
    if report_rows:
        with report_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
    print(f"[overlap] wrote {groups_out}")
    print(f"[overlap] wrote {report_out}")


def extract_group_frames(args: argparse.Namespace) -> None:
    groups_path = args.out_dir / "reconstruction_groups.json"
    if not groups_path.exists():
        raise FileNotFoundError(f"Missing {groups_path}; run analyze-overlap first.")
    groups = json.loads(groups_path.read_text())
    if args.video_id:
        groups = [g for g in groups if g["video_id"] == args.video_id or g["group_id"] == args.video_id]
    if args.limit:
        groups = groups[: args.limit]
    dest_root = recon_root(args.out_dir, args.recon_subdir)
    dest_root.mkdir(parents=True, exist_ok=True)
    for idx, group in enumerate(groups, start=1):
        source_dir = recon_root(args.out_dir, group["source_recon_subdir"]) / group["video_id"]
        rows = load_frame_rows(source_dir)
        include = set(group["included_segment_ids"])
        group_dir = dest_root / group["group_id"]
        frames_dir = group_dir / "frames"
        frame_map_path = group_dir / "frames.csv"
        if frames_dir.exists() and any(frames_dir.glob("*.jpg")) and frame_map_path.exists() and not args.refresh:
            print(f"[group-frames {idx}/{len(groups)}] skip existing {group['group_id']}")
            continue
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        new_rows = []
        next_frame = 1
        for row in rows:
            if row["segment_id"] not in include:
                continue
            src = source_dir / "frames" / row["file"]
            dst = frames_dir / f"f_{next_frame:06d}.jpg"
            shutil.copy2(src, dst)
            new_rows.append({**row, "frame_index": next_frame, "file": dst.name})
            next_frame += 1
        with frame_map_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()) if new_rows else [
                "frame_index",
                "file",
                "video_file",
                "segment_id",
                "segment_index",
                "is_attack",
                "video_time_s",
                "segment_time_s",
            ])
            writer.writeheader()
            writer.writerows(new_rows)
        metadata = {
            **group,
            "frames": len(new_rows),
            "note": "Contains attack segment plus only earlier flight segments passing appearance-overlap gate.",
        }
        (group_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"[group-frames {idx}/{len(groups)}] {group['group_id']}: {len(new_rows)} frames")


def run_vggt(args: argparse.Namespace) -> None:
    try:
        from gradio_client import Client, handle_file
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Missing gradio_client. Use the prior VGGT venv or install: "
            "pip install gradio_client"
        ) from exc

    root = recon_root(args.out_dir, args.recon_subdir)
    video_dirs = sorted([p for p in root.iterdir() if (p / "frames").is_dir()])
    if args.video_id:
        video_dirs = [p for p in video_dirs if p.name == args.video_id]
    if args.limit:
        video_dirs = video_dirs[: args.limit]
    if not video_dirs:
        raise SystemExit(f"No frame directories found under {root}")

    print(f"[vggt] connecting to {args.space}", flush=True)
    hf_token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    if hf_token:
        import inspect

        print(f"[vggt] using Hugging Face token from ${args.hf_token_env}", flush=True)
        token_kwarg = "hf_token" if "hf_token" in inspect.signature(Client).parameters else "token"
        client = Client(args.space, **{token_kwarg: hf_token}, verbose=False)
    else:
        client = Client(args.space, verbose=False)
    for idx, video_dir in enumerate(video_dirs, start=1):
        out_glb = video_dir / "vggt_scene.glb"
        if out_glb.exists() and out_glb.stat().st_size > 0 and not args.refresh:
            print(f"[vggt {idx}/{len(video_dirs)}] skip existing {video_dir.name}")
            continue
        upload_mode = args.upload_mode
        input_video = video_dir / "vggt_input.mp4"
        if upload_mode == "auto":
            upload_mode = "video" if input_video.exists() else "images"
        if args.backend == "classic" and upload_mode == "video":
            print("[vggt] classic backend video uploads are sampled at 1fps by the demo; using images instead", flush=True)
            upload_mode = "images"
        if upload_mode == "video":
            if not input_video.exists():
                raise SystemExit(f"{input_video} does not exist; rerun frame extraction or use --upload-mode images")
            print(
                f"[vggt {idx}/{len(video_dirs)}] uploading video at {args.video_sample_fps:g} fps: {video_dir.name}",
                flush=True,
            )
            _, target_dir, preview, _ = client.predict(
                input_video={"video": handle_file(str(input_video)), "subtitles": None},
                input_images=[],
                video_sample_fps=float(args.video_sample_fps),
                api_name="/update_gallery_on_upload",
            )
        else:
            frame_files = sorted((video_dir / "frames").glob("*.jpg"))
            if args.max_frames and len(frame_files) > args.max_frames:
                step = math.ceil(len(frame_files) / args.max_frames)
                frame_files = frame_files[::step]
            print(f"[vggt {idx}/{len(video_dirs)}] uploading {len(frame_files)} frames: {video_dir.name}", flush=True)
            upload_files = [handle_file(str(f)) for f in frame_files]
            if args.backend == "classic":
                _, target_dir, preview, _ = client.predict(
                    input_video=None,
                    input_images=upload_files,
                    api_name="/update_gallery_on_upload",
                )
            else:
                _, target_dir, preview, _ = client.predict(
                    input_video=None,
                    input_images=upload_files,
                    video_sample_fps=2.0,
                    api_name="/update_gallery_on_upload",
                )
        print(f"[vggt] target_dir={target_dir}; preview_frames={len(preview) if preview else 0}", flush=True)
        t0 = time.time()
        if args.backend == "classic":
            job = client.submit(
                target_dir=target_dir,
                conf_thres=args.conf_thres,
                frame_filter="All",
                mask_black_bg=False,
                mask_white_bg=False,
                show_cam=True,
                mask_sky=bool(args.mask_sky),
                prediction_mode="Pointmap Regression",
                api_name="/gradio_demo",
            )
        else:
            job = client.submit(
                target_dir=target_dir,
                conf_thres=args.conf_thres,
                mask_black_bg=False,
                mask_white_bg=False,
                show_cam=True,
                mask_sky=bool(args.mask_sky),
                max_points_k=args.max_points_k,
                api_name="/gradio_demo",
            )
        print(f"[vggt] waiting for /gradio_demo result (timeout={args.vggt_timeout}s)", flush=True)
        result = job.result(timeout=args.vggt_timeout)
        glb_path, vis_log = result[0], result[1]
        shutil.copy(glb_path, out_glb)
        (video_dir / "vggt_log.txt").write_text((vis_log or "")[:5000])
        print(f"[vggt] saved {out_glb} in {time.time() - t0:.0f}s", flush=True)


def parse_glb_camera_path(glb_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Missing trimesh. Install with: pip install trimesh") from exc

    scene = trimesh.load(glb_path)
    name_to_transform = {}
    for node in scene.graph.nodes:
        try:
            transform, geom = scene.graph[node]
        except Exception:
            continue
        if geom is not None:
            name_to_transform[geom] = np.asarray(transform)

    cam_names = sorted(
        [name for name in scene.geometry if name != "geometry_0"],
        key=lambda name: int(name.split("_")[1]),
    )
    centers = []
    for name in cam_names:
        geom = scene.geometry[name]
        counts = np.bincount(geom.faces.reshape(-1), minlength=len(geom.vertices))
        apex = geom.vertices[int(np.argmax(counts))]
        transform = name_to_transform.get(name, np.eye(4))
        centers.append((transform @ np.array([*apex, 1.0]))[:3])
    centers_arr = np.asarray(centers, dtype=float)

    pts = np.empty((0, 3), dtype=np.float32)
    cols = np.empty((0, 3), dtype=np.uint8)
    if "geometry_0" in scene.geometry:
        pc = scene.geometry["geometry_0"]
        transform = name_to_transform.get("geometry_0", np.eye(4))
        pts = trimesh.transformations.transform_points(pc.vertices, transform)
        ok = np.isfinite(pts).all(axis=1)
        pts = pts[ok]
        if hasattr(pc.visual, "vertex_colors") and len(pc.visual.vertex_colors):
            cols = np.asarray(pc.visual.vertex_colors)[:, :3][ok]
        else:
            cols = np.zeros((len(pts), 3), dtype=np.uint8)
    return centers_arr, pts.astype(np.float32), cols.astype(np.uint8)


def extract_vggt(args: argparse.Namespace) -> None:
    root = recon_root(args.out_dir, args.recon_subdir)
    video_dirs = sorted([p for p in root.iterdir() if (p / "vggt_scene.glb").exists()])
    if args.video_id:
        video_dirs = [p for p in video_dirs if p.name == args.video_id]
    if args.limit:
        video_dirs = video_dirs[: args.limit]
    if not video_dirs:
        raise SystemExit(f"No VGGT .glb files found under {root}")

    for idx, video_dir in enumerate(video_dirs, start=1):
        path_csv = video_dir / "relative_path.csv"
        if path_csv.exists() and not args.refresh:
            print(f"[extract {idx}/{len(video_dirs)}] skip existing {video_dir.name}")
            continue
        centers, pts, cols = parse_glb_camera_path(video_dir / "vggt_scene.glb")
        frame_rows = list(csv.DictReader((video_dir / "frames.csv").open()))
        n = min(len(centers), len(frame_rows))
        if n == 0:
            print(f"[extract] no cameras in {video_dir.name}", file=sys.stderr)
            continue
        np.save(video_dir / "relative_path.npy", centers[:n])
        if len(pts):
            max_points = args.max_points
            if len(pts) > max_points:
                rng = np.random.default_rng(0)
                sample = rng.choice(len(pts), max_points, replace=False)
                pts = pts[sample]
                cols = cols[sample]
            np.savez_compressed(video_dir / "point_cloud.npz", pts=pts, cols=cols)
        with path_csv.open("w", newline="") as f:
            frame_fieldnames = list(frame_rows[0].keys()) if frame_rows else []
            fieldnames = frame_fieldnames + [name for name in ["x", "y", "z"] if name not in frame_fieldnames]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, xyz in zip(frame_rows[:n], centers[:n]):
                writer.writerow({**row, "x": xyz[0], "y": xyz[1], "z": xyz[2]})
        print(f"[extract {idx}/{len(video_dirs)}] {video_dir.name}: {n} camera centers")


def load_relative_path(path_csv: Path) -> list[dict[str, str]]:
    with path_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def path_metrics(rows: list[dict[str, str]], up_axis: str = "y") -> dict[str, float | str | int]:
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    times = np.array([float(r["video_time_s"]) for r in rows], dtype=float)
    attack = np.array([r["is_attack"] == "true" for r in rows], dtype=bool)
    if len(xyz) < 2:
        return {}
    dt = np.diff(times)
    # Across non-contiguous annotated intervals, video time jumps through removed
    # pause/replay material. Do not let those seam jumps dominate speed metrics.
    same_segment = np.array(
        [rows[i]["segment_id"] == rows[i + 1]["segment_id"] for i in range(len(rows) - 1)],
        dtype=bool,
    )
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    valid_steps = robust_step_mask(rows, xyz, times)
    speeds = steps[valid_steps] / dt[valid_steps] if valid_steps.any() else np.array([])
    path_len = float(steps[valid_steps].sum()) if valid_steps.any() else 0.0
    axis_idx = {"x": 0, "y": 1, "z": 2}[up_axis]
    terminal = xyz[-1]
    up_sign = 1.0
    attack_idx = np.where(attack)[0]
    if len(attack_idx) >= 2:
        first_attack = xyz[attack_idx[0], axis_idx]
        last_attack = xyz[attack_idx[-1], axis_idx]
        up_sign = 1.0 if first_attack >= last_attack else -1.0
        terminal = xyz[attack_idx[-1]]
    rel_height_units = up_sign * (xyz[:, axis_idx] - terminal[axis_idx])
    return {
        "frames": len(rows),
        "segments": len({r["segment_id"] for r in rows}),
        "duration_s": float(sum_segment_durations(rows)),
        "relative_path_units": path_len,
        "relative_displacement_units": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "median_relative_speed_units_s": float(np.median(speeds)) if len(speeds) else 0.0,
        "p90_relative_speed_units_s": float(np.quantile(speeds, 0.9)) if len(speeds) else 0.0,
        "rejected_step_outliers": int(np.count_nonzero(((dt > 0) & (dt < 2.0) & same_segment) & ~valid_steps)),
        "height_proxy_units_min": float(np.min(rel_height_units)),
        "height_proxy_units_max": float(np.max(rel_height_units)),
        "up_axis": up_axis,
        "up_sign": int(up_sign),
    }


def robust_step_mask(rows: list[dict[str, str]], xyz: np.ndarray, times: np.ndarray, factor: float = 6.0) -> np.ndarray:
    """Return usable inter-frame steps, dropping seams and extreme pose jumps."""
    dt = np.diff(times)
    same_segment = np.array(
        [rows[i]["segment_id"] == rows[i + 1]["segment_id"] for i in range(len(rows) - 1)],
        dtype=bool,
    )
    valid = (dt > 0) & (dt < 2.0) & same_segment
    if not valid.any():
        return valid
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    speeds = np.zeros_like(steps)
    speeds[valid] = steps[valid] / dt[valid]
    base = np.median(speeds[valid])
    if base <= 0:
        return valid
    return valid & (speeds <= factor * base)


def sum_segment_durations(rows: list[dict[str, str]]) -> float:
    by_segment: dict[str, list[float]] = {}
    for row in rows:
        by_segment.setdefault(row["segment_id"], []).append(float(row["segment_time_s"]))
    total = 0.0
    for vals in by_segment.values():
        if vals:
            total += max(vals) - min(vals)
    return total


def scale_report(args: argparse.Namespace) -> None:
    root = recon_root(args.out_dir, args.recon_subdir)
    path_files = sorted(root.glob("*/relative_path.csv"))
    report_dir = args.out_dir / "reports" / args.recon_subdir
    report_dir.mkdir(parents=True, exist_ok=True)
    constraints_path = args.constraints or (args.out_dir / "scale_constraints.csv")
    constraints = load_scale_constraints(constraints_path)
    write_scale_constraint_template(args.out_dir / "scale_constraints_template.csv")
    summary_rows = []
    for path_csv in path_files:
        rows = load_relative_path(path_csv)
        metrics = path_metrics(rows, up_axis=args.up_axis)
        if not metrics:
            continue
        median_rel_speed = float(metrics["median_relative_speed_units_s"])
        video_id = path_csv.parent.name
        base = {
            "video_id": video_id,
            "video_file": rows[0]["video_file"],
            **metrics,
        }
        summary_rows.append(
            {
                **base,
                "scale_method": "unscaled_vggt",
                "scale_m_per_unit": "",
                "scale_confidence": "relative_only",
                "assumption": "VGGT/COLMAP-style monocular reconstruction has arbitrary global scale.",
            }
        )
        metric_scales: list[float] = []
        for constraint in constraints.get(video_id, []):
            observed = float(constraint["observed_units"])
            known = float(constraint["known_meters"])
            if observed <= 0 or known <= 0:
                continue
            scale = known / observed
            metric_scales.append(scale)
            summary_rows.append(
                metric_row(
                    base=base,
                    scale=scale,
                    method=f"constraint_{constraint['method']}_{constraint['constraint_id']}",
                    confidence=constraint.get("confidence", "manual"),
                    assumption=constraint.get("notes", "Manual metric scale constraint."),
                    terminal_agl_m=args.terminal_agl_m,
                )
            )
        if metric_scales:
            summary_rows.append(
                metric_row(
                    base=base,
                    scale=float(np.median(metric_scales)),
                    method="constraint_median",
                    confidence="measurement_based",
                    assumption="Median of available manual metric constraints for this reconstruction.",
                    terminal_agl_m=args.terminal_agl_m,
                )
            )
        for speed in args.speed_priors:
            if median_rel_speed <= 0:
                continue
            scale = speed / median_rel_speed
            summary_rows.append(
                metric_row(
                    base=base,
                    scale=scale,
                    method=f"speed_prior_{speed:g}mps",
                    confidence="sanity_check_only",
                    assumption=(
                        f"Scale chosen so median in-segment camera speed is {speed:g} m/s. "
                        "This is not used as measured scale."
                    ),
                    terminal_agl_m=args.terminal_agl_m,
                )
            )

    csv_path = report_dir / "scale_report.csv"
    if summary_rows:
        fieldnames = sorted({k for row in summary_rows for k in row})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    md_path = report_dir / "scale_report.md"
    write_scale_markdown(md_path, summary_rows, path_files, args, constraints_path)
    print(f"[scale] paths with VGGT output: {len(path_files)}")
    print(f"[scale] wrote {md_path}")
    if summary_rows:
        print(f"[scale] wrote {csv_path}")


def metric_row(
    base: dict,
    scale: float,
    method: str,
    confidence: str,
    assumption: str,
    terminal_agl_m: float,
) -> dict:
    return {
        **base,
        "scale_method": method,
        "scale_m_per_unit": scale,
        "scale_confidence": confidence,
        "assumption": assumption,
        "path_length_m": float(base["relative_path_units"]) * scale,
        "height_proxy_m_max": float(base["height_proxy_units_max"]) * scale + terminal_agl_m,
        "terminal_agl_m": terminal_agl_m,
    }


def load_scale_constraints(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        return {}
    by_video: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("video_id"):
                continue
            by_video.setdefault(row["video_id"], []).append(row)
    return by_video


def write_scale_constraint_template(path: Path) -> None:
    if path.exists():
        return
    rows = [
        {
            "video_id": "example_video_or_group_id",
            "constraint_id": "merkava_length_01",
            "method": "object_3d_distance",
            "observed_units": "0.0123",
            "known_meters": "7.6",
            "confidence": "manual_high",
            "notes": "Distance between two picked VGGT/scene points spanning a known object dimension.",
        },
        {
            "video_id": "example_video_or_group_id",
            "constraint_id": "map_road_width_01",
            "method": "map_anchor_distance",
            "observed_units": "0.0045",
            "known_meters": "5.0",
            "confidence": "manual_medium",
            "notes": "Distance between two reconstructed points matched to satellite/map features.",
        },
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_scale_markdown(
    path: Path,
    rows: list[dict],
    path_files: list[Path],
    args: argparse.Namespace,
    constraints_path: Path,
) -> None:
    lines = [
        "# Flight Path Scale Report",
        "",
        "This report separates measured relative geometry from metric assumptions.",
        "VGGT/monocular video produces camera paths in arbitrary scene units; meters require external constraints.",
        "",
        f"- Reconstruction set: `{args.recon_subdir}`",
        f"- Relative paths found: {len(path_files)}",
        f"- Scale constraints file: `{constraints_path}`",
        f"- Speed-prior scenarios, sanity only: {', '.join(str(v) + ' m/s' for v in args.speed_priors) if args.speed_priors else 'disabled'}",
        f"- Terminal AGL offset used for height proxy: {args.terminal_agl_m:g} m",
        "",
        "## Scale Methods",
        "",
        "| Method | Status | Notes |",
        "| --- | --- | --- |",
        "| VGGT relative path | available when `.glb` exists | Shape only; no meters. |",
        "| Manual metric constraints | preferred | Object dimensions, map distances, or georeferenced anchors. |",
        "| Speed prior | optional sanity check | Not suitable as the scale factor. Disabled by default. |",
        "| Terminal ground/impact height proxy | available after relative path | Assumes final attack camera is near target/ground and chooses vertical sign accordingly. |",
        "| Known object dimensions | pending constraints | Needs object/scene measurements in image or 3-D reconstruction. |",
        "| Map/satellite anchors | pending constraints | Needs at least two matched scene points, target coordinate alone is insufficient. |",
        "| DEM/DSM terrain alignment | pending georegistration | Needs georeferenced XY path plus local terrain/surface model. |",
        "",
    ]
    if rows:
        lines += [
            "## Per-Video Relative Metrics",
            "",
            "| Video | Frames | Segments | Relative length | Median rel speed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        seen = set()
        for row in rows:
            if row["scale_method"] != "unscaled_vggt" or row["video_id"] in seen:
                continue
            seen.add(row["video_id"])
            lines.append(
                f"| `{row['video_id']}` | {row['frames']} | {row['segments']} | "
                f"{float(row['relative_path_units']):.4f} | "
                f"{float(row['median_relative_speed_units_s']):.4f} |"
            )
        lines.append("")
    else:
        lines += [
            "## Current State",
            "",
            "No `relative_path.csv` files were found yet. Run `run-vggt` and `extract-vggt` after frame extraction.",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def visualize(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = args.out_dir / "plots"
    if args.recon_subdir != DEFAULT_RECON_SUBDIR:
        plot_dir = plot_dir / args.recon_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.out_dir)
    plot_segment_overview(manifest, plot_dir)

    for path_csv in sorted(recon_root(args.out_dir, args.recon_subdir).glob("*/relative_path.csv")):
        rows = load_relative_path(path_csv)
        if len(rows) < 2:
            continue
        xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
        attack = np.array([r["is_attack"] == "true" for r in rows])
        video_id = path_csv.parent.name
        fig = plt.figure(figsize=(8, 6), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xyz[~attack, 0], xyz[~attack, 2], xyz[~attack, 1], color="#1677b9", lw=2, label="flight")
        if attack.any():
            ax.plot(xyz[attack, 0], xyz[attack, 2], xyz[attack, 1], color="#d64b26", lw=3, label="attack segment")
        ax.scatter([xyz[0, 0]], [xyz[0, 2]], [xyz[0, 1]], color="#2a9d55", s=35, label="first frame")
        ax.scatter([xyz[-1, 0]], [xyz[-1, 2]], [xyz[-1, 1]], color="#111111", s=35, label="last frame")
        ax.set_title(f"Relative VGGT Camera Path: {video_id}")
        ax.set_xlabel("x units")
        ax.set_ylabel("z units")
        ax.set_zlabel("y units")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{video_id}_relative_path.png", dpi=160)
        plt.close(fig)

        plot_speed_height(rows, plot_dir / f"{video_id}_speed_height_proxy.png", args)
        plot_scene_path(path_csv.parent, rows, plot_dir / f"{video_id}_scene_path.png", args)
    print(f"[viz] wrote plots under {plot_dir}")


def plot_segment_overview(segments: list[FlightSegment], plot_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(segments, key=lambda s: (s.date, s.video_file, s.segment_index))
    labels = [s.segment_id.replace("_", "\n", 2) for s in ordered]
    durations = [s.duration_s for s in ordered]
    colors = ["#d64b26" if s.is_attack else "#1677b9" for s in ordered]
    fig, ax = plt.subplots(figsize=(max(12, len(ordered) * 0.22), 5.5), facecolor="white")
    ax.bar(range(len(ordered)), durations, color=colors, width=0.85)
    ax.set_ylabel("seconds")
    ax.set_title("Annotated Flight Segments; Last Segment Per Video Marked as Attack")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.margins(x=0.005)
    fig.tight_layout()
    fig.savefig(plot_dir / "flight_segment_durations.png", dpi=180)
    plt.close(fig)


def plot_speed_height(rows: list[dict[str, str]], out_path: Path, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    times = np.array([float(r["video_time_s"]) for r in rows], dtype=float)
    attack = np.array([r["is_attack"] == "true" for r in rows])
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    dt = np.diff(times)
    valid = robust_step_mask(rows, xyz, times)
    rel_speed = np.full(len(rows), np.nan)
    rel_speed[1:][valid] = steps[valid] / dt[valid]
    scale = args.plot_scale_m_per_unit
    metric = scale is not None and scale > 0
    plot_speed = rel_speed * scale if metric else rel_speed

    axis_idx = {"x": 0, "y": 1, "z": 2}[args.up_axis]
    up_sign = 1.0
    if attack.any():
        idx = np.where(attack)[0]
        up_sign = 1.0 if xyz[idx[0], axis_idx] >= xyz[idx[-1], axis_idx] else -1.0
        terminal_h = xyz[idx[-1], axis_idx]
    else:
        terminal_h = np.min(xyz[:, axis_idx])
    rel_height = up_sign * (xyz[:, axis_idx] - terminal_h)
    height_proxy = rel_height * scale + args.terminal_agl_m if metric else rel_height

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, facecolor="white")
    axes[0].plot(times, plot_speed, color="#1677b9", lw=1.8)
    axes[0].set_ylabel("m/s" if metric else "scene units/s")
    axes[0].set_title(
        f"Metric Speed/Height Proxy: scale={scale:g} m/unit" if metric else "Relative Speed/Height Proxy"
    )
    axes[1].plot(times, height_proxy, color="#d64b26", lw=1.8)
    axes[1].set_ylabel("height proxy m" if metric else "height proxy units")
    axes[1].set_xlabel("source video time s")
    axes[1].axhline(args.terminal_agl_m if metric else 0.0, color="#999999", lw=0.8, ls="--")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_scene_path(video_dir: Path, rows: list[dict[str, str]], out_path: Path, args: argparse.Namespace) -> None:
    point_cloud = video_dir / "point_cloud.npz"
    if not point_cloud.exists():
        return
    import matplotlib.pyplot as plt

    data = np.load(point_cloud)
    pts = data["pts"].astype(float)
    cols = data["cols"].astype(float) / 255.0
    path = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    if len(pts) == 0 or len(path) == 0:
        return
    pts, cols = crop_points_near_path(pts, cols, path)
    view_pts = to_plot_view(pts)
    view_path = to_plot_view(path)
    attack = np.array([r["is_attack"] == "true" for r in rows])
    groups = group_consecutive_segments(rows)

    bg = "#000000"
    fig = plt.figure(figsize=(11, 7), facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)
    ax.set_axis_off()
    ax.scatter(
        view_pts[:, 0],
        view_pts[:, 1],
        view_pts[:, 2],
        c=cols,
        s=args.scene_point_size,
        alpha=args.scene_point_alpha,
        linewidths=0,
        depthshade=False,
    )
    for start, end in groups:
        seg = view_path[start:end]
        if len(seg) < 2:
            continue
        is_attack = bool(attack[start:end].any())
        color = "#ffb000" if is_attack else "#3fc1ff"
        under = "white" if is_attack else "#a8e7ff"
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=under, lw=4.6 if is_attack else 3.8, alpha=0.45)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=2.7 if is_attack else 2.1, alpha=1.0)
    ax.scatter(
        [view_path[0, 0]],
        [view_path[0, 1]],
        [view_path[0, 2]],
        color="#70e080",
        s=70,
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
    )
    ax.scatter(
        [view_path[-1, 0]],
        [view_path[-1, 1]],
        [view_path[-1, 2]],
        color="#ffb000",
        s=90,
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
    )
    set_equalish_limits(ax, np.vstack([view_pts, view_path]))
    try:
        ax.set_box_aspect((1.15, 1.0, 0.55))
    except Exception:
        pass
    ax.view_init(elev=args.scene_elev, azim=args.scene_azim)
    fig.text(
        0.5,
        0.965,
        f"VGGT Scene Point Cloud + Recovered Camera Path: {video_dir.name}",
        color="#f4f4f4",
        ha="center",
        va="top",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.035,
        "cyan = overlapping flight context, amber = attack segment, green = first frame, amber dot = terminal frame",
        color="#9ba4b4",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.07)
    fig.savefig(out_path, dpi=170, facecolor=bg, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def crop_points_near_path(pts: np.ndarray, cols: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    span = np.maximum(path.max(axis=0) - path.min(axis=0), 1e-6)
    pad = np.maximum(span * 0.65, 0.025)
    lo = path.min(axis=0) - pad
    hi = path.max(axis=0) + pad
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    if mask.sum() < 1500:
        # Fallback: keep points nearest the path bounding-box center.
        center = path.mean(axis=0)
        dist = np.linalg.norm(pts - center, axis=1)
        keep = np.argsort(dist)[: min(len(pts), 80_000)]
        return pts[keep], cols[keep]
    pts = pts[mask]
    cols = cols[mask]
    if len(pts) > 90_000:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(pts), 90_000, replace=False)
        pts = pts[keep]
        cols = cols[keep]
    return pts, cols


def to_plot_view(points: np.ndarray) -> np.ndarray:
    return np.stack([points[:, 0], points[:, 2], points[:, 1]], axis=1)


def group_consecutive_segments(rows: list[dict[str, str]]) -> list[tuple[int, int]]:
    groups = []
    start = 0
    for idx in range(1, len(rows)):
        if rows[idx]["segment_id"] != rows[idx - 1]["segment_id"]:
            groups.append((start, idx))
            start = idx
    groups.append((start, len(rows)))
    return groups


def set_equalish_limits(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    span = np.max(points.max(axis=0) - points.min(axis=0))
    span = max(float(span), 1e-6)
    for axis, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(center[axis] - span * 0.55, center[axis] + span * 0.55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="Parse annotations into flight segment manifests")
    p.add_argument("--annotations", nargs="*", type=Path, default=sorted(ANNOTATION_DIR.glob("*.json")))

    p = sub.add_parser("extract-frames", help="Sample annotated flight frames per video")
    p.add_argument("--recon-subdir", default=DEFAULT_RECON_SUBDIR)
    p.add_argument("--video-cache-dir", type=Path, default=DEFAULT_VIDEO_CACHE)
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--jpeg-quality", type=int, default=3)
    p.add_argument("--download-timeout", type=int, default=240)
    p.add_argument("--ffmpeg-timeout", type=int, default=180)
    p.add_argument("--video-file")
    p.add_argument("--limit", type=int)
    p.add_argument("--refresh", action="store_true")

    p = sub.add_parser("analyze-overlap", help="Detect which earlier flight segments overlap the attack scene")
    p.add_argument("--source-recon-subdir", default=DEFAULT_RECON_SUBDIR)
    p.add_argument("--threshold", type=float, default=0.72)
    p.add_argument("--samples-per-segment", type=int, default=8)
    p.add_argument("--video-id")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("extract-group-frames", help="Build attack-overlap frame groups from overlap decisions")
    p.add_argument("--recon-subdir", default=ATTACK_RECON_SUBDIR)
    p.add_argument("--video-id")
    p.add_argument("--limit", type=int)
    p.add_argument("--refresh", action="store_true")

    p = sub.add_parser("run-vggt", help="Run public VGGT Space on extracted frame sequences")
    p.add_argument("--recon-subdir", default=DEFAULT_RECON_SUBDIR)
    p.add_argument("--space", default="facebook/vggt-omega")
    p.add_argument("--video-id")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-frames", type=int, default=180)
    p.add_argument("--conf-thres", type=float, default=50.0)
    p.add_argument("--max-points-k", type=float, default=1000.0)
    p.add_argument("--vggt-timeout", type=int, default=900)
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument("--backend", choices=["omega", "classic"], default="omega")
    p.add_argument("--upload-mode", choices=["auto", "images", "video"], default="auto")
    p.add_argument("--video-sample-fps", type=float, default=10.0)
    p.add_argument("--mask-sky", action="store_true", help="Enable the Space sky-segmentation mask before reconstruction")
    p.add_argument("--refresh", action="store_true")

    p = sub.add_parser("extract-vggt", help="Parse VGGT .glb files into relative paths")
    p.add_argument("--recon-subdir", default=DEFAULT_RECON_SUBDIR)
    p.add_argument("--video-id")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-points", type=int, default=1_000_000)
    p.add_argument("--refresh", action="store_true")

    p = sub.add_parser("scale-report", help="Create scale method report from relative paths")
    p.add_argument("--recon-subdir", default=DEFAULT_RECON_SUBDIR)
    p.add_argument("--constraints", type=Path)
    p.add_argument("--speed-priors", nargs="*", type=float, default=[])
    p.add_argument("--terminal-agl-m", type=float, default=1.5)
    p.add_argument("--up-axis", choices=["x", "y", "z"], default="y")

    p = sub.add_parser("visualize", help="Render segment/path/speed/height plots")
    p.add_argument("--recon-subdir", default=DEFAULT_RECON_SUBDIR)
    p.add_argument("--terminal-agl-m", type=float, default=1.5)
    p.add_argument("--up-axis", choices=["x", "y", "z"], default="y")
    p.add_argument("--plot-scale-m-per-unit", type=float)
    p.add_argument("--scene-elev", type=float, default=18.0)
    p.add_argument("--scene-azim", type=float, default=-145.0)
    p.add_argument("--scene-point-size", type=float, default=0.35)
    p.add_argument("--scene-point-alpha", type=float, default=0.28)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "manifest":
        write_manifest(load_segments(args.annotations), args.out_dir)
    elif args.command == "extract-frames":
        extract_frames(args)
    elif args.command == "analyze-overlap":
        analyze_overlap(args)
    elif args.command == "extract-group-frames":
        extract_group_frames(args)
    elif args.command == "run-vggt":
        run_vggt(args)
    elif args.command == "extract-vggt":
        extract_vggt(args)
    elif args.command == "scale-report":
        scale_report(args)
    elif args.command == "visualize":
        visualize(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
