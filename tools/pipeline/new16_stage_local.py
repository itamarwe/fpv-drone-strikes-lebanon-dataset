#!/usr/bin/env python3
"""Stage locally-produced base scenes into the undistort batch layout.

The undistort batch (starred_run_batch.py / starred_prepare.py) normally downloads
each scene's *published* viewer from CloudFront. For brand-new scenes there is no
published product yet: they have only been run through the base pipeline locally
(run_vggt_batch_from_annotations.py -> :8766), which writes

    scenes/<slug(video_file)>/<scene_id>/viewer/{scene_meta.json, <frames>, <point bins>}

This script mirrors starred_prepare.stage() but copies from that local base output
instead of downloading, producing

    scenes/<BATCH>/<video_id>/published/{scene_meta.json, images/, positions.bin, colors.bin}
    scenes/<BATCH>/<video_id>/pinhole/{frames.csv, metadata.json}

and a benchmarks/<BATCH>/starred_scenes.json spec the batch driver consumes.

Usage:
    FPV_UNDISTORT_BATCH=new16_undistort \
    python3 tools/pipeline/new16_stage_local.py \
        --base-results benchmarks/new16_undistort/base_results.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import os

BATCH = os.environ.get("FPV_UNDISTORT_BATCH", "new16_undistort")
OUT = ROOT / "scenes" / BATCH
SPEC = ROOT / "benchmarks" / BATCH / "starred_scenes.json"


def slugify(value: str) -> str:
    value = Path(value).stem if value else "video"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return value[:120] or "video"


def copy_raw(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def copy_expand_gzip(src: Path, dest: Path) -> None:
    """Copy a point-buffer, decompressing if it is gzip (CloudFront serves .gz; local may be raw)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(src, "rb") as r, open(dest, "wb") as w:
            shutil.copyfileobj(r, w)
    else:
        shutil.copyfile(src, dest)


def stage_scene(video_id: str, scene_id: str, viewer_dir: Path, title: str) -> dict:
    if not (viewer_dir / "scene_meta.json").exists():
        raise FileNotFoundError(f"{video_id}: no scene_meta.json under {viewer_dir}")
    base = OUT / video_id
    pub = base / "published"
    pub.mkdir(parents=True, exist_ok=True)
    copy_raw(viewer_dir / "scene_meta.json", pub / "scene_meta.json")
    meta = json.loads((pub / "scene_meta.json").read_text())
    entries = meta["path"]
    for e in entries:
        rel = e["frame_image"]
        src = (viewer_dir / rel)
        copy_raw(src, pub / "images" / Path(rel).name)
    for k in ("positions", "colors"):
        rel = meta["assets"][k]
        copy_expand_gzip(viewer_dir / rel, pub / Path(rel).name)

    # frames.csv + metadata.json for the pinhole run (same frame order as the base path)
    pin = base / "pinhole"
    pin.mkdir(parents=True, exist_ok=True)
    cols = ["frame_index", "file", "video_file", "video_time_s", "segment_time_s", "sequence_time_s", "segment_id", "segment_index", "is_attack"]
    with (pin / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in entries:
            w.writerow({"frame_index": e["frame"], "file": Path(e["frame_image"]).name, "video_file": e.get("video_file", ""),
                        "video_time_s": e.get("video_time_s", ""), "segment_time_s": e.get("segment_time_s", ""),
                        "sequence_time_s": e.get("sequence_time_s", ""), "segment_id": e.get("segment_id", ""),
                        "segment_index": e.get("segment_index", ""), "is_attack": str(e.get("is_attack", "")).lower()})
    md = {"scene_id": f"{video_id}_pinhole", "source_scene": scene_id, "published_scene_path": f"{video_id}/{scene_id}",
          "frame_count_target": len(entries), "default_scale_m_per_unit": meta.get("default_scale_m_per_unit"),
          "calibration": meta.get("calibration"), "sample_fps": meta.get("sample_fps"),
          "note": "local base-pipeline clean-crop frames, lens self-calibrated (GLOMAP OPENCV_FISHEYE) and remapped to a pinhole before VGGT-Omega"}
    (pin / "metadata.json").write_text(json.dumps(md, indent=2))

    from PIL import Image
    w_, h_ = Image.open(pub / "images" / Path(entries[0]["frame_image"]).name).size
    return {
        "video_id": video_id,
        "scene_id": meta["scene_id"],
        "scene_path": f"{video_id}/{scene_id}",
        "title": title,
        "published_frames": len(entries),
        "published_scale_m_per_unit": meta.get("default_scale_m_per_unit") or 117.6,
        "published_calibration": meta.get("calibration"),
        "starred": False,
        "image_size": [w_, h_],
    }


def _write_pinhole_meta(base: Path, video_id: str, scene_id: str, rows: list[dict], scale, sample_fps, calibration) -> None:
    pin = base / "pinhole"
    pin.mkdir(parents=True, exist_ok=True)
    cols = ["frame_index", "file", "video_file", "video_time_s", "segment_time_s", "sequence_time_s", "segment_id", "segment_index", "is_attack"]
    with (pin / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            # rows are path dicts keyed frame/frame_image; frames.csv needs frame_index/file.
            out = {c: r.get(c, "") for c in cols}
            out["frame_index"] = r.get("frame_index", r.get("frame", ""))
            out["file"] = r.get("file") or Path(r.get("frame_image", "")).name
            w.writerow(out)
    md = {"scene_id": f"{video_id}_pinhole", "source_scene": scene_id, "published_scene_path": f"{video_id}/{scene_id}",
          "frame_count_target": len(rows), "default_scale_m_per_unit": scale,
          "calibration": calibration, "sample_fps": sample_fps,
          "note": "local base-pipeline clean-crop frames, lens self-calibrated (GLOMAP OPENCV_FISHEYE) and remapped to a pinhole before VGGT-Omega"}
    (pin / "metadata.json").write_text(json.dumps(md, indent=2))


def stage_scene_from_frames(video_id: str, scene_id: str, scene_dir: Path, title: str) -> dict:
    """Stage a scene that has extracted frames + frames.csv but no VGGT viewer (base VGGT was skipped).

    No point cloud / before reconstruction: the undistortion pipeline is the standard and there is no
    before/after comparison, so only the frames and per-frame labels are needed."""
    fcsv = scene_dir / "frames.csv"
    fdir = scene_dir / "frames"
    if not (fcsv.exists() and fdir.is_dir()):
        raise FileNotFoundError(f"{video_id}: no frames.csv/frames under {scene_dir}")
    rows = list(csv.DictReader(fcsv.open()))
    if not rows:
        raise ValueError(f"{video_id}: empty frames.csv")
    smeta = {}
    mpath = scene_dir / "metadata.json"
    if mpath.exists():
        smeta = json.loads(mpath.read_text())
    sample_fps = smeta.get("sample_fps", 10.0)
    scale = 117.6
    base = OUT / video_id
    pub = base / "published"
    (pub / "images").mkdir(parents=True, exist_ok=True)
    path = []
    for r in rows:
        name = r["file"]
        copy_raw(fdir / name, pub / "images" / name)
        path.append({
            "frame": int(r.get("frame_index") or 0),
            "frame_image": f"images/{name}",
            "video_file": r.get("video_file", ""),
            "video_time_s": r.get("video_time_s", ""),
            "segment_time_s": r.get("segment_time_s", ""),
            "sequence_time_s": r.get("sequence_time_s", ""),
            "segment_id": r.get("segment_id", ""),
            "segment_index": r.get("segment_index", ""),
            "is_attack": str(r.get("is_attack", "")).lower() == "true",
        })
    scene_meta = {"scene_id": scene_id, "default_scale_m_per_unit": scale, "sample_fps": sample_fps,
                  "calibration": None, "path": path, "assets": {}}
    (pub / "scene_meta.json").write_text(json.dumps(scene_meta, indent=2))
    _write_pinhole_meta(base, video_id, scene_id, path, scale, sample_fps, None)
    from PIL import Image
    w_, h_ = Image.open(pub / "images" / rows[0]["file"]).size
    return {
        "video_id": video_id, "scene_id": scene_id, "scene_path": f"{video_id}/{scene_id}",
        "title": title, "published_frames": len(rows), "published_scale_m_per_unit": scale,
        "published_calibration": None, "starred": False, "image_size": [w_, h_],
    }


def find_scene_dir(video_id: str) -> tuple[str, Path] | None:
    root = ROOT / "scenes" / video_id
    if not root.is_dir():
        return None
    subs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(video_id)]
    if not subs:
        return None
    # prefer one with a viewer/scene_meta.json, else the first
    for p in subs:
        if (p / "viewer" / "scene_meta.json").exists():
            return p.name, p
    return subs[0].name, subs[0]


def annotation_title(video_file: str) -> str:
    ann = ROOT / "annotations" / f"{Path(video_file).stem}_annotations.json"
    if ann.exists():
        d = json.loads(ann.read_text())
        return d.get("description") or Path(video_file).stem
    return Path(video_file).stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation-list", type=Path, default=ROOT / "benchmarks" / BATCH / "annotation_list.txt",
                    help="text file of annotation paths, one per line (the scenes to stage)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these video_ids")
    args = ap.parse_args()

    ann_paths = [Path(line.strip()) for line in args.annotation_list.read_text().splitlines() if line.strip()]
    scenes: list[dict] = []
    skipped: list[str] = []
    for ann_path in ann_paths:
        ann = json.loads((ROOT / ann_path).read_text())
        video_file = ann["video_file"]
        video_id = slugify(video_file)
        if args.only and video_id not in args.only:
            continue
        found = find_scene_dir(video_id)
        if not found:
            skipped.append(f"{video_id} (no scene dir)")
            print(f"SKIP {video_id}: no scene dir", flush=True)
            continue
        scene_id, scene_dir = found
        title = annotation_title(video_file)
        try:
            if (scene_dir / "viewer" / "scene_meta.json").exists():
                entry = stage_scene(video_id, scene_id, scene_dir / "viewer", title)
                mode = "viewer"
            else:
                entry = stage_scene_from_frames(video_id, scene_id, scene_dir, title)
                mode = "frames-only"
            scenes.append(entry)
            print(f"staged {video_id} [{mode}]: {entry['published_frames']} frames {entry['image_size']}", flush=True)
        except Exception as exc:
            skipped.append(f"{scene_id} (stage error: {exc})")
            print(f"SKIP {video_id}: {exc}", flush=True)

    spec = {
        "schema_version": 1,
        "source": "local base pipeline (run_vggt_batch_from_annotations.py); staged by new16_stage_local.py",
        "cdn_base": None,
        "scenes": scenes,
    }
    SPEC.parent.mkdir(parents=True, exist_ok=True)
    SPEC.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"\nwrote {SPEC} with {len(scenes)} scene(s)")
    if skipped:
        print("skipped:")
        for s in skipped:
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
