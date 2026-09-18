#!/usr/bin/env python3
"""Stage the published starred scenes locally for the undistort-to-pinhole re-run.

For every scene in benchmarks/starred_undistort/starred_scenes.json this downloads
the published viewer (scene_meta.json, the exact frames that fed the published
VGGT-Omega run, and the published point buffers) into

  scenes/starred_undistort/<video_id>/published/{scene_meta.json, images/, points_*.bin}

and writes frames.csv / metadata.json for the later pinhole run under
  scenes/starred_undistort/<video_id>/pinhole/

Nothing here touches a GPU. Re-runs are incremental (existing files are kept
unless --refresh).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = __import__("os").environ.get("FPV_UNDISTORT_BATCH", "starred_undistort")  # batch name: benchmarks/<BATCH>, scenes/<BATCH>, reports/<BATCH>
SPEC = ROOT / "benchmarks" / BATCH / "starred_scenes.json"
OUT = ROOT / "scenes" / BATCH


def download(url: str, dest: Path, refresh: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "fpv-starred-undistort/1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest.with_suffix(dest.suffix + ".part"), "wb") as f:
        shutil.copyfileobj(r, f)
    dest.with_suffix(dest.suffix + ".part").replace(dest)


def expand_gzip(path: Path) -> None:
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        tmp = path.with_suffix(path.suffix + ".raw")
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(path)


def stage(scene: dict, cdn: str, refresh: bool) -> dict:
    vid = scene["video_id"]
    base = OUT / vid
    pub = base / "published"
    viewer_url = urllib.parse.urljoin(cdn, f"scenes/{scene['scene_path']}/viewer/")
    download(viewer_url + "scene_meta.json", pub / "scene_meta.json", refresh)
    meta = json.loads((pub / "scene_meta.json").read_text())
    if meta["scene_id"] != scene["scene_id"]:
        raise SystemExit(f"{vid}: scene id mismatch {meta['scene_id']} vs {scene['scene_id']}")
    entries = meta["path"]
    for e in entries:
        rel = e["frame_image"]
        download(viewer_url + rel, pub / "images" / Path(rel).name, refresh)
    for k in ("positions", "colors"):
        rel = meta["assets"][k]
        dest = pub / Path(rel).name
        download(viewer_url + rel, dest, refresh)
        expand_gzip(dest)
    # frames.csv + metadata.json for the pinhole run (same frame order as the published path)
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
    md = {"scene_id": f"{vid}_pinhole", "source_scene": scene["scene_id"], "published_scene_path": scene["scene_path"],
          "frame_count_target": len(entries), "default_scale_m_per_unit": meta.get("default_scale_m_per_unit"),
          "calibration": meta.get("calibration"), "sample_fps": meta.get("sample_fps"),
          "note": "published clean-crop frames, lens self-calibrated (GLOMAP OPENCV_FISHEYE) and remapped to a pinhole before VGGT-Omega"}
    (pin / "metadata.json").write_text(json.dumps(md, indent=2))
    from PIL import Image
    w_, h_ = Image.open(pub / "images" / Path(entries[0]["frame_image"]).name).size
    return {"video_id": vid, "frames": len(entries), "image_size": [w_, h_], "scale": meta.get("default_scale_m_per_unit")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="video ids to stage (default all)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    scenes = [s for s in spec["scenes"] if not args.only or s["video_id"] in args.only]
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda s: stage(s, spec["cdn_base"], args.refresh), scenes))
    for r in results:
        print(f"{r['frames']:4d} frames {r['image_size'][0]}x{r['image_size'][1]} scale {r['scale']}  {r['video_id']}")
    (OUT / "staged.json").write_text(json.dumps(results, indent=2))
    print("staged", len(results), "scenes under", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
