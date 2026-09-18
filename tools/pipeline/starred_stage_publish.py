#!/usr/bin/env python3
"""Stage the improved (undistorted) reconstructions as replacements for the published scenes, and publish them.

For every selected scene (benchmarks/starred_undistort/publish_selection.json) this writes the viewer directory
the site consumes, at the published scene's own key:

  scenes/<video_id>/<scene_id>/viewer/
      scene_meta.json                      title tagged "lens-corrected", pipeline block, metric scale carried over
      points_positions.<tag>.bin/colors    point buffers (versioned names: the old ones are cached as immutable)
      camera_view_assets_<tag>/            actual / render / overlay frames of the new run (versioned directory)

The old product stays intact locally under scenes/starred_undistort/<video_id>/published/ (and on S3 under its
old asset names). --publish syncs only these directories to the bucket; nothing else in scenes/ is touched.

  starred_stage_publish.py                       # stage only
  AWS_PROFILE=admin starred_stage_publish.py --publish
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = __import__("os").environ.get("FPV_UNDISTORT_BATCH", "starred_undistort")  # batch name: benchmarks/<BATCH>, scenes/<BATCH>, reports/<BATCH>
SPEC = ROOT / "benchmarks" / BATCH / "starred_scenes.json"
SEL = ROOT / "benchmarks" / BATCH / "publish_selection.json"
SRC = ROOT / "scenes" / BATCH
BUCKET = os.environ.get("FPV_BUCKET", "s3://fpv-drone-strikes-lebanon-dataset")
TAG = "lenscorr1"


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def stage(scene: dict) -> tuple[Path, dict]:
    vid = scene["video_id"]; base = SRC / vid
    pin_v = base / "pinhole" / "viewer"; pub_meta = json.loads((base / "published" / "scene_meta.json").read_text())
    meta = json.loads((pin_v / "scene_meta.json").read_text())
    metrics = json.loads((base / "metrics.json").read_text())
    out = ROOT / "scenes" / vid / scene["scene_id"] / "viewer"
    out.mkdir(parents=True, exist_ok=True)
    # point buffers under versioned names
    for k in ("positions", "colors"):
        src = pin_v / Path(meta["assets"][k]).name; name = f"points_{k}.{TAG}.bin"
        link(src, out / name); meta["assets"][k] = name
    # frames: actual (undistorted input), render, overlay
    cva = out / f"camera_view_assets_{TAG}"
    for e in meta["path"]:
        for key in ("frame_image", "actual_image", "render_image", "overlay_image"):
            rel = e.get(key)
            if not rel:
                continue
            src = pin_v / rel; new_rel = f"camera_view_assets_{TAG}/{Path(rel).name}"
            link(src, out / new_rel); e[key] = new_rel
    # metric scale: metres per new unit = published metres per unit x (new -> published Sim(3) scale)
    s = metrics["after_to_before_alignment"]["scale_to_reference"]
    pub_scale = pub_meta.get("default_scale_m_per_unit") or 117.6
    meta["default_scale_m_per_unit"] = float(pub_scale) * float(s)
    if pub_meta.get("calibration"):
        cal = dict(pub_meta["calibration"]); cal["scale_m_per_vggt_unit"] = float(pub_scale) * float(s)
        cal["note"] = "manual calibration of the published product transferred through the camera-path Sim(3) to the lens-corrected run"
        meta["calibration"] = cal
    meta["title"] = f"{scene['title']} - VGGT scene (lens-corrected)"
    meta["source_label"] = f"{scene['scene_id']} lens-corrected"
    meta["video_dir"] = ""; meta["scene_state_url"] = pub_meta.get("scene_state_url", "")
    meta["scene_id"] = scene["scene_id"]
    b, a = metrics.get("before_vs_reference", {}), metrics.get("after_vs_reference", {})
    meta["pipeline"] = {
        "tag": TAG, "lens": "per-video self-calibration (GLOMAP OPENCV_FISHEYE) and remap to a pinhole before VGGT-Omega",
        "replaced_published_product_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "path_disagreement_vs_sfm": {"published": b.get("rmse_fraction_of_path_length"), "lens_corrected": a.get("rmse_fraction_of_path_length")},
        "local_scale_std": {"published": b.get("local_scale_std"), "lens_corrected": a.get("local_scale_std")},
        "focal_ratio_vggt_over_calibrated": metrics.get("after_focal", {}).get("fx_ratio_vggt_over_reference"),
        "calibration_camera": metrics.get("calibration_camera"),
    }
    (out / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    return out, meta


def publish(out: Path) -> None:
    rel = out.relative_to(ROOT).as_posix()
    run = lambda *a: subprocess.run(["aws", "s3", "sync", f"{out}/", f"{BUCKET}/{rel}/", *a], check=True)
    run("--exclude", "*", "--include", "*.bin", "--content-type", "application/octet-stream", "--cache-control", "public,max-age=31536000")
    subprocess.run(["aws", "s3", "sync", f"{out}/camera_view_assets_{TAG}/", f"{BUCKET}/{rel}/camera_view_assets_{TAG}/",
                    "--content-type", "image/jpeg", "--cache-control", "public,max-age=31536000,immutable"], check=True)
    # the meta is the commit point: assets are in place before readers can see the new manifest
    run("--exclude", "*", "--include", "scene_meta.json", "--content-type", "application/json", "--cache-control", "public,max-age=300")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    spec = {s["video_id"]: s for s in json.loads(SPEC.read_text())["scenes"]}
    selected = json.loads(SEL.read_text())["selected"]
    record = {"tag": TAG, "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "published": args.publish, "scenes": []}
    for vid in selected:
        if args.only and vid not in args.only:
            continue
        out, meta = stage(spec[vid])
        if args.publish:
            publish(out)
        record["scenes"].append({"video_id": vid, "scene_id": spec[vid]["scene_id"], "viewer": out.relative_to(ROOT).as_posix(),
                                 "default_scale_m_per_unit": meta["default_scale_m_per_unit"], "pipeline": meta["pipeline"]})
        print(f"{'published' if args.publish else 'staged'} {vid} -> {out.relative_to(ROOT)}  scale {meta['default_scale_m_per_unit']:.1f} m/unit", flush=True)
    (ROOT / "benchmarks" / BATCH / "published_replacements.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
