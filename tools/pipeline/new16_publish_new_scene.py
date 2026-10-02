#!/usr/bin/env python3
"""Publish a new16 reconstruction as a BRAND-NEW site scene (no prior published product to replace).

The dataset site discovers 3D scenes by scanning scenes/<video_id>/<scene_id>/viewer/scene_meta.json
(tools/publishing/build_web_data.mjs). For a genuinely new scene there is no before/after Sim(3) transfer
and no manual metric scale, so we stage the lens-corrected pinhole/viewer directory as-is (hardlinked, no
extra disk) at the site key and only fix the site-facing meta fields. Real-world scale/orientation is set
later by the DTM/orthophoto ground registration.

  FPV_UNDISTORT_BATCH=new16_undistort python3 tools/pipeline/new16_publish_new_scene.py --only <video_id> ...
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = os.environ.get("FPV_UNDISTORT_BATCH", "new16_undistort")
SPEC = ROOT / "benchmarks" / BATCH / "starred_scenes.json"
SRC = ROOT / "scenes" / BATCH


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def stage_new(scene: dict) -> Path:
    vid, sid = scene["video_id"], scene["scene_id"]
    pin_v = SRC / vid / "pinhole" / "viewer"
    meta = json.loads((pin_v / "scene_meta.json").read_text())
    out = ROOT / "scenes" / vid / sid / "viewer"
    out.mkdir(parents=True, exist_ok=True)
    # point buffers + every referenced camera_view_assets frame, hardlinked
    for name in (Path(meta["assets"]["positions"]).name, Path(meta["assets"]["colors"]).name):
        link(pin_v / name, out / name)
    for e in meta["path"]:
        for key in ("frame_image", "actual_image", "render_image", "overlay_image"):
            rel = e.get(key)
            if rel:
                link(pin_v / rel, out / rel)
    # site-facing fields
    meta["scene_id"] = sid
    meta["source_label"] = sid
    meta["video_dir"] = str(ROOT / "scenes" / vid / sid)
    meta["scene_state_url"] = f"/api/scenes/{sid}/state"
    meta["title"] = f"{scene['title']} - VGGT scene"  # drop the "(undistorted to pinhole)" working label
    (out / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    spec = {s["video_id"]: s for s in json.loads(SPEC.read_text())["scenes"]}
    targets = args.only or list(spec)
    for vid in targets:
        out = stage_new(spec[vid])
        n = len(json.loads((out / "scene_meta.json").read_text())["path"])
        print(f"staged NEW scene {vid} -> {out.relative_to(ROOT)}  ({n} frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
