#!/usr/bin/env python3
"""Camera-view overlays for the standard scene viewer: real frames with the 3D model drawn over them.

The site viewer's camera-view mode shows, per frame, the actual frame, the reconstruction rendered
through that frame's camera, and the two blended (keys actual_image / render_image / overlay_image in
scene_meta.json). This script gives every starred scene both versions in the local viewer:

  published/viewer/   the published product, with the site's own per-frame renders and overlays (downloaded)
  pinhole/viewer/     the undistorted re-run, with renders and overlays generated from its predictions

Open them side by side with tools/local_scene_viewer_server.py:
  http://127.0.0.1:8766/scenes/starred_undistort/<video_id>/published/viewer/
  http://127.0.0.1:8766/scenes/starred_undistort/<video_id>/pinhole/viewer/

  starred_camera_overlays.py [--only <video_id> ...] [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from starred_overlay_video import load_cloud, render_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "benchmarks" / "starred_undistort" / "starred_scenes.json"
SCENES = ROOT / "scenes" / "starred_undistort"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fpv/1"}), timeout=60) as r:
        dest.write_bytes(r.read())


def published_viewer(scene: dict, cdn: str, force: bool) -> Path:
    """Assemble published/viewer/ from the downloaded product plus the site's per-frame renders and overlays."""
    base = SCENES / scene["video_id"]; pub = base / "published"; vd = pub / "viewer"
    meta_src = json.loads((pub / "scene_meta.json").read_text())
    if (vd / "scene_meta.json").exists() and not force and all((vd / e["overlay_image"]).exists() for e in meta_src["path"] if e.get("overlay_image")):
        return vd
    vd.mkdir(exist_ok=True)
    for k in ("positions", "colors"):
        src = pub / Path(meta_src["assets"][k]).name; dst = vd / src.name
        if not dst.exists():
            try:
                dst.hardlink_to(src)
            except Exception:
                shutil.copy2(src, dst)
    assets = vd / "camera_view_assets"; assets.mkdir(exist_ok=True)
    viewer_url = f"{cdn}scenes/{scene['scene_path']}/viewer/"
    jobs = []
    for e in meta_src["path"]:
        name = Path(e["frame_image"]).name
        actual = assets / name
        if not actual.exists():
            try:
                actual.hardlink_to(pub / "images" / name)
            except Exception:
                shutil.copy2(pub / "images" / name, actual)
        for key in ("render_image", "overlay_image"):
            rel = e.get(key)
            if rel:
                jobs.append((viewer_url + rel, vd / rel))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))
    meta = dict(meta_src)
    meta["title"] = f"{scene['title']} - published product (raw frames)"
    meta["source_label"] = "published product"
    meta["scene_state_url"] = ""
    (vd / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    return vd


def pinhole_overlays(scene: dict, force: bool) -> Path | None:
    """Render per-frame renders and overlays for the undistorted run and register them in its viewer meta."""
    pin = SCENES / scene["video_id"] / "pinhole"; vd = pin / "viewer"
    if not (vd / "scene_meta.json").exists() or not (pin / "runpod_artifacts" / "predictions.npz").exists():
        return None
    meta = json.loads((vd / "scene_meta.json").read_text())
    if not force and all(e.get("overlay_image") and (vd / e["overlay_image"]).exists() for e in meta["path"]):
        return vd
    P, C, K, ext, size = load_cloud(pin)
    frames = sorted((pin / "frames").glob("*.jpg"))
    assets = vd / "camera_view_assets"; assets.mkdir(exist_ok=True)
    by_name = {Path(f).name: f for f in frames}
    for i, e in enumerate(meta["path"]):
        name = Path(e["frame_image"]).name
        fp = by_name.get(name)
        if fp is None or i >= len(ext):
            continue
        act = np.asarray(Image.open(fp).convert("RGB")); H, W = act.shape[:2]
        w2c = np.vstack([ext[i], [0, 0, 0, 1]])
        render, hit = render_frame(P, C, K[i], w2c, size, (W, H))
        overlay = np.asarray(Image.blend(Image.fromarray(act), Image.fromarray(render), 0.5)).copy()
        overlay[hit] = (0.45 * act[hit] + 0.75 * render[hit]).clip(0, 255).astype(np.uint8)  # same blend as the site pipeline
        stem = Path(name).stem
        Image.fromarray(render).save(assets / f"{stem}_vggt_render.jpg", quality=90)
        Image.fromarray(overlay).save(assets / f"{stem}_overlay.jpg", quality=90)
        e["render_image"] = f"camera_view_assets/{stem}_vggt_render.jpg"
        e["overlay_image"] = f"camera_view_assets/{stem}_overlay.jpg"
        e.setdefault("actual_image", e["frame_image"])
    meta["title"] = f"{scene['title']} - undistorted to pinhole"
    (vd / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    return vd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    for s in spec["scenes"]:
        if args.only and s["video_id"] not in args.only:
            continue
        try:
            pv = published_viewer(s, spec["cdn_base"], args.force)
            uv = pinhole_overlays(s, args.force)
            print(f"{s['video_id']}: published viewer {'ok' if pv else '-'}, undistorted overlays {'ok' if uv else 'no run yet'}", flush=True)
        except Exception as exc:
            print(f"{s['video_id']}: FAILED {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
