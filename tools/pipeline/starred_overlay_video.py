#!/usr/bin/env python3
"""Reprojection overlay video: the 3D reconstruction drawn through each frame's own camera, over the real frame.

Left: BEFORE, the published product (its per-frame overlays as served by the site viewer).
Right: AFTER, the undistorted re-run rendered the same way from its predictions (depth + intrinsics + poses):
       every frame's depth is unprojected into the scene once, then the whole cloud is projected into each frame.
The overlay opacity oscillates so the eye can compare the model against the footage; a correct model sits on the
footage, a wrong one drifts off it. Output: reports/starred_undistort/<video_id>/overlay.mp4 (+ overlay_after/ frames).

  starred_overlay_video.py --only <video_id> ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_before_after import font  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BATCH = __import__("os").environ.get("FPV_UNDISTORT_BATCH", "starred_undistort")  # batch name: benchmarks/<BATCH>, scenes/<BATCH>, reports/<BATCH>
SPEC = ROOT / "benchmarks" / BATCH / "starred_scenes.json"
SCENES = ROOT / "scenes" / BATCH
REPORTS = ROOT / "reports" / BATCH
RED, CYAN = (255, 77, 109), (54, 228, 255)


def load_cloud(pin: Path, stride: int = 2, conf_q: float = 0.5, max_pts: int = 4_000_000):
    """World points from every frame's depth map (raw prediction frame), coloured from the undistorted frames."""
    z = np.load(pin / "runpod_artifacts" / "predictions.npz", allow_pickle=True)
    depth = np.asarray(z["depth"], np.float32)[..., 0]; conf = np.asarray(z["depth_conf"], np.float32)
    K = np.asarray(z["intrinsic"], np.float64); ext = np.asarray(z["extrinsic"], np.float64)
    S, H, W = depth.shape
    frames = sorted((pin / "frames").glob("*.jpg"))
    sky = sorted((pin / "runpod_artifacts" / "sky_masks").glob("*"))
    thr = np.quantile(conf, conf_q)
    pts, cols = [], []
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    for i in range(S):
        d = depth[i, ::stride, ::stride]; c = conf[i, ::stride, ::stride]
        keep = np.isfinite(d) & (d > 0) & (c >= thr)
        if i < len(sky):
            m = np.asarray(Image.open(sky[i]).convert("L").resize((W, H), Image.NEAREST))[::stride, ::stride]
            keep &= m >= 128
        if not keep.any():
            continue
        img = np.asarray(Image.open(frames[i]).convert("RGB").resize((W, H), Image.BILINEAR))[::stride, ::stride]
        fx, fy, cx, cy = K[i, 0, 0], K[i, 1, 1], K[i, 0, 2], K[i, 1, 2]
        x = (xs[keep] + 0.5 - cx) / fx * d[keep]; y = (ys[keep] + 0.5 - cy) / fy * d[keep]
        Pc = np.stack([x, y, d[keep]], 1)
        R, t = ext[i, :3, :3], ext[i, :3, 3]
        pts.append((R.T @ (Pc - t).T).T); cols.append(img[keep])
    P = np.concatenate(pts); C = np.concatenate(cols)
    if len(P) > max_pts:
        k = np.random.default_rng(0).choice(len(P), max_pts, replace=False); P, C = P[k], C[k]
    return P, C, K, ext, (W, H)


def render_frame(P, C, K, w2c, size, out_size, splat=1):
    W, H = size; ow, oh = out_size
    Pc = (w2c[:3, :3] @ P.T).T + w2c[:3, 3]
    zc = Pc[:, 2]; ok = zc > 1e-4
    sx, sy = ow / W, oh / H
    u = (K[0, 0] * Pc[ok, 0] / zc[ok] + K[0, 2]) * sx; v = (K[1, 1] * Pc[ok, 1] / zc[ok] + K[1, 2]) * sy
    z = zc[ok]; col = C[ok]
    inside = (u >= 0) & (u < ow - 1) & (v >= 0) & (v < oh - 1)
    u, v, z, col = u[inside].astype(int), v[inside].astype(int), z[inside], col[inside]
    order = np.argsort(-z); u, v, col = u[order], v[order], col[order]
    canvas = np.zeros((oh, ow, 3), np.uint8); hit = np.zeros((oh, ow), bool)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            uu, vv = np.clip(u + dx, 0, ow - 1), np.clip(v + dy, 0, oh - 1)
            canvas[vv, uu] = col; hit[vv, uu] = True
    return canvas, hit


def blend(actual: np.ndarray, render: np.ndarray, hit: np.ndarray, alpha: float) -> np.ndarray:
    out = actual.astype(np.float32).copy()
    a = alpha * hit[..., None]
    out = out * (1 - 0.65 * a) + render.astype(np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def published_overlay(vid: str, scene: dict, name: str, cdn: str) -> Path:
    d = SCENES / vid / "published" / "overlays"; d.mkdir(exist_ok=True)
    dest = d / f"{name}_overlay.jpg"
    if not dest.exists():
        url = f"{cdn}scenes/{scene['scene_path']}/viewer/camera_view_assets/{name}_overlay.jpg"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fpv/1"}), timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def make(scene: dict, cdn: str, fps: float, period_s: float) -> Path | None:
    vid = scene["video_id"]; base = SCENES / vid; pin = base / "pinhole"
    if not (pin / "runpod_artifacts" / "predictions.npz").exists():
        return None
    out_dir = REPORTS / vid; out_dir.mkdir(parents=True, exist_ok=True)
    P, C, K, ext, size = load_cloud(pin)
    frames = sorted((pin / "frames").glob("*.jpg")); actual_pub = sorted((base / "published" / "images").glob("*.jpg"))
    W, H = Image.open(frames[0]).size
    fr_dir = out_dir / "overlay_frames"; fr_dir.mkdir(exist_ok=True)
    for f in fr_dir.glob("*.png"):
        f.unlink()
    fnt, fnt_s = font(22), font(16)
    for i, fp in enumerate(frames):
        alpha = 0.5 * (1 + np.sin(2 * np.pi * (i / fps) / period_s - np.pi / 2))  # 0 -> 1 -> 0
        # BEFORE: the site's own overlay (rendered at full opacity); fade it against the raw frame
        raw = np.asarray(Image.open(actual_pub[i]).convert("RGB"))
        try:
            pub = np.asarray(Image.open(published_overlay(vid, scene, fp.stem, cdn)).convert("RGB").resize((W, H)))
            before = (raw * (1 - alpha) + pub * alpha).clip(0, 255).astype(np.uint8)
        except Exception:
            before = raw
        # AFTER: our render through this frame's predicted camera, over the undistorted frame
        act = np.asarray(Image.open(fp).convert("RGB"))
        w2c = np.vstack([ext[i], [0, 0, 0, 1]])
        render, hit = render_frame(P, C, K[i], w2c, size, (W, H))
        after = blend(act, render, hit, alpha)
        panels = []
        for arr, tag, col in ((before, "BEFORE  published product", RED), (after, "AFTER  undistorted to pinhole", CYAN)):
            im = Image.fromarray(arr).resize((W * 2, H * 2), Image.BILINEAR); d = ImageDraw.Draw(im)
            d.rounded_rectangle([10, 10, 10 + 20 + d.textlength(tag, font=fnt), 44], radius=6, fill=(12, 13, 15)); d.text((20, 15), tag, fill=col, font=fnt)
            panels.append(im)
        cv = Image.new("RGB", (W * 4 + 8, H * 2 + 60), (12, 13, 15)); cv.paste(panels[0], (0, 0)); cv.paste(panels[1], (W * 2 + 8, 0))
        d = ImageDraw.Draw(cv)
        d.text((12, H * 2 + 10), f"{scene['title'][:80]}   frame {i + 1}/{len(frames)}   overlay {int(alpha * 100):3d}%", fill=(200, 208, 216), font=fnt_s)
        d.text((12, H * 2 + 34), "3D reconstruction re-projected through each frame's own camera over the footage. Same VGGT-Omega model; the only change is undistorting the frames first.", fill=(139, 149, 161), font=fnt_s)
        cv.save(fr_dir / f"{i + 1:05d}.png")
    out = out_dir / "overlay.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(fr_dir / "%05d.png"), "-r", "24", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    for f in fr_dir.glob("*.png"):
        f.unlink()
    fr_dir.rmdir()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--fps", type=float, default=10.0, help="published sample rate of the frames")
    ap.add_argument("--period-s", type=float, default=4.0, help="overlay fade in/out period")
    args = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    for s in spec["scenes"]:
        if args.only and s["video_id"] not in args.only:
            continue
        try:
            out = make(s, spec["cdn_base"], args.fps, args.period_s)
            print(f"{s['video_id']}: {'wrote ' + str(out) if out else 'no pinhole run yet'}", flush=True)
        except Exception as exc:
            print(f"{s['video_id']}: FAILED {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
