#!/usr/bin/env python3
"""Remap the synthetic flyover's equisolid fisheye frames (and GT depth) to a pinhole camera.

Output pinhole: same 800x600 grid, fx = fy = W/2 (90 degree horizontal FOV,
73.7 degree vertical), principal point at the centre. Everything inside that
cone is sampled from the fisheye image with bilinear interpolation; the GT
depth is resampled (nearest) and converted from ray length to camera-z for
the pinhole rays. Writes:
  <scene>/frames_pinhole/f_NNNNNN.jpg
  <scene>/ground_truth/gt_depth_pinhole/NNNNN.npy
  <scene>/ground_truth/gt_cameras_pinhole.json  (same poses, pinhole intrinsics)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_dir", type=Path)
    p.add_argument("--hfov-deg", type=float, default=90.0)
    args = p.parse_args()
    sd = args.scene_dir
    gt = json.loads((sd / "ground_truth" / "gt_cameras.json").read_text())
    intr = gt["intrinsics"]
    W, H = intr["resolution_px"]
    f_mm = intr["focal_length_mm"]; k = W / intr["sensor_width_mm"]  # px per mm on the fisheye sensor
    fx = (W / 2) / np.tan(np.radians(args.hfov_deg) / 2); fy = fx; cx, cy = W / 2, H / 2
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dx, dy, dz = (u - cx) / fx, (v - cy) / fy, np.ones_like(u)
    n = np.sqrt(dx**2 + dy**2 + dz**2); dx, dy, dz = dx / n, dy / n, dz / n
    theta = np.arccos(np.clip(dz, -1, 1)); phi = np.arctan2(dy, dx)
    r = 2 * f_mm * np.sin(theta / 2) * k
    su = W / 2 + r * np.cos(phi) - 0.5; sv = H / 2 + r * np.sin(phi) - 0.5  # source pixel coords (array index space)
    inside = (su >= 0) & (su <= W - 1) & (sv >= 0) & (sv <= H - 1)
    out_frames = sd / "frames_pinhole"; out_frames.mkdir(exist_ok=True)
    out_depth = sd / "ground_truth" / "gt_depth_pinhole"; out_depth.mkdir(exist_ok=True)
    frames = sorted((sd / "frames").glob("f_*.jpg"))
    for fp in frames:
        img = np.asarray(Image.open(fp).convert("RGB")).astype(np.float32)
        out = np.stack([map_coordinates(img[..., c], [sv, su], order=1, mode="nearest") for c in range(3)], axis=-1)
        out[~inside] = 0
        Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_frames / fp.name, quality=95)
        fid = int(fp.stem.split("_")[1])
        dp = sd / "ground_truth" / "gt_depth" / f"{fid:05d}.npy"
        if dp.exists():
            ray = np.load(dp)
            ray_s = map_coordinates(np.nan_to_num(ray, nan=-1.0), [sv, su], order=0, mode="nearest")
            z = np.where((ray_s > 0) & inside, ray_s * dz, np.nan).astype(np.float32)
            np.save(out_depth / f"{fid:05d}.npy", z)
    pin = dict(gt)
    pin["intrinsics"] = {"model": "pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy, "resolution_px": [W, H],
                         "hfov_deg": args.hfov_deg, "vfov_deg": float(np.degrees(2 * np.arctan((H / 2) / fy))),
                         "source": "remapped from the equisolid fisheye frames; pixels outside the fisheye image are black",
                         "valid_fraction": float(inside.mean())}
    (sd / "ground_truth" / "gt_cameras_pinhole.json").write_text(json.dumps(pin, indent=2))
    print(f"wrote {len(frames)} pinhole frames, fx={fx:.1f}, valid fraction {inside.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
