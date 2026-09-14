#!/usr/bin/env python3
"""Remap real FPV frames to a pinhole camera using a COLMAP-calibrated camera model.

Supported COLMAP models (cameras.txt): SIMPLE_RADIAL, RADIAL, OPENCV,
SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, OPENCV_FISHEYE. The output pinhole keeps
the calibrated focal length and image size with the principal point at the
centre, so nothing is cropped for barrel-distorted input; pixels whose source
falls outside the frame are black.

  undistort_real_scene.py --cameras cameras.txt --input frames/ --output frames_pinhole/ [--scale 1.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def read_camera(path: Path) -> dict:
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        return {"model": parts[1], "width": int(parts[2]), "height": int(parts[3]), "params": [float(x) for x in parts[4:]]}
    raise ValueError(f"no camera in {path}")


def intrinsics(cam: dict) -> tuple[float, float, float, float, list[float]]:
    m, p = cam["model"], cam["params"]
    if m in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        return p[0], p[0], p[1], p[2], p[3:]
    if m in ("RADIAL", "RADIAL_FISHEYE"):
        return p[0], p[0], p[1], p[2], p[3:]
    if m in ("OPENCV", "OPENCV_FISHEYE"):
        return p[0], p[1], p[2], p[3], p[4:]
    raise ValueError(f"unsupported model {m}")


def distort(model: str, x: np.ndarray, y: np.ndarray, k: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Map normalised pinhole coordinates (x, y) = (X/Z, Y/Z) to distorted normalised coordinates."""
    if model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
        r2 = x * x + y * y
        k1 = k[0]; k2 = k[1] if len(k) > 1 else 0.0
        radial = 1 + k1 * r2 + k2 * r2 * r2
        xd, yd = x * radial, y * radial
        if model == "OPENCV":
            p1, p2 = k[2], k[3]
            xd = xd + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = yd + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd
    # fisheye family: theta-based
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    if model == "SIMPLE_RADIAL_FISHEYE":
        thetad = theta * (1 + k[0] * theta**2)
    elif model == "RADIAL_FISHEYE":
        thetad = theta * (1 + k[0] * theta**2 + k[1] * theta**4)
    elif model == "OPENCV_FISHEYE":
        k1, k2, k3, k4 = k
        thetad = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    else:
        raise ValueError(model)
    scale = np.where(r > 1e-9, thetad / np.maximum(r, 1e-9), 1.0)
    return x * scale, y * scale


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cameras", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fov-scale", type=float, default=1.0, help="multiply the output focal length (>1 narrows the FOV)")
    p.add_argument("--keep-principal-point", action="store_true", help="keep the calibrated principal point in the output (content stays in place; use when it is off-centre)")
    args = p.parse_args()
    cam = read_camera(args.cameras)
    fx, fy, cx, cy, k = intrinsics(cam)
    W, H = cam["width"], cam["height"]
    ofx, ofy = fx * args.fov_scale, fy * args.fov_scale
    ocx, ocy = (cx, cy) if args.keep_principal_point else (W / 2, H / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x, y = (u - ocx) / ofx, (v - ocy) / ofy
    xd, yd = distort(cam["model"], x, y, k)
    su, sv = fx * xd + cx - 0.5, fy * yd + cy - 0.5
    inside = (su >= 0) & (su <= W - 1) & (sv >= 0) & (sv <= H - 1)
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(q for q in args.input.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for fp in files:
        img = np.asarray(Image.open(fp).convert("RGB")).astype(np.float32)
        if img.shape[:2] != (H, W):
            raise SystemExit(f"{fp.name} is {img.shape[1]}x{img.shape[0]}, camera is {W}x{H}")
        out = np.stack([map_coordinates(img[..., c], [sv, su], order=1, mode="nearest") for c in range(3)], axis=-1)
        out[~inside] = 0
        Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(args.output / fp.name, quality=95)
    meta = {"source_camera": cam, "output_pinhole": {"fx": ofx, "fy": ofy, "cx": ocx, "cy": ocy, "width": W, "height": H,
            "hfov_deg": float(np.degrees(2 * np.arctan(W / 2 / ofx))), "vfov_deg": float(np.degrees(2 * np.arctan(H / 2 / ofy)))},
            "valid_fraction": float(inside.mean()), "frames": len(files)}
    (args.output / "undistortion.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["output_pinhole"]), "valid", round(meta["valid_fraction"], 4), "frames", len(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
