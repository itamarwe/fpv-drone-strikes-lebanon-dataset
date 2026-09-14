#!/usr/bin/env python3
"""Measure the Scene B enclosure wall height in VGGT-Omega reconstructions.

Per frame: a polygon on the concrete floor (fitted plane) and pixels along the
wall's top edge. Height = point-to-plane distance of each top point, in the
reconstruction's units. Pixels are given in the original 660x282 frame and
mapped through the variant's undistortion when needed. Metric calibration
follows from the known 7 m wall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from undistort_real_scene import read_camera, intrinsics, distort  # noqa: E402

# Original-frame pixel picks (x, y) for Scene B. Floor polygons avoid the vehicle and HUD ring.
BASE_Y = {30: 196, 45: 205, 60: 170}  # wall foot line (original-frame y) below the picked top pixels
PICKS = {
    30: {"floor": [(180, 195), (420, 195), (430, 245), (150, 245)],
         "top": [(260, 150), (300, 149), (340, 149), (380, 149), (170, 157), (200, 155), (235, 152)]},
    45: {"floor": [(180, 205), (440, 205), (470, 270), (130, 270)],
         "top": [(260, 124), (300, 123), (340, 123), (380, 123), (400, 124), (165, 128), (200, 126), (235, 125)]},
    60: {"floor": [(200, 170), (430, 170), (440, 280), (120, 280)],
         "top": [(260, 71), (300, 71), (340, 71), (380, 72), (415, 73), (165, 63), (200, 66), (235, 69)]},
}


def undistort_map(cam_path: Path, keep_pp: bool):
    """Return f(us, vs) -> (u, v): original pixel to undistorted-output pixel."""
    cam = read_camera(cam_path); fx, fy, cx, cy, k = intrinsics(cam); W, H = cam["width"], cam["height"]
    ocx, ocy = (cx, cy) if keep_pp else (W / 2, H / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xd, yd = distort(cam["model"], (u - ocx) / fx, (v - ocy) / fy, k)
    su, sv = fx * xd + cx, fy * yd + cy  # source pixel for each output pixel
    tree = cKDTree(np.stack([su.ravel(), sv.ravel()], axis=1))
    def f(us, vs):
        _, idx = tree.query([us, vs]); return float(u.ravel()[idx]), float(v.ravel()[idx])
    return f


def polygon_mask(poly, W, H):
    from PIL import Image, ImageDraw
    im = Image.new("L", (W, H), 0); ImageDraw.Draw(im).polygon(poly, fill=1); return np.asarray(im).astype(bool)


def fit_plane(P):
    c = P.mean(axis=0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); n = vt[-1]
    d = (P - c) @ n; keep = np.abs(d) < 3 * np.median(np.abs(d)) + 1e-9  # drop gross outliers, refit
    c = P[keep].mean(axis=0); _, _, vt = np.linalg.svd(P[keep] - c, full_matrices=False); n = vt[-1]
    return c, n, float(np.median(np.abs((P[keep] - c) @ n)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_dir", type=Path)
    p.add_argument("--cameras", type=Path, help="COLMAP cameras.txt used for this variant's undistortion (omit for original frames)")
    p.add_argument("--keep-principal-point", action="store_true")
    p.add_argument("--known-height-m", type=float, default=7.0)
    p.add_argument("--label", default=None)
    args = p.parse_args()
    # Heights are rigid-invariant, so work in each frame's camera frame from depth + intrinsics
    # (world_points_from_depth is 2.3 GB float64 and swaps this machine).
    z = np.load(args.scene_dir / "runpod_artifacts" / "predictions.npz", allow_pickle=True)
    d_raw = np.asarray(z["depth"], dtype=np.float32); depth = d_raw.reshape(d_raw.shape[0], d_raw.shape[1], d_raw.shape[2]); del d_raw; K = np.asarray(z["intrinsic"], dtype=np.float64)
    S, H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    def frame_points(i):
        d = depth[i].astype(np.float64); k = K[i]
        return np.stack([(uu - k[0, 2]) / k[0, 0] * d, (vv - k[1, 2]) / k[1, 1] * d, d], axis=-1)
    W0, H0 = 660, 282
    remap = undistort_map(args.cameras, args.keep_principal_point) if args.cameras else (lambda us, vs: (us, vs))
    def to_pred(us, vs):
        u, v = remap(us, vs); return int(round(u * W / W0)), int(round(v * H / H0))
    rows = []
    for frame, picks in PICKS.items():
        pts = frame_points(frame - 1)
        poly = [to_pred(*q) for q in picks["floor"]]
        m = polygon_mask(poly, W, H)
        floor = pts[m]; floor = floor[np.isfinite(floor).all(axis=1)]
        c, n, resid = fit_plane(floor)
        heights = []
        for (us, vs) in picks["top"]:
            # Sample 1-3 px below the picked top edge (in original-frame pixels) to stay on the wall face,
            # take the median so a single sky/background pixel does not dominate.
            hs = []
            for dv in (1, 2, 3):
                pu, pv = to_pred(us, vs + dv); P = pts[min(pv, H - 1), min(pu, W - 1)]
                if np.isfinite(P).all(): hs.append(float(abs((P - c) @ n)))
            heights.append(float(np.median(hs)) if hs else np.nan)
        heights = np.array(heights)
        # Column-profile estimate: along each column from 25% to 75% of the wall face, height above the
        # floor plane should grow linearly with row; fit and extrapolate to the top edge row. Immune to
        # edge bleed from the background behind the wall top.
        prof = []
        for (us, vs) in picks["top"]:
            vb = BASE_Y[frame]; span = vb - vs
            rows_o = np.arange(vs + 0.25 * span, vs + 0.75 * span, 1.0)
            hs, rs = [], []
            for r in rows_o:
                pu, pv = to_pred(us, r); P = pts[min(pv, H - 1), min(pu, W - 1)]
                if np.isfinite(P).all(): hs.append(float((P - c) @ n)); rs.append(r)
            if len(hs) >= 5:
                a, b = np.polyfit(rs, hs, 1); prof.append(abs(a * vs + b))
        prof = np.array(prof)
        rows.append({"frame": frame, "floor_points": int(len(floor)), "floor_plane_residual_units": resid,
                     "heights_units": heights.round(5).tolist(), "median_height_units": float(np.median(heights)),
                     "spread_pct": float(100 * (np.percentile(heights, 75) - np.percentile(heights, 25)) / np.median(heights)),
                     "profile_heights_units": prof.round(5).tolist(), "profile_median_units": float(np.median(prof)) if prof.size else None,
                     "profile_spread_pct": float(100 * (np.percentile(prof, 75) - np.percentile(prof, 25)) / np.median(prof)) if prof.size else None})
    med = float(np.median(np.concatenate([np.array(r["heights_units"]) for r in rows])))
    pmed = float(np.median(np.concatenate([np.array(r["profile_heights_units"]) for r in rows])))
    ext = np.asarray(z["extrinsic"], dtype=np.float64)
    C = np.array([-(e[:3, :3].T @ e[:3, 3]) for e in ext]); path_len = float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1)))
    out = {"label": args.label or args.scene_dir.name, "known_height_m": args.known_height_m, "per_frame": rows,
           "median_height_units": med, "metres_per_unit_from_wall": args.known_height_m / med,
           "profile_median_height_units": pmed, "metres_per_unit_from_wall_profile": args.known_height_m / pmed,
           "camera_path_length_units": path_len, "wall_over_path_length": pmed / path_len,
           "per_frame_metres_per_unit": [args.known_height_m / r["median_height_units"] for r in rows]}
    (args.scene_dir / "wall_height_measurement.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{out['label']}: edge-sample {med:.5f} u -> {out['metres_per_unit_from_wall']:.1f} m/u | column-profile {pmed:.5f} u -> {out['metres_per_unit_from_wall_profile']:.1f} m/u | wall/path {out['wall_over_path_length']:.5f} | per frame " +
          ", ".join(f"f{r['frame']}: profile {r['profile_median_units']:.5f} u (IQR {r['profile_spread_pct']:.0f}%)" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
