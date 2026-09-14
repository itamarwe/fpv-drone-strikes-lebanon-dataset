#!/usr/bin/env python3
"""Independent arbiter for the Scene B wall: height / camera-path length in COLMAP sparse models.

Uses the same original-frame picks as measure_wall_height.py, selects triangulated
points observed inside the floor polygon (plane) and on the wall face (between the
top edge and the base line), fits height against fractional position down the face
and extrapolates to the top edge. Ratio to the registered path length is unit-free,
so it can be compared with the VGGT runs directly.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_moge3_colmap_scale import qvec_to_rotation, read_images  # noqa: E402
from measure_wall_height import PICKS, BASE_Y  # noqa: E402

C = Path("benchmarks/synthetic_flyover/real_calibration/scene_b")
rows = list(csv.DictReader(open("benchmarks/3d_pipeline_two_scene/work/bundle/scenes/scene_b/profiles/dense_train/frames.csv")))
src_to_prof = {int(r["source_index"]): int(r["profile_index"]) for r in rows}


def parse_obs(s):
    v = np.array(s.split(), float).reshape(-1, 3)
    return v[:, :2], v[:, 2].astype(int)


def main() -> int:
    out = {}
    for fit in ("colmap_ba_radial_cropcenter", "colmap_ba_fisheye_pp", "colmap_ba_radialfisheye_pp", "colmap_ba_pp"):
        d = C / fit
        if not (d / "points3D.txt").exists():
            continue
        P3 = {}
        for line in (d / "points3D.txt").read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            q = line.split(); P3[int(q[0])] = np.array([float(q[1]), float(q[2]), float(q[3])])
        images = read_images(d / "images.txt"); by_name = {im.name: im for im in images}
        centres = np.array([-(qvec_to_rotation(im.qvec).T @ im.tvec) for im in sorted(images, key=lambda im: im.name)])
        path_len = float(np.sum(np.linalg.norm(np.diff(centres, axis=0), axis=1)))
        per = {}
        for frame, picks in PICKS.items():
            prof = src_to_prof.get(frame - 1); im = by_name.get(f"frame_{prof:06d}.jpg") if prof is not None else None
            if im is None:
                per[frame] = "not registered / not in dense profile"; continue
            xys, ids = parse_obs(im.observations); ok = ids >= 0; xys, ids = xys[ok], ids[ok]
            mask_img = Image.new("L", (660, 282), 0); ImageDraw.Draw(mask_img).polygon(picks["floor"], fill=1)
            mask = np.asarray(mask_img).astype(bool)
            inside = np.array([mask[min(int(y), 281), min(int(x), 659)] for x, y in xys])
            floor = np.array([P3[i] for i in ids[inside] if i in P3])
            if len(floor) < 8:
                per[frame] = f"only {len(floor)} floor points"; continue
            c = floor.mean(axis=0); _, _, vt = np.linalg.svd(floor - c, full_matrices=False); n = vt[-1]
            dist = np.abs((floor - c) @ n); keep = dist < 3 * np.median(dist) + 1e-12
            c = floor[keep].mean(axis=0); _, _, vt = np.linalg.svd(floor[keep] - c, full_matrices=False); n = vt[-1]
            st = sorted(picks["top"]); y_top = np.interp(xys[:, 0], [x for x, _ in st], [y for _, y in st]); base_y = BASE_Y[frame]
            face = (xys[:, 0] >= st[0][0] - 10) & (xys[:, 0] <= st[-1][0] + 10) & (xys[:, 1] > y_top + 1) & (xys[:, 1] < base_y - 1)
            fp = [(P3[i], xys[j, 1], y_top[j]) for j, i in enumerate(ids) if face[j] and i in P3]
            if len(fp) < 5:
                per[frame] = f"floor {int(keep.sum())}, only {len(fp)} face points"; continue
            h = np.array([abs((P - c) @ n) for P, _, _ in fp]); frac = np.array([(base_y - y) / (base_y - yt) for _, y, yt in fp])
            a, b = np.polyfit(frac, h, 1)
            per[frame] = {"floor_points": int(keep.sum()), "face_points": len(fp), "wall_units": float(a + b), "slope": float(a), "intercept": float(b),
                          "median_face_h": float(np.median(h)), "median_frac": float(np.median(frac))}
        hs = [v["wall_units"] for v in per.values() if isinstance(v, dict)]
        out[fit] = {"path_length_units": path_len, "per_frame": per, "wall_units_median": float(np.median(hs)) if hs else None,
                    "wall_over_path": float(np.median(hs) / path_len) if hs else None}
        print(f"== {fit}: path {path_len:.4f} u, wall {np.median(hs) if hs else float('nan'):.4f} u, wall/path {out[fit]['wall_over_path'] if hs else float('nan'):.5f}")
        for f, v in per.items():
            print("   ", f, v if isinstance(v, str) else {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
    (C / "wall_height_colmap.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
