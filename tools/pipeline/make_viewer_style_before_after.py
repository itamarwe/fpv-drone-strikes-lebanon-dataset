#!/usr/bin/env python3
"""Viewer-style before/after stills (ground grid, camera frusta, path) for every scene.

Real scenes: before = the published product reconstruction, after = the best
undistorted run, both Sim(3)-aligned into the original run's viewer frame and
drawn with that frame's ground grid. Synthetic scenes: before = fisheye-input
run, after = pinhole-remap run, in Blender metres with a metric grid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_before_after import quat_to_R, look_at, project, splat, draw_polyline, font  # noqa: E402
from evaluate_3d_trajectory import umeyama  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "lens_undistortion_before_after"
CYAN, ORANGE, RED, GRID_MAJOR, GRID_MINOR = (54, 228, 255), (255, 176, 0), (255, 77, 109), (75, 100, 114), (37, 49, 58)


def frustum_pts(cam, length):
    c, f, r, d = (np.asarray(cam[k], float) for k in ("position", "forward", "right", "down"))
    n = c + f * length; hw, hh = length * 0.58, length * 0.34
    k = [n - r * hw - d * hh, n + r * hw - d * hh, n + r * hw + d * hh, n - r * hw + d * hh]
    return [c, k[0], k[1], c, k[2], k[1], k[2], k[3], c, k[0], k[3]]


def draw_grid(im, R, eye, fpx, origin, u, v, size, minor, major):
    d = ImageDraw.Draw(im); W, H = im.size
    origin, u, v = np.asarray(origin, float), np.asarray(u, float) / np.linalg.norm(u), np.asarray(v, float) / np.linalg.norm(v)
    half = size / 2; n = int(np.ceil(half / minor)); every = max(1, round(major / minor))
    for i in range(-n, n + 1):
        off = i * minor; col = GRID_MAJOR if i % every == 0 else GRID_MINOR
        for a, b in ((origin - u * half + v * off, origin + u * half + v * off), (origin - v * half + u * off, origin + v * half + u * off)):
            uu, vv, zz, ok = project(np.stack([a, b]), R, eye, fpx, W, H)
            if ok.all(): d.line([(uu[0], vv[0]), (uu[1], vv[1])], fill=col, width=1)


def render_scene_panel(P, C, cams, grid, W=1400, H=900, view=None, size=2, fpx=1100):
    path = np.array([c["position"] for c in cams]); ext = float(np.linalg.norm(path[-1] - path[0]))
    ctr = path.mean(0) + np.array([0, -0.12 * ext, 0])  # path only, so before/after share the viewpoint exactly
    d = path[-1] - path[0]; d[1] = 0; d /= np.linalg.norm(d); left = np.cross([0, 1, 0], d)
    eye, tgt = view(path, ext, ctr, d, left) if view else (ctr - d * 0.55 * ext + left * 0.75 * ext + np.array([0, 0.75 * ext, 0]), ctr)
    R, eye = look_at(eye, tgt)
    img = np.full((H, W, 3), (12, 13, 15), np.uint8); zbuf = np.full((H, W), np.inf)
    splat(img, zbuf, P, C, R, eye, fpx, size)
    im = Image.fromarray(img)
    if grid: draw_grid(im, R, eye, fpx, grid["origin"], grid["u"], grid["v"], grid["size_units"], grid["minor_step_units"], grid["major_step_units"])
    draw_polyline(im, path, R, eye, fpx, CYAN, 3)
    fl = ext * 0.045
    for i in range(0, len(cams), 10): draw_polyline(im, frustum_pts(cams[i], fl), R, eye, fpx, CYAN, 1)
    draw_polyline(im, frustum_pts(cams[-1], fl * 1.4), R, eye, fpx, RED, 2)
    draw_polyline(im, frustum_pts(cams[0], fl * 1.6), R, eye, fpx, ORANGE, 3)
    dr = ImageDraw.Draw(im)
    for pt, col, rad in ((path[0], ORANGE, 9), (path[-1], RED, 7)):
        uu, vv, zz, ok = project(pt[None], R, eye, fpx, W, H)
        if ok[0]: dr.ellipse([uu[0] - rad, vv[0] - rad, uu[0] + rad, vv[0] + rad], fill=col)
    return im


def compose(panels, labels, caps, title, sub, out):
    W, H = panels[0].size; gap, header, footer = 12, 118, 140
    cv = Image.new("RGB", (W * 2 + gap, H + header + footer), (12, 13, 15)); d = ImageDraw.Draw(cv)
    d.text((28, 20), title, fill=(235, 238, 242), font=font(40)); d.text((28, 70), sub, fill=(139, 149, 161), font=font(24))
    for i, (p, lab, cap) in enumerate(zip(panels, labels, caps)):
        x = i * (W + gap); cv.paste(p, (x, header))
        tag, col = ("BEFORE", RED) if i == 0 else ("AFTER", CYAN)
        d.rounded_rectangle([x + 20, header + 18, x + 210, header + 74], radius=10, fill=col); d.text((x + 44, header + 24), tag, fill=(12, 13, 15), font=font(38))
        d.text((x + 230, header + 30), lab, fill=(235, 238, 242), font=font(30))
        y = header + H + 16
        for line in cap.split("\n"): d.text((x + 24, y), line, fill=(200, 208, 216), font=font(26)); y += 34
    d.text((W * 2 + gap - 720, header + H + 100), "cyan = camera path & frusta   orange = first frame   red = final frame", fill=(139, 149, 161), font=font(22))
    cv.save(out, quality=94); small = cv.copy(); small.thumbnail((1600, 1600)); small.save(str(out).replace(".jpg", "_1600.jpg"), quality=92); print("wrote", out)


def load_layer(base, L, Rq):
    P = np.fromfile(base / f"{L['key']}_points.bin", dtype="<f4").reshape(-1, 3).astype(float) @ Rq.T
    C = np.fromfile(base / f"{L['key']}_colors.bin", dtype=np.uint8).reshape(-1, 3)
    cams = [{k: (np.asarray(c[k]) @ Rq.T).tolist() for k in ("position", "right", "down", "forward")} for c in L["cameras"]]
    return P, C, cams


def published_layer(scene, ref_cams_raw, Rq, max_pts=1_500_000):
    b = ROOT / "benchmarks" / "3d_pipeline_two_scene" / "work" / "bundle" / "scenes" / f"scene_{scene}"
    P = np.fromfile(b / "baseline" / "points_positions.bin", dtype="<f4").reshape(-1, 3).astype(float)
    C = np.fromfile(b / "baseline" / "points_colors.bin", dtype=np.uint8).reshape(-1, 3)
    meta = json.loads((b / "source" / "scene_meta.json").read_text()); path = meta["path"]
    src = np.array([e["position"] for e in path]); ref = np.array([c["position"] for c in ref_cams_raw])
    n = min(len(src), len(ref)); s, R, t = umeyama(src[:n], ref[:n])
    P = (s * (R @ P.T)).T + t
    if len(P) > max_pts:
        k = np.random.default_rng(2).choice(len(P), size=max_pts, replace=False); P, C = P[k], C[k]
    cams = [{"position": (s * (R @ np.asarray(e["position"])) + t), "right": R @ np.asarray(e["right"]), "down": R @ np.asarray(e["down"]), "forward": R @ np.asarray(e["forward"])} for e in path]
    cams = [{k: (np.asarray(v) @ Rq.T).tolist() for k, v in c.items()} for c in cams]
    return P @ Rq.T, C, cams, {"scale": float(s), "matched": n}


def grid_in_aligned_frame(grid, Rq):
    g = dict(grid)
    for k in ("origin", "u", "v", "normal"): g[k] = (np.asarray(grid[k], float) @ Rq.T).tolist()
    return g


def real_scene(scene, after_idx, after_label, caps, title):
    base = ROOT / "scenes" / "real_two_scene_lens_test" / f"overlay_scene_{scene}"; m = json.loads((base / "meta.json").read_text())
    Rq = quat_to_R(m["scene_alignment_quaternion"]); grid = grid_in_aligned_frame(m["ground_grid"], Rq)
    before = published_layer(scene, m["layers"][0]["cameras"], Rq)
    after = load_layer(base, m["layers"][after_idx], Rq)
    panels = [render_scene_panel(before[0], before[1], before[2], grid), render_scene_panel(after[0], after[1], after[2], grid)]
    compose(panels, ["published product (VGGT-Omega on raw frames)", after_label], caps, title,
            "same VGGT-Omega model; the only change is a per-video lens calibration and undistortion before inference. Both panels: same viewpoint, aligned on the camera path.",
            OUT / f"viewer_scene_{scene}_before_after.jpg")


def synthetic_scene(before_dir, after_dir, title, caps, out_name, labels=("raw fisheye frames", "same frames, undistorted to pinhole")):
    S = ROOT / "scenes" / "synthetic_zofa_al_bayada"; Z2Y = lambda A: np.stack([A[:, 0], A[:, 2], -A[:, 1]], axis=1); data = []
    for name in (before_dir, after_dir):
        m = json.loads((S / name / "gt_overlay" / "meta.json").read_text())
        P = Z2Y(np.fromfile(S / name / "gt_overlay" / "vggt_points.bin", dtype="<f4").reshape(-1, 3).astype(float)); C = np.fromfile(S / name / "gt_overlay" / "vggt_colors.bin", dtype=np.uint8).reshape(-1, 3)
        cams = [{k: Z2Y(np.asarray(c[k])[None])[0].tolist() for k in ("position", "right", "down", "forward")} for c in m["vggt_cameras"]]
        gt = [{k: Z2Y(np.asarray(c[k])[None])[0].tolist() for k in ("position", "right", "down", "forward")} for c in m["gt_cameras"]]
        data.append((P, C, cams, gt))
    # metric grid on the ground under the path: 10 m minor, 50 m major, 400 m square
    P0 = data[0][0]; gy = np.percentile(P0[:, 1], 8); ctr = np.array([c["position"] for c in data[0][3]]).mean(0)
    grid = {"origin": [ctr[0], gy, ctr[2]], "u": [1, 0, 0], "v": [0, 0, 1], "size_units": 400, "minor_step_units": 10, "major_step_units": 50}
    panels = []
    for P, C, cams, gt in data:
        im = render_scene_panel(P, C, cams, grid, size=2)
        panels.append(im)
    # true path in white on both panels (re-draw with same view: recompute view identically)
    compose(panels, list(labels), caps, title, "exact Blender ground truth; grid 10 m / 50 m; cyan = reconstructed camera path", OUT / out_name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    real_scene("a", 2, "lens self-calibrated + undistorted (COLMAP radial model)",
               ["camera path disagreement vs independent SfM 1.15%\nlocal scale drift std 0.061\nVGGT focal estimate 1.63x the calibrated lens",
                "camera path disagreement 0.78%\nlocal scale drift std 0.039\nVGGT focal estimate 0.97x the calibrated lens"],
               "Real FPV footage, Scene A")
    real_scene("b", 1, "lens self-calibrated + undistorted (GLOMAP fisheye model)",
               ["wall height / flight-path length 0.060 (SfM says 0.079)\nlocal scale drift std 0.083\nVGGT focal estimate 1.47x the calibrated lens",
                "wall height / flight-path length 0.084, matches SfM\nlocal scale drift std 0.047\nVGGT focal estimate 1.17x the calibrated lens"],
               "Real FPV footage, Scene B")
    synthetic_scene("synthetic_zofa_fpv_roof_flyover_120f", "synthetic_zofa_fpv_roof_flyover_120f_pinhole", "Synthetic FPV flyover: House 01 roof approach",
                    ["camera path error 2.2% of path (max 18 m)\ndepth error 43% vs ground truth\nlocal scale drift 0.77 - 1.18",
                     "camera path error 0.24% (max 1.9 m)\ndepth error 2%\nlocal scale drift 0.97 - 1.01"], "viewer_synthetic_house01_before_after.jpg")
    synthetic_scene("synthetic_variant_approach_house21_high", "synthetic_variant_approach_house21_high_pinhole", "Synthetic FPV flight: House 21 high approach (45 m AGL, 20 m/s)",
                    ["camera path error 2.8% of path (max 26 m)\ndepth error 33% vs ground truth\nlocal scale drift 0.76 - 1.36",
                     "camera path error 0.15% (max 1.8 m)\ndepth error 3%\nlocal scale drift 0.98 - 1.02"], "viewer_synthetic_house21_before_after.jpg")
    synthetic_scene("synthetic_variant_lateral_house03", "synthetic_variant_lateral_house03_pinhole", "Synthetic FPV flight: House 03 lateral pass (camera yawed 60 degrees)",
                    ["camera path error 6.9% of path (max 62 m)\ndepth error 18% vs ground truth\nlocal scale drift 0.89 - 1.09",
                     "camera path error 0.53% (max 3.3 m)\ndepth error 3%\nlocal scale drift 0.96 - 1.02"], "viewer_synthetic_house03_lateral_before_after.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
