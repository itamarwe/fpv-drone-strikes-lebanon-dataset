#!/usr/bin/env python3
"""Feasibility of a decomposed, exhaustive search (research check; uses the truth to SCORE only).

Idea (Habbecke & Kobbelt 2010, oblique images vs cadastral maps; OrienterNet 2023):
  1. The vertical vanishing point (VP) of building edges fixes camera pitch and roll
     for a given focal length.
  2. With pitch, roll and focal fixed, the photo's detections can be rectified onto
     the ground: a bird's-eye point pattern known up to scale (the camera height).
  3. Matching that pattern to the map is a 2D problem - position, heading and scale -
     searched EXHAUSTIVELY: every heading step x every scale step, with all positions
     evaluated at once by shifting rasters. No random sampling, no needle.
  Score per cell: the chance-corrected idea in Gaussian form. For each layer
  (houses, crossings), hits = detections landing within r of a map feature,
  expected = sum of the local map coverage at those spots, z = (hits - expected) /
  sqrt(expected); total = z_houses + z_crossings.

Checks:
  (a) detect the vertical VP from line segments; compare the implied pitch/roll (at
      the true focal) with the true pose.
  (b) run the exhaustive 2D search in the 2x2 km box at the true focal/pitch/roll,
      and at focal +-10% (pitch/roll re-derived from the VP for that focal); report
      where the best peak lands and how the truth ranks.
"""
from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter

import chamaa_constellation as cc
from road_junctions import image_junctions, osm_junctions

GRID = 4.0                 # metres per raster cell
R_HOUSE, R_CROSS = 12.0, 16.0
COVER_WIN = 200.0          # metres: window for local map coverage (the chance rate)
HEADINGS = np.arange(0, 360, 2.0)
HEIGHTS = np.exp(np.arange(np.log(60), np.log(900), np.log(1.04)))


def rot_from(pitch, roll):
    return cc.rotation(0.0, pitch, roll)


def vertical_vp(photo):
    """RANSAC vertical vanishing point from near-vertical line segments (below the image)."""
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    try:
        lsd = cv2.createLineSegmentDetector(); segs = lsd.detect(g)[0]
        segs = segs.reshape(-1, 4) if segs is not None else np.zeros((0, 4))
    except Exception:
        e = cv2.Canny(g, 60, 160)
        segs = cv2.HoughLinesP(e, 1, np.pi / 360, 25, minLineLength=14, maxLineGap=2)
        segs = segs.reshape(-1, 4) if segs is not None else np.zeros((0, 4))
    segs = segs[segs[:, 1] < cc.IMAGE_H - 14]
    d = segs[:, 2:] - segs[:, :2]; L = np.linalg.norm(d, axis=1)
    ang = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))       # 0 = vertical
    keep = (L >= 12) & (ang < 25)
    S = segs[keep]; L = L[keep]
    lines = np.cross(np.c_[S[:, :2], np.ones(len(S))], np.c_[S[:, 2:], np.ones(len(S))])
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    rng = np.random.default_rng(0); best = (-1, None)
    mids = (S[:, :2] + S[:, 2:]) / 2
    for _ in range(4000):
        i, j = rng.choice(len(S), 2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if abs(vp[2]) < 1e-9:
            continue
        vp = vp[:2] / vp[2]
        if vp[1] < 2 * cc.IMAGE_H:                     # the nadir VP lies far below the image
            continue
        dirs = S[:, 2:] - S[:, :2]; to_vp = vp - mids
        cosang = np.abs(np.sum(dirs * to_vp, 1)) / (np.linalg.norm(dirs, axis=1) * np.linalg.norm(to_vp, axis=1) + 1e-9)
        inl = cosang > math.cos(math.radians(1.5))
        support = L[inl].sum()
        if support > best[0]:
            best = (support, vp, inl)
    _, vp, inl = best
    # least-squares refine on inliers
    A = lines[inl]; w = L[inl]
    sol = np.linalg.lstsq(A[:, :2] * w[:, None], -A[:, 2] * w, rcond=None)[0]
    return sol, int(inl.sum()), len(S)


def pitch_roll_from_vp(vp, f):
    up = -np.array([(vp[0] - cc.CX) / f, (vp[1] - cc.CY) / f, 1.0]); up /= np.linalg.norm(up)
    fn = lambda x: np.sum((rot_from(*x)[:, 2] - up) ** 2)
    res = min((minimize(fn, x0, method="Nelder-Mead") for x0 in ([-30, 0], [-50, 0], [-15, 0])), key=lambda r: r.fun)
    return float(res.x[0]), float(res.x[1])


def bev_unit(uv, f, pitch, roll):
    """Ground-plane positions (camera height 1, heading 0) of image points."""
    R = rot_from(pitch, roll)
    rays = (np.c_[(uv[:, 0] - cc.CX) / f, (uv[:, 1] - cc.CY) / f, np.ones(len(uv))]) @ R   # camera->world
    ok = rays[:, 2] < -1e-3
    t = 1.0 / -rays[ok, 2]
    return rays[ok, :2] * t[:, None], ok


def rasters(points_xy, bounds, radius):
    e0, n0, e1, n1 = bounds
    W = int((e1 - e0) / GRID); H = int((n1 - n0) / GRID)
    I = np.zeros((H, W), np.uint8)
    for x, y in points_xy:
        cv2.circle(I, (int((x - e0) / GRID), int((n1 - y) / GRID)), max(1, int(radius / GRID)), 1, -1)
    Q = uniform_filter(I.astype(np.float32), size=int(COVER_WIN / GRID))
    return I.astype(np.float32), Q


def search(bev_h, bev_c, c0, maps, box, pad):
    """Exhaustive over heading x height; all ground positions of the image centre in the box at once."""
    (Ih, Qh), (Ic, Qc) = maps
    e0, n0, e1, n1 = box
    ox, oy = pad                                        # raster origin offset (cells) of the box
    gw, gh = int((e1 - e0) / GRID), int((n1 - n0) / GRID)
    best = np.full((gh, gw), -1e9, np.float32); arg = np.zeros((gh, gw, 2), np.int16)
    for a, hd in enumerate(HEADINGS):
        th = math.radians(hd); Rm = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
        # rotate unit-height BEV so that image 'forward' points to heading hd (east=x, north=y)
        bh = bev_h @ Rm.T; bc = bev_c @ Rm.T; cc0 = c0 @ Rm.T
        for b, h in enumerate(HEIGHTS):
            z = np.zeros((gh, gw), np.float32)
            for pts, I, Q in ((bh, Ih, Qh), (bc, Ic, Qc)):
                d = (pts - cc0) * h                            # offsets from the image-centre ground point
                hits = np.zeros((gh, gw), np.float32); expc = np.zeros((gh, gw), np.float32)
                for dx, dy in d:
                    cx, cy = int(round(dx / GRID)), int(round(-dy / GRID))
                    ys, xs = oy + cy, ox + cx
                    if ys < 0 or xs < 0 or ys + gh > I.shape[0] or xs + gw > I.shape[1]:
                        continue
                    hits += I[ys:ys + gh, xs:xs + gw]; expc += Q[ys:ys + gh, xs:xs + gw]
                z += (hits - expc) / np.sqrt(expc + 1.0)
            upd = z > best
            best[upd] = z[upd]; arg[upd, 0] = a; arg[upd, 1] = b
    return best, arg


def main():
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    box = areas["2x2km"]["bounds_utm"]
    truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    tp = json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())
    f_true = tp["true_pose"]["focal_px"]; yaw_t, pitch_t, roll_t = tp["true_pose"]["yaw_pitch_roll_deg"]
    h_true = tp["true_pose"]["height_above_ground_m"]
    photo = cv2.imread(str(cc.DATA / "chamaa_photo1.png"))
    out = {}

    # (a) vertical vanishing point
    vp, n_inl, n_seg = vertical_vp(photo)
    p_vp, r_vp = pitch_roll_from_vp(vp, f_true)
    Rt = cc.rotation(yaw_t, pitch_t, roll_t)
    up_true = Rt[:, 2]; vp_true = np.array([cc.CX, cc.CY]) + f_true * up_true[:2] / up_true[2]
    print(f"(a) vertical VP detected at ({vp[0]:.0f}, {vp[1]:.0f}) from {n_inl} of {n_seg} near-vertical segments;"
          f" true VP ({vp_true[0]:.0f}, {vp_true[1]:.0f})")
    print(f"    at the true focal {f_true:.0f}px the VP implies pitch {p_vp:.1f}, roll {r_vp:.1f}"
          f"  (true {pitch_t:.1f}, {roll_t:.1f})")
    out["vp"] = dict(detected=vp.tolist(), true=vp_true.tolist(), inliers=n_inl, segments=n_seg,
                     implied_pitch_roll_at_true_focal=[p_vp, r_vp], true_pitch_roll=[pitch_t, roll_t])

    # map rasters: box + generous margin for far detections
    pad_m = 2500.0
    big = (box[0] - pad_m, box[1] - pad_m, box[2] + pad_m, box[3] + pad_m)
    extra = cc.Problem(dict(name="big", bounds_utm=list(big)))
    houses_xy = [np.mean(np.array(m["polygon_utm"]), 0) for m in extra.map]
    _, ojs = osm_junctions(); cross_xy = [j["utm"] for j in ojs]
    maps = (rasters(houses_xy, big, R_HOUSE), rasters(cross_xy, big, R_CROSS))
    pad = (int(pad_m / GRID), int(pad_m / GRID))
    _, _, _, ijs, _ = image_junctions()
    det_h = extra.uv; det_c = np.array([j.xy for j in ijs])

    runs = [("true focal, true pitch/roll", f_true, pitch_t, roll_t)]
    for fac in (0.9, 1.1):
        f = f_true * fac; p, r = pitch_roll_from_vp(vp, f)
        runs.append((f"focal x{fac}, pitch/roll from VP", f, p, r))
    p, r = pitch_roll_from_vp(vp, f_true)
    runs.insert(1, ("true focal, pitch/roll from VP", f_true, p, r))
    out["runs"] = []
    for label, f, p, r in runs:
        t0 = time.perf_counter()
        bh, _ = bev_unit(det_h, f, p, r); bc, _ = bev_unit(det_c, f, p, r)
        c0, _ = bev_unit(np.array([[cc.CX, cc.CY]]), f, p, r)
        best, arg = search(bh, bc, c0[0], maps, box, pad)
        dt = time.perf_counter() - t0
        gh, gw = best.shape
        ty, tx = int((box[3] - truth[1]) / GRID), int((truth[0] - box[0]) / GRID)
        yy, xx = np.unravel_index(np.argmax(best), best.shape)
        peak_xy = (box[0] + (xx + .5) * GRID, box[3] - (yy + .5) * GRID)
        # rank of the truth: best score within 40 m of the truth vs distinct peaks elsewhere
        r_c = int(40 / GRID)
        near = best[max(0, ty - r_c):ty + r_c + 1, max(0, tx - r_c):tx + r_c + 1].max()
        mask = np.ones_like(best, bool); mask[max(0, ty - r_c):ty + r_c + 1, max(0, tx - r_c):tx + r_c + 1] = False
        # count distinct places (100 m apart) scoring above the truth-neighbourhood best
        tmp = np.where(mask, best, -1e9).copy(); higher = 0
        while True:
            yy2, xx2 = np.unravel_index(np.argmax(tmp), tmp.shape)
            if tmp[yy2, xx2] <= near:
                break
            higher += 1; k = int(100 / GRID)
            tmp[max(0, yy2 - k):yy2 + k + 1, max(0, xx2 - k):xx2 + k + 1] = -1e9
            if higher > 50:
                break
        a, b = arg[yy, xx]
        err = math.dist(peak_xy, truth)
        print(f"(b) {label:<34} f {f:5.0f} pitch {p:6.1f} roll {r:5.1f} | best peak z {best.max():6.2f} at "
              f"{err:5.0f} m from truth (heading {HEADINGS[a]:.0f}, height {HEIGHTS[b]:.0f} m) | "
              f"truth-area best z {near:6.2f}, distinct places above it: {higher} | {dt:.0f} s")
        out["runs"].append(dict(label=label, focal=f, pitch=p, roll=r, best_z=float(best.max()),
                                best_error_m=err, best_heading=float(HEADINGS[a]), best_height=float(HEIGHTS[b]),
                                truth_area_z=float(near), distinct_places_above_truth=higher, seconds=dt))
        if label.startswith("true focal, pitch/roll from VP"):
            v = np.clip((best - np.percentile(best, 50)) / (best.max() - np.percentile(best, 50) + 1e-9), 0, 1)
            im = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            cv2.drawMarker(im, (tx, ty), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.circle(im, (int(xx), int(yy)), 10, (80, 255, 80), 2)
            cv2.imwrite(str(cc.RESULTS / "figures/feasibility_bev_heatmap.jpg"), cv2.resize(im, (gw * 2, gh * 2), interpolation=cv2.INTER_NEAREST))
    (cc.RESULTS / "diagnostics/feasibility_bev_search.json").write_text(json.dumps(dict(
        warning="research feasibility check; the truth is used only to score", **out), indent=2, default=float))


if __name__ == "__main__":
    main()
