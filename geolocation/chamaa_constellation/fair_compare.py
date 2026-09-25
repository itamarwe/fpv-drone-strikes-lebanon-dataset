#!/usr/bin/env python3
"""Fair comparison of the true pose with the wrong poses that 'beat' it, plus pictures.

The wrong poses were produced by an optimiser; the true pose was not (it is a fixed
estimate from the LoFTR alignment, 3.6 px RMS, never tuned to these features). So
every pose here - truth included - gets the same local refinement on each objective
before scores are compared. Uses the ground truth; a diagnostic, not a result.

Objectives (lower is better):
  A  crossings only       1 - crossing_F1
  B  houses + crossings   house/9 + 1 - crossing_F1
  J  houses + roads + crossings (bimodal.py)

Figures per pose: the photo with SAM detections and the projected OSM layers and
match lines; the photo orthorectified at that pose over the orthophoto, with its
footprint and the true footprint. Plus an overview map of all footprints.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
from scipy.optimize import differential_evolution

import chamaa_constellation as cc
from diagnose_crossings_only import Cross
from ortho_io import cut_ortho
from road_junctions import image_junctions

FIG = cc.RESULTS / "figures" / "fair_compare"
RADIUS = [60, 60, 5, 3, 3, .20, .20]
MAGENTA, CYAN, YELLOW, GREEN, BLUE, WHITE, RED = (255, 60, 255), (255, 230, 40), (40, 220, 255), (80, 255, 80), (255, 120, 40), (255, 255, 255), (60, 60, 255)


def text(im, s, org, scale=.55, col=WHITE, th=1):
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def refine(pr, fn, p, seed=71):
    lb = [max(lo, v - r) for (lo, hi), v, r in zip(pr.bounds, p, RADIUS)]
    ub = [min(hi, v + r) for (lo, hi), v, r in zip(pr.bounds, p, RADIUS)]
    fit = differential_evolution(fn, list(zip(lb, ub)), seed=seed, popsize=12, maxiter=150, tol=1e-4,
                                 polish=False, x0=np.clip(p, lb, ub))
    return fit.x if fn(fit.x) < fn(p) else np.array(p)


def footprint(pr, p):
    """Ground quadrilateral seen by the photo at pose p (rays to the plane at the ground point's height)."""
    f, c, R, g = pr.unpack(p)
    out = []
    for u, v in ((0, 0), (cc.IMAGE_W, 0), (cc.IMAGE_W, cc.IMAGE_H), (0, cc.IMAGE_H)):
        ray = R.T @ np.array([(u - cc.CX) / f, (v - cc.CY) / f, 1.0])
        if ray[2] >= -1e-3:                        # above the horizon: clip far away
            ray[2] = -1e-3
        t = (g[2] - c[2]) / ray[2]
        t = min(t, 3000.0)
        out.append((c + t * ray)[:2] + pr.origin[:2])
    return np.array(out)


def photo_panel(pr, p, label, closed):
    f, c, R, _ = pr.unpack(p)
    im = cv2.imread(str(cc.DATA / "chamaa_photo1.png"))
    tint = im.copy(); tint[closed > 0] = BLUE; im = cv2.addWeighted(tint, .30, im, .70, 0)
    # projected OSM roads
    v = (pr.road_world - c) @ R.T; ok = v[:, 2] > 20
    uv = f * v[:, :2] / np.maximum(v[:, 2:], 1) + [cc.CX, cc.CY]
    for q in uv[ok & (uv[:, 0] > -5) & (uv[:, 0] < cc.IMAGE_W + 5) & (uv[:, 1] > -5) & (uv[:, 1] < cc.IMAGE_H + 5)].astype(int):
        cv2.circle(im, tuple(q), 1, (255, 170, 60), -1)
    # houses: detections, projected centres, one-to-one pairs
    _, pairs = pr.assignment(p, np.arange(len(pr.uv)))
    for o in pr.uv:
        cv2.circle(im, tuple(int(x) for x in o), 6, CYAN, 2)
    huv, _, _ = pr.project(p)
    for q in huv:
        cv2.drawMarker(im, tuple(int(x) for x in q), YELLOW, cv2.MARKER_CROSS, 10, 2)
    for q in pairs:
        cv2.line(im, tuple(int(x) for x in q["observed_xy"]), tuple(int(x) for x in q["projected_xy"]), YELLOW, 1, cv2.LINE_AA)
    # crossings: detected (green), projected OSM (magenta), matched pairs (white lines)
    juv = pr._in_frame(pr.cross_world, f, c, R)
    for q in juv:
        cv2.drawMarker(im, tuple(int(x) for x in q), MAGENTA, cv2.MARKER_SQUARE, 16, 2)
    for q in pr.photo_crossings:
        cv2.circle(im, tuple(int(x) for x in q), 11, GREEN, 3)
    if len(juv):
        D = np.linalg.norm(pr.photo_crossings[:, None] - juv[None], axis=2)
        from scipy.optimize import linear_sum_assignment
        C = np.where(D <= 30, D, 1e6); r, k = linear_sum_assignment(C)
        for a, b in zip(r, k):
            if C[a, b] < 1e6:
                cv2.line(im, tuple(int(x) for x in pr.photo_crossings[a]), tuple(int(x) for x in juv[b]), WHITE, 2)
    text(im, label, (10, 26), .65, WHITE, 2)
    text(im, "cyan o SAM houses, yellow + OSM houses; green O SAM crossings, magenta [] OSM crossings, white = matched;"
             " orange dots OSM roads, blue tint SAM roads", (10, 50), .42)
    return im


def ortho_panel(pr, p, truth_fp, label, size=1100):
    fp = footprint(pr, p)
    allp = np.vstack([fp, truth_fp])
    cx, cy = allp.mean(0); half = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) / 2 + 120
    b = (cx - half, cy - half, cx + half, cy + half); gsd = 2 * half / size
    tmp = FIG / "_o.tif"; t = cut_ortho(b, tmp, gsd=gsd); om = cv2.imread(str(t.path))
    for x in FIG.glob("_o*"):
        x.unlink()
    H, W = om.shape[:2]
    xs = b[0] + (np.arange(W) + .5) * gsd; ys = b[3] - (np.arange(H) + .5) * gsd
    gx, gy = np.meshgrid(xs, ys)
    z = pr.terrain(np.c_[gy.ravel(), gx.ravel()]).reshape(H, W)
    f, c, R, _ = pr.unpack(p)
    P = np.stack([gx - pr.origin[0], gy - pr.origin[1], z], -1).reshape(-1, 3)
    v = (P - c) @ R.T; dep = v[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = f * v[:, 0] / dep + cc.CX; vv = f * v[:, 1] / dep + cc.CY
    ok = (dep > 20) & (u >= 0) & (u < cc.IMAGE_W - 1) & (vv >= 0) & (vv < cc.IMAGE_H - 13)
    photo = cv2.imread(str(cc.DATA / "chamaa_photo1.png"))
    mu = np.where(ok, u, -1).astype(np.float32).reshape(H, W); mv = np.where(ok, vv, -1).astype(np.float32).reshape(H, W)
    warped = cv2.remap(photo, mu, mv, cv2.INTER_LINEAR)
    m = ok.reshape(H, W)
    out = om.copy(); out[m] = (.55 * warped[m] + .45 * om[m]).astype(np.uint8)
    px = lambda e, n: (int((e - b[0]) / gsd), int((b[3] - n) / gsd))
    cv2.polylines(out, [np.array([px(*q) for q in truth_fp], np.int32)], True, MAGENTA, 3)
    cv2.polylines(out, [np.array([px(*q) for q in fp], np.int32)], True, GREEN, 3)
    text(out, label, (10, 26), .65, WHITE, 2)
    text(out, "photo orthorectified at this pose over the orthophoto; green = this pose's footprint, magenta = true footprint",
         (10, 50), .42)
    return out, fp


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    pr = Cross(areas["2x2km"])
    _, closed, _, _, _ = image_junctions()
    tp = np.array(json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())["parameters"], float)
    objs = {"A crossings only": pr.obj_a, "B houses+crossings": pr.obj_b, "J houses+roads+crossings": pr.objective}
    # the poses that beat or approached the truth: rerun the same (deterministic) DE seeds
    cands = {}
    for key, fn, seeds in (("A crossings only", pr.obj_a, (41, 7)), ("B houses+crossings", pr.obj_b, (41, 19))):
        for s in seeds:
            fit = differential_evolution(fn, pr.bounds, seed=s, popsize=16, maxiter=240, tol=.0002,
                                         polish=False, updating="immediate")
            cands[f"{key} search, seed {s}"] = fit.x
    h = min(json.loads((cc.RESULTS / "2x2km/bimodal/search.json").read_text())["hypotheses"], key=lambda r: r["train_cost"])
    cands["J search winner (2x2 km)"] = np.array(h["parameters"][:7])
    rows = []
    truth_fp = footprint(pr, tp)
    for name, fn in objs.items():
        rt = refine(pr, fn, tp)
        rows.append(dict(objective=name, pose="TRUE pose, refined", error_m=math.dist(rt[:2] + pr.origin[:2], truth),
                         raw=fn(tp), refined=fn(rt), params=rt.tolist()))
        for lab, q in cands.items():
            rq = refine(pr, fn, q)
            rows.append(dict(objective=name, pose=lab, error_m=math.dist(rq[:2] + pr.origin[:2], truth),
                             raw=fn(q), refined=fn(rq), params=rq.tolist()))
    print(f"{'objective':<26}{'pose':<34}{'err m':>7}{'as found':>10}{'refined':>9}")
    for r in rows:
        print(f"{r['objective']:<26}{r['pose']:<34}{r['error_m']:>7.0f}{r['raw']:>10.3f}{r['refined']:>9.3f}")
    (cc.RESULTS / "diagnostics/fair_compare.json").write_text(json.dumps(dict(
        warning="DIAGNOSTIC - includes the ground-truth pose; every pose refined identically", rows=rows), indent=2))

    # pictures: truth (refined on J) and each wrong candidate (refined on its own objective)
    shows = [("TRUE pose (refined on J)", next(r for r in rows if r["objective"].startswith("J") and r["pose"].startswith("TRUE")))]
    for lab in cands:
        obj = "A crossings only" if lab.startswith("A") else ("B houses+crossings" if lab.startswith("B") else "J houses+roads+crossings")
        shows.append((lab, next(r for r in rows if r["objective"] == obj and r["pose"] == lab)))
    fps = []
    for i, (lab, r) in enumerate(shows):
        p = np.array(r["params"])
        f1, m, n = pr.crossing_f1(p); rp, rr, rf = pr.roads(p); hc, _ = pr.assignment(p, pr.train)
        head = (f"{lab}: error {r['error_m']:.0f} m | {r['objective'].split()[0]} score {r['refined']:.3f} | "
                f"crossings {m}/{n} (F1 {f1:.2f}) | road F1 {rf:.2f} | houses {hc:.2f}")
        a = photo_panel(pr, p, head, closed)
        b, fp = ortho_panel(pr, p, truth_fp, lab)
        fps.append((lab, fp, r["error_m"]))
        b = cv2.resize(b, (int(b.shape[1] * a.shape[0] / b.shape[0]), a.shape[0]))
        cv2.imwrite(str(FIG / f"pose_{i}.jpg"), np.hstack([a, np.full((a.shape[0], 8, 3), 30, np.uint8), b]),
                    [cv2.IMWRITE_JPEG_QUALITY, 84])
    # overview: all footprints on the 2x2 km box
    bnd = areas["2x2km"]["bounds_utm"]; g = (bnd[2] - bnd[0]) / 1400
    t = cut_ortho(tuple(bnd), FIG / "_v.tif", gsd=g); ov = cv2.imread(str(t.path))
    for x in FIG.glob("_v*"):
        x.unlink()
    px = lambda e, n: (int((e - bnd[0]) / g), int((bnd[3] - n) / g))
    cols = [MAGENTA, GREEN, YELLOW, CYAN, (0, 165, 255), RED, WHITE]
    for (lab, fp, e), col in zip(fps, cols):
        P = np.array([px(*q) for q in fp], np.int32)
        cv2.polylines(ov, [P], True, col, 3); cen = P.mean(0).astype(int)
        text(ov, f"{lab} ({e:.0f} m)", (int(cen[0]) - 60, int(cen[1])), .5, col, 2)
    text(ov, "photo footprints of the true pose and of the wrong poses that beat or approached it (2x2 km box)", (10, 26), .6)
    cv2.imwrite(str(FIG / "overview_footprints.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
