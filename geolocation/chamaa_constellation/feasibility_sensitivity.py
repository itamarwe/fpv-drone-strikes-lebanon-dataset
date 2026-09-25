#!/usr/bin/env python3
"""How accurate must pitch, roll and focal be for the exhaustive BEV search? And can the
vertical vanishing point be found from building walls only? (research check; truth used to score)

Faster coarse settings than feasibility_bev_search.py: 8 m raster, 3 deg headings, 6% heights.
"""
import json, math, time
import cv2, numpy as np
import chamaa_constellation as cc
import feasibility_bev_search as fb
from road_junctions import image_junctions, osm_junctions

fb.GRID = 8.0; fb.R_HOUSE, fb.R_CROSS = 16.0, 20.0
fb.HEADINGS = np.arange(0, 360, 3.0)
fb.HEIGHTS = np.exp(np.arange(np.log(60), np.log(900), np.log(1.06)))


def vp_from_buildings(photo):
    """Vertical VP using only line segments on (dilated) SAM building masks."""
    seg = json.loads((cc.DATA / "sam_photo1/segments.json").read_text())["ground"]["features"]
    M = np.zeros((cc.IMAGE_H, cc.IMAGE_W), np.uint8)
    for f in seg:
        m = cv2.imread(str(cc.DATA / "sam_photo1/ground_masks" / f"{f['id']}.png"), 0)
        M |= (cv2.resize(m, (cc.IMAGE_W, cc.IMAGE_H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    M = cv2.dilate(M, np.ones((7, 7), np.uint8))
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    segs = cv2.createLineSegmentDetector().detect(g)[0].reshape(-1, 4)
    mid = ((segs[:, :2] + segs[:, 2:]) / 2).astype(int)
    on = M[np.clip(mid[:, 1], 0, cc.IMAGE_H - 1), np.clip(mid[:, 0], 0, cc.IMAGE_W - 1)] > 0
    d = segs[:, 2:] - segs[:, :2]; L = np.linalg.norm(d, axis=1)
    ang = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    keep = on & (L >= 8) & (ang < 20)
    S = segs[keep]; L = L[keep]
    lines = np.cross(np.c_[S[:, :2], np.ones(len(S))], np.c_[S[:, 2:], np.ones(len(S))])
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    mids = (S[:, :2] + S[:, 2:]) / 2; dirs = S[:, 2:] - S[:, :2]
    rng = np.random.default_rng(0); best = (-1, None, None)
    for _ in range(8000):
        i, j = rng.choice(len(S), 2, replace=False)
        v = np.cross(lines[i], lines[j])
        if abs(v[2]) < 1e-9: continue
        v = v[:2] / v[2]
        if v[1] < 1.5 * cc.IMAGE_H: continue
        to = v - mids
        cosang = np.abs(np.sum(dirs * to, 1)) / (np.linalg.norm(dirs, axis=1) * np.linalg.norm(to, axis=1) + 1e-9)
        inl = cosang > math.cos(math.radians(2.0))
        if L[inl].sum() > best[0]: best = (L[inl].sum(), v, inl)
    _, v, inl = best
    A = lines[inl]; w = L[inl]
    sol = np.linalg.lstsq(A[:, :2] * w[:, None], -A[:, 2] * w, rcond=None)[0]
    return sol, int(inl.sum()), len(S)


def run(label, f, p, r, ctx):
    bh, _ = fb.bev_unit(ctx["det_h"], f, p, r); bc, _ = fb.bev_unit(ctx["det_c"], f, p, r)
    c0, _ = fb.bev_unit(np.array([[cc.CX, cc.CY]]), f, p, r)
    t0 = time.perf_counter(); best, arg = fb.search(bh, bc, c0[0], ctx["maps"], ctx["box"], ctx["pad"]); dt = time.perf_counter() - t0
    box, truth = ctx["box"], ctx["truth"]
    ty, tx = int((box[3] - truth[1]) / fb.GRID), int((truth[0] - box[0]) / fb.GRID)
    yy, xx = np.unravel_index(np.argmax(best), best.shape)
    peak = (box[0] + (xx + .5) * fb.GRID, box[3] - (yy + .5) * fb.GRID); err = math.dist(peak, truth)
    k = int(40 / fb.GRID); near = best[max(0, ty - k):ty + k + 1, max(0, tx - k):tx + k + 1].max()
    tmp = best.copy(); tmp[max(0, ty - k):ty + k + 1, max(0, tx - k):tx + k + 1] = -1e9; above = 0
    while above <= 30:
        y2, x2 = np.unravel_index(np.argmax(tmp), tmp.shape)
        if tmp[y2, x2] <= near: break
        above += 1; kk = int(100 / fb.GRID); tmp[max(0, y2 - kk):y2 + kk + 1, max(0, x2 - kk):x2 + kk + 1] = -1e9
    a, b = arg[yy, xx]
    print(f"  {label:<30} best peak {err:5.0f} m from truth (heading {fb.HEADINGS[a]:.0f}, h {fb.HEIGHTS[b]:.0f})"
          f" | places above the truth: {above:>2} | {dt:.0f} s", flush=True)
    return dict(label=label, focal=f, pitch=p, roll=r, best_error_m=err, places_above_truth=above, seconds=dt)


def main():
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    box = areas["2x2km"]["bounds_utm"]; truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    tp = json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())["true_pose"]
    f0 = tp["focal_px"]; _, p0, r0 = tp["yaw_pitch_roll_deg"]
    photo = cv2.imread(str(cc.DATA / "chamaa_photo1.png"))
    vp, ni, ns = vp_from_buildings(photo)
    R = cc.rotation(0, p0, r0); up = R[:, 2]; vpt = np.array([cc.CX, cc.CY]) + f0 * up[:2] / up[2]
    pv, rv = fb.pitch_roll_from_vp(vp, f0)
    print(f"VP from building walls only: ({vp[0]:.0f}, {vp[1]:.0f}) from {ni} of {ns} segments; true ({vpt[0]:.0f}, {vpt[1]:.0f})")
    print(f"  implied at the true focal: pitch {pv:.1f}, roll {rv:.1f} (true {p0:.1f}, {r0:.1f})")
    pad_m = 2500.0; big = (box[0] - pad_m, box[1] - pad_m, box[2] + pad_m, box[3] + pad_m)
    extra = cc.Problem(dict(name="big", bounds_utm=list(big)))
    _, ojs = osm_junctions(); _, _, _, ijs, _ = image_junctions()
    ctx = dict(box=box, truth=truth, pad=(int(pad_m / fb.GRID),) * 2, det_h=extra.uv,
               det_c=np.array([j.xy for j in ijs]),
               maps=(fb.rasters([np.mean(np.array(m["polygon_utm"]), 0) for m in extra.map], big, fb.R_HOUSE),
                     fb.rasters([j["utm"] for j in ojs], big, fb.R_CROSS)))
    rows = [dict(vp_detected=vp.tolist(), vp_true=vpt.tolist(), vp_inliers=ni, implied=[pv, rv], true=[p0, r0])]
    print("\nsensitivity (coarse settings):")
    rows.append(run("exact", f0, p0, r0, ctx))
    for dp in (-6, -3, 3, 6):
        rows.append(run(f"pitch {dp:+d} deg", f0, p0 + dp, r0, ctx))
    for dr in (-4, 4):
        rows.append(run(f"roll {dr:+d} deg", f0, p0, r0 + dr, ctx))
    for fac in (0.85, 0.93, 1.07, 1.15):
        rows.append(run(f"focal x{fac}", f0 * fac, p0, r0, ctx))
    rows.append(run("building-wall VP", f0, pv, rv, ctx))
    (cc.RESULTS / "diagnostics/feasibility_sensitivity.json").write_text(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
