#!/usr/bin/env python3
"""Does a camera-height estimate from apparent house size remove the degenerate low-height
edge, and how much pitch/roll/focal error does the exhaustive BEV search then tolerate?
(research check; the truth is used only to score)

Height from size: each detection's bbox width w (px) vs the median OSM footprint width W (m)
gives range ~ f W / w; with pitch/roll known, height = range * sin(depression of its ray).
The search then only considers heights within [0.6, 1.7] x the median estimate.
"""
import json, math
import cv2, numpy as np
import chamaa_constellation as cc
import feasibility_bev_search as fb
import feasibility_sensitivity as fs
from road_junctions import image_junctions, osm_junctions

fb.GRID = 8.0; fb.R_HOUSE, fb.R_CROSS = 16.0, 20.0
fb.HEADINGS = np.arange(0, 360, 3.0)
ALL_HEIGHTS = np.exp(np.arange(np.log(60), np.log(900), np.log(1.06)))


def height_from_size(det_uv, det_w, W_m, f, p, r):
    R = fb.rot_from(p, r)
    rays = np.c_[(det_uv[:, 0] - cc.CX) / f, (det_uv[:, 1] - cc.CY) / f, np.ones(len(det_uv))]
    rays_w = rays @ R; n = np.linalg.norm(rays_w, axis=1)
    sin_dep = -rays_w[:, 2] / n
    rng = f * W_m / det_w * n              # distance along the ray (w measured perpendicular-ish)
    h = rng * sin_dep
    return float(np.median(h[sin_dep > 0.05]))


def main():
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    box = areas["2x2km"]["bounds_utm"]; truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    tp = json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())["true_pose"]
    f0 = tp["focal_px"]; _, p0, r0 = tp["yaw_pitch_roll_deg"]; h0 = tp["height_above_ground_m"]
    pad_m = 2500.0; big = (box[0] - pad_m, box[1] - pad_m, box[2] + pad_m, box[3] + pad_m)
    extra = cc.Problem(dict(name="big", bounds_utm=list(big)))
    widths = [2 * max(np.linalg.norm(a), np.linalg.norm(b)) for a, b in extra.axes]
    W_m = float(np.median(widths))
    _, ojs = osm_junctions(); _, _, _, ijs, _ = image_junctions()
    ctx = dict(box=box, truth=truth, pad=(int(pad_m / fb.GRID),) * 2, det_h=extra.uv,
               det_c=np.array([j.xy for j in ijs]),
               maps=(fb.rasters([np.mean(np.array(m["polygon_utm"]), 0) for m in extra.map], big, fb.R_HOUSE),
                     fb.rasters([j["utm"] for j in ojs], big, fb.R_CROSS)))
    print(f"median OSM footprint length {W_m:.1f} m; true camera height {h0:.0f} m")
    cases = [("exact", f0, p0, r0)] + [(f"pitch {d:+d}", f0, p0 + d, r0) for d in (-8, -6, -3, 3, 6, 8)] + \
            [(f"roll {d:+d}", f0, p0, r0 + d) for d in (-6, -4, 4, 6)] + \
            [(f"focal x{k}", f0 * k, p0, r0) for k in (0.8, 0.9, 1.1, 1.2)]
    rows = []
    for lab, f, p, r in cases:
        he = height_from_size(extra.uv, extra.wh[:, 0], W_m, f, p, r)
        fb.HEIGHTS = ALL_HEIGHTS[(ALL_HEIGHTS >= 0.6 * he) & (ALL_HEIGHTS <= 1.7 * he)]
        res = fs.run(f"{lab} (h est {he:.0f} m)", f, p, r, ctx)
        res["height_estimate_m"] = he; rows.append(res)
    (cc.RESULTS / "diagnostics/feasibility_size_prior.json").write_text(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
