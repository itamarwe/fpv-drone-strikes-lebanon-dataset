#!/usr/bin/env python3
"""FULLY BLIND decomposed search: grid over the camera's tilt, roll and focal; for each,
rectify detections to a bird's-eye view, estimate camera height from apparent house sizes,
and search heading x height x position exhaustively in the box. Rank everything by the
chance-corrected z score. The truth is read only at the end, to report the error.

Grid spacing follows the measured tolerance of the 2D search (tilt ~+-6 deg, roll ~+-6 deg,
focal ~+-20%): tilt -15..-75 in 6 deg steps, roll -6/0/+6, focal 450-2900 px in x1.44 steps.
"""
import json, math, time
import numpy as np
import chamaa_constellation as cc
import feasibility_bev_search as fb
from feasibility_size_prior import ALL_HEIGHTS, height_from_size
from road_junctions import image_junctions, osm_junctions

fb.GRID = 8.0; fb.R_HOUSE, fb.R_CROSS = 16.0, 20.0
fb.HEADINGS = np.arange(0, 360, 3.0)
PITCHES = np.arange(-15, -76, -6.0)
ROLLS = (-6.0, 0.0, 6.0)
FOCALS = 450 * 1.44 ** np.arange(6)


def peaks(best, arg, box, n=3, sep_m=150):
    out = []; tmp = best.copy(); k = int(sep_m / fb.GRID)
    for _ in range(n):
        y, x = np.unravel_index(np.argmax(tmp), tmp.shape)
        a, b = arg[y, x]
        out.append(dict(z=float(tmp[y, x]), e=box[0] + (x + .5) * fb.GRID, n=box[3] - (y + .5) * fb.GRID,
                        heading=float(fb.HEADINGS[a]), height=float(fb.HEIGHTS[b])))
        tmp[max(0, y - k):y + k + 1, max(0, x - k):x + k + 1] = -1e9
    return out


def main():
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    box = areas["2x2km"]["bounds_utm"]
    pad_m = 2500.0; big = (box[0] - pad_m, box[1] - pad_m, box[2] + pad_m, box[3] + pad_m)
    extra = cc.Problem(dict(name="big", bounds_utm=list(big)))
    W_m = float(np.median([2 * max(np.linalg.norm(a), np.linalg.norm(b)) for a, b in extra.axes]))
    _, ojs = osm_junctions(); _, _, _, ijs, _ = image_junctions()
    det_h, det_c = extra.uv, np.array([j.xy for j in ijs])
    maps = (fb.rasters([np.mean(np.array(m["polygon_utm"]), 0) for m in extra.map], big, fb.R_HOUSE),
            fb.rasters([j["utm"] for j in ojs], big, fb.R_CROSS))
    pad = (int(pad_m / fb.GRID),) * 2
    results = []; t0 = time.perf_counter()
    combos = [(p, r, f) for p in PITCHES for r in ROLLS for f in FOCALS]
    for i, (p, r, f) in enumerate(combos):
        he = height_from_size(det_h, extra.wh[:, 0], W_m, f, p, r)
        fb.HEIGHTS = ALL_HEIGHTS[(ALL_HEIGHTS >= 0.6 * he) & (ALL_HEIGHTS <= 1.7 * he)]
        if len(fb.HEIGHTS) == 0:
            continue
        bh, _ = fb.bev_unit(det_h, f, p, r); bc, _ = fb.bev_unit(det_c, f, p, r)
        c0, ok = fb.bev_unit(np.array([[cc.CX, cc.CY]]), f, p, r)
        if not ok.all() or len(bh) < 15:
            continue
        best, arg = fb.search(bh, bc, c0[0], maps, box, pad)
        for k, pk in enumerate(peaks(best, arg, box)):
            results.append(dict(pitch=float(p), roll=float(r), focal=float(f), height_est=he, peak_rank_in_combo=k + 1, **pk))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(combos)} combos, {time.perf_counter() - t0:.0f} s", flush=True)
    results.sort(key=lambda d: -d["z"])
    elapsed = time.perf_counter() - t0
    # ---- evaluation only below this line
    truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    for d in results:
        d["error_m"] = math.dist((d["e"], d["n"]), truth)
    distinct = []
    for d in results:
        if all(math.dist((d["e"], d["n"]), (q["e"], q["n"])) > 150 for q in distinct):
            distinct.append(d)
    rank = next((i for i, d in enumerate(distinct, 1) if d["error_m"] <= 50), None)
    print(f"\n{len(combos)} camera combos searched in {elapsed / 60:.1f} min")
    print(f"{'rank':>4}{'z':>8}{'error m':>9}{'heading':>9}{'height':>8}{'pitch':>7}{'roll':>6}{'focal':>7}")
    for i, d in enumerate(distinct[:8], 1):
        print(f"{i:>4}{d['z']:>8.2f}{d['error_m']:>9.0f}{d['heading']:>9.0f}{d['height']:>8.0f}{d['pitch']:>7.0f}{d['roll']:>6.0f}{d['focal']:>7.0f}")
    print(f"rank of the first place within 50 m of the truth: {rank}")
    (cc.RESULTS / "blind_grid_search.json").write_text(json.dumps(dict(
        box="2x2km", combos=len(combos), minutes=elapsed / 60, rank_of_correct=rank,
        top=distinct[:20], grid=dict(pitch=PITCHES.tolist(), roll=list(ROLLS), focal=FOCALS.tolist())),
        indent=2, default=float))


if __name__ == "__main__":
    main()
