#!/usr/bin/env python3
"""Final stage: refine the grid search's top candidates in the full 3D model and judge them
with the exact chance-corrected significance S (significance.py). Blind until the last
column (error), which reads the truth."""
import json, math
import numpy as np
import chamaa_constellation as cc
from fair_compare import refine
from significance import Sig

areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
pr = Sig(areas["2x2km"])
top = json.loads((cc.RESULTS / "blind_grid_search.json").read_text())["top"][:8]
rows = []
for i, d in enumerate(top, 1):
    p = np.array([d["e"] - pr.origin[0], d["n"] - pr.origin[1], d["heading"], d["pitch"], d["roll"],
                  math.log(d["focal"]), math.log(d["height"])])
    p = np.clip(p, [b[0] for b in pr.bounds], [b[1] for b in pr.bounds])
    q = refine(pr, pr.o_sig, p)
    S = -pr.o_sig(q); sh, kh, _, qh = pr.houses_sig(q); sc, kc, _, qc = pr.crossings_sig(q)
    rows.append(dict(grid_rank=i, grid_z=d["z"], S=S, houses=kh, houses_expected=len(pr.uv) * qh, crossings=kc,
                     params=q.tolist(), ground=(q[:2] + pr.origin[:2]).tolist()))
rows.sort(key=lambda r: -r["S"])
truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
print(f"{'final':>5}{'grid':>6}{'S':>7}{'houses (chance)':>18}{'cross':>7}{'heading':>9}{'height':>8}{'focal':>7}{'error m':>9}")
for k, r in enumerate(rows, 1):
    q = r["params"]; r["error_m"] = math.dist(r["ground"], truth)
    print(f"{k:>5}{r['grid_rank']:>6}{r['S']:>7.1f}{r['houses']:>9} ({r['houses_expected']:4.1f}){r['crossings']:>7}"
          f"{q[2] % 360:>9.0f}{math.exp(q[6]):>8.0f}{math.exp(q[5]):>7.0f}{r['error_m']:>9.0f}")
print(f"margin S(best) - S(second): {rows[0]['S'] - rows[1]['S']:.1f}")
(cc.RESULTS / "blind_grid_verified.json").write_text(json.dumps(rows, indent=2, default=float))
