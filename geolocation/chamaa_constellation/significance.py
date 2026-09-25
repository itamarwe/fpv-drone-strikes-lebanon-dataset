#!/usr/bin/env python3
"""Score matches by how UNLIKELY they are by chance (houses + crossings, no roads).

Counting matches (or F1) can be gamed: a zoomed-in, low pose makes each tolerance
circle huge, so everything 'matches'. The fix is to compare the number of matches k
with the number expected by chance for that pose:

  q = fraction of the image covered by the tolerance discs around the map features
      projected into the frame (rasterised, overlaps counted once)
  k = one-to-one matches (the same matching as before)
  n = number of detections
  chance model: each detection lands in the covered area with probability q, so
      K ~ Binomial(n, q);  significance S = -log10 P(K >= k)

S rewards more matches and automatically penalises dense map areas, loose
tolerances and zoomed-in poses. Layers add: S_total = S_houses + S_crossings.
Objective (lower is better): -S_total.

  houses:    matched = one-to-one pairs with assignment cost < 9 (metre-sigma); the
             disc radius for map building j is 3 x its combined sigma (the same test)
  crossings: matched = one-to-one within 30 px; disc radius 30 px

Diagnostic (uses the ground truth): fair local refinement of every pose, then blind
DE (2x2 km) - does any wrong pose beat the refined truth?
"""
import json
import math

import cv2
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import binom

import chamaa_constellation as cc
from fair_compare import refine
from symmetric_houses import Sym

SCALE = 4                                   # coverage raster at 1/4 resolution
RW, RH = cc.IMAGE_W // SCALE, (cc.IMAGE_H - 12) // SCALE


def coverage(centres, radii):
    m = np.zeros((RH, RW), np.uint8)
    for (u, v), r in zip(centres, radii):
        cv2.circle(m, (int(u / SCALE), int(v / SCALE)), max(1, int(r / SCALE)), 1, -1)
    return float(m.mean())


def significance(k, n, q):
    if k <= 0:
        return 0.
    q = min(max(q, 1e-6), 1 - 1e-9)
    return float(-binom.logsf(k - 1, n, q) / math.log(10))


class Sig(Sym):
    def houses_sig(self, p):
        f, c, R, _ = self.unpack(p)
        _, pairs = self.assignment(p, np.arange(len(self.uv)))
        uv, _, ids = self.project(p)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < cc.IMAGE_W) & (uv[:, 1] >= 0) & (uv[:, 1] < cc.IMAGE_H - 12)
        if inside.sum() == 0:
            return 0., 0, 0, 1.
        depth = ((self.xyz[ids[inside]] - c) @ R.T)[:, 2]
        med = np.median(self.wh, axis=0)                              # typical detection size
        sig_obs = np.maximum(4, med * [.16, .30]).mean()
        sig_map = f * bm_sigma / np.maximum(depth, 1)
        radii = 3 * np.sqrt(sig_obs ** 2 + sig_map ** 2)
        q = coverage(uv[inside], radii)
        k = len(pairs)
        return significance(k, len(self.uv), q), k, int(inside.sum()), q

    def crossings_sig(self, p):
        f, c, R, _ = self.unpack(p)
        juv = self._in_frame(self.cross_world, f, c, R)
        _, k, n_in = self.crossing_f1(p)
        if n_in == 0:
            return 0., 0, 0, 1.
        q = coverage(juv, np.full(len(juv), 30.0))
        return significance(k, len(self.photo_crossings), q), k, n_in, q

    def o_sig(self, p):
        self.calls += 1
        return -(self.houses_sig(p)[0] + self.crossings_sig(p)[0])


from bimodal import SIGMA_MAP_M as bm_sigma  # noqa: E402


def main():
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    pr = Sig(areas["2x2km"])
    tp = np.array(json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())["parameters"], float)
    cand = {}
    for r in json.loads((cc.RESULTS / "diagnostics/fair_compare.json").read_text())["rows"]:
        if not r["pose"].startswith("TRUE") and r["pose"] not in cand:
            cand[r["pose"]] = np.array(r["params"])
    # the zoomed houses-F1 winner
    fit = differential_evolution(pr.o_h, pr.bounds, seed=7, popsize=16, maxiter=240, tol=.0002, polish=False,
                                 updating="immediate")
    cand["houses-F1 blind winner (zoomed)"] = refine(pr, pr.o_h, fit.x)

    def describe(q):
        sh, kh, nh, qh = pr.houses_sig(q); sc, kc, nc, qc = pr.crossings_sig(q)
        return (f"houses {kh}/{len(pr.uv)} matched, coverage {qh:.0%}, chance {len(pr.uv) * qh:.1f} -> S {sh:5.1f} | "
                f"crossings {kc}/{len(pr.photo_crossings)}, coverage {qc:.0%} -> S {sc:4.1f} | total {sh + sc:5.1f}")

    rows = []
    print("1) every pose refined identically on -S (higher S = more significant)")
    rt = refine(pr, pr.o_sig, tp)
    print(f"   {'TRUE pose':<36} err {math.dist(rt[:2] + pr.origin[:2], truth):5.0f} m | {describe(rt)}")
    rows.append(dict(pose="TRUE pose", error_m=math.dist(rt[:2] + pr.origin[:2], truth), S=-pr.o_sig(rt)))
    for lab, q in cand.items():
        rq = refine(pr, pr.o_sig, q)
        e = math.dist(rq[:2] + pr.origin[:2], truth)
        print(f"   {lab[:36]:<36} err {e:5.0f} m | {describe(rq)}")
        rows.append(dict(pose=lab, error_m=e, S=-pr.o_sig(rq)))
    print("\n2) blind DE on -S (2x2 km, 3 seeds x popsize 16 x 240), each refined like the truth")
    runs = []
    for s in (7, 19, 41):
        fit = differential_evolution(pr.o_sig, pr.bounds, seed=s, popsize=16, maxiter=240, tol=.0002,
                                     polish=False, updating="immediate")
        rq = refine(pr, pr.o_sig, fit.x); e = math.dist(rq[:2] + pr.origin[:2], truth)
        runs.append(dict(seed=s, S=-pr.o_sig(rq), error_m=e))
        print(f"   seed {s:>2}: err {e:5.0f} m | {describe(rq)}")
    (cc.RESULTS / "diagnostics/significance.json").write_text(json.dumps(dict(
        warning="DIAGNOSTIC - includes the ground-truth pose", truth_S=rows[0]["S"], fixed=rows, blind=runs,
        beat_truth=sum(r["S"] > rows[0]["S"] for r in runs)), indent=2, default=float))


if __name__ == "__main__":
    main()
