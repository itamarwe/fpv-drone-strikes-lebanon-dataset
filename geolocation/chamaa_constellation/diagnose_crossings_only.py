#!/usr/bin/env python3
"""Could crossings alone (or houses + crossings, no road lines) carry the geolocation?

Two objectives, lower is better:
  A  crossings only:   1 - crossing_F1
  B  houses+crossings: house/9 + 1 - crossing_F1       (house term as in bimodal.py)
crossing_F1: one-to-one matches within 30 px between the 10 detected photo
crossings and the OSM crossings projected into the frame; precision =
matched / OSM crossings in frame, recall = matched / detected.

Checks (the first uses the ground truth; it is a diagnostic, not a result):
  1. at fixed poses: the true pose vs every earlier blind winner vs 3,000 random poses
  2. blind DE on each objective (same protocol as before, 2x2 km box): does the
     search find poses that score BETTER than the truth? If so, the objective
     itself would mislead any search, including a proposal-driven one.
"""
import json
import math

import numpy as np
from scipy.optimize import differential_evolution, linear_sum_assignment

import bimodal as bm
import chamaa_constellation as cc

TOL = 30.0


class Cross(bm.Bimodal):
    def crossing_f1(self, p):
        f, c, R, _ = self.unpack(p)
        uv = self._in_frame(self.cross_world, f, c, R)
        n_det = len(self.photo_crossings)
        if len(uv) == 0 or n_det == 0:
            return 0., 0, len(uv)
        D = np.linalg.norm(self.photo_crossings[:, None] - uv[None], axis=2)
        C = np.where(D <= TOL, D, 1e6); r, k = linear_sum_assignment(C); m = int((C[r, k] < 1e6).sum())
        if m == 0:
            return 0., 0, len(uv)
        prec, rec = m / len(uv), m / n_det
        return 2 * prec * rec / (prec + rec), m, len(uv)

    def obj_a(self, p):
        self.calls += 1
        return 1 - self.crossing_f1(p)[0]

    def obj_b(self, p):
        self.calls += 1
        return self.assignment(p, self.train)[0] / 9 + 1 - self.crossing_f1(p)[0]


def main():
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    pr = Cross(areas["2x2km"])
    tp = np.array(json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())["parameters"], float)
    poses = [("TRUE pose", 0.0, tp)]
    for name, sub in (("2x2km", ""), ("2x2km", "bimodal"), ("2x2km", "bimodal_stratified"),
                      ("4x4km", "bimodal"), ("village", "bimodal")):
        h = min(json.loads((cc.RESULTS / name / sub / "search.json").read_text())["hypotheses"], key=lambda r: r["train_cost"])
        q = np.array(h["parameters"][:7], float); b = areas[name]["bounds_utm"]
        q[0] += b[0] - pr.origin[0]; q[1] += b[1] - pr.origin[1]
        poses.append((f"winner {name} {sub or 'house-only'}", math.dist(h["footprint_centre_utm"], truth), q))
    print("1) fixed poses")
    print(f"   {'pose':<34}{'err m':>7}{'cross F1':>10}{'matched':>9}{'in frame':>10}{'A':>8}{'B':>8}")
    for lab, e, q in poses:
        f1, m, n = pr.crossing_f1(q)
        print(f"   {lab:<34}{e:>7.0f}{f1:>10.2f}{m:>9}{n:>10}{pr.obj_a(q):>8.3f}{pr.obj_b(q):>8.3f}")
    rng = np.random.default_rng(5)
    R = [np.array([rng.uniform(lo, hi) for lo, hi in pr.bounds]) for _ in range(3000)]
    a = np.array([pr.obj_a(q) for q in R]); b = np.array([pr.obj_b(q) for q in R])
    ta, tb = pr.obj_a(tp), pr.obj_b(tp)
    print(f"   3,000 random poses: A best {a.min():.3f} ({(a <= ta).sum()} at or below the truth's {ta:.3f}); "
          f"B best {b.min():.3f} ({(b <= tb).sum()} at or below the truth's {tb:.3f})")

    print("\n2) blind DE (2x2 km, 3 seeds x popsize 16 x 240), same protocol as the earlier searches")
    out = dict(fixed_poses=[], blind={})
    for key, fn, tval in (("A_crossings_only", pr.obj_a, ta), ("B_houses_plus_crossings", pr.obj_b, tb)):
        found = []
        for seed in (7, 19, 41):
            fit = differential_evolution(fn, pr.bounds, seed=seed, popsize=16, maxiter=240, tol=.0002,
                                         polish=False, updating="immediate")
            g = fit.x[:2] + pr.origin[:2]
            f1, m, n = pr.crossing_f1(fit.x)
            found.append(dict(seed=seed, score=float(fit.fun), error_m=math.dist(g, truth), crossing_f1=f1,
                              matched=m, in_frame=n, yaw=float(fit.x[2])))
            print(f"   {key:<26} seed {seed:>2}: score {fit.fun:.3f} (truth {tval:.3f}) | error {math.dist(g, truth):6.0f} m | "
                  f"F1 {f1:.2f} ({m} matched, {n} in frame) | heading {fit.x[2]:.0f}")
        out["blind"][key] = dict(truth_score=tval, runs=found,
                                 better_than_truth=sum(r["score"] < tval for r in found))
    for lab, e, q in poses:
        f1, m, n = pr.crossing_f1(q)
        out["fixed_poses"].append(dict(pose=lab, error_m=e, crossing_f1=f1, matched=m, in_frame=n,
                                       A=pr.obj_a(q), B=pr.obj_b(q)))
    out["random_poses"] = dict(n=3000, A_min=float(a.min()), A_at_or_below_truth=int((a <= ta).sum()),
                               B_min=float(b.min()), B_at_or_below_truth=int((b <= tb).sum()))
    out["warning"] = "DIAGNOSTIC - fixed-pose rows include the ground-truth pose"
    (cc.RESULTS / "diagnostics/crossings_only.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
