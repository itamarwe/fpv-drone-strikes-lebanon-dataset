#!/usr/bin/env python3
"""Houses scored symmetrically (F1), so a pose cannot win by flooding the frame.

The house term inherited from Sainte-Maxime charges only for DETECTED houses left
unmatched; projected map buildings with nothing there cost nothing. A pose that
crams a dense area into view gets more chances and scores better. Here every layer
is scored the same simple way:

  recall    = matched / detected
  precision = matched / map features projected inside the frame
  F1        = 2 P R / (P + R)

  houses:    one-to-one pairs with the metre-sigma assignment cost below the
             'unmatched' cost 9 (bimodal.py's house term), all 51 detections
  crossings: one-to-one within 30 px (diagnose_crossings_only.py)
  roads:     road F1 at 8 px (bimodal.py)

Objectives (lower is better):
  H    1 - F1_houses
  HC   (1 - F1_houses) + 0.3 (1 - F1_crossings)                  (no roads)
  HRC  (1 - F1_houses) + (1 - F1_roads) + 0.3 (1 - F1_crossings)

Diagnostic (uses the ground truth): every pose, truth included, gets the same local
refinement on each objective; then blind DE (2x2 km) checks whether any wrong pose
beats the refined truth.
"""
import json
import math

import numpy as np
from scipy.optimize import differential_evolution

import chamaa_constellation as cc
from diagnose_crossings_only import Cross
from fair_compare import RADIUS, refine

ALL = None


class Sym(Cross):
    def house_f1(self, p):
        _, pairs = self.assignment(p, np.arange(len(self.uv)))
        uv, _, ids = self.project(p)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < cc.IMAGE_W) & (uv[:, 1] >= 0) & (uv[:, 1] < cc.IMAGE_H - 12)).sum()
        m = len(pairs)
        if m == 0 or inside == 0:
            return 0., m, int(inside)
        prec, rec = m / max(inside, m), m / len(self.uv)
        return 2 * prec * rec / (prec + rec), m, int(inside)

    def o_h(self, p):
        self.calls += 1
        return 1 - self.house_f1(p)[0]

    def o_hc(self, p):
        self.calls += 1
        return 1 - self.house_f1(p)[0] + .3 * (1 - self.crossing_f1(p)[0])

    def o_hrc(self, p):
        self.calls += 1
        return 1 - self.house_f1(p)[0] + (1 - self.roads(p)[2]) + .3 * (1 - self.crossing_f1(p)[0])


def main():
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    pr = Sym(areas["2x2km"])
    fc = json.loads((cc.RESULTS / "diagnostics/fair_compare.json").read_text())["rows"]
    cand = {}
    for r in fc:                                    # the wrong poses from the fair comparison, as found
        if not r["pose"].startswith("TRUE") and r["pose"] not in cand:
            cand[r["pose"]] = np.array(r["params"])
    tp = np.array(json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())["parameters"], float)
    objs = {"H houses F1": pr.o_h, "HC houses+crossings (no roads)": pr.o_hc, "HRC houses+roads+crossings": pr.o_hrc}
    rows = []
    print("1) every pose refined identically on each objective")
    for name, fn in objs.items():
        rt = refine(pr, fn, tp)
        f1, m, n = pr.house_f1(rt)
        rows.append(dict(objective=name, pose="TRUE pose", error_m=math.dist(rt[:2] + pr.origin[:2], truth),
                         score=fn(rt), house_matched=m, house_in_frame=n))
        for lab, q in cand.items():
            rq = refine(pr, fn, q); f1, m, n = pr.house_f1(rq)
            rows.append(dict(objective=name, pose=lab, error_m=math.dist(rq[:2] + pr.origin[:2], truth),
                             score=fn(rq), house_matched=m, house_in_frame=n))
        tr = next(r for r in rows if r["objective"] == name and r["pose"] == "TRUE pose")
        wr = min((r for r in rows if r["objective"] == name and r["pose"] != "TRUE pose"), key=lambda r: r["score"])
        print(f"   {name:<32} truth {tr['score']:.3f} (houses {tr['house_matched']}/{tr['house_in_frame']} in frame) | "
              f"best wrong {wr['score']:.3f} at {wr['error_m']:.0f} m (houses {wr['house_matched']}/{wr['house_in_frame']})"
              f" -> {'TRUTH wins' if tr['score'] < wr['score'] else 'WRONG wins'} by {abs(wr['score'] - tr['score']):.3f}")
    print("\n2) blind DE (2x2 km, 3 seeds x popsize 16 x 240), each best refined like the truth")
    blind = {}
    for name, fn in objs.items():
        tr = next(r for r in rows if r["objective"] == name and r["pose"] == "TRUE pose")["score"]
        runs = []
        for s in (7, 19, 41):
            fit = differential_evolution(fn, pr.bounds, seed=s, popsize=16, maxiter=240, tol=.0002,
                                         polish=False, updating="immediate")
            rq = refine(pr, fn, fit.x); e = math.dist(rq[:2] + pr.origin[:2], truth)
            f1, m, n = pr.house_f1(rq)
            runs.append(dict(seed=s, score=fn(rq), error_m=e, house_matched=m, house_in_frame=n))
            print(f"   {name:<32} seed {s:>2}: {fn(rq):.3f} (truth {tr:.3f}) | error {e:6.0f} m | houses {m}/{n} in frame")
        blind[name] = dict(truth_score=tr, runs=runs, beat_truth=sum(r["score"] < tr for r in runs),
                           found_truth_within_50m=sum(r["error_m"] <= 50 for r in runs))
    (cc.RESULTS / "diagnostics/symmetric_houses.json").write_text(json.dumps(dict(
        warning="DIAGNOSTIC - includes the ground-truth pose", fixed=rows, blind=blind), indent=2, default=float))
    # picture the wrong pose that houses-F1 prefers over the truth, next to the truth
    import cv2
    from fair_compare import FIG, footprint, ortho_panel, photo_panel
    from road_junctions import image_junctions
    _, closed, _, _, _ = image_junctions()
    FIG.mkdir(parents=True, exist_ok=True)
    fit = differential_evolution(pr.o_h, pr.bounds, seed=7, popsize=16, maxiter=240, tol=.0002, polish=False,
                                 updating="immediate")
    wrong = refine(pr, pr.o_h, fit.x)
    truth_ref = refine(pr, pr.o_h, tp)
    tfp = footprint(pr, tp)
    for tag, q in (("truth", truth_ref), ("wrong", wrong)):
        f1, m, n = pr.house_f1(q)
        e = math.dist(q[:2] + pr.origin[:2], truth)
        head = (f"houses-F1 {'TRUE pose' if tag == 'truth' else 'blind winner'}: error {e:.0f} m | score {pr.o_h(q):.3f} | "
                f"houses matched {m} of {n} in frame, {m} of {len(pr.uv)} detected | road F1 {pr.roads(q)[2]:.2f}")
        a = photo_panel(pr, q, head, closed); b, _ = ortho_panel(pr, q, tfp, f"houses-F1 {tag}")
        b = cv2.resize(b, (int(b.shape[1] * a.shape[0] / b.shape[0]), a.shape[0]))
        cv2.imwrite(str(FIG / f"houses_f1_{tag}.jpg"), np.hstack([a, np.full((a.shape[0], 8, 3), 30, np.uint8), b]),
                    [cv2.IMWRITE_JPEG_QUALITY, 84])
    print("figures written")


if __name__ == "__main__":
    main()
