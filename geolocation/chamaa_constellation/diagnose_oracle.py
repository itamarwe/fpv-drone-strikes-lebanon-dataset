#!/usr/bin/env python3
"""ORACLE DIAGNOSTIC - uses the ground truth. NOT part of the blind result.

Question: did the blind search fail because the optimiser never reached the
right place (a SEARCH failure), or because the cost function does not prefer the
right place even when you are there (a MODEL failure)?

Method: the identical optimiser and cost, but with the footprint centre confined
to +/-60 m of the truth. Heading, pitch, roll, focal and height stay free and
unconstrained. The best cost reachable there is compared with the cost of the
blind winner. If the truth region scores clearly better, more search would fix
it; if it scores the same or worse, the model cannot see this scene.
"""
import json
import math

import numpy as np
from scipy.optimize import differential_evolution, minimize

import chamaa_constellation as cc

RADIUS_M = 60.0


def main():
    truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    areas = json.loads((cc.DATA / "search_areas.json").read_text())["areas"]
    area = next(a for a in areas if a["name"] == "2x2km")
    pr = cc.Problem(area)
    tx, ty = truth[0] - pr.origin[0], truth[1] - pr.origin[1]
    bounds = list(pr.bounds)
    bounds[0] = (tx - RADIUS_M, tx + RADIUS_M); bounds[1] = (ty - RADIUS_M, ty + RADIUS_M)
    pool = []
    for seed in (7, 19, 41):
        fit = differential_evolution(pr.coarse, bounds, seed=seed, popsize=16, maxiter=240,
                                     tol=.0002, polish=False, updating="immediate")
        pool.extend(fit.population[np.argsort(fit.population_energies)[:8]].tolist())
        print("oracle global", seed, round(float(fit.fun), 3), flush=True)
    best = None
    for p in sorted(pool, key=pr.coarse)[:12]:
        f = minimize(lambda q: pr.assignment(q, pr.train)[0], p, method="Powell", bounds=bounds,
                     options=dict(maxiter=45, xtol=1e-4, ftol=1e-5))
        q = f.x if pr.assignment(f.x, pr.train)[0] < pr.assignment(p, pr.train)[0] else np.array(p)
        if best is None or pr.assignment(q, pr.train)[0] < pr.assignment(best, pr.train)[0]:
            best = q
    radius = [RADIUS_M, RADIUS_M, 5, 3, 3, .20, .20]
    lb = [max(lo, v - r) for (lo, hi), v, r in zip(bounds, best, radius)]
    ub = [min(hi, v + r) for (lo, hi), v, r in zip(bounds, best, radius)]
    fit = differential_evolution(lambda x: pr.assignment(x, pr.train)[0], list(zip(lb, ub)), seed=71,
                                 popsize=12, maxiter=150, tol=.0001, polish=False, x0=np.clip(best, lb, ub))
    if pr.assignment(fit.x, pr.train)[0] < pr.assignment(best, pr.train)[0]:
        best = fit.x
    out = pr.payload(best)
    blind = json.loads((cc.RESULTS / "2x2km/search.json").read_text())["hypotheses"][0]
    res = dict(warning="ORACLE DIAGNOSTIC - footprint centre constrained to +/-60 m of the truth; not a blind result",
               constraint_radius_m=RADIUS_M,
               oracle=dict(train_cost=out["train_cost"], check_cost=out["check_cost"],
                           inlier_buildings=out["overall_matches"], train_matches=out["train_matches"],
                           check_matches=out["check_matches"], focal_px=out["focal_px"],
                           yaw_pitch_roll_deg=out["yaw_pitch_roll_deg"],
                           height_above_ground_m=out["height_above_ground_m"],
                           error_m=math.dist(out["footprint_centre_utm"], truth)),
               blind_winner_2x2=dict(train_cost=blind["train_cost"], check_cost=blind["check_cost"],
                                     inlier_buildings=blind["overall_matches"],
                                     error_m=math.dist(blind["footprint_centre_utm"], truth)),
               pairs=out["pairs"], parameters=out["parameters"])
    dst = cc.RESULTS / "diagnostics"; dst.mkdir(parents=True, exist_ok=True)
    (dst / "oracle_near_truth.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("pairs", "parameters")}, indent=1))


if __name__ == "__main__":
    main()
