#!/usr/bin/env python3
"""Coarse-to-fine bimodal search: the same evidence, scored with an annealed tolerance.

Why: with the 8 px tolerance of bimodal.py the good basin is a needle (10 m of
position error, 2 deg of pitch or 5% of focal already score worse than wrong poses),
so no global sampler lands in it. Scoring every term softly with a tolerance tau
that starts wide and shrinks gives a funnel instead of a needle.

  soft(d, tau) = mean(min(d, tau) / tau)               0 = perfect, 1 = nothing within tau
  J_tau = W_HOUSE * soft(detection -> nearest projected building centre, tau)
        + W_ROAD  * (soft(projected road -> SAM road mask, tau)
                     + soft(SAM road skeleton -> projected road, tau)) / 2
        + W_JUNCTION * soft(detected crossing -> nearest projected OSM crossing, 2 tau)
  (fewer than 15 buildings or 20 road samples in view scores 1.0 for that term)

Schedule: global DE at tau = TAU_SCHEDULE[0], then Powell at each smaller tau, then
the final candidates are ranked by the FROZEN bimodal objective from bimodal.py
(8 px road F1, metre-sigma one-to-one houses, one-to-one crossings), so the definition
of a good answer does not change. Blind: reads only the search box. Weights are
bimodal.py's (houses 1, roads 1, crossings 0.3).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

import bimodal as bm
import chamaa_constellation as cc

TAU_SCHEDULE = (60.0, 30.0, 15.0, 8.0)
W_HOUSE, W_ROAD, W_JUNCTION = 1.0, bm.W_ROAD, bm.W_JUNCTION


class C2F(bm.Bimodal):
    def soft(self, p, tau):
        f, c, R, _ = self.unpack(p)
        # houses
        uv, w, ids = self.project(p)
        if len(ids) >= 15:
            d, _ = cKDTree(uv).query(self.uv[self.train], k=1)
            house = float(np.mean(np.minimum(d, tau) / tau))
        else:
            house = 1.0
        # roads, both directions
        ruv = self._in_frame(self.road_world, f, c, R)
        if len(ruv) >= 20:
            d1 = self.dist_to_road_mask[ruv[:, 1].astype(int), ruv[:, 0].astype(int)]
            d2, _ = cKDTree(ruv).query(self.skeleton_pts, k=1)
            road = float((np.mean(np.minimum(d1, tau) / tau) + np.mean(np.minimum(d2, tau) / tau)) / 2)
        else:
            road = 1.0
        # crossings
        juv = self._in_frame(self.cross_world, f, c, R)
        if len(juv) and len(self.photo_crossings):
            d, _ = cKDTree(juv).query(self.photo_crossings, k=1)
            cross = float(np.mean(np.minimum(d, 2 * tau) / (2 * tau)))
        else:
            cross = 1.0
        return W_HOUSE * house + W_ROAD * road + W_JUNCTION * cross

    def coarse_tau(self, tau):
        def fn(p):
            self.calls += 1
            return self.soft(p, tau)
        return fn


def search(area, iterations=240, seeds=(7, 19, 41), keep=12):
    pr = C2F(area); t0 = time.perf_counter(); timings = {}
    print(area["name"], "coarse-to-fine, tau", TAU_SCHEDULE, flush=True)
    pool, runs = [], []
    g = pr.coarse_tau(TAU_SCHEDULE[0])
    for seed in seeds:
        t = time.perf_counter()
        fit = differential_evolution(g, pr.bounds, seed=seed, popsize=16, maxiter=iterations,
                                     tol=.0002, polish=False, updating="immediate")
        pool.extend(fit.population[np.argsort(fit.population_energies)[:8]].tolist())
        runs.append(dict(seed=seed, seconds=time.perf_counter() - t, cost=float(fit.fun), evaluations=int(fit.nfev)))
        print("  global", runs[-1], flush=True)
    timings["global_de_s"] = time.perf_counter() - t0
    t = time.perf_counter(); unique = []
    for p in sorted(pool, key=g):
        if all(np.linalg.norm(np.array(p[:2]) - np.array(q[:2])) > 35 or abs(p[2] - q[2]) > 1 for q in unique):
            unique.append(np.array(p))
    cands = unique[:keep]
    for tau in TAU_SCHEDULE[1:]:                           # anneal: tighten and re-polish
        fn = pr.coarse_tau(tau); nxt = []
        for p in cands:
            fit = minimize(fn, p, method="Powell", bounds=pr.bounds, options=dict(maxiter=60, xtol=1e-4, ftol=1e-5))
            nxt.append(fit.x if fn(fit.x) < fn(p) else p)
        cands = nxt
    timings["anneal_powell_s"] = time.perf_counter() - t
    t = time.perf_counter(); refined = []
    for index, q in enumerate(sorted(cands, key=pr.objective)[:2]):   # final local refine on the FROZEN objective
        radius = [60, 60, 3, 2, 2, .08, .10]
        lb = [max(lo, v - r) for (lo, hi), v, r in zip(pr.bounds, q, radius)]
        ub = [min(hi, v + r) for (lo, hi), v, r in zip(pr.bounds, q, radius)]
        fit = differential_evolution(pr.objective, list(zip(lb, ub)), seed=71 + index, popsize=12, maxiter=150,
                                     tol=.0001, polish=False, x0=np.clip(q, lb, ub))
        refined.append(fit.x if pr.objective(fit.x) < pr.objective(q) else q)
    timings["local_refine_s"] = time.perf_counter() - t
    hyps = [dict(pr.payload(p), stage="c2f_annealed") for p in cands] + \
           [dict(pr.payload(p), stage="local_refine") for p in refined]
    hyps.sort(key=lambda r: r["train_cost"])                 # ranked by the frozen bimodal objective
    timings["total_s"] = time.perf_counter() - t0
    return pr, dict(area=area, mode="bimodal_c2f", seconds=timings, global_runs=runs,
                    objective_evaluations=pr.calls, parameter_names=cc.PARAMETER_NAMES, bounds=pr.bounds,
                    map_buildings=len(pr.map), road_samples=pr.n_road_samples, osm_crossings=pr.n_osm_crossings,
                    photo_crossings=len(pr.photo_crossings), detections=len(pr.observed),
                    train_indices=pr.train.tolist(), check_indices=pr.check.tolist(), observations=pr.observed,
                    hypotheses=hyps,
                    search=dict(strategy="coarse-to-fine annealed tolerance", tau_schedule_px=TAU_SCHEDULE,
                                de=dict(seeds=list(seeds), popsize=16, maxiter=iterations), kept=keep,
                                final_ranking="frozen bimodal objective (bimodal.py)"),
                    assumptions=dict(truth_used=False, heading_prior="none (0-360)"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--area", required=True)
    ap.add_argument("--iterations", type=int, default=240); args = ap.parse_args()
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    pr, out = search(areas[args.area], iterations=args.iterations)
    dst = cc.RESULTS / args.area / "bimodal_c2f"; dst.mkdir(parents=True, exist_ok=True)
    (dst / "search.json").write_text(json.dumps(out, indent=2))
    (dst / "map_buildings.json").write_text(json.dumps(pr.map))
    top = out["hypotheses"][0]
    print(json.dumps(dict(seconds=out["seconds"], J=top["train_cost"], road_f1=top["road_f1"],
                          crossings=top["crossings_matched"], footprint=top["footprint_centre_utm"]), indent=1))


if __name__ == "__main__":
    main()
