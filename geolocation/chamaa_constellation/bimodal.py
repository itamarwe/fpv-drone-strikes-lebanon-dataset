#!/usr/bin/env python3
"""Blind BIMODAL constellation search: houses + roads (+ crossings).

Same camera model, search box and protocol as chamaa_constellation.py (joint
mode), with an objective that adds road evidence. Everything below was fixed
before any bimodal run, and none of it reads the ground truth.

  J = house_cost / 9 + W_ROAD * (1 - road_F1) + W_JUNCTION * (1 - crossing_match)

  house_cost   the original one-to-one assignment cost (dummy 'unmatched' = 9), with
               the pixel tolerance widened by a map-accuracy term: sigma^2 =
               sigma_original^2 + (f * SIGMA_MAP_M / depth)^2. SIGMA_MAP_M = 4 m is
               typical OSM positional error. (A fixed-pose table showed no sigma
               makes houses alone prefer the truth, so this choice is not what
               could make the truth win.)
  road_F1      agreement between projected OSM road centrelines (DEM-lifted,
               sampled every 4 m) and the photo's SAM road mask: precision = share
               of projected samples within ROAD_TOL_PX of the mask; recall = share of
               photo road-skeleton points within ROAD_TOL_PX of a projected sample.
  crossing_match  share of detected photo crossings with a projected OSM crossing
               within JUNCTION_TOL_PX (nearest-neighbour in the coarse stage,
               one-to-one in refinement).

Roads within the search box expanded by ROAD_MARGIN_M are used, because the
photo's far field extends beyond the footprint; buildings stay box-limited as
before. Houses keep the 70/30 train/reserve split; the reserved houses remain an
independent held-out check. There is no held-out split for roads.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import numpy as np
from scipy.optimize import differential_evolution, linear_sum_assignment, minimize
from scipy.spatial import cKDTree

import chamaa_constellation as cc
from road_junctions import image_junctions, osm_junctions

SIGMA_MAP_M = 4.0
W_ROAD, W_JUNCTION = 1.0, 0.3
ROAD_TOL_PX, JUNCTION_TOL_PX = 8.0, 30.0
ROAD_SAMPLE_M, ROAD_MARGIN_M = 4.0, 1000.0


class Bimodal(cc.Problem):
    def __init__(self, area):
        super().__init__(area, "joint")
        _, closed, skel, ijs, _ = image_junctions()
        self.dist_to_road_mask = cv2.distanceTransform((closed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        sk = np.argwhere(skel)[:, ::-1].astype(float)
        self.skeleton_pts = sk[::3]                                   # ~uniform subsample
        self.photo_crossings = np.array([j.xy for j in ijs]) if ijs else np.zeros((0, 2))
        ways, ojs = osm_junctions()
        e0, n0, e1, n1 = self.bounds_utm; m = ROAD_MARGIN_M
        samples = []
        for w in ways:
            for a, b in zip(w["xy"][:-1], w["xy"][1:]):
                samples.append(np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / ROAD_SAMPLE_M))))
        s = np.vstack(samples)
        s = s[(s[:, 0] >= e0 - m) & (s[:, 0] <= e1 + m) & (s[:, 1] >= n0 - m) & (s[:, 1] <= n1 + m)]
        self.road_world = np.c_[s[:, 0] - self.origin[0], s[:, 1] - self.origin[1],
                                [self.terrain([[y, x]])[0] for x, y in s]]
        jxy = np.array([j["utm"] for j in ojs])
        jxy = jxy[(jxy[:, 0] >= e0 - m) & (jxy[:, 0] <= e1 + m) & (jxy[:, 1] >= n0 - m) & (jxy[:, 1] <= n1 + m)]
        self.cross_world = np.c_[jxy[:, 0] - self.origin[0], jxy[:, 1] - self.origin[1],
                                 [self.terrain([[y, x]])[0] for x, y in jxy]]
        self.n_road_samples, self.n_osm_crossings = len(self.road_world), len(self.cross_world)

    # ---------- road and crossing evidence
    def _in_frame(self, X, f, c, R):
        v = (X - c) @ R.T
        uv = f * v[:, :2] / np.maximum(v[:, 2:], 1) + [cc.CX, cc.CY]
        ok = (v[:, 2] > 20) & (uv[:, 0] >= 0) & (uv[:, 0] < cc.IMAGE_W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < cc.IMAGE_H - 13)
        return uv[ok]

    def roads(self, p):
        f, c, R, _ = self.unpack(p)
        uv = self._in_frame(self.road_world, f, c, R)
        if len(uv) < 20:
            return 0., 0., 0.
        prec = float((self.dist_to_road_mask[uv[:, 1].astype(int), uv[:, 0].astype(int)] <= ROAD_TOL_PX).mean())
        d, _ = cKDTree(uv).query(self.skeleton_pts, k=1, distance_upper_bound=ROAD_TOL_PX + 1)
        rec = float((d <= ROAD_TOL_PX).mean())
        return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.)

    def crossings(self, p, one_to_one=False):
        if len(self.photo_crossings) == 0:
            return 0., 0
        f, c, R, _ = self.unpack(p)
        uv = self._in_frame(self.cross_world, f, c, R)
        if len(uv) == 0:
            return 0., 0
        if not one_to_one:
            d, _ = cKDTree(uv).query(self.photo_crossings, k=1)
            m = int((d <= JUNCTION_TOL_PX).sum())
        else:
            D = np.linalg.norm(self.photo_crossings[:, None] - uv[None], axis=2)
            C = np.where(D <= JUNCTION_TOL_PX, D, 1e6)
            r, k = linear_sum_assignment(C); m = int((C[r, k] < 1e6).sum())
        return m / len(self.photo_crossings), m

    # ---------- houses with a metre-based tolerance
    def assignment(self, p, subset, excluded_map_indices=None):
        if len(subset) == 0:
            raise ValueError("Assignment requires a nonempty observation subset")
        uv, w, ids = self.project(p)
        f, c, R, _ = self.unpack(p)
        if excluded_map_indices:
            keep = np.array([int(i) not in excluded_map_indices for i in ids], dtype=bool)
            uv, w, ids = uv[keep], w[keep], ids[keep]
        if len(ids) == 0:
            return 9., []
        depth = ((self.xyz[ids] - c) @ R.T)[:, 2]
        observed = self.uv[subset]; wh = self.wh[subset]
        sig_obs = np.maximum([4, 4], wh * [.16, .30])
        sig_map = np.maximum(1e-6, f * SIGMA_MAP_M / np.maximum(depth, 1))
        sigma = np.sqrt(sig_obs[:, None, :] ** 2 + sig_map[None, :, None] ** 2)
        delta = (observed[:, None] - uv[None]) / sigma
        cost = np.sum(delta * delta, axis=2) + .35 * (np.log(wh[:, 0, None] / w[None]) / .65) ** 2
        aug = np.c_[cost, np.full((len(subset), len(subset)), 9.)]
        row, col = linear_sum_assignment(aug); values = aug[row, col]; pairs = []
        for a, b, value in zip(row, col, values):
            if b >= len(ids) or value >= 9:
                continue
            oi = int(subset[a]); mi = int(ids[b])
            pairs.append(dict(observed_index=oi, observed_id=self.observed[oi]["id"], map_index=mi,
                              osm_id=self.map[mi]["osm_id"], observed_xy=self.uv[oi].tolist(),
                              projected_xy=uv[b].tolist(), observed_width_px=float(wh[a, 0]),
                              projected_width_px=float(w[b]), cost=float(value),
                              pixel_distance=float(np.linalg.norm(self.uv[oi] - uv[b]))))
        return float(values.mean()), pairs

    # ---------- objectives
    def coarse(self, p):
        self.calls += 1
        house = super().coarse(p)                      # original coarse house term (20 if < 15 visible)
        if house >= 20:
            return 5.
        _, _, f1 = self.roads(p)
        jm, _ = self.crossings(p)
        return house / 9 + W_ROAD * (1 - f1) + W_JUNCTION * (1 - jm)

    def objective(self, p):
        house, _ = self.assignment(p, self.train)
        _, _, f1 = self.roads(p)
        jm, _ = self.crossings(p, one_to_one=True)
        return house / 9 + W_ROAD * (1 - f1) + W_JUNCTION * (1 - jm)

    def payload(self, p):
        out = super().payload(p)                        # houses: train/check/overall with metre sigma
        prec, rec, f1 = self.roads(p); jm, jn = self.crossings(p, one_to_one=True)
        out.update(house_train_cost=out["train_cost"], train_cost=self.objective(p),
                   road_precision=prec, road_recall=rec, road_f1=f1,
                   crossings_matched=jn, crossings_detected=len(self.photo_crossings), crossing_match=jm)
        return out


def search(area, iterations=240, seeds=(7, 19, 41)):
    pr = Bimodal(area); t0 = time.perf_counter(); timings = {}
    print(area["name"], "detections", len(pr.observed), "map", len(pr.map), "road samples", pr.n_road_samples,
          "osm crossings", pr.n_osm_crossings, "photo crossings", len(pr.photo_crossings), flush=True)
    pool, runs = [], []
    for seed in seeds:
        t = time.perf_counter()
        fit = differential_evolution(pr.coarse, pr.bounds, seed=seed, popsize=16, maxiter=iterations,
                                     tol=.0002, polish=False, updating="immediate")
        pool.extend(fit.population[np.argsort(fit.population_energies)[:8]].tolist())
        runs.append(dict(seed=seed, seconds=time.perf_counter() - t, cost=float(fit.fun), evaluations=int(fit.nfev)))
        print("  global", runs[-1], flush=True)
    timings["global_de_s"] = time.perf_counter() - t0
    t = time.perf_counter(); unique = []
    for p in sorted(pool, key=pr.coarse):
        if all(np.linalg.norm(np.array(p[:2]) - np.array(q[:2])) > 35 or abs(p[2] - q[2]) > 1 for q in unique):
            unique.append(p)
    polished = []
    for p in unique[:12]:
        fit = minimize(pr.objective, p, method="Powell", bounds=pr.bounds, options=dict(maxiter=45, xtol=1e-4, ftol=1e-5))
        polished.append(fit.x if pr.objective(fit.x) < pr.objective(p) else np.array(p))
    timings["powell_s"] = time.perf_counter() - t
    t = time.perf_counter(); refined = []
    for index, q in enumerate(sorted(polished, key=pr.objective)[:2]):
        radius = [350, 350, 5, 3, 3, .20, .20]
        lb = [max(lo, v - r) for (lo, hi), v, r in zip(pr.bounds, q, radius)]
        ub = [min(hi, v + r) for (lo, hi), v, r in zip(pr.bounds, q, radius)]
        fit = differential_evolution(pr.objective, list(zip(lb, ub)), seed=71 + index, popsize=12, maxiter=150,
                                     tol=.0001, polish=False, x0=np.clip(q, lb, ub))
        refined.append(fit.x if pr.objective(fit.x) < pr.objective(q) else q)
    timings["local_refine_s"] = time.perf_counter() - t
    hyps = [dict(pr.payload(p), stage="global+powell") for p in polished] + \
           [dict(pr.payload(p), stage="local_refine") for p in refined]
    hyps.sort(key=lambda r: r["train_cost"])
    timings["total_s"] = time.perf_counter() - t0
    return pr, dict(area=area, mode="bimodal", seconds=timings, global_runs=runs, objective_evaluations=pr.calls,
                    parameter_names=cc.PARAMETER_NAMES, bounds=pr.bounds, map_buildings=len(pr.map),
                    road_samples=pr.n_road_samples, osm_crossings=pr.n_osm_crossings,
                    photo_crossings=len(pr.photo_crossings), detections=len(pr.observed),
                    train_indices=pr.train.tolist(), check_indices=pr.check.tolist(),
                    observations=pr.observed, hypotheses=hyps,
                    objective=dict(formula="house/9 + W_ROAD*(1-road_F1) + W_JUNCTION*(1-crossing_match)",
                                   sigma_map_m=SIGMA_MAP_M, w_road=W_ROAD, w_junction=W_JUNCTION,
                                   road_tol_px=ROAD_TOL_PX, junction_tol_px=JUNCTION_TOL_PX,
                                   road_sample_m=ROAD_SAMPLE_M, road_margin_m=ROAD_MARGIN_M),
                    assumptions=dict(truth_used=False, heading_prior="none (0-360)"))


def search_stratified(area, iterations=240, sectors=8, keep_per_sector=3, refine_top=3):
    """Heading-stratified multi-start: one DE per heading sector, so every heading is
    searched. Blind: no sector is favoured; candidates compete on the full objective."""
    pr = Bimodal(area); t0 = time.perf_counter(); timings = {}
    width = 360.0 / sectors; pool, runs = [], []
    print(area["name"], "stratified:", sectors, "heading sectors", flush=True)
    for k in range(sectors):
        b = list(pr.bounds); b[2] = (k * width, (k + 1) * width)
        t = time.perf_counter()
        fit = differential_evolution(pr.coarse, b, seed=1000 + k, popsize=16, maxiter=iterations,
                                     tol=.0002, polish=False, updating="immediate")
        order = np.argsort(fit.population_energies)
        kept = []
        for i in order:                                  # best distinct states of this sector
            q = fit.population[i]
            if all(np.linalg.norm(q[:2] - r[:2]) > 35 for r in kept):
                kept.append(q)
            if len(kept) == keep_per_sector:
                break
        pool.extend(kept)
        runs.append(dict(sector_deg=[k * width, (k + 1) * width], seconds=time.perf_counter() - t,
                         cost=float(fit.fun), evaluations=int(fit.nfev)))
        print("  sector", runs[-1], flush=True)
    timings["global_de_s"] = time.perf_counter() - t0
    t = time.perf_counter(); polished = []
    for p in pool:
        fit = minimize(pr.objective, p, method="Powell", bounds=pr.bounds, options=dict(maxiter=45, xtol=1e-4, ftol=1e-5))
        polished.append(fit.x if pr.objective(fit.x) < pr.objective(p) else np.array(p))
    timings["powell_s"] = time.perf_counter() - t
    t = time.perf_counter(); refined = []
    for index, q in enumerate(sorted(polished, key=pr.objective)[:refine_top]):
        radius = [350, 350, 5, 3, 3, .20, .20]
        lb = [max(lo, v - r) for (lo, hi), v, r in zip(pr.bounds, q, radius)]
        ub = [min(hi, v + r) for (lo, hi), v, r in zip(pr.bounds, q, radius)]
        fit = differential_evolution(pr.objective, list(zip(lb, ub)), seed=71 + index, popsize=12, maxiter=150,
                                     tol=.0001, polish=False, x0=np.clip(q, lb, ub))
        refined.append(fit.x if pr.objective(fit.x) < pr.objective(q) else q)
    timings["local_refine_s"] = time.perf_counter() - t
    hyps = [dict(pr.payload(p), stage="sector+powell") for p in polished] + \
           [dict(pr.payload(p), stage="local_refine") for p in refined]
    hyps.sort(key=lambda r: r["train_cost"])
    timings["total_s"] = time.perf_counter() - t0
    return pr, dict(area=area, mode="bimodal_stratified", seconds=timings, global_runs=runs,
                    objective_evaluations=pr.calls, parameter_names=cc.PARAMETER_NAMES, bounds=pr.bounds,
                    map_buildings=len(pr.map), road_samples=pr.n_road_samples, osm_crossings=pr.n_osm_crossings,
                    photo_crossings=len(pr.photo_crossings), detections=len(pr.observed),
                    train_indices=pr.train.tolist(), check_indices=pr.check.tolist(), observations=pr.observed,
                    hypotheses=hyps,
                    search=dict(strategy="heading-stratified multi-start", sectors=sectors,
                                keep_per_sector=keep_per_sector, refine_top=refine_top,
                                de_per_sector=dict(popsize=16, maxiter=iterations)),
                    objective=dict(formula="house/9 + W_ROAD*(1-road_F1) + W_JUNCTION*(1-crossing_match)",
                                   sigma_map_m=SIGMA_MAP_M, w_road=W_ROAD, w_junction=W_JUNCTION),
                    assumptions=dict(truth_used=False, heading_prior="none; all 8 sectors searched equally"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--area", required=True)
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--iterations", type=int, default=240); args = ap.parse_args()
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    if args.stratified:
        pr, out = search_stratified(areas[args.area], iterations=args.iterations)
        dst = cc.RESULTS / args.area / "bimodal_stratified"
    else:
        pr, out = search(areas[args.area], iterations=args.iterations)
        dst = cc.RESULTS / args.area / "bimodal"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "search.json").write_text(json.dumps(out, indent=2))
    (dst / "map_buildings.json").write_text(json.dumps(pr.map))
    top = out["hypotheses"][0]
    print(json.dumps(dict(seconds=out["seconds"], J=top["train_cost"], house=top["house_train_cost"],
                          road_f1=top["road_f1"], crossings=top["crossings_matched"],
                          footprint=top["footprint_centre_utm"]), indent=1))


if __name__ == "__main__":
    main()
