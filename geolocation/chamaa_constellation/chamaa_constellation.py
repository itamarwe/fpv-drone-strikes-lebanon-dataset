#!/usr/bin/env python3
"""Blind building-constellation camera search on the Chamaa aerial photo.

A port of the Sainte-Maxime method (docs/constellation_geolocation_method.md,
original code in original_sainte_maxime/joint_building_constellation.py and
refine_building_constellations.py), run blind inside one search box.

What is reproduced unchanged
  * rotation() convention: world east/north/up, camera rows right/down/forward,
    Xc = R (Xw - C), u = f Xc.x / Xc.z + cx.
  * Image side: SAM 3 masks with prompts building/house/roof, threshold 0.3,
    mask-IoU dedup 0.72 (produced by original_sainte_maxime/segment_cross_modal_anchors.py);
    mask centroids as observations, bbox width as a loose size cue; the same
    near-duplicate rule; a fixed 70/30 train/reserve split with seed 7.
  * Map side: footprint centroid, minAreaRect half-axes for projected width,
    the same 45-2500 m2 area filter.
  * Coarse cost: nearest neighbour in (u/8, v/6, log w/0.65), d2 capped at 9, a
    uniqueness penalty and the log-count term; < 15 visible map points costs 20.
  * Refinement cost: one-to-one linear_sum_assignment with dedicated dummy
    columns at cost 9, sigma = max(4 px, [0.16, 0.30] x bbox), plus the
    0.35 x (log width ratio / 0.65)^2 width term.
  * Search protocol: differential evolution with 3 seeds (7, 19, 41), popsize 16,
    maxiter 240, tol 2e-4, no polish, immediate updating; the best 8 of each
    population pooled; states deduplicated; the best 12 polished with Powell on
    the training assignment cost (keeping the start if Powell got worse); the best
    2 refined with a local DE (popsize 12, maxiter 150, seed 71+i) inside
    +/-350 m, +/-5 deg yaw, +/-3 deg pitch/roll, +/-0.20 log focal.
  * Selection uses TRAINING detections only; reserved detections may not reuse map
    identities consumed by training assignments.

What had to change, and why (all stated in results/README.md)
  1. Camera height. Sainte-Maxime fixed the camera at terrain + 2 m (a phone on
     the ground). This is an aerial photo, so height above the viewed ground is a
     searched parameter (40-1200 m).
  2. Parametrisation. The search state is the ground point hit by the image-centre
     ray (kept inside the search box), not the camera position; the camera is
     placed back along that ray at the searched height. This makes "the search
     area" mean where the photo looks, and the estimate is directly the footprint
     centre that the ground truth describes.
  3. Heading. Sainte-Maxime used a northward prior (yaw -20..35 deg). The blind
     test here has no heading prior: yaw is the full 0..360 deg. Pitch is
     -80..-12 deg (looking down), roll -10..10, focal 400-3000 px.
  4. Map heights. IGN supplied roof elevations. Here roof centre = 10 m DEM terrain
     + building height, from OSM building:levels x 3 m when tagged, else 7 m.
  5. Image filters are rescaled to this photo (see DETECTION_FILTER), fixed from
     detection statistics alone before any search.

The search reads data/search_areas.json (box bounds) and nothing else about
location. data/truth.json is read only by evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution, linear_sum_assignment, minimize
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
DEM = Path.home() / "Documents/code/mappa/layers/height_evoPilot.tif"
IMAGE_W, IMAGE_H = 1194, 906
CX, CY = IMAGE_W / 2, IMAGE_H / 2
DEFAULT_BUILDING_HEIGHT_M = 7.0
DETECTION_FILTER = dict(min_score=0.35, min_area=100, max_area=9000, min_w=8, max_w=160,
                        max_h=200, edge_px=5, bottom_band_px=12)
BOUNDS_ANGLES = [(0.0, 360.0), (-80.0, -12.0), (-10.0, 10.0)]
LOG_FOCAL = (np.log(400.0), np.log(3000.0))
LOG_HEIGHT = (np.log(40.0), np.log(1200.0))
PARAMETER_NAMES = ["ground_east_offset", "ground_north_offset", "yaw_deg", "pitch_deg",
                   "roll_deg", "log_focal", "log_height_above_ground"]
MODES = ("joint", "joint_offset")   # the write-up's 'fixed' mode needs a prior focal; none exists here


def rotation(yaw, pitch, roll):
    """Unchanged from the Sainte-Maxime implementation."""
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.]); down = np.cross(forward, right)
    return np.array([right * np.cos(roll) + down * np.sin(roll), down * np.cos(roll) - right * np.sin(roll), forward])


def load_detections(seed=7):
    raw = json.loads((DATA / "sam_photo1/segments.json").read_text())["ground"]["features"]
    F = DETECTION_FILTER
    selected = []
    for p in sorted(raw, key=lambda a: -a["score"]):
        x0, y0, x1, y1 = p["bbox_xyxy"]; w = x1 - x0; h = y1 - y0
        if not (p["score"] >= F["min_score"] and F["min_area"] <= p["area_px"] <= F["max_area"]
                and F["min_w"] < w < F["max_w"] and h < F["max_h"] and x0 > F["edge_px"]
                and x1 < IMAGE_W - F["edge_px"] and y0 > F["edge_px"]
                and y1 < IMAGE_H - F["bottom_band_px"]):
            continue
        # Same near-duplicate rule as the original.
        if any(np.linalg.norm(np.array(p["centroid_xy"]) - np.array(q["centroid_xy"]))
               < max(5, .2 * min(w, q["bbox_xyxy"][2] - q["bbox_xyxy"][0])) for q in selected):
            continue
        selected.append(p)
    if len(selected) < 12:
        raise ValueError("At least 12 usable building detections are required")
    uv = np.array([p["centroid_xy"] for p in selected])
    wh = np.array([[p["bbox_xyxy"][2] - p["bbox_xyxy"][0], p["bbox_xyxy"][3] - p["bbox_xyxy"][1]] for p in selected])
    rng = np.random.default_rng(seed); order = rng.permutation(len(selected))
    train = np.sort(order[:round(.7 * len(order))]); check = np.sort(order[round(.7 * len(order)):])
    return selected, uv, wh, train, check


def load_terrain(bounds, margin=300.0):
    e0, n0, e1, n1 = bounds
    with rasterio.open(DEM) as ds:
        win = rasterio.windows.from_bounds(e0 - margin, n0 - margin, e1 + margin, n1 + margin, ds.transform)
        z = ds.read(1, window=win).astype(float)
        tr = ds.window_transform(win)
    ys = tr.f + (np.arange(z.shape[0]) + .5) * tr.e
    xs = tr.c + (np.arange(z.shape[1]) + .5) * tr.a
    if ys[0] > ys[-1]:
        ys = ys[::-1]; z = z[::-1]
    z[~np.isfinite(z)] = np.nanmedian(z)
    return RegularGridInterpolator((ys, xs), z, bounds_error=False, fill_value=float(np.nanmedian(z)))


def load_map(bounds, terrain, exclude_ids=()):
    """OSM building footprints whose centroid lies in the box -> roof-centre points."""
    data = json.loads((DATA / "osm_buildings.json").read_text())
    exclude_ids = set(exclude_ids)
    tf = Transformer.from_crs(4326, 32636, always_xy=True)
    e0, n0, e1, n1 = bounds
    rows, xyz, axes = [], [], []
    for el in data["elements"]:
        g = el.get("geometry")
        if el["type"] != "way" or not g or len(g) < 4 or el["id"] in exclude_ids:
            continue
        v = np.array([tf.transform(p["lon"], p["lat"]) for p in g])
        local = (v - v[0]).astype(np.float32); m = cv2.moments(local)
        if not 45 <= m["m00"] <= 2500:        # same area filter as the original
            continue
        c = np.array([m["m10"], m["m01"]]) / m["m00"] + v[0]
        if not (e0 <= c[0] <= e1 and n0 <= c[1] <= n1):
            continue
        box = cv2.boxPoints(cv2.minAreaRect(local)); a = (box[1] - box[0]) / 2; b = (box[2] - box[1]) / 2
        tags = el.get("tags", {})
        try:
            height = float(tags["building:levels"]) * 3.0
        except (KeyError, ValueError):
            height = DEFAULT_BUILDING_HEIGHT_M
        ground = float(terrain([[c[1], c[0]]])[0])
        xyz.append([c[0], c[1], ground + height]); axes.append([a, b])
        rows.append(dict(osm_id=el["id"], area_m2=float(m["m00"]), height_m=height,
                         levels_tagged="building:levels" in tags, polygon_utm=v.round(2).tolist()))
    return rows, np.array(xyz), np.array(axes)


class Problem:
    def __init__(self, area, mode="joint"):
        self.area = area; self.bounds_utm = area["bounds_utm"]; self.mode = mode
        self.observed, self.uv, self.wh, self.train, self.check = load_detections()
        self.terrain = load_terrain(self.bounds_utm)
        self.map, xyz, self.axes = load_map(self.bounds_utm, self.terrain, area.get("exclude_osm_ids", ()))
        self.origin = np.array([self.bounds_utm[0], self.bounds_utm[1], 0.])
        self.xyz = xyz - self.origin
        self.heights = np.array([m["height_m"] for m in self.map])
        e0, n0, e1, n1 = self.bounds_utm
        self.bounds = [(0.0, e1 - e0), (0.0, n1 - n0)] + BOUNDS_ANGLES + [LOG_FOCAL, LOG_HEIGHT]
        if mode == "joint_offset":
            self.bounds.append((0, .75))   # same range as the write-up
        self.obs_feat = np.c_[self.uv / [8, 6], np.log(self.wh[:, 0]) / .65]
        self.calls = 0

    def ground_z(self, e_local, n_local):
        return float(self.terrain([[n_local + self.origin[1], e_local + self.origin[0]]])[0])

    def unpack(self, p):
        f = np.exp(p[5]); height = np.exp(p[6]); R = rotation(*p[2:5])
        g = np.array([p[0], p[1], self.ground_z(p[0], p[1])])
        forward = R[2]
        t = height / max(-forward[2], 1e-6)     # distance along the ray to the ground point
        c = g - t * forward                       # camera: back along the image-centre ray
        return f, c, R, g

    def alpha(self, p):
        return float(p[7]) if self.mode == "joint_offset" else 0.

    def project(self, p):
        f, c, R, _ = self.unpack(p)
        xyz = self.xyz.copy(); xyz[:, 2] -= self.alpha(p) * self.heights   # facade-centre correction
        v = (xyz - c) @ R.T; z = v[:, 2]; safe = np.maximum(z, 1)
        uv = f * v[:, :2] / safe[:, None] + [CX, CY]
        widths = 2 * f * np.sum(np.abs(self.axes @ R[0, :2]), axis=1) / safe
        keep = ((z > 30) & (z < 6000) & (uv[:, 0] > -60) & (uv[:, 0] < IMAGE_W + 60)
                & (uv[:, 1] > -40) & (uv[:, 1] < IMAGE_H + 40) & (widths > 6) & (widths < 250))
        ids = np.where(keep)[0]
        return uv[keep], widths[keep], ids

    def coarse(self, p):
        """Unchanged cost; only the projection geometry differs."""
        self.calls += 1; uv, w, ids = self.project(p)
        if len(ids) < 15:
            return 20.
        feat = np.c_[uv / [8, 6], np.log(w) / .65]
        d, i = cKDTree(feat).query(self.obs_feat[self.train], k=1)
        d2 = np.minimum(d * d, 9); unique = len(np.unique(i[d < 3]))
        return float(d2.mean() + .6 * (1 - unique / len(i)) + .08 * np.log(max(len(ids), 1) / 150))

    def assignment(self, p, subset, excluded_map_indices=None):
        """Unchanged one-to-one assignment with dummy (unmatched) options."""
        if len(subset) == 0:
            raise ValueError("Assignment requires a nonempty observation subset")
        uv, w, ids = self.project(p)
        if excluded_map_indices:
            keep = np.array([int(i) not in excluded_map_indices for i in ids], dtype=bool)
            uv, w, ids = uv[keep], w[keep], ids[keep]
        if len(ids) == 0:
            return 9., []
        observed = self.uv[subset]; wh = self.wh[subset]
        sigma = np.maximum([4, 4], wh * [.16, .30])
        delta = (observed[:, None] - uv[None]) / sigma[:, None]
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

    def payload(self, p):
        f, c, R, g = self.unpack(p); C = c + self.origin; G = g + self.origin
        to_ll = Transformer.from_crs(32636, 4326, always_xy=True)
        glon, glat = to_ll.transform(*G[:2]); clon, clat = to_ll.transform(*C[:2])
        train, tp = self.assignment(p, self.train)
        check, cp = self.assignment(p, self.check, {q["map_index"] for q in tp})
        overall, allpairs = self.assignment(p, np.arange(len(self.uv)))
        return dict(parameters=np.asarray(p).tolist(), mode=self.mode, focal_px=float(f),
                    roof_to_facade_height_fraction=self.alpha(p),
                    footprint_centre_utm=G[:2].tolist(), footprint_centre_lat=glat, footprint_centre_lon=glon,
                    camera_utm_xyz=C.tolist(), camera_lat=clat, camera_lon=clon,
                    height_above_ground_m=float(np.exp(p[6])), yaw_pitch_roll_deg=np.asarray(p[2:5]).tolist(),
                    train_cost=train, train_matches=len(tp), check_cost=check, check_matches=len(cp),
                    overall_cost=overall, overall_matches=len(allpairs), pairs=allpairs,
                    train_pairs=tp, check_pairs=cp, coarse_cost=self.coarse(p))


def search(area, iterations=240, seeds=(7, 19, 41), mode="joint"):
    pr = Problem(area, mode); timings = {}; t0 = time.perf_counter()
    print(area["name"], "observed", len(pr.observed), "train", len(pr.train), "check", len(pr.check),
          "map", len(pr.map), flush=True)
    # Stage 1: global DE, exactly the original protocol.
    pool, runs = [], []
    for seed in seeds:
        t = time.perf_counter()
        fit = differential_evolution(pr.coarse, pr.bounds, seed=seed, popsize=16, maxiter=iterations,
                                     tol=.0002, polish=False, updating="immediate")
        order = np.argsort(fit.population_energies)[:8]
        pool.extend(fit.population[order].tolist())
        runs.append(dict(seed=seed, seconds=time.perf_counter() - t, cost=float(fit.fun), evaluations=int(fit.nfev)))
        print("  global", runs[-1], flush=True)
    timings["global_de_s"] = time.perf_counter() - t0
    # Stage 2: dedupe (35 m / 1 deg, as the original) and Powell-polish the best 12.
    t = time.perf_counter(); unique = []
    for p in sorted(pool, key=pr.coarse):
        if all(np.linalg.norm(np.array(p[:2]) - np.array(q[:2])) > 35 or abs(p[2] - q[2]) > 1 for q in unique):
            unique.append(p)
    polished = []
    for p in unique[:12]:
        fit = minimize(lambda q: pr.assignment(q, pr.train)[0], p, method="Powell", bounds=pr.bounds,
                       options=dict(maxiter=45, xtol=1e-4, ftol=1e-5))
        best = fit.x if pr.assignment(fit.x, pr.train)[0] < pr.assignment(p, pr.train)[0] else np.array(p)
        polished.append(best)
    timings["powell_s"] = time.perf_counter() - t
    # Stage 3: local DE refinement of the best 2 (the original's refine script, joint mode).
    t = time.perf_counter(); refined = []
    starts = sorted(polished, key=lambda q: pr.assignment(q, pr.train)[0])
    for index, q in enumerate(starts[:2]):
        radius = [350, 350, 5, 3, 3, .20, .20] + ([.45] if mode == "joint_offset" else [])
        lb = [max(lo, v - r) for (lo, hi), v, r in zip(pr.bounds, q, radius)]
        ub = [min(hi, v + r) for (lo, hi), v, r in zip(pr.bounds, q, radius)]
        fit = differential_evolution(lambda x: pr.assignment(x, pr.train)[0], list(zip(lb, ub)),
                                     seed=71 + index, popsize=12, maxiter=150, tol=.0001, polish=False,
                                     x0=np.clip(q, lb, ub))
        best = fit.x if pr.assignment(fit.x, pr.train)[0] < pr.assignment(q, pr.train)[0] else q
        refined.append(best)
    timings["local_refine_s"] = time.perf_counter() - t
    hypotheses = [dict(pr.payload(p), stage="global+powell") for p in polished] + \
                 [dict(pr.payload(p), stage="local_refine") for p in refined]
    hypotheses.sort(key=lambda r: r["train_cost"])
    timings["total_s"] = time.perf_counter() - t0
    return pr, dict(area=area, mode=mode, seconds=timings, global_runs=runs, objective_evaluations=pr.calls,
                    parameter_names=PARAMETER_NAMES + (["height_fraction"] if mode == "joint_offset" else []),
                    bounds=pr.bounds, map_buildings=len(pr.map),
                    map_buildings_with_levels_tag=sum(m["levels_tagged"] for m in pr.map),
                    detections=len(pr.observed), train_indices=pr.train.tolist(), check_indices=pr.check.tolist(),
                    observations=pr.observed, hypotheses=hypotheses,
                    assumptions=dict(principal_point_px=[CX, CY], distortion=0,
                                     camera_height="searched, 40-1200 m above the viewed ground point",
                                     yaw_prior="none (0-360 deg)", pitch_deg=BOUNDS_ANGLES[1],
                                     roll_deg=BOUNDS_ANGLES[2], focal_px=[400, 3000],
                                     map_heights=f"DEM + building:levels x 3 m, else {DEFAULT_BUILDING_HEIGHT_M} m",
                                     detection_filter=DETECTION_FILTER, truth_used=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--area", required=True)
    ap.add_argument("--iterations", type=int, default=240)
    ap.add_argument("--mode", choices=MODES, default="joint"); args = ap.parse_args()
    areas = {a["name"]: a for a in json.loads((DATA / "search_areas.json").read_text())["areas"]}
    pr, out = search(areas[args.area], iterations=args.iterations, mode=args.mode)
    # primary (joint) at results/<area>/, the facade-offset variant at results/<area>/joint_offset/
    dst = RESULTS / args.area / ("" if args.mode == "joint" else args.mode); dst.mkdir(parents=True, exist_ok=True)
    (dst / "search.json").write_text(json.dumps(out, indent=2))
    (dst / "map_buildings.json").write_text(json.dumps(pr.map))
    top = out["hypotheses"][0]
    print(json.dumps(dict(seconds=out["seconds"], top_train_cost=top["train_cost"],
                          top_footprint=top["footprint_centre_utm"], top_matches=top["overall_matches"]), indent=1))


if __name__ == "__main__":
    main()
