#!/usr/bin/env python3
"""Post-search evaluation and figures. The ONLY script that reads data/truth.json.

For each area: distinct candidates (hypotheses whose footprint centres lie
within 50 m of a better-scoring one are merged), their error from the truth,
the rank of the first correct candidate, the score margin, and the same
descriptive null as the Sainte-Maxime review (100 x-shuffled controls at the
frozen best pose, not re-optimised). "Correct" is defined here, before looking
at the results, as a footprint centre within 50 m of the truth (the truth itself
carries ~10 m uncertainty); errors are reported so any other threshold can be
applied.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import chamaa_constellation as cc
from ortho_io import cut_ortho

HERE = Path(__file__).resolve().parent
CORRECT_M = 50.0
MERGE_M = 50.0
FIG = cc.RESULTS / "figures"
YELLOW, CYAN, RED, MAGENTA, WHITE, GREEN = (43, 199, 255), (252, 228, 3), (85, 85, 255), (220, 56, 236), (255, 255, 255), (80, 220, 80)


def text(im, s, org, scale=0.6, col=WHITE, th=1):
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def distinct(hypotheses):
    out = []
    for h in sorted(hypotheses, key=lambda r: r["train_cost"]):
        g = np.array(h["footprint_centre_utm"])
        if all(np.linalg.norm(g - np.array(q["footprint_centre_utm"])) > MERGE_M for q in out):
            out.append(h)
    return out


def ortho_tile(bounds, gsd, name):
    tmp = FIG / f"_{name}.tif"
    t = cut_ortho(tuple(bounds), tmp, gsd=gsd)
    im = cv2.imread(str(t.path))
    for f in FIG.glob(f"_{name}*"):
        f.unlink()
    return im


def to_px(bounds, gsd, e, n):
    return int(round((e - bounds[0]) / gsd)), int(round((bounds[3] - n) / gsd))


def fig_area_map(name, area, cands, truth, map_rows, out):
    b = area["bounds_utm"]; side = b[2] - b[0]; gsd = side / 1600
    im = ortho_tile(b, gsd, "area")
    im = (im * 0.75).astype(np.uint8)
    for m in map_rows:
        c = np.mean(np.array(m["polygon_utm"]), axis=0)
        cv2.circle(im, to_px(b, gsd, *c), 2, (200, 200, 200), -1)
    costs = [c["train_cost"] for c in cands]; lo, hi = min(costs), max(costs)
    for rank, c in reversed(list(enumerate(cands, 1))):
        t = (c["train_cost"] - lo) / max(hi - lo, 1e-9)
        col = (int(255 * t), int(80 + 120 * (1 - t)), int(255 * (1 - t)))
        p = to_px(b, gsd, *c["footprint_centre_utm"])
        cv2.circle(im, p, 13 if rank == 1 else 9, col, 3)
        text(im, str(rank), (p[0] + 12, p[1] - 8), 0.7, col, 2)
    T = to_px(b, gsd, *truth)
    cv2.drawMarker(im, T, MAGENTA, cv2.MARKER_CROSS, 36, 4)
    text(im, "truth", (T[0] + 16, T[1] + 22), 0.8, MAGENTA, 2)
    text(im, f"{name}: {len(cands)} distinct candidates (1 = best score; circle colour = score, red is worse)", (14, 32), 0.75)
    text(im, f"grey dots: {len(map_rows)} OSM reference buildings  |  magenta cross: truth", (14, 62), 0.65)
    cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 85])


def fig_matched_on_ortho(name, best, map_rows, truth, out):
    pairs = best["pairs"]; train_ids = {q["observed_id"] for q in best["train_pairs"]}
    pts = np.array([np.mean(np.array(map_rows[q["map_index"]]["polygon_utm"]), axis=0) for q in pairs])
    g = np.array(best["footprint_centre_utm"])
    allp = np.vstack([pts, g[None], np.array(truth)[None]])
    pad = 90
    b = [allp[:, 0].min() - pad, allp[:, 1].min() - pad, allp[:, 0].max() + pad, allp[:, 1].max() + pad]
    side = max(b[2] - b[0], b[3] - b[1]); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    b = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]; gsd = side / 1400
    im = ortho_tile(b, gsd, "match")
    for q in pairs:
        poly = np.array(map_rows[q["map_index"]]["polygon_utm"])
        P = np.array([to_px(b, gsd, *p) for p in poly], np.int32)
        col = YELLOW if q["observed_id"] in train_ids else CYAN
        cv2.polylines(im, [P], True, col, 3)
    G = to_px(b, gsd, *g); T = to_px(b, gsd, *truth)
    cv2.drawMarker(im, G, GREEN, cv2.MARKER_TILTED_CROSS, 30, 4); text(im, "estimated footprint centre", (G[0] + 14, G[1] - 10), 0.65, GREEN, 2)
    cv2.drawMarker(im, T, MAGENTA, cv2.MARKER_CROSS, 30, 4); text(im, "truth", (T[0] + 14, T[1] + 22), 0.65, MAGENTA, 2)
    text(im, f"{name}: top candidate's matched OSM buildings (yellow = training, cyan = reserved)", (14, 32), 0.7)
    text(im, f"{len(pairs)} assigned buildings  |  error {math.dist(g, truth):.0f} m", (14, 62), 0.7)
    cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 85])


def fig_side_by_side(name, pr, best, map_rows, truth, out):
    photo = cv2.imread(str(cc.DATA / "chamaa_photo1.png"))
    train_ids = {q["observed_id"] for q in best["train_pairs"]}
    assigned = {q["observed_id"] for q in best["pairs"]}
    pairs = sorted(best["pairs"], key=lambda q: q["observed_xy"][0])
    for k, q in enumerate(pairs, 1):
        a = tuple(int(v) for v in q["observed_xy"]); p = tuple(int(v) for v in q["projected_xy"])
        col = YELLOW if q["observed_id"] in train_ids else CYAN
        cv2.circle(photo, a, 7, col, 2); cv2.drawMarker(photo, p, col, cv2.MARKER_CROSS, 12, 2)
        cv2.line(photo, a, p, col, 1, cv2.LINE_AA); text(photo, str(k), (a[0] + 7, a[1] - 7), 0.5, col, 1)
    for o in pr.observed:
        if o["id"] not in assigned:
            cv2.drawMarker(photo, tuple(int(v) for v in o["centroid_xy"]), RED, cv2.MARKER_TILTED_CROSS, 14, 2)
    text(photo, "photo: detections o, projected map centres +, red x unmatched", (10, 26), 0.6)
    # map side, same numbering
    pts = np.array([np.mean(np.array(map_rows[q["map_index"]]["polygon_utm"]), axis=0) for q in pairs])
    pad = 80; b = [pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad]
    side = max(b[2] - b[0], b[3] - b[1]); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    b = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]; gsd = side / photo.shape[0]
    im = ortho_tile(b, gsd, "side")
    for k, q in enumerate(pairs, 1):
        poly = np.array(map_rows[q["map_index"]]["polygon_utm"])
        P = np.array([to_px(b, gsd, *p) for p in poly], np.int32)
        col = YELLOW if q["observed_id"] in train_ids else CYAN
        cv2.polylines(im, [P], True, col, 2); c = P.mean(axis=0).astype(int)
        text(im, str(k), (int(c[0]) + 6, int(c[1]) - 6), 0.5, col, 1)
    T = to_px(b, gsd, *truth)
    if 0 <= T[0] < im.shape[1] and 0 <= T[1] < im.shape[0]:
        cv2.drawMarker(im, T, MAGENTA, cv2.MARKER_CROSS, 26, 3); text(im, "truth", (T[0] + 12, T[1] + 20), 0.55, MAGENTA, 2)
    text(im, "orthophoto: matched OSM buildings, same numbers", (10, 26), 0.6)
    im = cv2.resize(im, (int(im.shape[1] * photo.shape[0] / im.shape[0]), photo.shape[0]))
    strip = np.hstack([photo, np.full((photo.shape[0], 10, 3), 30, np.uint8), im])
    head = np.full((46, strip.shape[1], 3), 20, np.uint8)
    text(head, f"{name}: image vs matched constellation for the top candidate "
               f"(error {math.dist(best['footprint_centre_utm'], truth):.0f} m, focal {best['focal_px']:.0f} px, "
               f"yaw {best['yaw_pitch_roll_deg'][0]:.0f} deg)", (12, 30), 0.7)
    cv2.imwrite(str(out), np.vstack([head, strip]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def evaluate(name, truth_doc, mode="joint"):
    sub = cc.RESULTS / name / ("" if mode == "joint" else mode)
    run = json.loads((sub / "search.json").read_text())
    map_rows = json.loads((sub / "map_buildings.json").read_text())
    truth = truth_doc["footprint_centre_utm"]
    cands = distinct(run["hypotheses"])
    for c in cands:
        c["error_m"] = math.dist(c["footprint_centre_utm"], truth)
    correct = [i for i, c in enumerate(cands, 1) if c["error_m"] <= CORRECT_M]
    near100 = [i for i, c in enumerate(cands, 1) if c["error_m"] <= 100]
    best = cands[0]
    # Descriptive null, as in the original review: shuffle detection x-coordinates at the frozen pose.
    if mode in ("bimodal", "bimodal_stratified"):
        import bimodal
        pr = bimodal.Bimodal(run["area"])
    else:
        pr = cc.Problem(run["area"], mode)
    p = np.array(best["parameters"])
    rng = np.random.default_rng(119); original = pr.uv.copy(); controls = []
    used = {q["map_index"] for q in best["train_pairs"]}
    for _ in range(100):
        pr.uv[:, 0] = original[rng.permutation(len(original)), 0]
        controls.append(pr.assignment(p, pr.check, used)[0])
    pr.uv = original
    offsets = next((o for o in truth_doc["offsets"] if o["name"] == name), None)
    if offsets is None:   # map-defined areas (village sanity check): derive the same numbers here
        b = run["area"]["bounds_utm"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        offsets = dict(name=name, box_centre_minus_truth_m=[round(cx - truth[0], 1), round(cy - truth[1], 1)],
                       truth_distance_to_nearest_edge_m=round(min(truth[0] - b[0], b[2] - truth[0],
                                                                  truth[1] - b[1], b[3] - truth[1]), 1),
                       note="box defined from OSM (place node + building cluster), not drawn around the truth")
    ev = dict(area=name, mode=mode, bounds_utm=run["area"]["bounds_utm"], offset=offsets,
              reference_buildings=run["map_buildings"], detections=run["detections"],
              runtime_s=run["seconds"], objective_evaluations=run["objective_evaluations"],
              global_de_converged=[r["evaluations"] < 26992 for r in run["global_runs"]],
              n_distinct_candidates=len(cands), correct_threshold_m=CORRECT_M,
              rank_of_correct=correct[0] if correct else None,
              rank_within_100m=near100[0] if near100 else None,
              best_error_m=best["error_m"], min_error_any_candidate_m=min(c["error_m"] for c in cands),
              margin_train_cost=cands[1]["train_cost"] - best["train_cost"] if len(cands) > 1 else None,
              best_train_cost=best["train_cost"], best_check_cost=best["check_cost"],
              descriptive_null=dict(type="100 x-shuffled controls at the frozen best pose, not re-optimised",
                                    heldout_cost_median=float(np.median(controls)),
                                    heldout_cost_min=float(min(controls)), real_heldout_cost=best["check_cost"]),
              top5=[dict(rank=i, road_f1=c.get("road_f1"), crossings_matched=c.get("crossings_matched"),
                         house_train_cost=c.get("house_train_cost", c["train_cost"]), lat=c["footprint_centre_lat"], lon=c["footprint_centre_lon"],
                         footprint_centre_utm=[round(v, 1) for v in c["footprint_centre_utm"]],
                         train_cost=c["train_cost"], check_cost=c["check_cost"],
                         inlier_buildings=c["overall_matches"], train_matches=c["train_matches"],
                         check_matches=c["check_matches"], focal_px=c["focal_px"],
                         yaw_pitch_roll_deg=c["yaw_pitch_roll_deg"],
                         height_above_ground_m=c["height_above_ground_m"], error_m=c["error_m"],
                         stage=c["stage"]) for i, c in enumerate(cands[:5], 1)],
              all_candidates=[dict(rank=i, error_m=c["error_m"], train_cost=c["train_cost"],
                                   check_cost=c["check_cost"], inlier_buildings=c["overall_matches"])
                              for i, c in enumerate(cands, 1)])
    (sub / "evaluation.json").write_text(json.dumps(ev, indent=2))
    FIG.mkdir(parents=True, exist_ok=True)
    tag = name if mode == "joint" else f"{name}_{mode}"
    label = name if mode == "joint" else f"{name} ({mode})"
    fig_area_map(label, run["area"], cands, truth, map_rows, FIG / f"{tag}_area_candidates.jpg")
    fig_matched_on_ortho(label, best, map_rows, truth, FIG / f"{tag}_top_matched_on_ortho.jpg")
    fig_side_by_side(label, pr, best, map_rows, truth, FIG / f"{tag}_image_vs_constellation.jpg")
    return ev


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--areas", default="2x2km,4x4km")
    ap.add_argument("--mode", choices=cc.MODES + ("bimodal", "bimodal_stratified"), default="joint"); args = ap.parse_args()
    truth_doc = json.loads((cc.DATA / "truth.json").read_text())
    for name in args.areas.split(","):
        ev = evaluate(name, truth_doc, args.mode)
        print(json.dumps({k: ev[k] for k in ("area", "reference_buildings", "rank_of_correct", "rank_within_100m",
                                             "best_error_m", "min_error_any_candidate_m", "margin_train_cost",
                                             "best_train_cost", "best_check_cost", "descriptive_null",
                                             "n_distinct_candidates", "global_de_converged")}, indent=1))
        for t in ev["top5"]:
            print(f"  #{t['rank']} err {t['error_m']:7.1f} m  train {t['train_cost']:.3f}  check {t['check_cost']:.3f}  "
                  f"inliers {t['inlier_buildings']:>2}  focal {t['focal_px']:.0f}  yaw {t['yaw_pitch_roll_deg'][0]:.0f}  "
                  f"h {t['height_above_ground_m']:.0f} m")


if __name__ == "__main__":
    main()
