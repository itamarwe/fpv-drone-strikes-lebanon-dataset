#!/usr/bin/env python3
"""DIAGNOSTIC - does road geometry prefer the true pose where houses did not?

For a set of poses - the true pose (recovered from the alignment; uses the truth)
and the blind winners of every search run - project the OSM road centrelines and
junctions into the photo and measure agreement with the SAM 3 road mask:

  road precision = share of projected OSM road samples (in frame) lying within
                   TOL px of the photo's road mask
  road recall    = share of photo road-skeleton pixels within TOL px of a
                   projected OSM road
  road F1        = harmonic mean of the two
  junction matches = one-to-one detected crossings within 30 px of a projected
                   OSM junction

The house cost (the method's training cost) is shown alongside. This only asks
whether the road evidence separates truth from the house-cost winners; it is not a
search.
"""
import json
import math

import cv2
import numpy as np

import chamaa_constellation as cc
from road_junctions import EXCLUDED, image_junctions, osm_junctions

TOL = 8.0


def main():
    truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    mask, closed, skel, ijs, _ = image_junctions()
    ways, ojs = osm_junctions()
    dist_to_mask = cv2.distanceTransform((closed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    sk_pts = np.argwhere(skel)[:, ::-1].astype(float)
    iuv = np.array([j.xy for j in ijs])
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    pr = cc.Problem(areas["4x4km"])          # widest map: covers every pose below

    samples = []
    for w in ways:
        for a, b in zip(w["xy"][:-1], w["xy"][1:]):
            n = max(2, int(np.linalg.norm(b - a) / 3))
            samples.append(np.linspace(a, b, n))
    samples = np.vstack(samples)
    zs = np.array([pr.ground_z(x - pr.origin[0], y - pr.origin[1]) for x, y in samples])
    road_world = np.c_[samples[:, 0] - pr.origin[0], samples[:, 1] - pr.origin[1], zs]
    jxy = np.array([j["utm"] for j in ojs])
    jz = np.array([pr.ground_z(x - pr.origin[0], y - pr.origin[1]) for x, y in jxy])
    j_world = np.c_[jxy[:, 0] - pr.origin[0], jxy[:, 1] - pr.origin[1], jz]

    def score(params_local_ground, rest):
        # rest = yaw, pitch, roll, logf, logh ; ground point given in absolute UTM
        g = np.array(params_local_ground) - pr.origin[:2]
        p = np.r_[g, rest]
        f, c, R, _ = pr.unpack(p)
        v = (road_world - c) @ R.T; ok = v[:, 2] > 20
        uv = f * v[:, :2] / np.maximum(v[:, 2:], 1) + [cc.CX, cc.CY]
        inf = ok & (uv[:, 0] >= 0) & (uv[:, 0] < cc.IMAGE_W) & (uv[:, 1] >= 0) & (uv[:, 1] < cc.IMAGE_H - 12)
        if inf.sum() < 20:
            return dict(road_precision=0., road_recall=0., road_f1=0., junction_matches=0, osm_junctions_in_frame=0)
        P = uv[inf]
        prec = float((dist_to_mask[P[:, 1].astype(int), P[:, 0].astype(int)] <= TOL).mean())
        canvas = np.zeros(mask.shape, np.uint8)
        for q in P.astype(int):
            cv2.circle(canvas, tuple(q), 1, 255, -1)
        d_osm = cv2.distanceTransform((canvas == 0).astype(np.uint8), cv2.DIST_L2, 5)
        rec = float((d_osm[sk_pts[:, 1].astype(int), sk_pts[:, 0].astype(int)] <= TOL).mean())
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.
        vj = (j_world - c) @ R.T; okj = vj[:, 2] > 20
        juv = f * vj[:, :2] / np.maximum(vj[:, 2:], 1) + [cc.CX, cc.CY]
        infj = okj & (juv[:, 0] >= 0) & (juv[:, 0] < cc.IMAGE_W) & (juv[:, 1] >= 0) & (juv[:, 1] < cc.IMAGE_H - 12)
        J = juv[infj]; matched = 0
        if len(J) and len(iuv):
            D = np.linalg.norm(iuv[:, None] - J[None], axis=2); ua, ub = set(), set()
            for k in np.argsort(D, axis=None):
                a, b = np.unravel_index(k, D.shape)
                if D[a, b] > 30:
                    break
                if a in ua or b in ub:
                    continue
                ua.add(a); ub.add(b); matched += 1
        return dict(road_precision=prec, road_recall=rec, road_f1=f1, junction_matches=matched,
                    osm_junctions_in_frame=int(infj.sum()))

    rows = []
    tp = json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())
    ta = next(a for a in areas.values() if a["name"] == "2x2km")
    tprob = cc.Problem(ta); tpar = np.array(tp["parameters"])
    g_abs = tpar[:2] + tprob.origin[:2]
    rows.append(dict(pose="TRUE pose (uses the truth)", error_m=0.0,
                     house_train_cost=tp["constellation_cost_at_true_pose"]["train_cost"], **score(g_abs, tpar[2:7])))
    for name, sub in (("2x2km", ""), ("4x4km", ""), ("village", ""), ("village_no_dense_complex", ""),
                      ("2x2km", "joint_offset"), ("4x4km", "joint_offset")):
        path = cc.RESULTS / name / sub / "search.json"
        if not path.exists():
            continue
        hyps = json.loads(path.read_text())["hypotheses"]
        h = min(hyps, key=lambda r: r["train_cost"])
        prm = np.array(h["parameters"])
        rows.append(dict(pose=f"blind winner {name}{' ' + sub if sub else ''}",
                         error_m=math.dist(h["footprint_centre_utm"], truth), house_train_cost=h["train_cost"],
                         **score(h["footprint_centre_utm"], prm[2:7])))
    (cc.RESULTS / "diagnostics/road_score_by_pose.json").write_text(json.dumps(dict(
        warning="DIAGNOSTIC - the first row uses the ground-truth pose", tolerance_px=TOL, rows=rows), indent=2))
    print(f"{'pose':<40}{'err m':>7}{'house':>7}{'road P':>8}{'road R':>8}{'road F1':>8}{'junc':>6}")
    for r in rows:
        print(f"{r['pose']:<40}{r['error_m']:>7.0f}{r['house_train_cost']:>7.2f}{r['road_precision']:>8.2f}"
              f"{r['road_recall']:>8.2f}{r['road_f1']:>8.2f}{r['junction_matches']:>4}/{r['osm_junctions_in_frame']}")


if __name__ == "__main__":
    main()
