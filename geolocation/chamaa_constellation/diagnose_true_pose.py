#!/usr/bin/env python3
"""ORACLE DIAGNOSTIC - uses the ground truth. NOT part of the blind result.

Recovers the actual camera pose from the LoFTR alignment and scores it with the
constellation cost. The alignment homography maps photo pixels to ground; a grid
of pixels over the reliable central/lower part of the photo becomes ground
points (z from the DEM), and a PnP solve with a focal sweep recovers the camera.
Evaluating the constellation cost at that pose answers the decisive question:
does the method's objective prefer the true pose at all?
"""
import json
import math

import cv2
import numpy as np

import chamaa_constellation as cc

TILE_ORIGIN, TILE_GSD = (705174.0, 3670464.0), 0.4


def to_params(pr, R, C, f, footprint):
    fwd = R[2]
    yaw = math.degrees(math.atan2(fwd[0], fwd[1])) % 360
    pitch = math.degrees(math.asin(fwd[2]))
    y = math.radians(yaw)
    right_nom = np.array([math.cos(y), -math.sin(y), 0.])
    down_nom = np.cross(fwd, right_nom)
    roll = math.degrees(math.atan2(R[0] @ down_nom, R[0] @ right_nom))
    g_local = np.array(footprint) - pr.origin[:2]
    height = C[2] - pr.ground_z(*g_local)
    return np.array([g_local[0], g_local[1], yaw, pitch, roll, math.log(f), math.log(height)])


def main():
    al = json.loads((cc.DATA / "alignment_results.json").read_text())
    truth = json.loads((cc.DATA / "truth.json").read_text())["footprint_centre_utm"]
    area = json.loads((cc.DATA / "search_areas.json").read_text())["areas"][0]
    pr = cc.Problem(area)
    H = np.array(al["loftr_query_to_northwest_tile_homography"])
    # central/lower 60% of the photo, where the alignment README says the planar model holds
    us, vs = np.meshgrid(np.linspace(150, 1050, 13), np.linspace(330, 850, 9))
    img = np.c_[us.ravel(), vs.ravel()]
    t = np.c_[img, np.ones(len(img))] @ H.T
    tile = t[:, :2] / t[:, 2:]
    E = TILE_ORIGIN[0] + tile[:, 0] * TILE_GSD; N = TILE_ORIGIN[1] - tile[:, 1] * TILE_GSD
    Z = np.array([pr.ground_z(e - pr.origin[0], n - pr.origin[1]) for e, n in zip(E, N)])
    obj = np.c_[E - pr.origin[0], N - pr.origin[1], Z]
    best = None
    for f in np.geomspace(500, 3000, 120):
        K = np.array([[f, 0, cc.CX], [0, f, cc.CY], [0, 0, 1.]])
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        Rm, _ = cv2.Rodrigues(rv)
        if not ((Rm @ obj.T + tv)[2] > 0).all():          # cheirality
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        if best is None or rms < best[0]:
            best = (rms, f, Rm, (-Rm.T @ tv).ravel())
    rms, f, Rm, C = best
    p = to_params(pr, Rm, C, f, truth)
    fr, cr, Rr, gr = pr.unpack(p)
    out = pr.payload(p)
    uv, w, ids = pr.project(p)
    res = dict(
        warning="ORACLE DIAGNOSTIC - camera recovered from the ground-truth alignment; not a blind result",
        pnp=dict(points=len(img), focal_px=f, rms_px=rms, camera_utm_xyz=(C + pr.origin).tolist(),
                 reparametrisation_camera_mismatch_m=float(np.linalg.norm(cr - C))),
        true_pose=dict(yaw_pitch_roll_deg=p[2:5].tolist(), focal_px=f, height_above_ground_m=math.exp(p[6])),
        constellation_cost_at_true_pose=dict(train_cost=out["train_cost"], check_cost=out["check_cost"],
                                             inlier_buildings=out["overall_matches"],
                                             train_matches=out["train_matches"], check_matches=out["check_matches"],
                                             coarse_cost=out["coarse_cost"], map_points_in_view=int(len(ids))),
        median_pixel_distance_of_assigned_pairs=float(np.median([q["pixel_distance"] for q in out["pairs"]])) if out["pairs"] else None,
        pairs=out["pairs"], parameters=p.tolist())
    # The write-up's joint_offset variant: at the TRUE pose, sweep the facade-centre fraction alpha.
    po = cc.Problem(area, mode="joint_offset"); sweep = []
    for a in np.round(np.arange(0, .7501, .05), 2):
        q = np.r_[p, a]; tr, _ = po.assignment(q, po.train)
        tp = po.assignment(q, po.train)[1]; ck, cp_ = po.assignment(q, po.check, {x["map_index"] for x in tp})
        sweep.append(dict(alpha=float(a), train_cost=tr, check_cost=ck, train_matches=len(tp), check_matches=len(cp_)))
    res["joint_offset_alpha_sweep_at_true_pose"] = sweep
    best_a = min(sweep, key=lambda r: r["train_cost"])
    res["joint_offset_best_at_true_pose"] = best_a
    dst = cc.RESULTS / "diagnostics"; dst.mkdir(parents=True, exist_ok=True)
    (dst / "true_pose_cost.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("pairs", "parameters", "joint_offset_alpha_sweep_at_true_pose")}, indent=1))
    for r in res["joint_offset_alpha_sweep_at_true_pose"][::3]:
        print(f"  alpha {r['alpha']:.2f}: train {r['train_cost']:.2f}  check {r['check_cost']:.2f}  matches {r['train_matches']}+{r['check_matches']}")

    # Figure: at the TRUE pose, where do the map roof centres land relative to the detections?
    photo = cv2.imread(str(cc.DATA / "chamaa_photo1.png"))
    for (u, v), wd in zip(uv, w):
        cv2.drawMarker(photo, (int(u), int(v)), (43, 199, 255), cv2.MARKER_CROSS, 14, 2)
    # also the OSM footprint centre at GROUND level, to show the height effect
    Rg = rotation = Rr
    for m in pr.map:
        c = np.mean(np.array(m["polygon_utm"]), axis=0) - pr.origin[:2]
        X = np.array([c[0], c[1], pr.ground_z(*c)])
        v3 = Rg @ (X - cr)
        if v3[2] > 30:
            u, v = fr * v3[:2] / v3[2] + [cc.CX, cc.CY]
            if 0 <= u < cc.IMAGE_W and 0 <= v < cc.IMAGE_H:
                cv2.circle(photo, (int(u), int(v)), 4, (80, 220, 80), -1)
    for o in pr.observed:
        cv2.circle(photo, tuple(int(x) for x in o["centroid_xy"]), 8, (255, 80, 80), 2)
    for s, y, col in (("TRUE POSE (from the alignment):", 26, (255, 255, 255)),
                      ("blue o = SAM detection centroids   yellow + = OSM roof centres (DEM + 7 m)   green . = OSM footprint centre at ground", 50, (255, 255, 255)),
                      (f"train cost {out['train_cost']:.2f}, held-out {out['check_cost']:.2f}, {out['overall_matches']} assigned", 74, (255, 255, 255))):
        cv2.putText(photo, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(photo, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, col, 1, cv2.LINE_AA)
    (cc.RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cc.RESULTS / "figures" / "diagnostic_true_pose_projection.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
