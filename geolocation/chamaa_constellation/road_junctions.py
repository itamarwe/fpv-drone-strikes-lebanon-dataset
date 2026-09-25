#!/usr/bin/env python3
"""Road crossings (junctions) in the Chamaa photo and in OSM, for a bimodal constellation.

Image side: SAM 3 road instances (prompts road/street/driveway/golf cart path, the
original segmentation script's road family, threshold 0.3) are unioned, closed
with a 9x9 kernel (SAM fragments roads at trees/buildings, turning one crossing
into two facing ends), skeletonised with skimage, and junctions of degree >= 3
are taken from road_graph.junctions_from_mask.

Map side: OSM highway ways. A node is a junction when road segments meeting there
number >= 3 (a way passing through counts 2, a way ending counts 1). Footways,
paths and steps are excluded as not visible roads.

The third figure is a DIAGNOSTIC that uses the ground-truth camera pose (recovered
from the alignment in diagnose_true_pose.py) to project OSM roads and junctions
into the photo, so detected crossings can be checked against real ones. It is not
used by any search.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer
from skimage.morphology import skeletonize

import chamaa_constellation as cc
from ortho_io import cut_ortho
from road_graph import junctions_from_mask

DATA, FIG = cc.DATA, cc.RESULTS / "figures"
EXCLUDED = {"footway", "path", "steps", "cycleway", "pedestrian", "corridor"}
MINOR = {"service", "track"}


def text(im, s, org, scale=.55, col=(255, 255, 255), th=1):
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def image_junctions():
    seg = json.loads((DATA / "sam_photo1_roads/segments.json").read_text())["ground"]
    mask = np.zeros((cc.IMAGE_H, cc.IMAGE_W), np.uint8)
    for f in seg["features"]:
        m = cv2.imread(str(DATA / "sam_photo1_roads/ground_masks" / f"{f['id']}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != mask.shape:
            m = cv2.resize(m, (cc.IMAGE_W, cc.IMAGE_H), interpolation=cv2.INTER_NEAREST)
        mask |= (m > 127).astype(np.uint8)
    mask[cc.IMAGE_H - 12:] = 0                               # black band at the bottom
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    skel = skeletonize(closed > 0)
    js = junctions_from_mask(closed, walk_px=20, merge_px=10, min_branch_px=10)
    js = [j for j in js if j.degree >= 3]
    return mask, closed, skel, js, len(seg["features"])


def osm_junctions():
    data = json.loads((DATA / "osm_roads.json").read_text())
    tf = Transformer.from_crs(4326, 32636, always_xy=True)
    ends = defaultdict(int); classes = defaultdict(set); ways = []
    for el in data["elements"]:
        hw = el.get("tags", {}).get("highway")
        g = el.get("geometry")
        if el["type"] != "way" or not g or hw in EXCLUDED:
            continue
        keys = [(round(p["lon"], 7), round(p["lat"], 7)) for p in g]
        xy = np.array([tf.transform(*k) for k in keys])
        ways.append(dict(highway=hw, xy=xy, osm_id=el["id"]))
        for i, k in enumerate(keys):
            ends[k] += 1 if i in (0, len(keys) - 1) else 2
            classes[k].add(hw)
    js = []
    for k, n in ends.items():
        if n >= 3:
            e, nn = tf.transform(*k)
            js.append(dict(utm=[e, nn], degree=n, classes=sorted(classes[k]),
                           minor_only=classes[k] <= MINOR))
    return ways, js


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    mask, closed, skel, ijs, n_inst = image_junctions()
    ways, ojs = osm_junctions()

    # --- figure 1: photo, road mask, skeleton, detected junctions
    photo = cv2.imread(str(DATA / "chamaa_photo1.png"))
    over = photo.copy(); over[closed > 0] = (0.55 * over[closed > 0] + 0.45 * np.array([30, 190, 255])).astype(np.uint8)
    over[skel] = (255, 255, 255)
    for i, j in enumerate(ijs, 1):
        x, y = map(int, j.xy)
        cv2.circle(over, (x, y), 11, (255, 60, 255), 3)
        for d in j.dirs:
            cv2.line(over, (x, y), (int(x + 26 * d[0]), int(y + 26 * d[1])), (255, 60, 255), 2)
        text(over, str(i), (x + 12, y - 10), .6, (255, 60, 255), 2)
    text(over, f"photo: SAM 3 roads ({n_inst} instances, orange), skeleton (white), "
               f"{len(ijs)} crossings of degree >= 3 (magenta, with branch directions)", (10, 24))
    cv2.imwrite(str(FIG / "roads_image_junctions.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # --- figure 2: OSM roads and junctions over the orthophoto around the photo footprint
    truth = json.loads((DATA / "truth.json").read_text())      # only to centre the view
    t = truth["footprint_centre_utm"]; half = 700; g = 0.7
    b = (t[0] - half, t[1] - half, t[0] + half, t[1] + half)
    tile = cut_ortho(b, FIG / "_r.tif", gsd=g); om = cv2.imread(str(tile.path))
    for f in FIG.glob("_r*"):
        f.unlink()
    px = lambda e, n: (int((e - b[0]) / g), int((b[3] - n) / g))
    for w in ways:
        col = (140, 140, 140) if w["highway"] in MINOR else (30, 200, 255)
        cv2.polylines(om, [np.array([px(*p) for p in w["xy"]], np.int32)], False, col, 3 if col[0] == 30 else 2)
    shown = 0
    for j in ojs:
        p = px(*j["utm"])
        if 0 <= p[0] < om.shape[1] and 0 <= p[1] < om.shape[0]:
            shown += 1
            cv2.circle(om, p, 9, (160, 160, 160) if j["minor_only"] else (255, 60, 255), 3)
    corners = truth["footprint_corners_utm"]
    cv2.polylines(om, [np.array([px(*corners[k]) for k in ("top_left", "top_right", "bottom_right", "bottom_left")], np.int32)],
                  True, (80, 255, 80), 2)
    text(om, f"OSM: roads (orange; service/track grey) and {shown} junctions of degree >= 3 in view "
             f"(magenta; grey = service/track only)", (10, 24))
    text(om, "green: approximate photo footprint (alignment homography; far edge unreliable)", (10, 48))
    cv2.imwrite(str(FIG / "roads_osm_junctions.jpg"), om, [cv2.IMWRITE_JPEG_QUALITY, 86])

    # --- figure 3 (DIAGNOSTIC, uses the truth): OSM projected into the photo at the true pose
    diag = json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())
    area = json.loads((DATA / "search_areas.json").read_text())["areas"][0]
    pr = cc.Problem(area); p = np.array(diag["parameters"]); f, c, R, _ = pr.unpack(p)

    def project(xy):
        z = np.array([pr.ground_z(x - pr.origin[0], y - pr.origin[1]) for x, y in xy])
        v = (np.c_[xy[:, 0] - pr.origin[0], xy[:, 1] - pr.origin[1], z] - c) @ R.T
        ok = v[:, 2] > 20
        return f * v[:, :2] / np.maximum(v[:, 2:], 1) + [cc.CX, cc.CY], ok

    d3 = photo.copy()
    for w in ways:
        dense = np.vstack([np.linspace(a, b2, max(2, int(np.linalg.norm(b2 - a) / 4)))
                           for a, b2 in zip(w["xy"][:-1], w["xy"][1:])])
        uv, ok = project(dense)
        col = (140, 140, 140) if w["highway"] in MINOR else (30, 200, 255)
        for seg in np.split(np.arange(len(uv)), np.where(~ok)[0]):
            seg = seg[ok[seg]] if len(seg) else seg
            if len(seg) > 1:
                cv2.polylines(d3, [uv[seg].astype(np.int32)], False, col, 2)
    ouv, ok = project(np.array([j["utm"] for j in ojs]))
    inframe = ok & (ouv[:, 0] >= 0) & (ouv[:, 0] < cc.IMAGE_W) & (ouv[:, 1] >= 0) & (ouv[:, 1] < cc.IMAGE_H - 12)
    for (u, v), j, inside in zip(ouv, ojs, inframe):
        if inside:
            cv2.drawMarker(d3, (int(u), int(v)), (160, 160, 160) if j["minor_only"] else (255, 60, 255), cv2.MARKER_SQUARE, 18, 2)
    iuv = np.array([j.xy for j in ijs]) if ijs else np.zeros((0, 2))
    for x, y in iuv:
        cv2.circle(d3, (int(x), int(y)), 10, (80, 255, 80), 2)
    # agreement within a tolerance, one-to-one greedy by distance
    tol = 30.0
    P = ouv[inframe]; used = set(); matched = 0; pairs = []
    if len(P) and len(iuv):
        D = np.linalg.norm(iuv[:, None] - P[None], axis=2)
        for k in np.argsort(D, axis=None):
            a, bb = np.unravel_index(k, D.shape)
            if D[a, bb] > tol:
                break
            if a in {x[0] for x in pairs} or bb in used:
                continue
            pairs.append((int(a), int(bb))); used.add(bb); matched += 1
    text(d3, "DIAGNOSTIC (true pose): OSM roads projected (orange; grey service/track), OSM junctions (magenta squares),", (10, 24))
    text(d3, f"detected crossings (green). {matched} of {len(iuv)} detections within {tol:.0f} px of one of "
             f"{int(inframe.sum())} OSM junctions in frame", (10, 48))
    cv2.imwrite(str(FIG / "roads_diagnostic_true_pose.jpg"), d3, [cv2.IMWRITE_JPEG_QUALITY, 88])

    out = dict(
        image=dict(road_instances=n_inst, junctions_deg3plus=len(ijs),
                   junctions=[dict(id=i, xy=j.xy.tolist(), degree=int(j.degree),
                                   branch_dirs=j.dirs.tolist()) for i, j in enumerate(ijs, 1)],
                   method="union of SAM 3 road masks, 9x9 close, skeletonize, junctions_from_mask(walk 20, merge 10, min branch 10)"),
        osm=dict(road_ways=len(ways), junctions_deg3plus=len(ojs),
                 junctions_minor_only=sum(j["minor_only"] for j in ojs), junctions=ojs,
                 excluded_highway=sorted(EXCLUDED)),
        diagnostic_true_pose=dict(osm_junctions_in_frame=int(inframe.sum()),
                                  osm_junctions_in_frame_major=int(sum(1 for j, i in zip(ojs, inframe) if i and not j["minor_only"])),
                                  detections_matched_within_px=tol, detections_matched=matched,
                                  detections_total=len(iuv),
                                  precision=matched / len(iuv) if len(iuv) else None,
                                  recall=matched / int(inframe.sum()) if inframe.sum() else None))
    (DATA / "road_junctions.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("junctions",)} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
