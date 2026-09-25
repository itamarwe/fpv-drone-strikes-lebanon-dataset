#!/usr/bin/env python3
"""SANITY OVERLAY (uses the ground-truth pose): does SAM agree with OSM on this photo?

Draws, on the Chamaa photo:
  SAM side  - building masks (outlined; those kept by the method's filter in solid
              cyan, rejected ones dashed grey), the road mask, detected crossings
  OSM side  - building footprints projected at ground level (yellow) and at roof
              level (DEM + assumed height, thin orange), roads, and crossings
projected with the camera recovered from the LoFTR alignment
(results/diagnostics/true_pose_cost.json, PnP 3.6 px RMS). Panels: SAM only, OSM
only, and both together, so a mismatch is visible before any score is trusted.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
from pyproj import Transformer

import chamaa_constellation as cc
from road_junctions import image_junctions, osm_junctions

CYAN, GREY, YELLOW, ORANGE, BLUE, MAGENTA, GREEN, WHITE = (
    (255, 230, 40), (170, 170, 170), (40, 220, 255), (0, 140, 255), (255, 120, 40), (255, 60, 255), (80, 255, 80), (255, 255, 255))


def text(im, s, org, scale=.55, col=WHITE, th=1):
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def main():
    photo = cv2.imread(str(cc.DATA / "chamaa_photo1.png"))
    area = next(a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"] if a["name"] == "4x4km")
    pr = cc.Problem(area)
    diag = json.loads((cc.RESULTS / "diagnostics/true_pose_cost.json").read_text())
    # the stored parameters are local to the 2x2 box; convert the ground point to this box
    two = next(a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"] if a["name"] == "2x2km")
    p = np.array(diag["parameters"], float)
    p[0] += two["bounds_utm"][0] - pr.origin[0]; p[1] += two["bounds_utm"][1] - pr.origin[1]
    f, c, R, _ = pr.unpack(p)

    def proj(xyz_local):
        v = (xyz_local - c) @ R.T
        ok = v[:, 2] > 20
        return f * v[:, :2] / np.maximum(v[:, 2:], 1) + [cc.CX, cc.CY], ok

    def ground(xy):
        return np.array([pr.ground_z(x - pr.origin[0], y - pr.origin[1]) for x, y in xy])

    # ---------------- SAM layer
    sam = photo.copy()
    seg = json.loads((cc.DATA / "sam_photo1/segments.json").read_text())["ground"]["features"]
    kept = {o["id"] for o in pr.observed}
    for feat in seg:
        m = cv2.imread(str(cc.DATA / "sam_photo1/ground_masks" / f"{feat['id']}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (cc.IMAGE_H, cc.IMAGE_W):
            m = cv2.resize(m, (cc.IMAGE_W, cc.IMAGE_H), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if feat["id"] in kept:
            cv2.drawContours(sam, cs, -1, CYAN, 2)
            cv2.circle(sam, tuple(int(v) for v in feat["centroid_xy"]), 3, CYAN, -1)
        else:
            cv2.drawContours(sam, cs, -1, GREY, 1)
    rmask, closed, skel, ijs, _ = image_junctions()
    tint = sam.copy(); tint[closed > 0] = BLUE
    sam = cv2.addWeighted(tint, .35, sam, .65, 0)
    for j in ijs:
        x, y = map(int, j.xy)
        cv2.circle(sam, (x, y), 10, GREEN, 3)

    # ---------------- OSM layer (projected at the true pose)
    osm = photo.copy()
    tf = Transformer.from_crs(4326, 32636, always_xy=True)
    buildings = json.loads((cc.DATA / "osm_buildings.json").read_text())["elements"]
    heights = {m["osm_id"]: m["height_m"] for m in pr.map}
    n_bld = 0
    for el in buildings:
        g = el.get("geometry")
        if el["type"] != "way" or not g:
            continue
        xy = np.array([tf.transform(q["lon"], q["lat"]) for q in g])
        z0 = ground(xy)
        local = np.c_[xy[:, 0] - pr.origin[0], xy[:, 1] - pr.origin[1], z0]
        uv, ok = proj(local)
        if not ok.all() or not ((uv[:, 0] > -50) & (uv[:, 0] < cc.IMAGE_W + 50) & (uv[:, 1] > -50) & (uv[:, 1] < cc.IMAGE_H + 50)).any():
            continue
        n_bld += 1
        h = heights.get(el["id"], cc.DEFAULT_BUILDING_HEIGHT_M)
        roof, _ = proj(local + [0, 0, h])
        cv2.polylines(osm, [roof.astype(np.int32)], True, ORANGE, 1, cv2.LINE_AA)
        cv2.polylines(osm, [uv.astype(np.int32)], True, YELLOW, 2, cv2.LINE_AA)
    ways, ojs = osm_junctions()
    for w in ways:
        dense = np.vstack([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 3))) for a, b in zip(w["xy"][:-1], w["xy"][1:])])
        uv, ok = proj(np.c_[dense[:, 0] - pr.origin[0], dense[:, 1] - pr.origin[1], ground(dense)])
        for seg_idx in np.split(np.arange(len(uv)), np.where(~ok)[0]):
            seg_idx = seg_idx[ok[seg_idx]] if len(seg_idx) else seg_idx
            if len(seg_idx) > 1:
                cv2.polylines(osm, [uv[seg_idx].astype(np.int32)], False, BLUE, 3, cv2.LINE_AA)
    jxy = np.array([j["utm"] for j in ojs])
    juv, jok = proj(np.c_[jxy[:, 0] - pr.origin[0], jxy[:, 1] - pr.origin[1], ground(jxy)])
    for (u, v), ok in zip(juv, jok):
        if ok and 0 <= u < cc.IMAGE_W and 0 <= v < cc.IMAGE_H:
            cv2.drawMarker(osm, (int(u), int(v)), MAGENTA, cv2.MARKER_SQUARE, 20, 3)

    # ---------------- both: OSM vectors over the SAM layer
    both = sam.copy()
    osm_only_vectors = cv2.absdiff(osm, photo).max(axis=2) > 25
    both[osm_only_vectors] = osm[osm_only_vectors]

    legend_sam = "SAM: kept building masks (cyan) / rejected (grey), road mask (blue tint), detected crossings (green circles)"
    legend_osm = "OSM at the TRUE pose: footprint at ground (yellow), at roof height (thin orange), roads (blue), crossings (magenta squares)"
    for im, title, leg in ((sam, "1. SAM segmentation only", legend_sam),
                           (osm, "2. OSM projected only", legend_osm),
                           (both, "3. Both together", legend_sam)):
        text(im, title, (10, 26), .75, WHITE, 2)
        text(im, leg, (10, 50), .48)
        if im is both:
            text(im, legend_osm, (10, 72), .48)
    text(both, f"pose: camera recovered from the LoFTR alignment (PnP, 3.6 px RMS): heading 223, pitch -31, "
               f"274 m above ground, f 1594 px", (10, 94), .48, (200, 200, 200))
    cv2.imwrite(str(cc.RESULTS / "figures/sanity_overlay_both.jpg"), both, [cv2.IMWRITE_JPEG_QUALITY, 90])
    grid = np.vstack([np.hstack([sam, osm]), np.hstack([both, np.full_like(both, 25)])])
    cv2.imwrite(str(cc.RESULTS / "figures/sanity_overlay_panels.jpg"),
                cv2.resize(grid, (grid.shape[1] * 3 // 4, grid.shape[0] * 3 // 4)), [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(f"OSM buildings drawn: {n_bld}; SAM building masks: {len(seg)} ({len(kept)} kept); "
          f"detected crossings: {len(ijs)}")


if __name__ == "__main__":
    main()
