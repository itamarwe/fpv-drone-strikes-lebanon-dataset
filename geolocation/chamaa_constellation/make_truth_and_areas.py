#!/usr/bin/env python3
"""Derive the Chamaa ground truth from the LoFTR alignment and draw the search areas.

Ground truth: the photo-to-orthophoto homography recorded by the earlier
alignment (`data/alignment_results.json`) maps raw photo pixels onto the
north-west 500 m quadrant of the 1 km orthophoto crop (0.4 m/px, origin
705174 E / 3670464 N, EPSG:32636). The true footprint centre is where the image
centre pixel lands.

Search areas: a 2x2 km and a 4x4 km box, each centred at the truth plus a random
offset (fixed seed) so that the truth lies inside but not at the centre.

Two outputs, kept apart on purpose:
  data/search_areas.json  - box bounds only. This is the ONLY file the search reads.
  data/truth.json         - truth and offsets. Read only by evaluate.py.
"""
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20260925
TILE_ORIGIN = (705174.0, 3670464.0)   # north-west quadrant, top-left corner
TILE_GSD = 0.4


def photo_px_to_utm(H, u, v):
    p = np.asarray(H) @ np.array([u, v, 1.0])
    x, y = p[:2] / p[2]
    return TILE_ORIGIN[0] + x * TILE_GSD, TILE_ORIGIN[1] - y * TILE_GSD


def main():
    al = json.loads((DATA / "alignment_results.json").read_text())
    H = al["loftr_query_to_northwest_tile_homography"]
    w, h = 1194, 906
    centre = photo_px_to_utm(H, (w - 1) / 2, (h - 1) / 2)
    # Check the convention against the two numbers the alignment recorded.
    supplied = photo_px_to_utm(H, *al["supplied_coordinate_in_photo1_px"])
    assert abs(supplied[0] - 705674.0) < 0.5 and abs(supplied[1] - 3669964.0) < 0.5, supplied
    rec = al["consensus_photo1_center"]
    assert abs(centre[0] - rec["easting"]) < 2 and abs(centre[1] - rec["northing"]) < 2, centre
    to_ll = Transformer.from_crs(32636, 4326, always_xy=True)
    lon, lat = to_ll.transform(*centre)
    corners = {k: photo_px_to_utm(H, u, v) for k, (u, v) in
               {"top_left": (0, 0), "top_right": (w - 1, 0), "bottom_right": (w - 1, h - 1),
                "bottom_left": (0, h - 1)}.items()}

    rng = np.random.default_rng(SEED)
    areas, offsets = [], []
    for side in (2000.0, 4000.0):
        while True:
            dx, dy = rng.uniform(-0.35 * side, 0.35 * side, 2)
            if np.hypot(dx, dy) >= 0.10 * side:
                break
        cx, cy = centre[0] + dx, centre[1] + dy
        half = side / 2
        bounds = [cx - half, cy - half, cx + half, cy + half]
        name = f"{int(side / 1000)}x{int(side / 1000)}km"
        areas.append({"name": name, "side_m": side, "bounds_utm": [round(b, 2) for b in bounds]})
        offsets.append({"name": name, "box_centre_minus_truth_m": [round(dx, 1), round(dy, 1)],
                        "offset_norm_m": round(float(np.hypot(dx, dy)), 1),
                        "truth_distance_to_nearest_edge_m": round(float(min(
                            centre[0] - bounds[0], bounds[2] - centre[0],
                            centre[1] - bounds[1], bounds[3] - centre[1])), 1)})

    (DATA / "search_areas.json").write_text(json.dumps({
        "epsg": 32636,
        "note": "Box bounds only. The constellation search reads nothing else about location.",
        "areas": areas}, indent=2))
    (DATA / "truth.json").write_text(json.dumps({
        "epsg": 32636,
        "use": "EVALUATION ONLY - never read by the search",
        "source": "LoFTR photo->orthophoto homography, data/alignment_results.json "
                  "(225 inliers at 180 deg; control tile abstained; ~10 m conservative uncertainty)",
        "footprint_centre_utm": [round(centre[0], 2), round(centre[1], 2)],
        "footprint_centre_lonlat": [lon, lat],
        "image_centre_px": [(w - 1) / 2, (h - 1) / 2],
        "footprint_corners_utm_caveat": "planar-homography extrapolation; unsafe at the far edge (relief)",
        "footprint_corners_utm": {k: [round(a, 1), round(b, 1)] for k, (a, b) in corners.items()},
        "random_seed": SEED,
        "offsets": offsets}, indent=2))
    print("truth", [round(c, 1) for c in centre], f"({lat:.6f} N, {lon:.6f} E)")
    for a, o in zip(areas, offsets):
        print(a["name"], a["bounds_utm"], o)


if __name__ == "__main__":
    main()
