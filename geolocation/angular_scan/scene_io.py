#!/usr/bin/env python3
"""Map-side and image-side inputs shared by the scene scripts (buildings, DEM grids, SAM detections)."""
import json
from pathlib import Path

import cv2
import numpy as np


def ign_buildings(geojson, bounds=None, z_mode="roof_median"):
    """IGN BD TOPO buildings -> centre xy, target z, extent. Area filter 45-2500 m2 (as the original).
    z_mode 'roof_median': median z of the 3D footprint ring (what the Sainte-Maxime code used);
    'mid': ground + 0.6 x height."""
    feats = json.loads(Path(geojson).read_text())["features"]
    xy, z, ext, ids = [], [], [], []
    for ft in feats:
        g = ft.get("geometry")
        if not g:
            continue
        ring = np.array(g["coordinates"][0][0] if g["type"] == "MultiPolygon" else g["coordinates"][0], float)
        if ring.shape[0] < 4 or ring.shape[1] < 3 or not np.isfinite(ring).all():
            continue
        loc = (ring[:, :2] - ring[0, :2]).astype(np.float32); m = cv2.moments(loc)
        if not 45 <= m["m00"] <= 2500:
            continue
        c = np.array([m["m10"], m["m01"]]) / m["m00"] + ring[0, :2]
        if bounds is not None and not (bounds[0] <= c[0] <= bounds[2] and bounds[1] <= c[1] <= bounds[3]):
            continue
        pr = ft["properties"]; hgt = pr.get("hauteur") or 5.0
        if z_mode == "roof_median":
            zz = float(np.median(ring[:, 2]))
        else:
            zz = float((pr.get("altitude_minimale_sol") or np.median(ring[:, 2]) - hgt)) + .6 * hgt
        if not -50 < zz < 3000:
            continue
        xy.append(c); z.append(zz); ext.append(float(np.max(np.ptp(ring[:, :2], axis=0)))); ids.append(pr.get("cleabs"))
    return np.array(xy), np.array(z), np.array(ext), ids


def osm_buildings(osm_json, to_crs, bounds=None, dem=None, default_h=7.0):
    """Overpass JSON (ways with geometry) -> centre xy (to_crs), target z = ground + 0.6 x height, extent."""
    from pyproj import Transformer
    tf = Transformer.from_crs(4326, to_crs, always_xy=True)
    xy, ext, hh, ids = [], [], [], []
    for el in json.loads(Path(osm_json).read_text())["elements"]:
        g = el.get("geometry")
        if el.get("type") != "way" or not g or len(g) < 4:
            continue
        v = np.array([tf.transform(p["lon"], p["lat"]) for p in g])
        loc = (v - v[0]).astype(np.float32); m = cv2.moments(loc)
        if not 45 <= m["m00"] <= 2500:
            continue
        c = np.array([m["m10"], m["m01"]]) / m["m00"] + v[0]
        if bounds is not None and not (bounds[0] <= c[0] <= bounds[2] and bounds[1] <= c[1] <= bounds[3]):
            continue
        tags = el.get("tags", {})
        try:
            h = float(tags["building:levels"]) * 3.0
        except (KeyError, ValueError):
            h = default_h
        xy.append(c); ext.append(float(np.max(np.ptp(v, axis=0)))); hh.append(h); ids.append(el["id"])
    xy = np.array(xy); hh = np.array(hh)
    z = dem.ground(xy[:, 0], xy[:, 1]) + .6 * hh if dem is not None else .6 * hh
    return xy, z, np.array(ext), ids


class Grid:
    """Regular DEM grid in a projected CRS: x, y ascending, z[y, x]."""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def ground(self, px, py):
        from scipy.ndimage import map_coordinates
        fx = (np.asarray(px) - self.x[0]) / (self.x[1] - self.x[0])
        fy = (np.asarray(py) - self.y[0]) / (self.y[1] - self.y[0])
        return map_coordinates(self.z, [np.atleast_1d(fy), np.atleast_1d(fx)], order=1, mode="nearest")


def warp_dem(src_paths, crs, bounds, res):
    """Mosaic + reproject DEM rasters (e.g. SRTM .hgt) onto a regular grid in `crs`."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_origin
    srcs = [rasterio.open(p) for p in src_paths]
    mosaic, tr = merge(srcs)
    W = int(np.ceil((bounds[2] - bounds[0]) / res)); H = int(np.ceil((bounds[3] - bounds[1]) / res))
    dst = np.zeros((H, W), np.float32)
    reproject(mosaic[0].astype(np.float32), dst, src_transform=tr, src_crs=srcs[0].crs,
              dst_transform=from_origin(bounds[0], bounds[3], res, res), dst_crs=f"EPSG:{crs}",
              resampling=Resampling.bilinear, src_nodata=-32768, dst_nodata=np.nan)
    dst[~np.isfinite(dst)] = np.nanmedian(dst)
    x = bounds[0] + (np.arange(W) + .5) * res; y = bounds[3] - (np.arange(H) + .5) * res
    return Grid(x, y[::-1].copy(), dst[::-1].copy())


def sam_detections(segments_json, flt):
    """Filtered SAM 3 building detections: centroids and box sizes (the Sainte-Maxime/Chamaa filter)."""
    raw = json.loads(Path(segments_json).read_text())["ground"]["features"]
    sel = []
    for p in sorted(raw, key=lambda a: -a["score"]):
        x0, y0, x1, y1 = p["bbox_xyxy"]; w, h = x1 - x0, y1 - y0
        if not (p["score"] >= flt["min_score"] and flt["min_area"] <= p["area_px"] <= flt["max_area"]
                and flt["min_w"] < w < flt["max_w"] and h < flt.get("max_h", 1e9)
                and x0 > flt.get("edge_px", 5) and x1 < flt["W"] - flt.get("edge_px", 5)
                and flt.get("y0_min", -1) < y0 < flt.get("y0_max", 1e9) and y1 < flt["H"] - flt.get("bottom_px", 5)):
            continue
        if any(np.linalg.norm(np.array(p["centroid_xy"]) - np.array(q["centroid_xy"]))
               < max(5, .2 * min(w, q["bbox_xyxy"][2] - q["bbox_xyxy"][0])) for q in sel):
            continue
        sel.append(p)
    uv = np.array([p["centroid_xy"] for p in sel], float)
    wh = np.array([[p["bbox_xyxy"][2] - p["bbox_xyxy"][0], p["bbox_xyxy"][3] - p["bbox_xyxy"][1]] for p in sel], float)
    return sel, uv, wh
