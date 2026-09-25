#!/usr/bin/env python3
"""Scene definitions for the angular scan. Each builder returns (Scene, truth) - truth is read only
by the evaluation step. Paths into the main repository's untracked benchmark folders are recorded
where the inputs are too large to copy."""
import json
import math
from pathlib import Path

import numpy as np
from pyproj import Transformer

import scene_io as io
from angular_scan import Scene

HERE = Path(__file__).resolve().parent
MAIN = Path.home() / "Documents/code/fpv-drone-strikes-lebanon-dataset"


def geometric(lo, hi, step):
    return [float(lo * step ** k) for k in range(int(math.floor(math.log(hi / lo) / math.log(step))) + 1)]


def sainte_maxime():
    base = MAIN / "benchmarks/france_phone_geolocation"
    W, H = 1280, 960
    # the Sainte-Maxime detection filter, unchanged (score, area, width, y0 band 180-570)
    sel, uv, wh = io.sam_detections(base / "sam_photo1/segments.json",
                                    dict(W=W, H=H, min_score=.35, min_area=100, max_area=2500, min_w=8, max_w=100,
                                         y0_min=180, y0_max=570, bottom_px=-1e9))
    box = (994750.0, 6251500.0, 997250.0, 6255000.0)          # the original camera search box (terrain extent)
    bxy, bz, bext, ids = io.ign_buildings(base / "pnp_reference/ign_buildings.geojson")
    dem = io.warp_dem([base / "dem/N43E006.hgt"], 2154, (box[0] - 6000, box[1] - 6000, box[2] + 6000, box[3] + 6000), 30.0)
    valid = np.zeros((H, W), bool); valid[180:620] = True       # the rows where detections are allowed
    sc = Scene(name="sainte_maxime", W=W, H=H, cx=W / 2, cy=H / 2, uv=uv, wh=wh, bxy=bxy, bz=bz, bext=bext,
               dem_x=dem.x, dem_y=dem.y, dem_z=dem.z, box=box, heights=[2.0],
               pitches=[-10.0, 0.0], rolls=[-3.0, 0.0, 3.0], focals=geometric(1300, 3500, 1.10),
               step_m=25.0, r_min=60.0, r_max=5000.0, bin_deg=0.2, tol_px=12.0, map_sigma_m=6.0, valid=valid,
               extra=dict(pitch_window=7.0, size_gate=True, crs=2154,
                          heading_range=(-20.0, 35.0)))   # the original run's prior; scan keeps both
    pin = Transformer.from_crs(4326, 2154, always_xy=True).transform(6.6452751, 43.3170973)
    truth = dict(kind="camera_position", xy=list(pin), source="user-supplied pin (Sainte-Maxime report)",
                 reference_pose=dict(source="Codex joint solve, refined.json", yaw=8.34, pitch=-3.54, roll=0.88,
                                     focal=2543.1, camera_xy=[995771.60, 6253269.15]))
    return sc, truth


def offset_box(truth_xy, side, seed):
    """A side x side box containing truth_xy, randomly offset (+-0.35 side, >= 0.1 side), as for Chamaa."""
    rng = np.random.default_rng(seed)
    while True:
        off = rng.uniform(-.35, .35, 2) * side
        if np.linalg.norm(off) >= .1 * side:
            break
    c = np.array(truth_xy) + off
    return (float(c[0] - side / 2), float(c[1] - side / 2), float(c[0] + side / 2), float(c[1] + side / 2)), off.tolist()


def fisheye_undistort(uv, f, cx, cy, k1):
    """COLMAP SIMPLE_RADIAL_FISHEYE pixel -> pinhole pixel with the same focal and centre."""
    d = (np.asarray(uv, float) - [cx, cy]) / f; rd = np.linalg.norm(d, axis=-1)
    th = rd.copy()
    for _ in range(20):                                # theta (1 + k1 theta^2) = rd
        th = th - (th * (1 + k1 * th ** 2) - rd) / (1 + 3 * k1 * th ** 2)
    ru = np.tan(np.clip(th, 0, 1.45))
    s = np.where(rd > 1e-9, ru / np.maximum(rd, 1e-9), 1.0)
    return np.c_[cx + f * d[..., 0] * s, cy + f * d[..., 1] * s] if d.ndim == 2 else np.array([cx + f * d[0] * s, cy + f * d[1] * s])


def dem_window(path, bounds):
    import rasterio
    with rasterio.open(path) as ds:
        win = rasterio.windows.from_bounds(*bounds, ds.transform)
        z = ds.read(1, window=win).astype(np.float32); tr = ds.window_transform(win)
    z[~np.isfinite(z) | (z < -1000)] = np.nan; z[np.isnan(z)] = np.nanmedian(z)
    x = tr.c + (np.arange(z.shape[1]) + .5) * tr.a; y = tr.f + (np.arange(z.shape[0]) + .5) * tr.e
    if y[0] > y[-1]:
        y, z = y[::-1].copy(), z[::-1].copy()
    return io.Grid(x, y, z)


def bint_jbeil():
    d = HERE / "scenes/bint_jbeil"
    Wd, Hd, F, CX, CY, K1 = 848, 480, 468.43, 424.0, 240.0, -0.0714        # calibrated lens (skyline_approach)
    # usable pixels of the distorted frame: no HUD reticle/bar, logo, blurred band, top sky edge
    vd = np.ones((Hd, Wd), bool)
    vd[335:400, 45:100] = False            # channel logo
    vd[320:420, 190:500] = False           # blurred band
    vd[125:150, 280:550] = False; vd[170:205, 385:445] = False   # HUD icons, reticle
    vd[:, :4] = vd[:, -4:] = False
    sel, uv_d, wh_d = io.sam_detections(d / "sam_x3/segments.json", dict(W=Wd, H=Hd, min_score=.35, min_area=4, max_area=9000,
                                                                       min_w=2, max_w=200, max_h=200, edge_px=3, bottom_px=3))
    keep = vd[uv_d[:, 1].astype(int), uv_d[:, 0].astype(int)]
    uv_d, wh_d = uv_d[keep], wh_d[keep]
    uv = fisheye_undistort(uv_d, F, CX, CY, K1)
    # box sizes to pinhole scale (local magnification along the radius)
    mag = np.linalg.norm(fisheye_undistort(uv_d + [1, 0], F, CX, CY, K1) - uv, axis=1)
    wh = wh_d * mag[:, None]
    # undistorted canvas and its valid mask
    corners = fisheye_undistort(np.array([[0, 0], [Wd - 1, 0], [0, Hd - 1], [Wd - 1, Hd - 1], [0, CY], [Wd - 1, CY]]), F, CX, CY, K1)
    half_w = int(np.ceil(np.abs(corners[:, 0] - CX).max())); half_h = int(np.ceil(np.abs(corners[:, 1] - CY).max()))
    W, H = 2 * half_w, 2 * half_h; cx, cy = half_w, half_h
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    pu = np.c_[(gx.ravel() - cx) / F, (gy.ravel() - cy) / F]; ru = np.linalg.norm(pu, axis=1); th = np.arctan(ru)
    rd = th * (1 + K1 * th ** 2); s = np.where(ru > 1e-9, rd / np.maximum(ru, 1e-9), 1.0)
    xd = CX + F * pu[:, 0] * s; yd = CY + F * pu[:, 1] * s
    inside = (xd >= 0) & (xd < Wd) & (yd >= 0) & (yd < Hd)
    valid = np.zeros(W * H, bool); valid[inside] = vd[yd[inside].astype(int), xd[inside].astype(int)]
    valid = valid.reshape(H, W)
    uv = uv + [cx - CX, cy - CY]
    cam_truth = [726433.36, 3669126.55]
    box, off = offset_box(cam_truth, 2000.0, 20260925)
    mb = (box[0] - 5000, box[1] - 5000, box[2] + 5000, box[3] + 5000)
    dem = dem_window(Path.home() / "Documents/code/mappa/layers/height_evoPilot.tif", mb)
    bxy, bz, bext, ids = io.osm_buildings(d / "osm_buildings.json", 32636, dem=dem)
    sc = Scene(name="bint_jbeil", W=W, H=H, cx=cx, cy=cy, uv=uv, wh=wh, bxy=bxy, bz=bz, bext=bext,
               dem_x=dem.x, dem_y=dem.y, dem_z=dem.z, box=box, heights=geometric(30, 300, 1.3),
               pitches=[-3.0, -11.0, -19.0, -27.0], rolls=[-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0], focals=[F],
               step_m=40.0, r_min=40.0, r_max=5000.0, bin_deg=0.2, tol_px=10.0, map_sigma_m=15.0, valid=valid,
               extra=dict(pitch_window=4.0, size_gate=False, min_building_px=1.0, crs=32636, lens="SIMPLE_RADIAL_FISHEYE f 468.43 k1 -0.0714",
                          box_offset_m=off))
    truth = dict(kind="camera_position", xy=cam_truth, camera_xy=cam_truth, agl_m=90.0, heading=157.6, depression=4.58,
                 source="SQPnP on the analyst's 6 correspondences (reports/real_scene/skyline_approach/result.json)")
    return sc, truth


SCENES = {"sainte_maxime": sainte_maxime, "bint_jbeil": bint_jbeil}
