#!/usr/bin/env python3
"""Ortho access for the geolocation experiments.

Cuts UTM-rectified crops straight out of the ECW mosaic (via ecw2tiff + gdalwarp)
and hands back both the pixels and the pixel->UTM affine, so any match in tile
pixels can be pushed to metres without a second bookkeeping layer.
"""

from __future__ import annotations

import math
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ECW = Path.home() / "Documents" / "code" / "mappa" / "layers" / "tzafon_3_26.ecw"
DEFAULT_DEM = Path.home() / "Documents" / "code" / "mappa" / "layers" / "height_evoPilot.tif"
DEFAULT_ECW2TIFF = Path.home() / "Documents" / "code" / "ecw2tiff" / "target" / "release" / "ecw2tiff"
UTM36N = "EPSG:32636"


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr[-2000:]}")
    return proc.stdout


@dataclass(frozen=True)
class EcwInfo:
    width: int
    height: int
    origin_x: float
    origin_y: float
    cell_x: float
    cell_y: float
    srs: str

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        return (lon - self.origin_x) / self.cell_x, (lat - self.origin_y) / self.cell_y


def ecw_info(ecw: Path = DEFAULT_ECW, tool: Path = DEFAULT_ECW2TIFF) -> EcwInfo:
    text = run([str(tool), "--info", str(ecw)])
    def grab(pattern: str) -> str:
        m = re.search(pattern, text)
        if not m:
            raise RuntimeError(f"cannot parse ECW info for {pattern!r}")
        return m.group(1)
    w, h = grab(r"Size:\s+(\d+) x (\d+)"), None
    m = re.search(r"Size:\s+(\d+) x (\d+)", text)
    cell = re.search(r"Cell size:\s+([-\d.e]+) x ([-\d.e]+)", text)
    origin = re.search(r"Origin:\s+([-\d.e]+), ([-\d.e]+)", text)
    proj = re.search(r"Projection:\s+(\S+)", text)
    return EcwInfo(
        width=int(m.group(1)), height=int(m.group(2)),
        origin_x=float(origin.group(1)), origin_y=float(origin.group(2)),
        cell_x=float(cell.group(1)), cell_y=float(cell.group(2)),
        srs=proj.group(1) if proj else "EPSG:4326",
    )


def utm_to_lonlat(pts: np.ndarray, srs: str = UTM36N) -> np.ndarray:
    """pts: (N,2) easting/northing -> (N,2) lon/lat, via gdaltransform."""
    stdin = "\n".join(f"{x} {y}" for x, y in pts)
    proc = subprocess.run(
        ["gdaltransform", "-s_srs", srs, "-t_srs", "EPSG:4326"],
        input=stdin, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gdaltransform failed: {proc.stderr}")
    out = [line.split()[:2] for line in proc.stdout.strip().splitlines()]
    return np.asarray(out, dtype=float)


@dataclass
class OrthoTile:
    """A UTM-rectified ortho crop plus the affine that turns its pixels into metres."""
    path: Path
    bounds: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax in UTM
    gsd: float
    width: int
    height: int

    def pixel_to_utm(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        xmin, _, _, ymax = self.bounds
        return np.stack([xmin + pts[:, 0] * self.gsd, ymax - pts[:, 1] * self.gsd], axis=1)

    def utm_to_pixel(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        xmin, _, _, ymax = self.bounds
        return np.stack([(pts[:, 0] - xmin) / self.gsd, (ymax - pts[:, 1]) / self.gsd], axis=1)


def cut_ortho(
    bounds: tuple[float, float, float, float],
    out_tif: Path,
    gsd: float = 0.35,
    ecw: Path = DEFAULT_ECW,
    tool: Path = DEFAULT_ECW2TIFF,
    srs: str = UTM36N,
    threads: int = 8,
    scale: int = 1,
    quality: int = 92,
    force: bool = False,
) -> OrthoTile:
    """Decode the ECW region covering `bounds` (UTM) and warp it to a north-up UTM tile."""
    xmin, ymin, xmax, ymax = bounds
    width = int(round((xmax - xmin) / gsd))
    height = int(round((ymax - ymin) / gsd))
    tile = OrthoTile(out_tif, bounds, gsd, width, height)
    if out_tif.exists() and not force:
        return tile

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    info = ecw_info(ecw, tool)
    corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]], dtype=float)
    ll = utm_to_lonlat(corners, srs)
    px = [info.lonlat_to_pixel(lon, lat) for lon, lat in ll]
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    margin = 96
    x0 = max(0, math.floor(min(xs)) - margin)
    x1 = min(info.width, math.ceil(max(xs)) + margin)
    y0 = max(0, math.floor(min(ys)) - margin)
    y1 = min(info.height, math.ceil(max(ys)) + margin)
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"requested bounds fall outside the ECW: {bounds}")

    region = out_tif.with_suffix(".region.tif")
    run([
        str(tool), "--threads", str(threads), "--compress", f"jpeg:{quality}",
        "--scale", str(scale), "--region", f"{x0},{y0},{x1 - x0},{y1 - y0}",
        str(ecw), str(region),
    ])
    run([
        "gdalwarp", "-overwrite", "-q", "-t_srs", srs,
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-tr", str(gsd), str(gsd), "-r", "cubic",
        "-b", "1", "-b", "2", "-b", "3",
        "-co", "COMPRESS=JPEG", "-co", f"JPEG_QUALITY={quality}", "-co", "TILED=YES",
        str(region), str(out_tif),
    ])
    region.unlink(missing_ok=True)
    return tile


def read_rgb(path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3])
    return np.transpose(arr, (1, 2, 0))
