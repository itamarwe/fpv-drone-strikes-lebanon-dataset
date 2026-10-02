#!/usr/bin/env python3
"""Convert a scene's viewer point buffers (points_positions.bin + points_colors.bin) to LAS/LAZ.

The web viewer stores each scene as a raw, headerless float32 XYZ buffer plus a uint8 RGB buffer
in arbitrary reconstruction units. GIS / photogrammetry tools (QGIS, ArcGIS, CloudCompare, PDAL)
cannot read that; they need a real container. This writes a coloured LAS (point format 2) or LAZ.

Coordinates are the reconstruction's own units by default. With --scale-meters the XYZ are multiplied
by the scene's default_scale_m_per_unit (NOTE: for most scenes that is a placeholder 117.6, not a
verified metric scale, so treat the result as approximate). There is no real-world georeferencing
(no CRS) until a scene is georegistered; the LAS carries local coordinates only.

Usage:
  bin_to_las.py <viewer_dir> [--out out.las] [--laz] [--scale-meters]
  bin_to_las.py scenes/<vid>/<sid>/viewer --laz
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import laspy


def convert(viewer_dir: Path, out: Path, laz: bool, scale_meters: bool) -> tuple[int, float]:
    meta = json.loads((viewer_dir / "scene_meta.json").read_text())
    pos = viewer_dir / Path(meta["assets"]["positions"]).name
    col = viewer_dir / Path(meta["assets"]["colors"]).name
    P = np.fromfile(pos, dtype="<f4").reshape(-1, 3).astype(np.float64)
    C = np.fromfile(col, dtype=np.uint8).reshape(-1, 3)
    if len(P) != len(C):
        n = min(len(P), len(C)); P, C = P[:n], C[:n]
    unit_scale = 1.0
    if scale_meters:
        unit_scale = float(meta.get("default_scale_m_per_unit") or 1.0)
        P = P * unit_scale
    header = laspy.LasHeader(point_format=2, version="1.4")  # format 2 = XYZ + RGB
    # store to ~1e-4 unit precision via integer scaling; offset at the cloud centre
    header.offsets = P.min(axis=0)
    header.scales = np.full(3, max((P.max(axis=0) - P.min(axis=0)).max() / 2_000_000.0, 1e-6))
    las = laspy.LasData(header)
    las.x, las.y, las.z = P[:, 0], P[:, 1], P[:, 2]
    las.red   = C[:, 0].astype(np.uint16) * 257  # 8-bit -> 16-bit LAS colour
    las.green = C[:, 1].astype(np.uint16) * 257
    las.blue  = C[:, 2].astype(np.uint16) * 257
    out.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out), do_compress=laz or out.suffix.lower() == ".laz")
    return len(P), unit_scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("viewer_dir", type=Path, help="a .../viewer directory containing scene_meta.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--laz", action="store_true", help="write compressed LAZ")
    ap.add_argument("--scale-meters", action="store_true", help="multiply by default_scale_m_per_unit (approximate)")
    args = ap.parse_args()
    vd = args.viewer_dir
    out = args.out or vd / f"point_cloud.{'laz' if args.laz else 'las'}"
    n, s = convert(vd, out, args.laz, args.scale_meters)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({n:,} points, {mb:.1f} MB, units={'metres x%.1f' % s if args.scale_meters else 'reconstruction'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
