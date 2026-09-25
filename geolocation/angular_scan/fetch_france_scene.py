#!/usr/bin/env python3
"""Fetch the map side of a French scene: IGN BD TOPO buildings (paged WFS, Lambert-93, with roof
altitudes) and the SRTM 1-arc-second tile(s) covering it. Prints progress; caches everything.

  python3 fetch_france_scene.py saint_paul 7.1225 43.6967 5000
     -> scenes/saint_paul/ign_buildings.geojson (+ source.json), scenes/saint_paul/dem/*.hgt
"""
import gzip
import json
import math
import sys
import time
from pathlib import Path

import requests
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:6.1f} s] {msg}", flush=True)


def main():
    name, lon, lat, half = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    out = HERE / "scenes" / name; (out / "dem").mkdir(parents=True, exist_ok=True)
    x, y = Transformer.from_crs(4326, 2154, always_xy=True).transform(lon, lat)
    bounds = [round(x - half), round(y - half), round(x + half), round(y + half)]
    gj = out / "ign_buildings.geojson"
    if not gj.exists():
        feats, start, page = [], 0, 5000
        while True:
            params = dict(SERVICE="WFS", VERSION="2.0.0", REQUEST="GetFeature", TYPENAMES="BDTOPO_V3:batiment",
                          OUTPUTFORMAT="application/json", SRSNAME="EPSG:2154",
                          BBOX=",".join(map(str, bounds)) + ",EPSG:2154", COUNT=page, STARTINDEX=start,
                          SORTBY="cleabs")
            for attempt in range(4):
                try:
                    r = requests.get("https://data.geopf.fr/wfs", params=params, timeout=180); r.raise_for_status()
                    batch = r.json()["features"]; break
                except Exception as e:                              # noqa: BLE001
                    log(f"page {start}: {type(e).__name__} {str(e)[:120]}; retry {attempt + 1}")
                    time.sleep(5)
            else:
                raise SystemExit("WFS failed")
            feats += batch; log(f"buildings: {len(feats)} (page at {start})")
            if len(batch) < page:
                break
            start += page
        gj.write_text(json.dumps(dict(type="FeatureCollection", features=feats)))
        (out / "ign_buildings_source.json").write_text(json.dumps(dict(
            service="https://data.geopf.fr/wfs BDTOPO_V3:batiment", bounds_l93=bounds, centre_lonlat=[lon, lat],
            count=len(feats), attribution="IGN / Geoplateforme / BD TOPO"), indent=2))
    log(f"buildings cached: {gj}")
    to_ll = Transformer.from_crs(2154, 4326, always_xy=True)
    lons, lats = zip(*[to_ll.transform(a, b) for a in bounds[::2] for b in bounds[1::2]])
    for la in range(math.floor(min(lats)), math.floor(max(lats)) + 1):
        for lo in range(math.floor(min(lons)), math.floor(max(lons)) + 1):
            tile = f"N{la:02d}E{lo:03d}"; dst = out / "dem" / f"{tile}.hgt"
            if not dst.exists():
                log(f"downloading SRTM {tile}")
                r = requests.get(f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/N{la:02d}/{tile}.hgt.gz", timeout=300)
                r.raise_for_status(); dst.write_bytes(gzip.decompress(r.content))
            log(f"DEM tile {dst}")


if __name__ == "__main__":
    main()
