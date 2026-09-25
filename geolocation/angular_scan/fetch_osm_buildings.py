#!/usr/bin/env python3
"""Fetch OSM building ways (with geometry) for a UTM 36N box via Overpass (mirrors, retries, progress).
Records the query bbox next to the cache (Overpass returns whole ways, so extents mislead).

  python3 fetch_osm_buildings.py bint_jbeil 721000 3663000 733000 3675000
"""
import json
import sys
import time
from pathlib import Path

import requests
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:6.1f} s] {msg}", flush=True)


def main():
    name = sys.argv[1]; e0, n0, e1, n1 = map(float, sys.argv[2:6])
    out = HERE / "scenes" / name; out.mkdir(parents=True, exist_ok=True)
    tf = Transformer.from_crs(32636, 4326, always_xy=True)
    lons, lats = zip(*[tf.transform(x, y) for x in (e0, e1) for y in (n0, n1)])
    bbox = (min(lats), min(lons), max(lats), max(lons))
    q = f'[out:json][timeout:170];way["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out geom;'
    for attempt in range(6):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            log(f"querying {url}")
            r = requests.post(url, data={"data": q}, timeout=200, headers={"User-Agent": "angular-scan-geoloc/0.1"})
            r.raise_for_status(); data = r.json(); break
        except Exception as e:                                       # noqa: BLE001
            log(f"failed: {type(e).__name__} {str(e)[:100]}"); time.sleep(5)
    else:
        raise SystemExit("all Overpass attempts failed")
    (out / "osm_buildings.json").write_text(json.dumps(data))
    (out / "osm_buildings.meta.json").write_text(json.dumps(dict(query_bbox_utm36n=[e0, n0, e1, n1], query_bbox_latlon=bbox,
                                                                 n_elements=len(data["elements"]),
                                                                 osm_base=data.get("osm3s", {}).get("timestamp_osm_base")), indent=2))
    log(f"{len(data['elements'])} building ways -> {out / 'osm_buildings.json'}")


if __name__ == "__main__":
    main()
