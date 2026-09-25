#!/usr/bin/env python3
"""Fetch OSM building footprints covering the search areas (no location prior beyond the boxes).

The write-up used official IGN 3D roof polygons. Lebanon has no public equivalent,
so OSM footprints stand in, lifted onto the 10 m DEM. The query box is the union of
the search areas, and it is recorded next to the cache so a small cache is never
mistaken for a large one (Overpass returns whole ways).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from pyproj import Transformer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]


def main():
    areas = json.loads((DATA / "search_areas.json").read_text())["areas"]
    e0 = min(a["bounds_utm"][0] for a in areas); n0 = min(a["bounds_utm"][1] for a in areas)
    e1 = max(a["bounds_utm"][2] for a in areas); n1 = max(a["bounds_utm"][3] for a in areas)
    to_ll = Transformer.from_crs(32636, 4326, always_xy=True)
    lo0, la0 = to_ll.transform(e0, n0); lo1, la1 = to_ll.transform(e1, n1)
    q = (f'[out:json][timeout:170];(way["building"]({la0},{lo0},{la1},{lo1});'
         f'relation["building"]({la0},{lo0},{la1},{lo1}););out tags geom;')
    data = None
    for attempt in range(3):
        for url in ENDPOINTS:
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                             headers={"User-Agent": "chamaa-constellation-test/0.1"})
                data = json.load(urllib.request.urlopen(req, timeout=180))
                print("ok via", url.split("/")[2]); break
            except Exception as ex:
                print("fail", url.split("/")[2], str(ex)[:80])
        if data: break
        time.sleep(10)
    if data is None:
        raise SystemExit("all Overpass endpoints failed")
    (DATA / "osm_buildings.json").write_text(json.dumps(data))
    (DATA / "osm_buildings.meta.json").write_text(json.dumps({
        "query_bbox_utm": [e0, n0, e1, n1], "query_bbox_lonlat": [lo0, la0, lo1, la1],
        "osm_base_timestamp": data.get("osm3s", {}).get("timestamp_osm_base"),
        "n_elements": len(data["elements"])}, indent=2))
    print(len(data["elements"]), "building elements")


if __name__ == "__main__":
    main()
