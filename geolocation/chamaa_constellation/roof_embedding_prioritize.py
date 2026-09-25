#!/usr/bin/env python3
"""How much do rooftop embeddings shrink each photo roof's candidate list? (DIAGNOSTIC)

Reads roof_embeddings.py's results (true-building ranks per photo roof) and reports, per
model / setting / crop size, in the 4x4 km gallery:
  recall@K%   share of the 27 true pairs inside a top-K% shortlist (K = 1, 5, 10, 20)
  bits        mean log2(gallery / rank): information each roof's embedding supplies
  3-seed x    speed-up of a 3-correspondence constellation search seeded only from
              top-10% shortlists: (N / K)^3 fewer seeds, times the chance all 3 true
              pairs survive (recall@10%^3) - the rate at which correct seeds still appear
Control: rank by footprint area alone (bigger first) - photo detections favour big houses,
so embeddings must beat this to be carrying real appearance information.
"""
import json
import math

import numpy as np

import chamaa_constellation as cc

OUT = cc.RESULTS / "roof_embeddings"


def summarise(ranks, n):
    r = np.asarray(ranks, float)
    rec = {k: float((r <= k / 100 * n).mean()) for k in (1, 5, 10, 20)}
    return dict(recall=rec, bits=float(np.mean(np.log2(n / r))), median_pct=float(np.median(r / n) * 100),
                seed_speedup=(1 / .10) ** 3 * rec[10] ** 3)


def main():
    res = json.loads((OUT / "results.json").read_text())
    areas = {a["name"]: a for a in json.loads((cc.DATA / "search_areas.json").read_text())["areas"]}
    pr = cc.Problem(areas["4x4km"])
    area = np.array([m["area_m2"] for m in pr.map])
    truth = [v["map_index"] for v in res["pairs"].values()]
    n = len(area)
    ctrl = [int((area > area[j]).sum()) + 1 for j in truth]
    rows = [dict(model="control: footprint area", setting="-", size="-", **summarise(ctrl, n))]
    for r in res["results"]:
        if r["box"] == "4x4km":
            rows.append(dict(model=r["model"], setting=r["setting"], size=r["size"], **summarise(r["ranks"], n)))
    print(f"4x4 km gallery, {n} buildings, {len(truth)} true photo-roof pairs; chance: recall@K% = K%, bits 1.4")
    print(f"{'model':<24}{'set':>4}{'size':>9}{'med %':>7}{'r@1%':>6}{'r@5%':>6}{'r@10%':>7}{'r@20%':>7}{'bits':>6}{'3-seed x':>10}")
    for d in sorted(rows, key=lambda d: -d["bits"]):
        R = d["recall"]
        print(f"{d['model']:<24}{d['setting']:>4}{d['size']:>9}{d['median_pct']:>7.1f}{R[1]:>6.0%}{R[5]:>6.0%}{R[10]:>7.0%}"
              f"{R[20]:>7.0%}{d['bits']:>6.2f}{d['seed_speedup']:>10.0f}")
    (OUT / "prioritize.json").write_text(json.dumps(dict(
        warning="DIAGNOSTIC - pairs come from the ground-truth homography", gallery=n, pairs=len(truth),
        rows=rows), indent=2))


if __name__ == "__main__":
    main()
