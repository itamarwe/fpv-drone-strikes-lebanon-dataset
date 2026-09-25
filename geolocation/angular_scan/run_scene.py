#!/usr/bin/env python3
"""Run the blind angular scan + 3D verification on one scene, then (only then) score against truth.

  python3 run_scene.py sainte_maxime [--check-reference] [--workers 8] [--verify 12]
"""
import argparse
import json
import math
import time

import numpy as np

import angular_scan as A
from scenes import HERE, SCENES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify", type=int, default=12); ap.add_argument("--check-reference", action="store_true")
    args = ap.parse_args()
    out = HERE / "results" / args.scene; out.mkdir(parents=True, exist_ok=True)
    A.log(f"building scene {args.scene}")
    sc, truth = SCENES[args.scene]()
    A.log(f"{len(sc.uv)} detections, {len(sc.bxy)} map buildings, camera box "
          f"{(sc.box[2] - sc.box[0]) / 1000:.1f} x {(sc.box[3] - sc.box[1]) / 1000:.1f} km")
    V = A.Verifier(sc)
    if args.check_reference and "reference_pose" in truth:          # geometry sanity check (uses truth)
        rp = truth["reference_pose"]
        p = [*rp["camera_xy"], 2.0, rp["yaw"], rp["pitch"], rp["roll"], math.log(rp["focal"])]
        S, k, nin, q, _ = V.score(p, detail=True)
        A.log(f"REFERENCE pose check: S {S:.1f}, {k} matches of {len(sc.uv)} detections, {nin} buildings in frame, "
              f"coverage {q:.0%}")
        return
    t0 = time.perf_counter()
    raw = A.scan(sc, workers=args.workers)
    scan_min = (time.perf_counter() - t0) / 60
    A.save(out / "scan_raw.json", raw[:5000])
    variants = [("blind", None)] + ([("heading_prior", sc.extra["heading_range"])] if sc.extra.get("heading_range") else [])
    for tag, hr in variants:
        run_variant(sc, truth, V, raw, hr, args, out / tag, scan_min)


def run_variant(sc, truth, V, raw, heading_range, args, out, scan_min):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    peaks = A.distinct(raw, n=max(20, args.verify), heading_range=heading_range)
    A.log(f"[{out.name}] verifying top {args.verify} of {len(peaks)} distinct peaks (heading range {heading_range})")
    rows = []
    for i, d in enumerate(peaks[:args.verify], 1):
        p, S = V.refine(A.peak_params(d))
        _, k, nin, q, pairs = V.score(p, detail=True)
        rows.append(dict(scan_rank=i, scan_z=d["z"], S=S, matches=k, in_frame=nin, coverage=q,
                         expected=len(sc.uv) * q, params=[float(x) for x in p], pairs=pairs))
        A.log(f"  peak {i}: z {d['z']:.2f} -> S {S:.1f} ({k} matches, chance {len(sc.uv) * q:.1f}) "
              f"heading {p[3] % 360:.0f} pitch {p[4]:.1f} roll {p[5]:.1f} f {math.exp(p[6]):.0f}")
    rows.sort(key=lambda r: -r["S"])
    verify_min = (time.perf_counter() - t0) / 60
    # ---------------- evaluation only below this line
    for r in rows:
        p = r["params"]
        if truth["kind"] == "camera_position":
            r["error_m"] = math.dist(p[:2], truth["xy"])
        else:
            g = A.centre_ground_point(sc, p); r["error_m"] = math.dist(g, truth["xy"]) if g is not None else float("inf")
            r["centre_ground_xy"] = None if g is None else [float(g[0]), float(g[1])]
        r["camera_error_m"] = math.dist(p[:2], truth["camera_xy"]) if "camera_xy" in truth else None
    for d in peaks:
        d["error_m"] = math.dist((d["e"], d["n"]), truth["xy"]) if truth["kind"] == "camera_position" else None
    rank = next((i for i, r in enumerate(rows, 1) if r["error_m"] <= truth.get("correct_m", 50)), None)
    A.log(f"{'final':>5}{'scan':>5}{'S':>7}{'matches (chance)':>18}{'heading':>8}{'pitch':>7}{'roll':>6}{'focal':>7}"
          f"{'height':>7}{'error m':>9}")
    for k, r in enumerate(rows, 1):
        p = r["params"]
        print(f"{k:>5}{r['scan_rank']:>5}{r['S']:>7.1f}{r['matches']:>10} ({r['expected']:4.1f}){p[3] % 360:>8.0f}"
              f"{p[4]:>7.1f}{p[5]:>6.1f}{math.exp(p[6]):>7.0f}{p[2]:>7.0f}{r['error_m']:>9.0f}", flush=True)
    margin = rows[0]["S"] - rows[1]["S"] if len(rows) > 1 else None
    A.log(f"rank of first result within {truth.get('correct_m', 50)} m: {rank}; S margin best - second: {margin:.1f}")
    A.save(out / "result.json", dict(scene=args.scene, variant=out.name, heading_range=heading_range, detections=len(sc.uv), map_buildings=len(sc.bxy),
                                     box=sc.box, scan_minutes=scan_min, verify_minutes=verify_min,
                                     grids=dict(pitches=sc.pitches, rolls=sc.rolls, focals=sc.focals,
                                                heights=sc.heights, step_m=sc.step_m, bin_deg=sc.bin_deg),
                                     truth=truth, rank_of_correct=rank, margin_S=margin, verified=rows,
                                     scan_peaks=peaks[:20]))


if __name__ == "__main__":
    main()
