#!/usr/bin/env python3
"""Score every synthetic flight (original flyover + variants, fisheye and pinhole) against Blender truth and tabulate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scenes" / "synthetic_zofa_al_bayada"
SCORER = ROOT / "tools" / "pipeline" / "compare_reconstruction_to_blender_gt.py"
RUNS = [
    ("House 01 roof approach, fisheye", "synthetic_zofa_fpv_roof_flyover_120f"),
    ("House 01 roof approach, pinhole remap", "synthetic_zofa_fpv_roof_flyover_120f_pinhole"),
    ("House 01 roof approach, true pinhole lens", "synthetic_variant_approach_house01_pinhole"),
    ("House 21 high approach, fisheye", "synthetic_variant_approach_house21_high"),
    ("House 21 high approach, pinhole remap", "synthetic_variant_approach_house21_high_pinhole"),
    ("House 03 lateral pass, fisheye", "synthetic_variant_lateral_house03"),
    ("House 03 lateral pass, pinhole remap", "synthetic_variant_lateral_house03_pinhole"),
]


def main() -> int:
    lines = ["# Synthetic flights: VGGT-Omega vs Blender truth", "",
             "| Flight | Cameras | Scale m/unit | Path RMSE % | Cam err med / max m | Orient med deg | Local scale range | Surface med / p90 m (<100 m) | Depth ratio full / central | Median rel depth err |",
             "|---|---:|---:|---:|---:|---:|---|---|---|---:|"]
    for label, name in RUNS:
        sd = S / name
        if not (sd / "runpod_artifacts" / "predictions.npz").exists() or not (sd / "ground_truth" / "gt_cameras.json").exists():
            lines.append(f"| {label} | not run |||||||||"); continue
        out = sd / "gt_comparison" / "comparison.json"
        if not out.exists() or out.stat().st_mtime < (sd / "runpod_artifacts" / "predictions.npz").stat().st_mtime:
            r = subprocess.run([sys.executable, str(SCORER), str(sd)], capture_output=True, text=True)
            if r.returncode != 0:
                lines.append(f"| {label} | scoring failed: {r.stderr.strip().splitlines()[-1][:80]} |||||||||"); continue
        r = json.loads(out.read_text()); sc = r["scale"]; cs = r["cloud_vs_surface"]; ls = r["local_scale"]; d = r.get("depth_vs_gt", {})
        near = [b for b in cs["by_range_to_nearest_camera"] if b["range_m"][1] and b["range_m"][1] <= 100]
        near_med = sum(b["median"] * b["count"] for b in near) / max(1, sum(b["count"] for b in near)); near_p90 = max(b["p90"] for b in near) if near else float("nan")
        f = lambda v, n=2: "n/a" if v is None else f"{v:.{n}f}"
        lines.append(f"| {label} | {r['frames']['matched']}/{r['frames']['ground_truth']} | {f(sc['metres_per_vggt_unit_sim3'],1)} | {f(100*sc['camera_centre_error_fraction_of_path_length'],2)} | "
                     f"{f(sc['camera_centre_error_m_after_sim3']['median'],1)} / {f(sc['camera_centre_error_m_after_sim3']['max'],1)} | {f(sc['camera_orientation_error_deg']['median'],1)} | "
                     f"{f(ls['relative_scale_range'][0])} – {f(ls['relative_scale_range'][1])} | {f(near_med,1)} / {f(near_p90,1)} | {f(d.get('global_median_ratio_pred_over_gt'))} / {f(d.get('median_central_30deg_ratio_across_frames'))} | {f(d.get('median_abs_rel_error_across_frames'))} |")
    lines += ["", "All numbers against exact Blender truth. Scale is the Sim(3) fit on all camera centres; path RMSE is the residual as a fraction of path length; ",
              "local scale is the 20-frame window scale over the global scale; surface error is nearest-neighbour distance from the aligned cloud to a 3M-point Blender surface sample for points within 100 m of the path; ",
              "depth ratio is predicted depth times the fitted scale over true camera-z, full frame and central 30 degrees."]
    (S / "SYNTHETIC_FLIGHTS_SCORES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
