#!/usr/bin/env python3
"""Score every method attempt in the synthetic-flyover benchmark against Blender truth.

Walks results/synthetic/<profile>/<method>/attempt-*/ and calls
compare_reconstruction_to_blender_gt.py for scal3r, moge3_glomap (as glomap,
with the MoGe-3 claimed scale) and querysplat feed-forward. VGGT-Omega direct
runs (fisheye and pinhole scene directories) are scored in place. Writes a
markdown summary table.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "tools" / "pipeline" / "compare_reconstruction_to_blender_gt.py"


def latest_complete(base: Path, sub: str = "") -> Path | None:
    if not base.is_dir():
        return None
    attempts = sorted(p for p in base.glob("attempt-*") if (p / "run.json").exists()
                      and json.loads((p / "run.json").read_text()).get("status") == "complete")
    for a in reversed(attempts):
        if not sub or (a / sub).exists():
            return a
    return None


def run_scorer(scene_dir: Path, method: str, run_dir: Path | None, label: str, claimed: float | None) -> dict | None:
    cmd = [sys.executable, str(SCORER), str(scene_dir), "--method", method, "--label", label]
    if run_dir:
        cmd += ["--run-dir", str(run_dir)]
    if claimed:
        cmd += ["--claimed-scale", str(claimed)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[score] {label} failed:\n{proc.stderr[-1500:]}", file=sys.stderr)
        return None
    out = scene_dir / "gt_comparison" / label / "comparison.json"
    return json.loads(out.read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "benchmarks" / "synthetic_flyover" / "results")
    p.add_argument("--scene-fisheye", type=Path, default=ROOT / "scenes/synthetic_zofa_al_bayada/synthetic_zofa_fpv_roof_flyover_120f")
    p.add_argument("--scene-pinhole", type=Path, default=ROOT / "scenes/synthetic_zofa_al_bayada/synthetic_zofa_fpv_roof_flyover_120f_pinhole")
    p.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "synthetic_flyover" / "SCORES.md")
    args = p.parse_args()
    rows = []

    def add(label, profile, method_name, rep, claimed=None, extra=""):
        if rep is None:
            rows.append({"label": label, "profile": profile, "method": method_name, "status": "not scored"})
            return
        sc = rep["scale"]; cs = rep["cloud_vs_surface"]; ls = rep["local_scale"]
        rows.append({
            "label": label, "profile": profile, "method": method_name, "status": "scored",
            "frames": f'{rep["frames"]["matched"]}/{rep["frames"]["ground_truth"]}',
            "scale": sc["metres_per_vggt_unit_sim3"],
            "path_rmse_pct": 100 * sc["camera_centre_error_fraction_of_path_length"],
            "cam_med_m": sc["camera_centre_error_m_after_sim3"]["median"],
            "orient_med": sc["camera_orientation_error_deg"]["median"],
            "local_range": ls["relative_scale_range"],
            "surf_med": cs["distance_to_surface_m_after_rigid_refinement"]["median"],
            "surf_p90": cs["distance_to_surface_m_after_rigid_refinement"]["p90"],
            "within3": cs["fraction_within_3m_after_refinement"],
            "points": cs["cloud_points_scored"],
            "depth_full": rep.get("depth_vs_gt", {}).get("global_median_ratio_pred_over_gt"),
            "depth_central": rep.get("depth_vs_gt", {}).get("median_central_30deg_ratio_across_frames"),
            "depth_relerr": rep.get("depth_vs_gt", {}).get("median_abs_rel_error_across_frames"),
            "claimed": rep.get("claimed_scale"),
            "extra": extra,
        })

    for profile, scene in (("full", args.scene_fisheye), ("pinhole", args.scene_pinhole)):
        if not (scene / "ground_truth" / "gt_cameras.json").exists():
            continue
        # VGGT-Omega direct run lives in the scene directory itself.
        if (scene / "runpod_artifacts" / "predictions.npz").exists():
            add(f"vggt_omega_direct_{profile}", profile, "VGGT-Omega direct (demo pipeline)", run_scorer(scene, "vggt_omega", None, f"vggt_omega_{profile}", None))
        base = args.results / "synthetic" / profile
        a = latest_complete(base / "scal3r_zju", "points/whole.ply")
        if a:
            add(f"scal3r_{profile}", profile, "Scal3R-ZJU", run_scorer(scene, "scal3r", a, f"scal3r_{profile}", None))
        a = latest_complete(base / "moge3_glomap", "sparse-txt/images.txt")
        if a:
            claimed = None
            ms = a / "metric_scale.json"
            if ms.exists():
                claimed = json.loads(ms.read_text()).get("global_scale_m_per_colmap_unit")
            add(f"glomap_{profile}", profile, "GLOMAP sparse + MoGe-3 scale", run_scorer(scene, "glomap", a, f"glomap_{profile}", claimed))
        a = latest_complete(base / "querysplat", "feed_forward/predicted_input_cameras.json")
        if a:
            add(f"querysplat_ff_{profile}", profile, "VGGT-Omega via QuerySplat (512 crop)", run_scorer(scene, "querysplat", a / "feed_forward", f"querysplat_ff_{profile}", None))

    f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"
    lines = ["# Synthetic flyover: methods vs Blender truth", "",
             "| Run | Profile | Cameras | Scale m/unit | Path RMSE % | Cam err med m | Orient med deg | Local scale range | Surface med / p90 m | Within 3 m | Points | Depth ratio full / central | Claimed scale / true |",
             "|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|"]
    for r in rows:
        if r["status"] != "scored":
            lines.append(f"| {r['label']} | {r['profile']} | {r['status']} |||||||||||")
            continue
        lr = r["local_range"]; cl = r["claimed"]
        lines.append(f"| {r['method']} | {r['profile']} | {r['frames']} | {f(r['scale'])} | {f(r['path_rmse_pct'],1)} | {f(r['cam_med_m'],1)} | {f(r['orient_med'],1)} | "
                     f"{f(lr[0]) if lr else 'n/a'} – {f(lr[1]) if lr else 'n/a'} | {f(r['surf_med'],1)} / {f(r['surf_p90'],1)} | {f(100*r['within3'],0)}% | {r['points']:,} | "
                     f"{f(r['depth_full'])} / {f(r['depth_central'])} | {(f(cl['claimed_metres_per_unit'])+' / '+f(cl['true_metres_per_unit_sim3'])+' ('+f(100*cl['relative_error'],0)+'%)') if cl else 'n/a'} |")
    lines += ["", "Scale is the Sim(3) fit of predicted camera centres to the exact Blender poses. Path RMSE is the residual as a fraction of the 330 m path. "
              "Surface distance is nearest-neighbour from the aligned cloud to a 3M-point sample of the Blender surface after rigid refinement. "
              "Depth ratios are predicted depth (times the fitted scale) over true camera-z, VGGT-Omega direct runs only. "
              "Claimed scale is MoGe-3's automatic metres-per-unit against the true value."]
    args.output.write_text("\n".join(lines) + "\n")
    (args.output.with_suffix(".json")).write_text(json.dumps(rows, indent=1, default=str) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
