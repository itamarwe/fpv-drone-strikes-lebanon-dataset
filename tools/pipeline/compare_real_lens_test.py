#!/usr/bin/env python3
"""Before/after undistortion comparison for a real scene without ground truth.

For each VGGT-Omega direct run (scene dir with runpod_artifacts/predictions.npz):
  - focal length VGGT estimated vs the COLMAP-calibrated focal (scaled to VGGT's input size)
  - Sim(3) agreement of VGGT camera centres with the COLMAP self-calibrated path
    (registered dense frames only), plus local-window scale drift relative to it
  - per-frame depth confidence and the fraction of near-zero-parallax frames
The COLMAP path is itself an estimate, so treat agreement as consistency, not accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_3d_trajectory import umeyama  # noqa: E402
from fit_moge3_colmap_scale import qvec_to_rotation, read_images  # noqa: E402


def colmap_centres(model_dir: Path) -> dict[int, np.ndarray]:
    out = {}
    for im in read_images(model_dir / "images.txt"):
        R = qvec_to_rotation(im.qvec)
        out[int(Path(im.name).stem.rsplit("_", 1)[1])] = -(R.T @ im.tvec)
    return out


def summary(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())} if v.size else {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True, help="scene dirs of VGGT-Omega direct runs, label=path")
    p.add_argument("--colmap-model", type=Path, required=True, help="calibration fit dir with cameras.txt/images.txt (dense frames)")
    p.add_argument("--dense-frames-csv", type=Path, required=True, help="profile frames.csv mapping profile_index to source_index")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = list(csv.DictReader(args.dense_frames_csv.open()))
    prof_to_src = {int(r["profile_index"]): int(r["source_index"]) for r in rows}
    ref = {prof_to_src[i]: c for i, c in colmap_centres(args.colmap_model).items() if i in prof_to_src}
    cam_line = next(l for l in (args.colmap_model / "cameras.txt").read_text().splitlines() if not l.startswith("#"))
    cparts = cam_line.split(); cmodel = cparts[1]; cw, ch = int(cparts[2]), int(cparts[3]); cparams = [float(x) for x in cparts[4:]]
    calib_fx = cparams[0]
    report = {"colmap_model": cmodel, "colmap_params": cparams, "colmap_image_size": [cw, ch], "registered_reference_frames": len(ref), "runs": {}}

    for item in args.runs:
        label, path = (str(item).split("=", 1) if "=" in str(item) else (item.name, str(item)))
        sd = Path(path)
        z = np.load(sd / "runpod_artifacts" / "predictions.npz", allow_pickle=True)
        ext = np.asarray(z["extrinsic"], float); K = np.asarray(z["intrinsic"], float)
        S, H, W = np.asarray(z["depth"]).shape[:3]
        P = np.array([np.linalg.inv(np.vstack([e, [0, 0, 0, 1]]))[:3, 3] for e in ext])
        # predictions are in published frame order; source_index i -> frame i
        common = sorted(i for i in ref if i < len(P))
        A = P[common]; B = np.vstack([ref[i] for i in common])
        s, R, t = umeyama(A, B)
        al = (s * (R @ A.T)).T + t
        err = np.linalg.norm(al - B, axis=1)
        ref_len = float(np.sum(np.linalg.norm(np.diff(B, axis=0), axis=1)))
        local = []
        w = args.window
        for st in range(0, len(common) - w + 1, max(1, w // 2)):
            sl = umeyama(A[st:st + w], B[st:st + w])[0]
            local.append({"frames": [common[st], common[st + w - 1]], "relative_to_global": float(sl / s)})
        rel = np.array([l["relative_to_global"] for l in local])
        undist = sd / "frames" / "undistortion.json"
        fx_truth_px = None
        if undist.exists():
            u = json.loads(undist.read_text())["output_pinhole"]; fx_truth_px = u["fx"] * W / u["width"]
        else:
            fx_truth_px = calib_fx * W / cw  # calibrated focal of the distorted frames (approximate for fisheye centre)
        report["runs"][label] = {
            "frames": int(S), "vggt_input_size": [W, H],
            "vggt_fx_px_median": float(np.median(K[:, 0, 0])), "vggt_fx_px_spread": float(K[:, 0, 0].max() - K[:, 0, 0].min()),
            "reference_fx_px_at_vggt_size": float(fx_truth_px), "fx_ratio_vggt_over_reference": float(np.median(K[:, 0, 0]) / fx_truth_px),
            "sim3_vs_colmap_path": {"matched": len(common), "rmse_fraction_of_path_length": float(np.sqrt(np.mean(err**2)) / ref_len), "error_units_colmap": summary(err)},
            "local_scale_relative": {"range": [float(rel.min()), float(rel.max())], "std": float(rel.std()), "windows": local},
            "depth_conf_median": float(np.median(np.asarray(z["depth_conf"]))),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for label, r in report["runs"].items():
        print(f"{label}: fx {r['vggt_fx_px_median']:.1f} vs ref {r['reference_fx_px_at_vggt_size']:.1f} (ratio {r['fx_ratio_vggt_over_reference']:.3f}) | "
              f"path RMSE vs COLMAP {100*r['sim3_vs_colmap_path']['rmse_fraction_of_path_length']:.2f}% | local scale {r['local_scale_relative']['range'][0]:.2f}-{r['local_scale_relative']['range'][1]:.2f} (std {r['local_scale_relative']['std']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
