#!/usr/bin/env python3
"""Convert AMB3R outputs and fit a scale against a VGGT relative path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_vggt_path(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    return rows, xyz


def camera_centers_from_poses(poses: np.ndarray) -> np.ndarray:
    """Return camera centers from AMB3R poses.

    AMB3R scripts save `pose` as per-frame 4x4 matrices. In the public scripts
    these are treated as camera-to-world transforms, so the center is the
    translation column.
    """
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose shape (T,4,4) or (1,T,4,4), got {poses.shape}")
    return poses[:, :3, 3].astype(float)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit dst ~= scale * src @ R.T + t."""
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {dst.shape}")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    cov = (dst0.T @ src0) / len(src)
    u, singular, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vt
    var_src = np.mean(np.sum(src0 * src0, axis=1))
    scale = float((singular * d).sum() / var_src)
    trans = dst_mean - scale * (rot @ src_mean)
    aligned = scale * (src @ rot.T) + trans
    return scale, rot, trans, aligned


def command_extract(args: argparse.Namespace) -> None:
    data = np.load(args.amb3r_npz)
    centers = camera_centers_from_poses(data["pose"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["amb3r_index", "x_m", "y_m", "z_m"])
        writer.writeheader()
        for idx, xyz in enumerate(centers, start=1):
            writer.writerow({"amb3r_index": idx, "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]})
    print(f"[amb3r] wrote {args.out_csv}")


def command_fit(args: argparse.Namespace) -> None:
    _, vggt_xyz = load_vggt_path(args.vggt_path)
    amb3r_rows = list(csv.DictReader(args.amb3r_path.open()))
    amb3r_xyz = np.array([[float(r["x_m"]), float(r["y_m"]), float(r["z_m"])] for r in amb3r_rows], dtype=float)

    manifest_rows = list(csv.DictReader(args.manifest.open()))
    pairs = []
    for idx, row in enumerate(manifest_rows):
        vggt_idx = int(row["source_frame_index"]) - 1
        amb3r_idx = idx
        if 0 <= vggt_idx < len(vggt_xyz) and 0 <= amb3r_idx < len(amb3r_xyz):
            pairs.append((vggt_idx, amb3r_idx))
    if len(pairs) < 3:
        raise SystemExit(f"Need at least 3 matched cameras, got {len(pairs)}")

    src = np.array([vggt_xyz[i] for i, _ in pairs], dtype=float)
    dst = np.array([amb3r_xyz[j] for _, j in pairs], dtype=float)
    scale, rot, trans, aligned = umeyama_similarity(src, dst)
    errors = np.linalg.norm(aligned - dst, axis=1)
    summary = {
        "status": "ok",
        "scale_m_per_vggt_unit": scale,
        "matched_cameras": len(pairs),
        "rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "median_error_m": float(np.median(errors)),
        "max_error_m": float(np.max(errors)),
        "rotation": rot.tolist(),
        "translation_m": trans.tolist(),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(f"[amb3r] summary={summary}")
    print(f"[amb3r] wrote {args.summary_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract-path")
    p.add_argument("--amb3r-npz", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)

    p = sub.add_parser("fit-vggt-scale")
    p.add_argument("--vggt-path", type=Path, required=True)
    p.add_argument("--amb3r-path", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "extract-path":
        command_extract(args)
    elif args.command == "fit-vggt-scale":
        command_fit(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
