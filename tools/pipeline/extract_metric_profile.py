#!/usr/bin/env python3
"""Scale a VGGT camera path into a metric flight profile."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_scale(args: argparse.Namespace) -> tuple[float, str]:
    if args.scale_m_per_unit is not None:
        return args.scale_m_per_unit, "explicit --scale-m-per-unit"
    if not args.scale_summary:
        raise SystemExit("Pass --scale-m-per-unit or --scale-summary")
    summary = json.loads(args.scale_summary.read_text())
    if args.scale_key not in summary:
        raise SystemExit(f"Scale key {args.scale_key!r} is missing from {args.scale_summary}")
    return float(summary[args.scale_key]), f"{args.scale_summary}:{args.scale_key}"


def terminal_index(rows: list[dict[str, str]], mode: str) -> int:
    if mode == "last":
        return len(rows) - 1
    if mode == "last_attack":
        attack = [i for i, row in enumerate(rows) if row.get("is_attack") == "true"]
        return attack[-1] if attack else len(rows) - 1
    raise ValueError(mode)


def height_sign_for(rows: list[dict[str, str]], xyz: np.ndarray, axis: int, terminal_idx: int, mode: str) -> float:
    if mode == "positive":
        return 1.0
    if mode == "negative":
        return -1.0
    attack = [i for i, row in enumerate(rows) if row.get("is_attack") == "true"]
    start_idx = attack[0] if attack else 0
    delta = xyz[start_idx, axis] - xyz[terminal_idx, axis]
    return 1.0 if delta >= 0 else -1.0


def build_profile(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = read_rows(args.vggt_path)
    if len(rows) < 2:
        raise SystemExit(f"Need at least two VGGT cameras in {args.vggt_path}")
    scale, scale_source = load_scale(args)
    xyz_units = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    times = np.array([float(r["video_time_s"]) for r in rows], dtype=float)
    axis = {"x": 0, "y": 1, "z": 2}[args.up_axis]
    horizontal_axes = [idx for idx in range(3) if idx != axis]
    term_idx = terminal_index(rows, args.zero_height_at)
    sign = height_sign_for(rows, xyz_units, axis, term_idx, args.height_sign)

    xyz_m = xyz_units * scale
    height_m = sign * (xyz_units[:, axis] - xyz_units[term_idx, axis]) * scale
    deltas_m = np.diff(xyz_m, axis=0)
    dt = np.diff(times)
    valid_dt = dt > 1e-9
    step_3d_m = np.linalg.norm(deltas_m, axis=1)
    step_ground_m = np.linalg.norm(deltas_m[:, horizontal_axes], axis=1)
    step_vertical_m = np.diff(height_m)

    speed_3d = np.full(len(rows), np.nan)
    ground_speed = np.full(len(rows), np.nan)
    vertical_rate = np.full(len(rows), np.nan)
    speed_3d[1:][valid_dt] = step_3d_m[valid_dt] / dt[valid_dt]
    ground_speed[1:][valid_dt] = step_ground_m[valid_dt] / dt[valid_dt]
    vertical_rate[1:][valid_dt] = step_vertical_m[valid_dt] / dt[valid_dt]

    cumulative_3d = np.concatenate([[0.0], np.cumsum(step_3d_m)])
    cumulative_ground = np.concatenate([[0.0], np.cumsum(step_ground_m)])
    profile_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        profile_rows.append(
            {
                "frame_index": int(row["frame_index"]),
                "file": row["file"],
                "video_file": row.get("video_file", ""),
                "segment_id": row.get("segment_id", ""),
                "is_attack": row.get("is_attack", ""),
                "video_time_s": float(row["video_time_s"]),
                "segment_time_s": float(row.get("segment_time_s", "nan")),
                "x_m": float(xyz_m[idx, 0]),
                "y_m": float(xyz_m[idx, 1]),
                "z_m": float(xyz_m[idx, 2]),
                "height_agl_proxy_m": float(height_m[idx]),
                "ground_distance_from_start_m": float(cumulative_ground[idx]),
                "distance_3d_from_start_m": float(cumulative_3d[idx]),
                "ground_speed_mps": "" if np.isnan(ground_speed[idx]) else float(ground_speed[idx]),
                "speed_3d_mps": "" if np.isnan(speed_3d[idx]) else float(speed_3d[idx]),
                "vertical_rate_mps": "" if np.isnan(vertical_rate[idx]) else float(vertical_rate[idx]),
            }
        )

    finite_ground = ground_speed[np.isfinite(ground_speed)]
    finite_3d = speed_3d[np.isfinite(speed_3d)]
    finite_vrate = vertical_rate[np.isfinite(vertical_rate)]
    summary = {
        "vggt_path": str(args.vggt_path),
        "scale_m_per_vggt_unit": scale,
        "scale_source": scale_source,
        "up_axis": args.up_axis,
        "height_sign": sign,
        "zero_height_frame_index": int(rows[term_idx]["frame_index"]),
        "zero_height_video_time_s": float(rows[term_idx]["video_time_s"]),
        "frames": len(rows),
        "start_video_time_s": float(times[0]),
        "end_video_time_s": float(times[-1]),
        "duration_s": float(times[-1] - times[0]),
        "height_start_m": float(height_m[0]),
        "height_max_m": float(np.max(height_m)),
        "height_min_m": float(np.min(height_m)),
        "total_ground_distance_m": float(cumulative_ground[-1]),
        "total_3d_distance_m": float(cumulative_3d[-1]),
        "net_horizontal_displacement_m": float(np.linalg.norm((xyz_m[-1] - xyz_m[0])[horizontal_axes])),
        "net_3d_displacement_m": float(np.linalg.norm(xyz_m[-1] - xyz_m[0])),
        "median_ground_speed_mps": float(np.median(finite_ground)) if len(finite_ground) else 0.0,
        "p90_ground_speed_mps": float(np.quantile(finite_ground, 0.9)) if len(finite_ground) else 0.0,
        "max_ground_speed_mps": float(np.max(finite_ground)) if len(finite_ground) else 0.0,
        "median_3d_speed_mps": float(np.median(finite_3d)) if len(finite_3d) else 0.0,
        "p90_3d_speed_mps": float(np.quantile(finite_3d, 0.9)) if len(finite_3d) else 0.0,
        "max_3d_speed_mps": float(np.max(finite_3d)) if len(finite_3d) else 0.0,
        "median_vertical_rate_mps": float(np.median(finite_vrate)) if len(finite_vrate) else 0.0,
        "note": "Height is terminal-frame-relative AGL proxy, not terrain-following ground clearance.",
    }
    return profile_rows, summary


def plot_profile(csv_rows: list[dict[str, object]], summary: dict[str, object], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([float(r["video_time_s"]) for r in csv_rows])
    h = np.array([float(r["height_agl_proxy_m"]) for r in csv_rows])
    ground_distance = np.array([float(r["ground_distance_from_start_m"]) for r in csv_rows])
    dist_3d = np.array([float(r["distance_3d_from_start_m"]) for r in csv_rows])
    ground_speed = np.array([np.nan if r["ground_speed_mps"] == "" else float(r["ground_speed_mps"]) for r in csv_rows])
    speed_3d = np.array([np.nan if r["speed_3d_mps"] == "" else float(r["speed_3d_mps"]) for r in csv_rows])
    vertical_rate = np.array([np.nan if r["vertical_rate_mps"] == "" else float(r["vertical_rate_mps"]) for r in csv_rows])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, facecolor="white")
    axes[0].plot(t, h, color="#d64b26", lw=2.2)
    axes[0].scatter([t[0], t[-1]], [h[0], h[-1]], color=["#2a9d55", "#111111"], s=45, zorder=3)
    axes[0].axhline(0.0, color="#777777", lw=1.0, ls="--")
    axes[0].set_ylabel("height proxy m")
    axes[0].set_title(
        f"VGGT metric flight profile, scale={float(summary['scale_m_per_vggt_unit']):.3g} m/unit"
    )

    axes[1].plot(t, ground_speed, color="#1677b9", lw=2.0, label="ground speed")
    axes[1].plot(t, speed_3d, color="#744fc6", lw=1.6, alpha=0.78, label="3D speed")
    axes[1].set_ylabel("m/s")
    axes[1].legend(loc="best")

    axes[2].plot(t, ground_distance, color="#1677b9", lw=2.0, label="ground distance")
    axes[2].plot(t, dist_3d, color="#744fc6", lw=1.6, alpha=0.78, label="3D distance")
    axes[2].plot(t, vertical_rate, color="#d64b26", lw=1.2, alpha=0.65, label="vertical rate")
    axes[2].set_ylabel("m or m/s")
    axes[2].set_xlabel("source video time s")
    axes[2].legend(loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vggt-path", type=Path, required=True)
    parser.add_argument("--scale-summary", type=Path)
    parser.add_argument("--scale-key", default="step_scale_inlier_median")
    parser.add_argument("--scale-m-per-unit", type=float)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--up-axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--height-sign", choices=["auto", "positive", "negative"], default="auto")
    parser.add_argument("--zero-height-at", choices=["last_attack", "last"], default="last_attack")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_rows, summary = build_profile(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "metric_profile.csv", profile_rows)
    (args.out_dir / "metric_profile_summary.json").write_text(json.dumps(summary, indent=2))
    plot_profile(profile_rows, summary, args.out_dir / "metric_profile.png")
    print(json.dumps(summary, indent=2))
    print(f"[metric-profile] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
