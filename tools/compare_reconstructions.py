#!/usr/bin/env python3
"""Compare VGGT and AMB3R reconstructions over the same image sequence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_vggt_path(vggt_dir: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows = read_csv_rows(vggt_dir / "relative_path.csv")
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    return rows, xyz


def camera_centers_from_poses(poses: np.ndarray) -> np.ndarray:
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected AMB3R pose shape (T,4,4), got {poses.shape}")
    return poses[:, :3, 3].astype(float)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit dst ~= scale * src @ rot.T + trans."""
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
    scale = float((singular * d).sum() / max(var_src, 1e-12))
    trans = dst_mean - scale * (rot @ src_mean)
    aligned = scale * (src @ rot.T) + trans
    return scale, rot, trans, aligned


def path_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def robust_mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    med = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - med)))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_vggt_points(vggt_dir: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cloud = np.load(vggt_dir / "point_cloud.npz")
    pts = cloud["pts"].astype(float)
    cols = cloud["cols"].astype(float) / 255.0
    ok = np.isfinite(pts).all(axis=1)
    pts = pts[ok]
    cols = cols[ok]
    return sample_points(pts, cols, max_points=max_points, seed=seed)


def load_amb3r_points(npz_path: Path, max_points: int, seed: int, conf_quantile: float) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    pts = data["pts"].astype(float).reshape(-1, 3)
    conf = data["conf"].astype(float).reshape(-1)
    images = np.moveaxis(data["images"].astype(float), 1, -1)
    cols = ((images + 1.0) * 0.5).reshape(-1, 3)
    cols = np.clip(cols, 0.0, 1.0)
    sky = data["sky_mask"].reshape(-1) if "sky_mask" in data.files else np.zeros(len(conf), dtype=bool)
    threshold = float(np.quantile(conf[np.isfinite(conf)], conf_quantile)) if np.isfinite(conf).any() else 0.0
    ok = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (conf >= threshold) & ~sky
    pts = pts[ok]
    cols = cols[ok]
    return sample_points(pts, cols, max_points=max_points, seed=seed)


def sample_points(pts: np.ndarray, cols: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) <= max_points:
        return pts, cols
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(pts), max_points, replace=False)
    return pts[keep], cols[keep]


def crop_near_path(
    pts: np.ndarray,
    cols: np.ndarray,
    path: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) == 0 or len(path) == 0:
        return pts, cols
    span = np.maximum(path.max(axis=0) - path.min(axis=0), 1e-6)
    pad = np.maximum(span * 1.6, 0.05)
    lo = path.min(axis=0) - pad
    hi = path.max(axis=0) + pad
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    if mask.sum() >= 3000:
        return sample_points(pts[mask], cols[mask], max_points=max_points, seed=seed)
    center = path.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    keep = np.argsort(dist)[: min(len(pts), max_points)]
    return pts[keep], cols[keep]


def to_plot_view(points: np.ndarray) -> np.ndarray:
    return np.stack([points[:, 0], points[:, 2], points[:, 1]], axis=1)


def set_equal_limits(ax, points: np.ndarray, pad_frac: float = 0.08) -> None:
    if len(points) == 0:
        return
    center = points.mean(axis=0)
    span = float(np.max(points.max(axis=0) - points.min(axis=0)))
    span = max(span * (1.0 + pad_frac), 1e-6)
    for axis, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(center[axis] - span * 0.5, center[axis] + span * 0.5)


def write_contact_sheet(vggt_dir: Path, rows: list[dict[str, str]], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    thumbs = []
    for row in rows:
        img = Image.open(vggt_dir / "frames" / row["file"]).convert("RGB")
        img.thumbnail((210, 118), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (210, 142), (16, 18, 22))
        tile.paste(img, ((210 - img.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        label = f"{int(row['frame_index']):02d}  t={float(row['video_time_s']):.1f}s"
        draw.text((8, 122), label, fill=(235, 238, 245), font=ImageFont.load_default())
        thumbs.append(tile)
    cols = 6
    rows_n = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 210, rows_n * 142), (8, 10, 14))
    for idx, tile in enumerate(thumbs):
        sheet.paste(tile, ((idx % cols) * 210, (idx // cols) * 142))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=94)


def plot_scene(
    pts: np.ndarray,
    cols: np.ndarray,
    path: np.ndarray,
    out_path: Path,
    *,
    title: str,
    path_color: str,
    elev: float,
    azim: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, cols = crop_near_path(pts, cols, path, max_points=80_000, seed=4)
    p_pts = to_plot_view(pts)
    p_path = to_plot_view(path)

    bg = "#05070a"
    fig = plt.figure(figsize=(12, 7.2), facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)
    ax.set_axis_off()
    if len(p_pts):
        ax.scatter(
            p_pts[:, 0],
            p_pts[:, 1],
            p_pts[:, 2],
            c=cols,
            s=0.55,
            alpha=0.42,
            linewidths=0,
            depthshade=False,
        )
    if len(p_path):
        ax.plot(p_path[:, 0], p_path[:, 1], p_path[:, 2], color="white", lw=5.2, alpha=0.32)
        ax.plot(p_path[:, 0], p_path[:, 1], p_path[:, 2], color=path_color, lw=2.8, alpha=0.98)
        ax.scatter(
            [p_path[0, 0]],
            [p_path[0, 1]],
            [p_path[0, 2]],
            color="#7ee787",
            s=90,
            edgecolors="white",
            linewidths=0.7,
            depthshade=False,
        )
        ax.scatter(
            [p_path[-1, 0]],
            [p_path[-1, 1]],
            [p_path[-1, 2]],
            color=path_color,
            s=110,
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
        )
    set_equal_limits(ax, np.vstack([p_pts, p_path]) if len(p_pts) else p_path)
    try:
        ax.set_box_aspect((1.15, 1.0, 0.55))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    fig.text(0.5, 0.965, title, color="#f4f7fb", ha="center", va="top", fontsize=15)
    fig.text(0.5, 0.035, "green = first attack frame, amber/red = terminal frame", color="#9aa4b2", ha="center", fontsize=9)
    fig.subplots_adjust(left=0, right=1, bottom=0.07, top=0.93)
    fig.savefig(out_path, dpi=180, facecolor=bg, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_comparison(
    vggt: np.ndarray,
    amb3r: np.ndarray,
    vggt_aligned: np.ndarray,
    residuals: np.ndarray,
    times: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.2, 8.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_top = fig.add_subplot(grid[0, 1])
    ax_res = fig.add_subplot(grid[1, 1])

    a = to_plot_view(amb3r)
    va = to_plot_view(vggt_aligned)
    ax3d.plot(a[:, 0], a[:, 1], a[:, 2], color="#e4572e", lw=2.8, label="AMB3R")
    ax3d.plot(va[:, 0], va[:, 1], va[:, 2], color="#1677b9", lw=2.4, label="VGGT aligned")
    ax3d.scatter([a[0, 0]], [a[0, 1]], [a[0, 2]], color="#2a9d55", s=45)
    ax3d.scatter([a[-1, 0]], [a[-1, 1]], [a[-1, 2]], color="#111111", s=45)
    ax3d.set_title("Camera paths after 7-DoF alignment")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("z")
    ax3d.set_zlabel("y")
    ax3d.legend(loc="best")
    set_equal_limits(ax3d, np.vstack([a, va]))
    try:
        ax3d.set_box_aspect((1.0, 1.0, 0.55))
    except Exception:
        pass
    ax3d.view_init(elev=25, azim=-62)

    ax_top.plot(amb3r[:, 0], amb3r[:, 2], color="#e4572e", lw=2.4, label="AMB3R")
    ax_top.plot(vggt_aligned[:, 0], vggt_aligned[:, 2], color="#1677b9", lw=2.0, label="VGGT aligned")
    ax_top.set_title("Top view")
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("z")
    ax_top.axis("equal")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="best")

    ax_res.plot(times, residuals, color="#444444", lw=1.8)
    ax_res.scatter(times, residuals, c=np.linspace(0, 1, len(times)), cmap="plasma", s=22)
    ax_res.set_title("Per-frame alignment residual")
    ax_res.set_xlabel("source video time s")
    ax_res.set_ylabel("AMB3R units")
    ax_res.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scale_vs_time(rows: list[dict[str, object]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [r for r in rows if r["used_for_scale"]]
    t = np.array([float(r["mid_video_time_s"]) for r in usable], dtype=float)
    step = np.array([float(r["amb3r_per_vggt_step_scale"]) for r in usable], dtype=float)
    cum = np.array([float(r["amb3r_per_vggt_cumulative_scale"]) for r in usable], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    if len(step):
        med = float(np.median(step))
        mad = robust_mad(step)
        axes[0].plot(t, step, color="#1677b9", lw=1.5)
        axes[0].scatter(t, step, c=t, cmap="viridis", s=24, zorder=3)
        axes[0].axhline(med, color="#111111", ls="--", lw=1.0, label=f"median {med:.3g}")
        axes[0].fill_between(t, med - mad, med + mad, color="#1677b9", alpha=0.12, label=f"MAD {mad:.3g}")
        axes[0].legend(loc="best")
        axes[1].hist(step, bins=min(14, max(5, len(step) // 2)), color="#e4572e", alpha=0.78)
    if len(cum):
        axes[0].plot(t, cum, color="#e4572e", lw=1.6, alpha=0.85, label="cumulative")
    axes[0].set_title("AMB3R/VGGT scale over time")
    axes[0].set_xlabel("source video time s")
    axes[0].set_ylabel("AMB3R units per VGGT unit")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Step-scale vote histogram")
    axes[1].set_xlabel("AMB3R units per VGGT unit")
    axes[1].set_ylabel("step votes")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_scale_rows(
    rows: list[dict[str, str]],
    vggt: np.ndarray,
    amb3r: np.ndarray,
    min_step_frac: float,
) -> list[dict[str, object]]:
    v_steps = np.linalg.norm(np.diff(vggt, axis=0), axis=1)
    a_steps = np.linalg.norm(np.diff(amb3r, axis=0), axis=1)
    v_cum = np.concatenate([[0.0], np.cumsum(v_steps)])
    a_cum = np.concatenate([[0.0], np.cumsum(a_steps)])
    threshold = max(float(np.median(v_steps) * min_step_frac), 1e-9) if len(v_steps) else 1e-9
    out = []
    for idx in range(len(v_steps)):
        used = bool(v_steps[idx] >= threshold and a_steps[idx] > 1e-9)
        cum_scale = a_cum[idx + 1] / v_cum[idx + 1] if v_cum[idx + 1] > 1e-9 else float("nan")
        out.append(
            {
                "from_frame": int(rows[idx]["frame_index"]),
                "to_frame": int(rows[idx + 1]["frame_index"]),
                "from_video_time_s": float(rows[idx]["video_time_s"]),
                "to_video_time_s": float(rows[idx + 1]["video_time_s"]),
                "mid_video_time_s": 0.5 * (float(rows[idx]["video_time_s"]) + float(rows[idx + 1]["video_time_s"])),
                "vggt_step_units": float(v_steps[idx]),
                "amb3r_step_units": float(a_steps[idx]),
                "amb3r_per_vggt_step_scale": float(a_steps[idx] / v_steps[idx]) if v_steps[idx] > 1e-12 else float("nan"),
                "vggt_from_start_units": float(v_cum[idx + 1]),
                "amb3r_from_start_units": float(a_cum[idx + 1]),
                "amb3r_per_vggt_cumulative_scale": float(cum_scale),
                "used_for_scale": used,
            }
        )
    return out


def compare(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_rows, vggt_xyz = load_vggt_path(args.vggt_dir)
    amb3r_data = np.load(args.amb3r_npz)
    amb3r_xyz = camera_centers_from_poses(amb3r_data["pose"])

    n = min(len(frame_rows), len(vggt_xyz), len(amb3r_xyz))
    if n < 3:
        raise SystemExit(f"Need at least 3 matched cameras; got {n}")
    frame_rows = frame_rows[:n]
    vggt_xyz = vggt_xyz[:n]
    amb3r_xyz = amb3r_xyz[:n]
    times = np.array([float(r["video_time_s"]) for r in frame_rows], dtype=float)

    scale, rot, trans, vggt_to_amb3r = umeyama_similarity(vggt_xyz, amb3r_xyz)
    inv_scale, inv_rot, inv_trans, amb3r_to_vggt = umeyama_similarity(amb3r_xyz, vggt_xyz)
    residuals = np.linalg.norm(vggt_to_amb3r - amb3r_xyz, axis=1)
    scale_rows = build_scale_rows(frame_rows, vggt_xyz, amb3r_xyz, args.min_step_frac)
    used_scales = np.array(
        [float(r["amb3r_per_vggt_step_scale"]) for r in scale_rows if r["used_for_scale"]],
        dtype=float,
    )
    used_cum = np.array(
        [float(r["amb3r_per_vggt_cumulative_scale"]) for r in scale_rows if r["used_for_scale"]],
        dtype=float,
    )
    step_median = float(np.median(used_scales)) if len(used_scales) else float("nan")
    step_mad = robust_mad(used_scales)
    if len(used_scales) and np.isfinite(step_mad) and step_mad > 0:
        inlier_mask = np.abs(used_scales - step_median) <= args.step_scale_mad_filter * step_mad
    else:
        inlier_mask = np.ones(len(used_scales), dtype=bool)
    inlier_scales = used_scales[inlier_mask]
    if len(used_scales) >= 2:
        slope = float(np.polyfit(np.array([float(r["mid_video_time_s"]) for r in scale_rows if r["used_for_scale"]]), used_scales, 1)[0])
    else:
        slope = float("nan")

    summary = {
        "vggt_dir": str(args.vggt_dir),
        "amb3r_npz": str(args.amb3r_npz),
        "matched_cameras": n,
        "video_file": frame_rows[0].get("video_file", ""),
        "segment_id": frame_rows[0].get("segment_id", ""),
        "start_video_time_s": float(times[0]),
        "end_video_time_s": float(times[-1]),
        "duration_s": float(times[-1] - times[0]),
        "vggt_path_length_units": path_length(vggt_xyz),
        "amb3r_path_length_units": path_length(amb3r_xyz),
        "vggt_to_amb3r_similarity_scale": scale,
        "amb3r_to_vggt_similarity_scale": inv_scale,
        "alignment_rmse_amb3r_units": float(np.sqrt(np.mean(residuals * residuals))),
        "alignment_median_residual_amb3r_units": float(np.median(residuals)),
        "alignment_p90_residual_amb3r_units": float(np.quantile(residuals, 0.9)),
        "alignment_max_residual_amb3r_units": float(np.max(residuals)),
        "alignment_rmse_over_amb3r_path_length": float(np.sqrt(np.mean(residuals * residuals)) / max(path_length(amb3r_xyz), 1e-12)),
        "step_scale_votes": int(len(used_scales)),
        "step_scale_median": step_median,
        "step_scale_mad": step_mad,
        "step_scale_cv": float(np.std(used_scales) / np.mean(used_scales)) if len(used_scales) and np.mean(used_scales) else float("nan"),
        "step_scale_p10": float(np.quantile(used_scales, 0.1)) if len(used_scales) else float("nan"),
        "step_scale_p90": float(np.quantile(used_scales, 0.9)) if len(used_scales) else float("nan"),
        "step_scale_inlier_filter": f"{args.step_scale_mad_filter:g} * MAD around all-step median",
        "step_scale_inlier_votes": int(len(inlier_scales)),
        "step_scale_inlier_median": float(np.median(inlier_scales)) if len(inlier_scales) else float("nan"),
        "step_scale_inlier_mad": robust_mad(inlier_scales),
        "step_scale_inlier_cv": float(np.std(inlier_scales) / np.mean(inlier_scales)) if len(inlier_scales) and np.mean(inlier_scales) else float("nan"),
        "step_scale_inlier_min": float(np.min(inlier_scales)) if len(inlier_scales) else float("nan"),
        "step_scale_inlier_max": float(np.max(inlier_scales)) if len(inlier_scales) else float("nan"),
        "cumulative_scale_start": float(used_cum[0]) if len(used_cum) else float("nan"),
        "cumulative_scale_end": float(used_cum[-1]) if len(used_cum) else float("nan"),
        "step_scale_slope_per_s": slope,
        "rotation_vggt_to_amb3r": rot.tolist(),
        "translation_vggt_to_amb3r": trans.tolist(),
    }

    (args.out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    write_rows(args.out_dir / "scale_vs_time.csv", scale_rows)
    write_rows(
        args.out_dir / "alignment_residuals.csv",
        [
            {
                "frame_index": int(row["frame_index"]),
                "file": row["file"],
                "video_time_s": float(row["video_time_s"]),
                "residual_amb3r_units": float(residual),
                "vggt_aligned_x": float(aligned[0]),
                "vggt_aligned_y": float(aligned[1]),
                "vggt_aligned_z": float(aligned[2]),
                "amb3r_x": float(amb[0]),
                "amb3r_y": float(amb[1]),
                "amb3r_z": float(amb[2]),
            }
            for row, residual, aligned, amb in zip(frame_rows, residuals, vggt_to_amb3r, amb3r_xyz)
        ],
    )

    write_contact_sheet(args.vggt_dir, frame_rows, args.out_dir / "attack36_contact_sheet.jpg")
    vggt_pts, vggt_cols = load_vggt_points(args.vggt_dir, max_points=args.max_scene_points, seed=1)
    amb_pts, amb_cols = load_amb3r_points(
        args.amb3r_npz,
        max_points=args.max_scene_points,
        seed=2,
        conf_quantile=args.amb3r_conf_quantile,
    )
    plot_scene(
        vggt_pts,
        vggt_cols,
        vggt_xyz,
        args.out_dir / "vggt_scene_path.png",
        title="VGGT scene reconstruction + attack segment 3 path",
        path_color="#ffb000",
        elev=args.scene_elev,
        azim=args.scene_azim,
    )
    plot_scene(
        amb_pts,
        amb_cols,
        amb3r_xyz,
        args.out_dir / "amb3r_scene_path.png",
        title="AMB3R scene reconstruction + attack segment 3 path",
        path_color="#ff5a3d",
        elev=args.scene_elev,
        azim=args.scene_azim,
    )
    plot_comparison(
        vggt_xyz,
        amb3r_xyz,
        vggt_to_amb3r,
        residuals,
        times,
        args.out_dir / "aligned_paths_and_residuals.png",
    )
    plot_scale_vs_time(scale_rows, args.out_dir / "scale_vs_time.png")
    print(json.dumps(summary, indent=2))
    print(f"[compare] wrote {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vggt-dir", type=Path, required=True)
    parser.add_argument("--amb3r-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-scene-points", type=int, default=220_000)
    parser.add_argument("--amb3r-conf-quantile", type=float, default=0.72)
    parser.add_argument("--min-step-frac", type=float, default=0.12)
    parser.add_argument("--step-scale-mad-filter", type=float, default=3.0)
    parser.add_argument("--scene-elev", type=float, default=22.0)
    parser.add_argument("--scene-azim", type=float, default=-62.0)
    return parser.parse_args()


def main() -> int:
    compare(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
