#!/usr/bin/env python3
"""Create an interactive Three.js viewer for a VGGT reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh


VIEWER_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path, help="VGGT reconstruction dir with point_cloud.npz, vggt_scene.glb, relative_path.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="VGGT Attack Segment 3")
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--calibration-npz", type=Path, default=None)
    parser.add_argument("--calibration-summary", type=Path, default=None)
    parser.add_argument("--default-scale-m-per-unit", type=float, default=117.6)
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--state-url", default="")
    return parser.parse_args()


def transform_points(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return trimesh.transformations.transform_points(np.asarray(vertices), transform)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


class VggtGlbCameras:
    def __init__(self, glb_path: Path):
        self.scene = trimesh.load(glb_path)
        self.transforms: dict[str, np.ndarray] = {}
        for node in self.scene.graph.nodes:
            try:
                transform, geom_name = self.scene.graph[node]
            except Exception:
                continue
            if geom_name is not None:
                self.transforms[geom_name] = np.asarray(transform)

    def camera_basis(self, frame_number: int, flip_y: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        name = f"geometry_{frame_number}"
        geom = self.scene.geometry[name]
        transform = self.transforms.get(name, np.eye(4))
        vertices = transform_points(geom.vertices, transform)
        counts = np.bincount(geom.faces.reshape(-1), minlength=len(vertices))
        center = vertices[int(np.argmax(counts))]
        corners = vertices[[0, 2, 3, 4]]
        forward = unit(corners.mean(axis=0) - center)
        right = unit(((corners[0] + corners[3]) * 0.5) - ((corners[1] + corners[2]) * 0.5))
        down = unit(((corners[0] + corners[1]) * 0.5) - ((corners[2] + corners[3]) * 0.5))
        right = unit(right - np.dot(right, forward) * forward)
        down = unit(down - np.dot(down, forward) * forward - np.dot(down, right) * right)
        if flip_y:
            down = -down
        return center, right, down, forward


def relative_asset_url(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_dir).replace(os.sep, "/")


def copy_viewer_asset(out_dir: Path, target: Path) -> str:
    asset_dir = out_dir / "camera_view_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    dest = asset_dir / target.name
    if target.resolve() != dest.resolve():
        shutil.copy2(target, dest)
    return relative_asset_url(out_dir, dest)


def load_path(video_dir: Path, cameras: VggtGlbCameras, out_dir: Path) -> list[dict[str, object]]:
    with (video_dir / "relative_path.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    path: list[dict[str, object]] = []
    centers: list[np.ndarray] = []
    for row in rows:
        frame = int(row["frame_index"])
        center, right, down, forward = cameras.camera_basis(frame)
        centers.append(center)
        frame_file = row.get("file") or ""
        frame_path = video_dir / "frames" / frame_file
        frame_stem = Path(frame_file).stem
        actual_view = video_dir / "camera_views" / f"{frame_stem}_actual.jpg"
        render_view = video_dir / "camera_views" / f"{frame_stem}_vggt_render.jpg"
        overlay_view = video_dir / "camera_views" / f"{frame_stem}_overlay.jpg"
        image_assets: dict[str, str] = {}
        if frame_path.exists():
            frame_url = copy_viewer_asset(out_dir, frame_path)
            image_assets["frame_image"] = frame_url
            image_assets["actual_image"] = frame_url
        if actual_view.exists():
            image_assets["actual_image"] = copy_viewer_asset(out_dir, actual_view)
        if render_view.exists():
            image_assets["render_image"] = copy_viewer_asset(out_dir, render_view)
        if overlay_view.exists():
            image_assets["overlay_image"] = copy_viewer_asset(out_dir, overlay_view)
        path.append(
            {
                "frame": frame,
                "file": row.get("file"),
                "video_file": row.get("video_file"),
                "segment_id": row.get("segment_id"),
                "segment_index": int(row["segment_index"]) if row.get("segment_index") else None,
                "is_attack": str(row.get("is_attack", "")).lower() == "true",
                "video_time_s": float(row["video_time_s"]),
                "segment_time_s": float(row["segment_time_s"]),
                "sequence_time_s": float(row.get("sequence_time_s") or row["segment_time_s"]),
                "position": center.astype(float).tolist(),
                "right": right.astype(float).tolist(),
                "down": down.astype(float).tolist(),
                "forward": forward.astype(float).tolist(),
                **image_assets,
            }
        )
    end = centers[-1]
    for row, center in zip(path, centers):
        row["distance_to_end_units"] = float(np.linalg.norm(center - end))
    return path


def load_calibration(npz_path: Path | None, summary_path: Path | None) -> dict[str, object] | None:
    if npz_path is None or not npz_path.exists():
        return None
    data = np.load(npz_path)
    calibration: dict[str, object] = {}
    if "axis_endpoints" in data.files:
        calibration["axis_endpoints"] = data["axis_endpoints"].astype(float).tolist()
    if "corners" in data.files:
        calibration["corners"] = data["corners"].astype(float).tolist()
    if "dims" in data.files:
        calibration["dims_vggt_units"] = data["dims"].astype(float).tolist()
    if summary_path is not None and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        calibration.update(
            {
                "label": "manual object calibration",
                "known_length_m": summary.get("known_length_m"),
                "length_vggt_units": summary.get("length_vggt_units"),
                "scale_m_per_vggt_unit": summary.get("scale_m_per_vggt_unit"),
                "box_dims_m_if_length_known": summary.get("box_dims_m_if_length_known"),
            }
        )
    return calibration or None


def quat_from_unit_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = unit(a)
    b = unit(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.999999:
        axis = unit(np.cross(np.array([1.0, 0.0, 0.0]), a))
        if float(np.linalg.norm(axis)) < 1e-8:
            axis = unit(np.cross(np.array([0.0, 0.0, 1.0]), a))
        return np.array([axis[0], axis[1], axis[2], 0.0])
    s = float(np.sqrt((1.0 + dot) * 2.0))
    cross = np.cross(a, b)
    return np.array([cross[0] / s, cross[1] / s, cross[2] / s, 0.5 * s])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def rotate_vec(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    qv = np.array([x, y, z])
    return vector + 2.0 * np.cross(qv, np.cross(qv, vector) + w * vector)


def scene_alignment_quaternion(normal: np.ndarray, u: np.ndarray) -> list[float]:
    y_up = np.array([0.0, 1.0, 0.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    q_align = quat_from_unit_vectors(normal, y_up)
    u_rot = rotate_vec(q_align, u)
    u_xz = np.array([u_rot[0], 0.0, u_rot[2]])
    if float(np.linalg.norm(u_xz)) < 1e-8:
        u_xz = x_axis.copy()
    else:
        u_xz = unit(u_xz)
    q_twist = quat_from_unit_vectors(u_xz, x_axis)
    q = quat_multiply(q_twist, q_align)
    return [float(v) for v in q]


def estimate_ground_grid(
    points: np.ndarray,
    colors: np.ndarray,
    calibration: dict[str, object] | None,
    path: list[dict[str, object]],
    default_scale_m_per_unit: float | None = None,
) -> dict[str, object] | None:
    cols = colors.astype(float)
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    groundish = (
        (r > 85)
        & (g > 70)
        & (b < 190)
        & (((r + g) / (b + 1)) > 1.7)
        & ~((g > r * 1.12) & (g > b * 1.35))
    )
    candidates = points[groundish].astype(np.float64)
    if len(candidates) < 2000:
        return None

    rng = np.random.default_rng(42)
    if len(candidates) > 80_000:
        candidates = candidates[rng.choice(len(candidates), 80_000, replace=False)]
    lo = np.percentile(candidates, 1, axis=0)
    hi = np.percentile(candidates, 99, axis=0)
    candidates = candidates[((candidates >= lo) & (candidates <= hi)).all(axis=1)]
    if len(candidates) < 2000:
        return None

    y_axis = np.array([0.0, 1.0, 0.0])
    threshold = 0.012
    subset = candidates[rng.choice(len(candidates), min(14_000, len(candidates)), replace=False)]
    best_count = -1
    best_model: tuple[np.ndarray, float] | None = None
    sample_idx = rng.choice(len(candidates), size=(4000, 3), replace=True)
    for tri in candidates[sample_idx]:
        a, b_point, c = tri
        normal = np.cross(b_point - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        if np.dot(normal, y_axis) < 0:
            normal = -normal
        if np.dot(normal, y_axis) < 0.28:
            continue
        d = -float(np.dot(normal, a))
        count = int((np.abs(subset @ normal + d) < threshold).sum())
        if count > best_count:
            best_count = count
            best_model = (normal, d)
    if best_model is None:
        return None

    normal, d = best_model
    for _ in range(3):
        distances = np.abs(candidates @ normal + d)
        inliers = candidates[distances < threshold]
        if len(inliers) < 1000:
            return None
        centroid = inliers.mean(axis=0)
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal = vh[-1]
        if np.dot(normal, y_axis) < 0:
            normal = -normal
        d = -float(np.dot(normal, centroid))

    distances = np.abs(candidates @ normal + d)
    inliers = candidates[distances < threshold]
    centroid = np.median(inliers, axis=0)
    origin = centroid - (float(np.dot(normal, centroid)) + d) * normal
    _, _, vh = np.linalg.svd(inliers - inliers.mean(axis=0), full_matrices=False)
    u = vh[0] - float(np.dot(vh[0], normal)) * normal
    u = unit(u)
    v = unit(np.cross(normal, u))

    scale = float(default_scale_m_per_unit or 0.0)
    if scale <= 0 and calibration:
        scale = float(calibration.get("scale_m_per_vggt_unit", 0.0))
    if scale > 0:
        minor_step_m = 2.0
        major_step_m = 8.0
        minor_step_units = minor_step_m / scale
        major_step_units = major_step_m / scale
    else:
        minor_step_units = 0.1
        major_step_units = minor_step_units * 4.0
        minor_step_m = None
        major_step_m = None

    if path:
        path_points = np.asarray([row["position"] for row in path], dtype=np.float64)
        start_end_midpoint = (path_points[0] + path_points[-1]) * 0.5
        origin = start_end_midpoint - (float(np.dot(normal, start_end_midpoint)) + d) * normal
        projected_path = np.column_stack([(path_points - origin) @ u, (path_points - origin) @ v])
        span_units = projected_path.max(axis=0) - projected_path.min(axis=0)
        size_units = float(np.ceil((max(span_units) + major_step_units * 2.0) / major_step_units) * major_step_units)
        size_units = max(size_units, float(major_step_units * 5.0))
    else:
        projected = np.column_stack([(points - origin) @ u, (points - origin) @ v])
        span_units = np.percentile(projected, 99, axis=0) - np.percentile(projected, 1, axis=0)
        size_units = float(np.ceil((max(span_units) * 1.25) / major_step_units) * major_step_units)
    size_m = float(size_units * scale) if scale > 0 else None

    return {
        "normal": normal.astype(float).tolist(),
        "d": float(d),
        "origin": origin.astype(float).tolist(),
        "u": u.astype(float).tolist(),
        "v": v.astype(float).tolist(),
        "inlier_count": int(len(inliers)),
        "candidate_count": int(len(candidates)),
        "threshold_units": threshold,
        "size_units": float(size_units),
        "fixed_size_units": float(size_units),
        "minor_step_units": float(minor_step_units),
        "major_step_units": float(major_step_units),
        "size_m": size_m,
        "minor_step_m": minor_step_m,
        "major_step_m": major_step_m,
        "path_span_units": span_units.astype(float).tolist(),
    }


def load_sample_fps(video_dir: Path) -> float | None:
    meta_path = video_dir / "metadata.json"
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text())
    fps = data.get("sample_fps")
    if fps is None:
        preprocess = (data.get("model_config") or {}).get("preprocess") or {}
        fps = preprocess.get("sample_fps_effective") or preprocess.get("sample_fps_requested")
    if fps is None:
        return None
    value = float(fps)
    return value if value > 0 else None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cloud = np.load(args.video_dir / "point_cloud.npz")
    points = cloud["pts"].astype(np.float32)
    colors = cloud["cols"].astype(np.uint8)
    if len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    positions_path = args.out_dir / "points_positions.bin"
    colors_path = args.out_dir / "points_colors.bin"
    points.astype("<f4").tofile(positions_path)
    colors.tofile(colors_path)

    cameras = VggtGlbCameras(args.video_dir / "vggt_scene.glb")
    path = load_path(args.video_dir, cameras, args.out_dir)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    calibration = load_calibration(args.calibration_npz, args.calibration_summary)
    ground_grid = estimate_ground_grid(points, colors, calibration, path, args.default_scale_m_per_unit)
    sample_fps = load_sample_fps(args.video_dir)
    align_quat = None
    if ground_grid:
        align_quat = scene_alignment_quaternion(
            np.asarray(ground_grid["normal"], dtype=np.float64),
            np.asarray(ground_grid["u"], dtype=np.float64),
        )

    meta = {
        "title": args.title,
        "source_label": args.video_dir.name,
        "video_dir": str(args.video_dir),
        "scene_id": args.scene_id,
        "scene_state_url": args.state_url or "",
        "default_scale_m_per_unit": args.default_scale_m_per_unit,
        "sample_fps": sample_fps,
        "point_count": int(len(points)),
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "assets": {
            "positions": positions_path.name,
            "colors": colors_path.name,
        },
        "path": path,
        "calibration": calibration,
        "ground_grid": ground_grid,
        "scene_alignment_quaternion": align_quat,
    }
    meta["viewer_version"] = VIEWER_VERSION
    (args.out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    print(args.out_dir / "scene_meta.json")


if __name__ == "__main__":
    main()
