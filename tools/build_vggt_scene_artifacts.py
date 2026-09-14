#!/usr/bin/env python3
"""Build registration artifacts from a VGGT-Omega GLB scene.

Inputs:
  scene_dir/vggt_scene.glb
  scene_dir/frames.csv
  scene_dir/frames/*.jpg

Outputs:
  scene_dir/point_cloud.npz
  scene_dir/relative_path.npy
  scene_dir/relative_path.csv
  scene_dir/viewer/points_positions.bin
  scene_dir/viewer/points_colors.bin
  scene_dir/viewer/scene_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


VIEWER_VERSION = 2


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def transform_points(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return trimesh.transformations.transform_points(np.asarray(vertices), transform)


class VggtGlbScene:
    def __init__(self, glb_path: Path):
        self.glb_path = glb_path
        self.scene = trimesh.load(glb_path)
        self.transforms: dict[str, np.ndarray] = {}
        for node in self.scene.graph.nodes:
            try:
                transform, geom_name = self.scene.graph[node]
            except Exception:
                continue
            if geom_name is not None:
                self.transforms[geom_name] = np.asarray(transform)

    def camera_names(self) -> list[str]:
        def key(name: str) -> int:
            try:
                return int(name.split("_", 1)[1])
            except Exception:
                return 10**9

        return sorted(
            [name for name in self.scene.geometry if name.startswith("geometry_") and name != "geometry_0"],
            key=key,
        )

    def point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        if "geometry_0" not in self.scene.geometry:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        pc = self.scene.geometry["geometry_0"]
        transform = self.transforms.get("geometry_0", np.eye(4))
        points = transform_points(pc.vertices, transform)
        ok = np.isfinite(points).all(axis=1)
        points = points[ok]
        if hasattr(pc.visual, "vertex_colors") and len(pc.visual.vertex_colors):
            colors = np.asarray(pc.visual.vertex_colors)[:, :3][ok]
        else:
            colors = np.zeros((len(points), 3), dtype=np.uint8)
        return points.astype(np.float32), colors.astype(np.uint8)

    def camera_basis(self, frame_number: int, flip_y: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        name = f"geometry_{frame_number}"
        if name not in self.scene.geometry:
            raise KeyError(f"{name} is missing from {self.glb_path}")
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


def read_metadata(scene_dir: Path) -> dict[str, Any]:
    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def write_metadata(scene_dir: Path, updates: dict[str, Any]) -> None:
    meta = read_metadata(scene_dir)
    meta.update(updates)
    (scene_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def read_frames(scene_dir: Path) -> list[dict[str, str]]:
    path = scene_dir / "frames.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    if "frame_index" not in rows[0]:
        raise SystemExit(f"{path} must contain a frame_index column")
    return rows


def sample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices], colors[indices]


def write_point_cloud(
    scene_dir: Path,
    glb_scene: VggtGlbScene,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    points, colors = glb_scene.point_cloud()
    raw_count = int(len(points))
    points, colors = sample_points(points, colors, max_points=max_points, seed=seed)
    np.savez_compressed(scene_dir / "point_cloud.npz", pts=points, cols=colors)
    return points, colors, raw_count


def frame_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def build_path_rows(
    rows: list[dict[str, str]],
    glb_scene: VggtGlbScene,
    flip_y: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    path_rows: list[dict[str, Any]] = []
    centers: list[np.ndarray] = []
    for row in rows:
        frame_index = int(row["frame_index"])
        center, right, down, forward = glb_scene.camera_basis(frame_index, flip_y=flip_y)
        centers.append(center)
        path_rows.append(
            {
                **row,
                "frame_index": frame_index,
                "video_time_s": frame_float(row, "video_time_s"),
                "segment_time_s": frame_float(row, "segment_time_s"),
                "sequence_time_s": frame_float(row, "sequence_time_s"),
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "position": center.astype(float).tolist(),
                "right": right.astype(float).tolist(),
                "down": down.astype(float).tolist(),
                "forward": forward.astype(float).tolist(),
            }
        )
    centers_arr = np.asarray(centers, dtype=np.float32)
    if len(path_rows):
        end = centers_arr[-1].astype(float)
        for path_row, center in zip(path_rows, centers_arr):
            path_row["distance_to_end_units"] = float(np.linalg.norm(center.astype(float) - end))
    return path_rows, centers_arr


def write_relative_path(scene_dir: Path, rows: list[dict[str, str]], path_rows: list[dict[str, Any]], centers: np.ndarray) -> None:
    np.save(scene_dir / "relative_path.npy", centers)
    frame_fields = list(rows[0].keys())
    extra_fields = [field for field in ("x", "y", "z") if field not in frame_fields]
    with (scene_dir / "relative_path.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_fields + extra_fields)
        writer.writeheader()
        for source_row, path_row in zip(rows, path_rows):
            out = dict(source_row)
            out.update({"x": path_row["x"], "y": path_row["y"], "z": path_row["z"]})
            writer.writerow(out)


def relative_asset_url(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_dir).replace(os.sep, "/")


def copy_viewer_asset(out_dir: Path, target: Path) -> str:
    asset_dir = out_dir / "camera_view_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    dest = asset_dir / target.name
    if target.resolve() != dest.resolve():
        shutil.copy2(target, dest)
    return relative_asset_url(out_dir, dest)


def path_for_viewer(scene_dir: Path, out_dir: Path, path_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    viewer_path: list[dict[str, Any]] = []
    for row in path_rows:
        frame_file = str(row.get("file") or "")
        frame_path = scene_dir / "frames" / frame_file
        image_assets: dict[str, str] = {}
        if frame_file and frame_path.exists():
            frame_url = copy_viewer_asset(out_dir, frame_path)
            image_assets["frame_image"] = frame_url
            image_assets["actual_image"] = frame_url
        viewer_path.append(
            {
                "frame": int(row["frame_index"]),
                "file": frame_file,
                "video_file": row.get("video_file"),
                "segment_id": row.get("segment_id"),
                "segment_index": int(row["segment_index"]) if row.get("segment_index") not in (None, "") else None,
                "is_attack": str(row.get("is_attack", "")).lower() == "true",
                "video_time_s": float(row["video_time_s"]) if row.get("video_time_s") is not None else None,
                "segment_time_s": float(row["segment_time_s"]) if row.get("segment_time_s") is not None else None,
                "sequence_time_s": float(row["sequence_time_s"]) if row.get("sequence_time_s") is not None else None,
                "position": row["position"],
                "right": row["right"],
                "down": row["down"],
                "forward": row["forward"],
                "distance_to_end_units": row["distance_to_end_units"],
                **image_assets,
            }
        )
    return viewer_path


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
    path: list[dict[str, Any]],
    scale_m_per_unit: float,
) -> dict[str, Any] | None:
    cols = colors.astype(float)
    if len(cols) == 0:
        return None
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

    if scale_m_per_unit > 0:
        minor_step_m: float | None = 2.0
        major_step_m: float | None = 8.0
        minor_step_units = minor_step_m / scale_m_per_unit
        major_step_units = major_step_m / scale_m_per_unit
    else:
        minor_step_m = None
        major_step_m = None
        minor_step_units = 0.1
        major_step_units = 0.4

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
        "size_m": float(size_units * scale_m_per_unit) if scale_m_per_unit > 0 else None,
        "minor_step_m": minor_step_m,
        "major_step_m": major_step_m,
        "path_span_units": span_units.astype(float).tolist(),
    }


def default_scale_from_metadata(meta: dict[str, Any]) -> float:
    if meta.get("default_scale_m_per_unit") is not None:
        return float(meta["default_scale_m_per_unit"])
    scale = (((meta.get("model_config") or {}).get("scale") or {}).get("default_scale_m_per_unit"))
    return float(scale) if scale is not None else 0.0


def sample_fps_from_metadata(meta: dict[str, Any]) -> float | None:
    fps = meta.get("sample_fps")
    if fps is None:
        preprocess = ((meta.get("model_config") or {}).get("preprocess") or {})
        fps = preprocess.get("sample_fps_effective") or preprocess.get("sample_fps_requested")
    if fps is None:
        return None
    value = float(fps)
    return value if value > 0 else None


def write_viewer(
    scene_dir: Path,
    out_dir: Path,
    meta: dict[str, Any],
    points: np.ndarray,
    colors: np.ndarray,
    path_rows: list[dict[str, Any]],
    title: str,
    scene_id: str,
    state_url: str,
    scale_m_per_unit: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    positions_path = out_dir / "points_positions.bin"
    colors_path = out_dir / "points_colors.bin"
    points.astype("<f4").tofile(positions_path)
    colors.astype(np.uint8).tofile(colors_path)

    viewer_path = path_for_viewer(scene_dir, out_dir, path_rows)
    bbox_min = points.min(axis=0) if len(points) else np.zeros(3, dtype=np.float32)
    bbox_max = points.max(axis=0) if len(points) else np.zeros(3, dtype=np.float32)
    ground_grid = estimate_ground_grid(points, colors, viewer_path, scale_m_per_unit)
    align_quat = None
    if ground_grid:
        align_quat = scene_alignment_quaternion(
            np.asarray(ground_grid["normal"], dtype=np.float64),
            np.asarray(ground_grid["u"], dtype=np.float64),
        )

    scene_meta = {
        "title": title,
        "source_label": scene_dir.name,
        "video_dir": str(scene_dir),
        "scene_id": scene_id,
        "scene_state_url": state_url,
        "default_scale_m_per_unit": scale_m_per_unit,
        "sample_fps": sample_fps_from_metadata(meta),
        "point_count": int(len(points)),
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "assets": {
            "positions": positions_path.name,
            "colors": colors_path.name,
        },
        "path": viewer_path,
        "calibration": None,
        "ground_grid": ground_grid,
        "scene_alignment_quaternion": align_quat,
        "viewer_version": VIEWER_VERSION,
    }
    (out_dir / "scene_meta.json").write_text(json.dumps(scene_meta, indent=2))
    return scene_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("scene_dir", type=Path, help="Scene directory containing vggt_scene.glb and frames.csv")
    parser.add_argument("--viewer-dir", type=Path, default=None, help="Output viewer directory")
    parser.add_argument("--title", default="VGGT Scene")
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--state-url", default="")
    parser.add_argument("--default-scale-m-per-unit", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flip-y", action="store_true", default=True)
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    glb_path = scene_dir / "vggt_scene.glb"
    if not glb_path.exists():
        raise SystemExit(f"Missing {glb_path}")

    meta = read_metadata(scene_dir)
    scene_id = args.scene_id or str(meta.get("scene_id") or scene_dir.name)
    scale = args.default_scale_m_per_unit
    if scale is None:
        scale = default_scale_from_metadata(meta)
    out_dir = (args.viewer_dir or (scene_dir / "viewer")).resolve()

    glb_scene = VggtGlbScene(glb_path)
    frame_rows = read_frames(scene_dir)
    expected_cameras = len(glb_scene.camera_names())
    if expected_cameras < len(frame_rows):
        raise SystemExit(f"GLB has {expected_cameras} cameras but frames.csv has {len(frame_rows)} rows")

    path_rows, centers = build_path_rows(frame_rows, glb_scene, flip_y=args.flip_y)
    write_relative_path(scene_dir, frame_rows, path_rows, centers)
    points, colors, raw_point_count = write_point_cloud(scene_dir, glb_scene, args.max_points, args.seed)
    scene_meta = write_viewer(
        scene_dir=scene_dir,
        out_dir=out_dir,
        meta=meta,
        points=points,
        colors=colors,
        path_rows=path_rows,
        title=args.title,
        scene_id=scene_id,
        state_url=args.state_url,
        scale_m_per_unit=float(scale or 0.0),
    )
    write_metadata(
        scene_dir,
        {
            "job_status": "artifacts_built",
            "job_step": "viewer_scene_meta_saved",
            "artifact_paths": {
                "relative_path_csv": str(scene_dir / "relative_path.csv"),
                "relative_path_npy": str(scene_dir / "relative_path.npy"),
                "point_cloud_npz": str(scene_dir / "point_cloud.npz"),
                "viewer_scene_meta": str(out_dir / "scene_meta.json"),
            },
            "artifact_summary": {
                "frames": int(len(path_rows)),
                "raw_point_count": raw_point_count,
                "point_count": int(len(points)),
                "ground_grid_found": scene_meta["ground_grid"] is not None,
                "default_scale_m_per_unit": float(scale or 0.0),
            },
        },
    )

    print(json.dumps({
        "scene_dir": str(scene_dir),
        "frames": int(len(path_rows)),
        "raw_point_count": raw_point_count,
        "point_count": int(len(points)),
        "viewer_scene_meta": str(out_dir / "scene_meta.json"),
        "ground_grid_found": scene_meta["ground_grid"] is not None,
        "default_scale_m_per_unit": float(scale or 0.0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
