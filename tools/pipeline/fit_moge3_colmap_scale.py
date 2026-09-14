#!/usr/bin/env python3
"""Fit and apply a global metric scale to a COLMAP text model using MoGe-3 depths."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from metric_scale import occupied_grid_cells, robust_log_scale


@dataclass
class ColmapImage:
    header: str
    observations: str
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="COLMAP TXT model directory")
    parser.add_argument("--depths", type=Path, required=True, help="MoGe NPZ directory")
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--reference-scale",
        type=float,
        help="Optional diagnostic reference; compared only when both coordinate-system IDs are identical",
    )
    parser.add_argument("--model-coordinate-system-id")
    parser.add_argument("--reference-coordinate-system-id")
    parser.add_argument("--min-votes", type=int, default=20)
    parser.add_argument("--min-spatial-cells", type=int, default=4)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-columns", type=int, default=8)
    parser.add_argument("--max-log-deviation", type=float, default=0.7)
    parser.add_argument("--votes-dir", type=Path)
    return parser.parse_args()


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def data_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text().splitlines() if line and not line.startswith("#")]


def read_points(path: Path) -> dict[int, np.ndarray]:
    points = {}
    for line in data_lines(path):
        fields = line.split()
        points[int(fields[0])] = np.array([float(fields[1]), float(fields[2]), float(fields[3])])
    return points


def read_cameras(path: Path) -> dict[int, ColmapCamera]:
    cameras = {}
    for line in data_lines(path):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Invalid COLMAP camera line: {line}")
        camera = ColmapCamera(
            camera_id=int(fields[0]),
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
        )
        cameras[camera.camera_id] = camera
    return cameras


def read_images(path: Path) -> list[ColmapImage]:
    raw_lines = [line.rstrip("\n") for line in path.read_text().splitlines() if not line.startswith("#")]
    while raw_lines and not raw_lines[0]:
        raw_lines.pop(0)
    images = []
    offset = 0
    while offset < len(raw_lines):
        if not raw_lines[offset]:
            offset += 1
            continue
        header = raw_lines[offset]
        observations = raw_lines[offset + 1] if offset + 1 < len(raw_lines) else ""
        offset += 2
        fields = header.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError(f"Invalid COLMAP image header: {header}")
        images.append(
            ColmapImage(
                header=header,
                observations=observations,
                image_id=int(fields[0]),
                qvec=np.array([float(value) for value in fields[1:5]]),
                tvec=np.array([float(value) for value in fields[5:8]]),
                camera_id=int(fields[8]),
                name=fields[9],
            )
        )
    return images


def observation_triplets(line: str) -> Iterable[tuple[float, float, int]]:
    fields = line.split()
    for offset in range(0, len(fields), 3):
        yield float(fields[offset]), float(fields[offset + 1]), int(fields[offset + 2])


def robust_scale(values: np.ndarray, max_log_deviation: float) -> tuple[float, np.ndarray]:
    """Backward-compatible wrapper used by existing tests and callers."""

    values = np.asarray(values, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0)]
    fit = robust_log_scale(positive, max_log_deviation)
    return fit.scale, positive[fit.keep_mask]


def frame_vote_data(
    image: ColmapImage,
    points: dict[int, np.ndarray],
    depth_path: Path,
    expected_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(depth_path) as payload:
        depth = np.asarray(payload["depth"], dtype=np.float64)
        mask = np.asarray(payload["mask"], dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError(f"{depth_path}: depth and mask must be matching 2D arrays")
    height, width = depth.shape
    if expected_size is not None and (width, height) != expected_size:
        raise ValueError(
            f"{depth_path}: depth is {width}x{height}, but COLMAP camera {image.camera_id} "
            f"is {expected_size[0]}x{expected_size[1]}; pixel observations cannot be sampled safely"
        )
    rotation = qvec_to_rotation(image.qvec)
    votes = []
    pixel_xs = []
    pixel_ys = []
    for x, y, point_id in observation_triplets(image.observations):
        if point_id < 0 or point_id not in points:
            continue
        pixel_x = int(round(x))
        pixel_y = int(round(y))
        if not (0 <= pixel_x < width and 0 <= pixel_y < height and mask[pixel_y, pixel_x]):
            continue
        sparse_depth = float((rotation @ points[point_id] + image.tvec)[2])
        metric_depth = float(depth[pixel_y, pixel_x])
        if sparse_depth > 0 and metric_depth > 0 and math.isfinite(metric_depth):
            votes.append(metric_depth / sparse_depth)
            pixel_xs.append(pixel_x)
            pixel_ys.append(pixel_y)
    return (
        np.asarray(votes, dtype=np.float64),
        np.asarray(pixel_xs, dtype=np.int32),
        np.asarray(pixel_ys, dtype=np.int32),
    )


def frame_votes(
    image: ColmapImage,
    points: dict[int, np.ndarray],
    depth_path: Path,
    expected_size: tuple[int, int] | None = None,
) -> np.ndarray:
    votes, _, _ = frame_vote_data(image, points, depth_path, expected_size)
    return votes


def scale_images_file(images: list[ColmapImage], destination: Path, scale: float) -> None:
    with destination.open("w") as handle:
        handle.write("# Metric-scaled COLMAP images generated by fit_moge3_colmap_scale.py\n")
        for image in images:
            fields = image.header.split(maxsplit=9)
            scaled_t = image.tvec * scale
            fields[5:8] = [f"{value:.17g}" for value in scaled_t]
            handle.write(" ".join(fields) + "\n")
            handle.write(image.observations + "\n")


def scale_points_file(source: Path, destination: Path, scale: float) -> None:
    with destination.open("w") as handle:
        handle.write("# Metric-scaled COLMAP points generated by fit_moge3_colmap_scale.py\n")
        for line in data_lines(source):
            fields = line.split()
            fields[1:4] = [f"{float(value) * scale:.17g}" for value in fields[1:4]]
            handle.write(" ".join(fields) + "\n")


def reference_scale_comparison(
    fitted_scale: float,
    reference_scale: float | None,
    model_coordinate_system_id: str | None,
    reference_coordinate_system_id: str | None,
) -> dict[str, object]:
    """Compare coefficients only when coordinate-system identity is explicit."""

    same_coordinate_system = bool(
        reference_scale is not None
        and model_coordinate_system_id
        and reference_coordinate_system_id
        and model_coordinate_system_id == reference_coordinate_system_id
    )
    return {
        "status": "evaluated_same_coordinate_system" if same_coordinate_system else "not_evaluated_unverified_coordinate_system",
        "reference_scale_m_per_unit": reference_scale,
        "model_coordinate_system_id": model_coordinate_system_id,
        "reference_coordinate_system_id": reference_coordinate_system_id,
        "relative_error": (
            abs(fitted_scale - reference_scale) / reference_scale if same_coordinate_system else None
        ),
        "warning": (
            None
            if same_coordinate_system
            else "Raw scale coefficients from different reconstruction coordinate systems are incomparable."
        ),
    }


def main() -> int:
    args = parse_args()
    images = read_images(args.model / "images.txt")
    points = read_points(args.model / "points3D.txt")
    cameras = read_cameras(args.model / "cameras.txt")
    votes_dir = args.votes_dir
    if votes_dir:
        votes_dir.mkdir(parents=True, exist_ok=True)
    frame_records = []
    frame_scales = []
    frame_scale_record_indices = []
    for image in images:
        depth_path = args.depths / f"{Path(image.name).stem}.npz"
        if not depth_path.exists():
            frame_records.append({"image": image.name, "status": "missing_depth"})
            continue
        if image.camera_id not in cameras:
            raise ValueError(f"Image {image.name} references missing camera {image.camera_id}")
        camera = cameras[image.camera_id]
        votes, pixel_x, pixel_y = frame_vote_data(
            image,
            points,
            depth_path,
            expected_size=(camera.width, camera.height),
        )
        spatial_mask = np.zeros((camera.height, camera.width), dtype=bool)
        spatial_mask[pixel_y, pixel_x] = True
        spatial_cells = occupied_grid_cells(spatial_mask, args.grid_rows, args.grid_columns)
        if len(votes) < args.min_votes:
            frame_records.append(
                {
                    "image": image.name,
                    "raw_votes": int(len(votes)),
                    "occupied_grid_cells": spatial_cells,
                    "status": "insufficient_votes",
                }
            )
            continue
        if spatial_cells < args.min_spatial_cells:
            frame_records.append(
                {
                    "image": image.name,
                    "raw_votes": int(len(votes)),
                    "occupied_grid_cells": spatial_cells,
                    "status": "insufficient_spatial_coverage",
                }
            )
            continue
        fit = robust_log_scale(votes, args.max_log_deviation)
        kept_spatial_mask = np.zeros((camera.height, camera.width), dtype=bool)
        kept_spatial_mask[pixel_y[fit.keep_mask], pixel_x[fit.keep_mask]] = True
        kept_spatial_cells = occupied_grid_cells(kept_spatial_mask, args.grid_rows, args.grid_columns)
        if votes_dir:
            np.savez_compressed(
                votes_dir / f"{Path(image.name).stem}_votes.npz",
                scale_m_per_colmap_unit=votes.astype(np.float32),
                kept=fit.keep_mask,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
            )
        record = {
            "image": image.name,
            "raw_votes": fit.input_count,
            "kept_votes": fit.kept_count,
            "rejected_votes": fit.input_count - fit.kept_count,
            "occupied_grid_cells": spatial_cells,
            "kept_occupied_grid_cells": kept_spatial_cells,
            "total_grid_cells": args.grid_rows * args.grid_columns,
            "scale_m_per_colmap_unit": fit.scale,
            "vote_log_mad": fit.log_mad,
            "status": "accepted" if kept_spatial_cells >= args.min_spatial_cells else "insufficient_kept_spatial_coverage",
        }
        if kept_spatial_cells >= args.min_spatial_cells:
            frame_scales.append(fit.scale)
            frame_scale_record_indices.append(len(frame_records))
        frame_records.append(record)
    if not frame_scales:
        raise SystemExit("No image produced enough valid MoGe/COLMAP scale votes")
    global_fit = robust_log_scale(np.asarray(frame_scales), args.max_log_deviation)
    global_scale = global_fit.scale
    for local_index, record_index in enumerate(frame_scale_record_indices):
        frame_records[record_index]["kept_for_global_fit"] = bool(global_fit.keep_mask[local_index])
    args.output_model.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model / "cameras.txt", args.output_model / "cameras.txt")
    scale_images_file(images, args.output_model / "images.txt", global_scale)
    scale_points_file(args.model / "points3D.txt", args.output_model / "points3D.txt", global_scale)
    reference_comparison = reference_scale_comparison(
        global_scale,
        args.reference_scale,
        args.model_coordinate_system_id,
        args.reference_coordinate_system_id,
    )
    summary = {
        "schema_version": 2,
        "global_scale_m_per_colmap_unit": global_scale,
        "depth_definition": "camera_z",
        "frame_scale_log_mad": global_fit.log_mad,
        "frames_with_scale_before_global_rejection": global_fit.input_count,
        "frames_with_scale": global_fit.kept_count,
        "registered_frames": len(images),
        "missing_depth_frames": sum(record["status"] == "missing_depth" for record in frame_records),
        "aggregation": "median in log space per frame, then equal-weight median across frames",
        "reference_scale_comparison": reference_comparison,
        "frames": frame_records,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
