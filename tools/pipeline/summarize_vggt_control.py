#!/usr/bin/env python3
"""Validate matched-input VGGT-Omega camera/depth exports and summarize control metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from summarize_colmap_diagnostics import numeric_summary
from summarize_scal3r_diagnostics import ply_vertex_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed-forward", type=Path, required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--trajectory-vs-published", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(args: argparse.Namespace) -> dict[str, object]:
    root = args.feed_forward
    cameras = json.loads((root / "predicted_input_cameras.json").read_text())
    camera_npz = np.load(root / "predicted_input_cameras.npz")
    pointcloud = json.loads((root / "vggt_depth_pointcloud.json").read_text())
    depth_info = json.loads((root / "vggt_input_depths/vggt_input_depths.json").read_text())
    depth_npz = np.load(root / "vggt_input_depths/vggt_input_depths.npz")
    trajectory = json.loads(args.trajectory_vs_published.read_text())
    with args.frames_csv.open(newline="") as handle:
        frame_rows = list(csv.DictReader(handle))

    expected_names = [Path(row["profile_file"]).stem for row in frame_rows]
    frames = cameras["frames"]
    names = [frame["name"] for frame in frames]
    sources = [frame["source"] for frame in frames]
    json_c2w = np.asarray([frame["c2w"] for frame in frames], dtype=np.float64)
    json_w2c = np.asarray([frame["w2c"] for frame in frames], dtype=np.float64)
    json_cam_view = np.asarray([frame["cam_view"] for frame in frames], dtype=np.float64)
    json_intrinsics = np.asarray([frame["intrinsics"] for frame in frames], dtype=np.float64)
    json_k = np.asarray([frame["K"] for frame in frames], dtype=np.float64)

    identity_errors = np.max(np.abs(json_c2w @ json_w2c - np.eye(4)), axis=(1, 2))
    transpose_errors = np.max(np.abs(json_cam_view - np.swapaxes(json_w2c, 1, 2)), axis=(1, 2))
    rotations = json_c2w[:, :3, :3]
    rotation_errors = np.max(
        np.abs(np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3)), axis=(1, 2)
    )
    rotation_determinants = np.linalg.det(rotations)
    expected_k = np.zeros_like(json_k)
    expected_k[:, 0, 0] = json_intrinsics[:, 0]
    expected_k[:, 1, 1] = json_intrinsics[:, 1]
    expected_k[:, 0, 2] = json_intrinsics[:, 2]
    expected_k[:, 1, 2] = json_intrinsics[:, 3]
    expected_k[:, 2, 2] = 1

    image_shapes = []
    for source in sources:
        with Image.open(root / source) as image:
            image_shapes.append([image.height, image.width])

    camera_npz_matches_json = bool(
        np.array_equal(camera_npz["frame_names"], np.asarray(names))
        and np.array_equal(camera_npz["frame_sources"], np.asarray(sources))
        and np.allclose(camera_npz["c2w"], json_c2w, atol=1e-7)
        and np.allclose(camera_npz["w2c"], json_w2c, atol=1e-7)
        and np.allclose(camera_npz["cam_view"], json_cam_view, atol=1e-7)
        and np.allclose(camera_npz["intrinsics"], json_intrinsics, atol=1e-7)
        and np.allclose(camera_npz["K"], json_k, atol=1e-7)
    )
    point_frames = pointcloud["frames"]
    pointcloud_cameras_match = bool(
        [frame["name"] for frame in point_frames] == names
        and np.allclose([frame["c2w"] for frame in point_frames], json_c2w, atol=1e-7)
        and np.allclose([frame["w2c"] for frame in point_frames], json_w2c, atol=1e-7)
        and np.allclose([frame["cam_view"] for frame in point_frames], json_cam_view, atol=1e-7)
        and np.allclose([frame["intrinsics"] for frame in point_frames], json_intrinsics, atol=1e-7)
    )
    depth_frames = depth_info["frames"]
    depth_contract_matches = bool(
        depth_npz["depth"].shape == (len(frames), 512, 512)
        and depth_npz["depth_confidence"].shape == (len(frames), 512, 512)
        and [frame["name"] for frame in depth_frames] == names
        and np.allclose(depth_npz["cam_view"], json_cam_view, atol=1e-7)
        and np.allclose(depth_npz["intrinsics"], json_intrinsics, atol=1e-7)
        and np.allclose(depth_npz["pose_enc"], camera_npz["pose_enc"], atol=1e-7)
    )
    ply_points = ply_vertex_count(root / pointcloud["pointcloud"])
    declared_valid_points = int(sum(int(frame["valid_points"]) for frame in point_frames))
    pointcloud_contract_matches = bool(
        ply_points == int(pointcloud["total_points"]) == declared_valid_points
        and pointcloud_cameras_match
    )

    all_contracts_match = bool(
        names == expected_names
        and [int(frame["index"]) for frame in frames] == list(range(len(frames)))
        and camera_npz_matches_json
        and max(identity_errors) < 2e-5
        and max(transpose_errors) < 2e-5
        and max(rotation_errors) < 2e-5
        and np.allclose(rotation_determinants, 1, atol=2e-5)
        and np.allclose(json_k, expected_k, atol=2e-5)
        and all(shape == [512, 512] for shape in image_shapes)
        and np.allclose(json_intrinsics[:, 2:], 256, atol=1e-6)
        and depth_contract_matches
        and pointcloud_contract_matches
    )
    depth = np.asarray(depth_npz["depth"], dtype=np.float64)
    confidence = np.asarray(depth_npz["depth_confidence"], dtype=np.float64)
    return {
        "schema_version": 1,
        "control": "matched_input_vggt_omega_camera_and_depth_head",
        "frames": len(frames),
        "contract": {
            "validated": all_contracts_match,
            "profile_names_and_indices_match": names == expected_names,
            "preprocessed_image_shapes": sorted({tuple(shape) for shape in image_shapes}),
            "camera_json_npz_match": camera_npz_matches_json,
            "c2w_w2c_inverse_max_abs_error": float(max(identity_errors)),
            "cam_view_equals_w2c_transpose_max_abs_error": float(max(transpose_errors)),
            "c2w_rotation_orthonormal_max_abs_error": float(max(rotation_errors)),
            "c2w_rotation_determinant": numeric_summary(rotation_determinants.tolist()),
            "K_matches_intrinsics": bool(np.allclose(json_k, expected_k, atol=2e-5)),
            "principal_points_at_512_center": bool(np.allclose(json_intrinsics[:, 2:], 256, atol=1e-6)),
            "depth_export_matches_camera_export": depth_contract_matches,
            "pointcloud_export_matches_camera_export": pointcloud_contract_matches,
        },
        "camera_intrinsics": {
            "fx_px": numeric_summary(json_intrinsics[:, 0].tolist()),
            "fy_px": numeric_summary(json_intrinsics[:, 1].tolist()),
            "interpretation": "These are VGGT-Omega predictions, not measured camera calibration.",
        },
        "depth_and_pointcloud": {
            "depth_shape": list(depth_npz["depth"].shape),
            "positive_finite_depth_fraction": float(np.mean(np.isfinite(depth) & (depth > 0))),
            "finite_confidence_fraction": float(np.mean(np.isfinite(confidence))),
            "pointcloud_points": ply_points,
            "points_per_frame": [int(frame["valid_points"]) for frame in point_frames],
            "sampling_stride": pointcloud["sampling_stride"],
            "mean_nearest_neighbor_distance_native_units": pointcloud["mean_nearest_neighbor_distance"],
            "interpretation": (
                "Coverage and spacing describe the exported control geometry in its native gauge; they are "
                "not geometric-accuracy or metric-scale measurements."
            ),
        },
        "trajectory_vs_published_vggt": trajectory,
        "metric_calibration": {
            "status": "unavailable_missing_control_geometry_endpoints",
            "published_scale_coefficient_reused": False,
            "raw_scale_coefficient_comparison_performed": False,
        },
        "warning": (
            "This controls input subset/density. Agreement with the published reconstruction is not survey "
            "ground truth and does not by itself establish lower distortion or better visual quality."
        ),
    }


def main() -> int:
    args = parse_args()
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
