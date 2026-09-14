#!/usr/bin/env python3
"""Validate a Scal3R export contract and aggregate its metric diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from fit_moge3_scal3r_scale import read_scal3r_depth, scal3r_image_transform
from summarize_colmap_diagnostics import numeric_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--scene-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def opencv_matrices(text: str, prefix: str) -> dict[str, np.ndarray]:
    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+): !!opencv-matrix\s+"
        r"rows: (\d+)\s+cols: (\d+)\s+dt: \w+\s+data: \[([^]]+)\]",
        re.MULTILINE,
    )
    result = {}
    for name, rows, columns, payload in pattern.findall(text):
        values = np.fromstring(payload, sep=",")
        shape = (int(rows), int(columns))
        if values.size != shape[0] * shape[1]:
            raise ValueError(f"Invalid {prefix}_{name} matrix size")
        result[name] = values.reshape(shape)
    return result


def opencv_scalars(text: str, prefix: str) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in re.findall(rf"^{re.escape(prefix)}_(\d+): ([^\s]+)$", text, re.MULTILINE)
    }


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise ValueError(f"No vertex count in PLY header: {path}")


def scale_condition(path: Path, trajectory_path: Path, segment_by_image: dict[str, str]) -> dict[str, object]:
    metric = json.loads(path.read_text())
    trajectory = json.loads(trajectory_path.read_text())
    frames = [frame for frame in metric["frames"] if frame.get("status") == "accepted"]
    scales = [float(frame["scale_m_per_scal3r_unit"]) for frame in frames]
    by_segment = {}
    for segment in sorted(set(segment_by_image.values())):
        selected = [frame for frame in frames if segment_by_image[frame["image"]] == segment]
        values = [float(frame["scale_m_per_scal3r_unit"]) for frame in selected]
        by_segment[segment] = {
            "frames": len(values),
            "kept_for_global_fit": int(sum(bool(frame.get("kept_for_global_fit")) for frame in selected)),
            "scale_summary_m_per_scal3r_unit": numeric_summary(values),
        }
    prediction_steps = trajectory["metric_rigid_alignment"]["prediction_trajectory_m"]
    published_steps = trajectory["metric_rigid_alignment"]["published_trajectory_m"]
    return {
        "scale_m_per_scal3r_unit": metric["scale_m_per_scal3r_unit"],
        "frames_accepted_before_global_rejection": len(frames),
        "frames_kept_for_global_fit": metric["frames_kept"],
        "frame_scale_log_mad": metric["frame_scale_log_mad"],
        "accepted_frame_scale_summary_m_per_scal3r_unit": numeric_summary(scales),
        "accepted_frame_scale_max_min_ratio": float(max(scales) / min(scales)),
        "by_segment": by_segment,
        "metric_trajectory": {
            "rigid_alignment_error_m": trajectory["metric_rigid_alignment"]["error_m"],
            "depth_scaled_median_step_m": prediction_steps["step_median"],
            "published_median_step_m": published_steps["step_median"],
            "median_step_ratio": prediction_steps["step_median"] / published_steps["step_median"],
        },
        "known_element_evaluation": metric["known_element_evaluation"],
    }


def summarize(args: argparse.Namespace) -> dict[str, object]:
    scene_metadata = json.loads(args.scene_meta.read_text())
    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    with args.frames_csv.open(newline="") as handle:
        frame_rows = list(csv.DictReader(handle))
    expected_names = [row["profile_file"] for row in frame_rows]
    segment_by_image = {row["profile_file"]: row["segment_id"] for row in frame_rows}
    manifest_names = [Path(line).name for line in (args.result / "runtime/image_manifest.txt").read_text().splitlines() if line]

    intrinsics_text = (args.result / "intri.yml").read_text()
    extrinsics_text = (args.result / "extri.yml").read_text()
    intrinsics = opencv_matrices(intrinsics_text, "K")
    heights = opencv_scalars(intrinsics_text, "H")
    widths = opencv_scalars(intrinsics_text, "W")
    rotations = opencv_matrices(extrinsics_text, "Rot")
    translations = opencv_matrices(extrinsics_text, "T")
    matrices = np.atleast_2d(np.loadtxt(args.result / "mat.txt", dtype=np.float64))

    transforms = []
    depth_shapes, mask_shapes, mask_valid_fractions = [], [], []
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            transform = scal3r_image_transform(image.width, image.height)
        transforms.append(transform)
        depth = read_scal3r_depth(args.result / f"depths/{index:06d}.exr")
        with Image.open(args.result / f"masks/{index:06d}.png") as mask_image:
            mask = np.asarray(mask_image)
        depth_shapes.append(list(depth.shape))
        mask_shapes.append(list(mask.shape))
        mask_valid_fractions.append(float(np.mean(mask > 0)))

    names = [f"{index:06d}" for index in range(len(image_paths))]
    expected_pose_matrices = []
    for name in names:
        world_to_camera_rotation = rotations[name]
        world_to_camera_translation = translations[name].reshape(3)
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = world_to_camera_rotation.T
        camera_to_world[:3, 3] = -(world_to_camera_rotation.T @ world_to_camera_translation)
        expected_pose_matrices.append(camera_to_world)
    expected_pose_matrices_array = np.asarray(expected_pose_matrices)
    if matrices.shape != (len(image_paths), 16):
        raise ValueError(f"Expected {len(image_paths)} flattened poses, got {matrices.shape}")

    crop_matches = all(
        int(widths[name]) == transform.output_width
        and int(heights[name]) == transform.output_height
        and intrinsics[name][0, 2] == transform.output_width / 2
        and intrinsics[name][1, 2] == transform.output_height / 2
        and depth_shapes[index] == [transform.output_height, transform.output_width]
        and mask_shapes[index] == [transform.output_height, transform.output_width]
        for index, (name, transform) in enumerate(zip(names, transforms))
    )
    pose_max_difference = float(
        np.max(np.abs(matrices.reshape(-1, 4, 4) - expected_pose_matrices_array))
    )

    trajectory = json.loads((args.result / "trajectory.json").read_text())
    estimated = scale_condition(
        args.result / "metric_scale_estimated_focal.json", args.result / "trajectory.json", segment_by_image
    )
    fixed = scale_condition(
        args.result / "metric_scale_diagnostic_fixed_focal.json",
        args.result / "trajectory_diagnostic_fixed_focal.json",
        segment_by_image,
    )
    estimated_frame_scales = [
        float(frame["scale_m_per_scal3r_unit"])
        for frame in json.loads((args.result / "metric_scale_estimated_focal.json").read_text())["frames"]
    ]
    fixed_frame_scales = [
        float(frame["scale_m_per_scal3r_unit"])
        for frame in json.loads((args.result / "metric_scale_diagnostic_fixed_focal.json").read_text())["frames"]
    ]
    return {
        "schema_version": 1,
        "scene": scene_metadata.get("scene_id"),
        "export": {
            "expected_frames": len(expected_names),
            "pose_frames": len(matrices),
            "depth_frames": len(depth_shapes),
            "dense_point_vertices": ply_vertex_count(args.result / "points/whole.ply"),
            "manifest_matches_exact_profile_order": manifest_names == expected_names,
        },
        "preprocessing_and_pose_contract": {
            "validated": bool(crop_matches and pose_max_difference < 2e-5),
            "crop_intrinsics_depth_and_mask_shapes_match": crop_matches,
            "unique_source_to_output_transforms": [
                dict(zip(transform.__dataclass_fields__, values))
                for values in sorted({tuple(transform.__dict__.values()) for transform in transforms})
                for transform in [type(transforms[0])(*values)]
            ],
            "intri_principal_points_equal_output_centers": crop_matches,
            "mat_matches_inverse_extri_max_abs_difference": pose_max_difference,
            "saved_intrinsics_fx_px": numeric_summary([float(intrinsics[name][0, 0]) for name in names]),
            "saved_intrinsics_fy_px": numeric_summary([float(intrinsics[name][1, 1]) for name in names]),
            "mask_valid_fraction": numeric_summary(mask_valid_fractions),
        },
        "trajectory_shape_vs_published_vggt": {
            "registered_frames": trajectory["registered_frames"],
            "sim3_error_vggt_units": trajectory["sim3_aligned_error_vggt_units"],
            "sim3_rmse_fraction_published_path_rms_radius": trajectory[
                "sim3_rmse_fraction_published_path_rms_radius"
            ],
            "error_by_segment_vggt_units": trajectory["sim3_aligned_error_by_segment_vggt_units"],
            "warning": trajectory["warning"],
        },
        "metric_scale_conditions": {
            "estimated_focal": estimated,
            "diagnostic_fixed_focal": fixed,
            "fixed_over_estimated_global_scale_ratio": (
                fixed["scale_m_per_scal3r_unit"] / estimated["scale_m_per_scal3r_unit"]
            ),
            "per_frame_log_scale_correlation": float(
                np.corrcoef(np.log(estimated_frame_scales), np.log(fixed_frame_scales))[0, 1]
            ),
            "interpretation": (
                "Both estimates inherit MoGe's metric-depth behavior. They calibrate the Scal3R native gauge "
                "but are not independent metric validation. Raw coefficients are not compared with VGGT."
            ),
        },
        "limitations": [
            "The published VGGT trajectory is a comparison baseline, not independent pose ground truth.",
            "The saved 3 m/7 m element has no corresponding Scal3R endpoints and is therefore not scored.",
            "Scal3R intrinsics are model outputs, not measured camera calibration.",
        ],
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
