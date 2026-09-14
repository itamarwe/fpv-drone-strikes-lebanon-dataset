#!/usr/bin/env python3
"""Fit MoGe-3 metric depth to an unchanged published VGGT reconstruction.

This evaluator is intentionally specific to the bundled published baseline:
the camera vectors, point cloud, and saved measured-element length all share
the same VGGT coordinate system.  It must not be used to compare a raw scale
coefficient from a newly reconstructed coordinate system with the VGGT value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from metric_scale import (
    crop_depth_to_vggt_view,
    intrinsics_pixels,
    occupied_grid_cells,
    project_camera_z,
    robust_log_scale,
    validate_camera_basis,
    vggt_supported_crop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True, help="Bundled scenes/<key> directory")
    parser.add_argument("--profile", default="common16")
    parser.add_argument("--depths", type=Path, required=True, help="MoGe-3 NPZ output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--votes-dir", type=Path)
    parser.add_argument(
        "--focal-source",
        choices=["fixed", "moge"],
        default="fixed",
        help="Use the documented VGGT diagnostic focal or MoGe's per-frame estimated intrinsics",
    )
    parser.add_argument("--focal-px", type=float, default=812.0)
    parser.add_argument("--depth-layout", choices=["auto", "full", "crop"], default="auto")
    parser.add_argument("--min-votes", type=int, default=100)
    parser.add_argument("--min-spatial-cells", type=int, default=4)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-columns", type=int, default=8)
    parser.add_argument("--max-log-deviation", type=float, default=0.7)
    parser.add_argument("--min-metric-depth-m", type=float, default=0.1)
    parser.add_argument("--max-metric-depth-m", type=float, default=200.0)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    return parser.parse_args()


def read_profile_rows(scene_dir: Path, profile: str) -> list[dict[str, str]]:
    path = scene_dir / "profiles" / profile / "frames.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Profile contains no frames: {path}")
    return rows


def depth_path_for_row(depths: Path, row: dict[str, str]) -> Path:
    candidates = [
        depths / f"{Path(row['profile_file']).stem}.npz",
        depths / f"{Path(row['source_file']).stem}.npz",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No depth NPZ for {row['profile_file']} (also tried {candidates[1].name})")
    return existing[0]


def bootstrap_frame_ci(frame_scales: np.ndarray, samples: int = 4000, seed: int = 20260908) -> list[float] | None:
    """A scene-internal uncertainty diagnostic; frames are not independent scenes."""

    if len(frame_scales) < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(frame_scales), size=(samples, len(frame_scales)))
    estimates = np.exp(np.median(np.log(frame_scales[indices]), axis=1))
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = args.scene_dir / "source" / "scene_meta.json"
    metadata = json.loads(metadata_path.read_text())
    depth_manifest_path = args.depths / "manifest.json"
    depth_manifest = json.loads(depth_manifest_path.read_text()) if depth_manifest_path.exists() else None
    calibration = metadata.get("calibration")
    if not calibration:
        raise ValueError(f"Published scene has no calibration record: {metadata_path}")
    expected_scale = float(calibration["scale_m_per_vggt_unit"])
    real_length = float(calibration["real_length_m"])
    measured_units = float(calibration["measured_vggt_units"])
    if not np.isclose(real_length / measured_units, expected_scale, rtol=1e-10, atol=0.0):
        raise ValueError("Calibration length and saved scale coefficient are inconsistent")

    points_path = args.scene_dir / "baseline" / "points_positions.bin"
    flat_points = np.fromfile(points_path, dtype="<f4")
    if len(flat_points) % 3:
        raise ValueError(f"Baseline positions do not contain XYZ triplets: {points_path}")
    points = flat_points.reshape(-1, 3)
    if len(points) > args.max_points:
        points = points[np.random.default_rng(20260908).choice(len(points), args.max_points, replace=False)]

    rows = read_profile_rows(args.scene_dir, args.profile)
    votes_dir = args.votes_dir or args.output.parent / f"{args.output.stem}_votes"
    votes_dir.mkdir(parents=True, exist_ok=True)
    frame_records: list[dict[str, Any]] = []
    accepted_scales: list[float] = []
    accepted_record_indices: list[int] = []
    basis_errors: list[float] = []
    layouts: set[str] = set()
    source_sizes: set[tuple[int, int]] = set()

    for row in rows:
        source_index = int(row["source_index"])
        entry = metadata["path"][source_index]
        source_image = args.scene_dir / "source" / "images" / row["source_file"]
        with Image.open(source_image) as image:
            source_size = image.size
        source_sizes.add(source_size)
        crop = vggt_supported_crop(*source_size)
        depth_path = depth_path_for_row(args.depths, row)
        with np.load(depth_path) as payload:
            metric_depth_raw = np.asarray(payload["depth"], dtype=np.float64)
            metric_mask_raw = np.asarray(payload["mask"], dtype=bool)
            intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64) if "intrinsics" in payload else None
        metric_depth, metric_mask, layout = crop_depth_to_vggt_view(
            metric_depth_raw,
            metric_mask_raw,
            source_size=source_size,
            crop=crop,
            layout=args.depth_layout,
        )
        layouts.add(layout)

        center = np.asarray(entry["position"], dtype=np.float64)
        right = np.asarray(entry["right"], dtype=np.float64)
        down = np.asarray(entry["down"], dtype=np.float64)
        forward = np.asarray(entry["forward"], dtype=np.float64)
        basis = validate_camera_basis(right, down, forward)
        basis_errors.append(basis["orthogonality_error"])

        source_width, source_height = source_size
        left, top, crop_width, crop_height = crop
        projection_width, projection_height = (source_width, source_height) if layout == "full" else (crop_width, crop_height)
        focal_meta: dict[str, float] | None = None
        if args.focal_source == "moge":
            if intrinsics is None:
                raise ValueError(f"{depth_path} has no intrinsics for --focal-source moge")
            focal_meta = intrinsics_pixels(intrinsics, projection_width, projection_height)
            focal_x = focal_meta["fx_px"]
            focal_y = focal_meta["fy_px"]
            principal_x = focal_meta["cx_px"]
            principal_y = focal_meta["cy_px"]
        else:
            focal_x = focal_y = float(args.focal_px)
            principal_x = projection_width / 2.0
            principal_y = projection_height / 2.0

        baseline_raw, baseline_mask_raw = project_camera_z(
            points,
            center,
            right,
            down,
            forward,
            projection_width,
            projection_height,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
        )
        baseline_depth, baseline_mask, _ = crop_depth_to_vggt_view(
            baseline_raw,
            baseline_mask_raw,
            source_size=source_size,
            crop=crop,
            layout=layout,
        )
        correspondence_mask = (
            baseline_mask
            & metric_mask
            & np.isfinite(metric_depth)
            & (metric_depth >= args.min_metric_depth_m)
            & (metric_depth <= args.max_metric_depth_m)
            & (baseline_depth > 0)
        )
        pixel_y, pixel_x = np.nonzero(correspondence_mask)
        scale_votes = metric_depth[correspondence_mask] / baseline_depth[correspondence_mask]
        cells = occupied_grid_cells(correspondence_mask, args.grid_rows, args.grid_columns)
        record: dict[str, Any] = {
            "profile_index": int(row["profile_index"]),
            "source_index": source_index,
            "profile_file": row["profile_file"],
            "source_file": row["source_file"],
            "segment_id": row.get("segment_id"),
            "video_time_s": float(row["video_time_s"]) if row.get("video_time_s") else None,
            "depth_file": depth_path.name,
            "depth_layout": layout,
            "depth_shape": list(metric_depth_raw.shape),
            "crop_left_top_width_height": list(crop),
            "focal_source": args.focal_source,
            "focal_x_px": focal_x,
            "focal_y_px": focal_y,
            "moge_intrinsics_pixels": focal_meta,
            "raw_votes": int(len(scale_votes)),
            "occupied_grid_cells": cells,
            "total_grid_cells": args.grid_rows * args.grid_columns,
        }
        keep = np.zeros(len(scale_votes), dtype=bool)
        if len(scale_votes) < args.min_votes:
            record["status"] = "insufficient_votes"
        elif cells < args.min_spatial_cells:
            record["status"] = "insufficient_spatial_coverage"
        else:
            fit = robust_log_scale(scale_votes, args.max_log_deviation)
            keep = fit.keep_mask
            kept_spatial_mask = np.zeros_like(correspondence_mask)
            kept_spatial_mask[pixel_y[keep], pixel_x[keep]] = True
            kept_cells = occupied_grid_cells(kept_spatial_mask, args.grid_rows, args.grid_columns)
            record.update(
                {
                    "status": "accepted" if kept_cells >= args.min_spatial_cells else "insufficient_kept_spatial_coverage",
                    "kept_votes": fit.kept_count,
                    "rejected_votes": fit.input_count - fit.kept_count,
                    "kept_occupied_grid_cells": kept_cells,
                    "scale_m_per_vggt_unit": fit.scale,
                    "vote_log_mad": fit.log_mad,
                }
            )
            if kept_cells >= args.min_spatial_cells:
                accepted_scales.append(fit.scale)
                accepted_record_indices.append(len(frame_records))
        vote_file = votes_dir / f"frame_{int(row['profile_index']):06d}_votes.npz"
        np.savez_compressed(
            vote_file,
            scale_m_per_vggt_unit=scale_votes.astype(np.float32),
            kept=keep,
            pixel_x=pixel_x.astype(np.int32),
            pixel_y=pixel_y.astype(np.int32),
            metric_camera_z_m=metric_depth[correspondence_mask].astype(np.float32),
            baseline_camera_z_vggt_units=baseline_depth[correspondence_mask].astype(np.float32),
        )
        record["votes_file"] = str(vote_file)
        frame_records.append(record)

    if not accepted_scales:
        raise ValueError("No frame met the scale-vote and spatial-coverage requirements")
    global_fit = robust_log_scale(np.asarray(accepted_scales), args.max_log_deviation)
    for local_index, record_index in enumerate(accepted_record_indices):
        frame_records[record_index]["kept_for_global_fit"] = bool(global_fit.keep_mask[local_index])
    predicted_length = global_fit.scale * measured_units
    relative_error = abs(predicted_length - real_length) / real_length
    accepted_array = np.asarray(accepted_scales)[global_fit.keep_mask]
    inference_fov = depth_manifest.get("fov_x") if depth_manifest else None
    expected_fixed_fovs = [
        math.degrees(2.0 * math.atan(width / (2.0 * args.focal_px))) for width, _ in sorted(source_sizes)
    ]
    if args.focal_source == "fixed":
        inference_projection_consistent = bool(
            inference_fov is not None
            and len(expected_fixed_fovs) == 1
            and math.isclose(float(inference_fov), expected_fixed_fovs[0], rel_tol=1e-5, abs_tol=1e-5)
        )
    else:
        inference_projection_consistent = inference_fov is None
    result: dict[str, Any] = {
        "schema_version": 2,
        "evaluation": "moge3_metric_depth_vs_unchanged_published_vggt_geometry",
        "scene_id": metadata["scene_id"],
        "profile": args.profile,
        "coordinate_system_id": f"published-vggt:{metadata['scene_id']}",
        "coordinate_comparison_valid": True,
        "depth_definition": "camera_z",
        "focal_contract": {
            "source": args.focal_source,
            "fixed_focal_px": args.focal_px if args.focal_source == "fixed" else None,
            "fixed_focal_status": "documented_reprojection_diagnostic_not_exported_intrinsics"
            if args.focal_source == "fixed"
            else None,
            "depth_layouts_seen": sorted(layouts),
            "vggt_preprocess": "center crop to height/width in [0.5, 2.0]",
            "scene_meta_contains_intrinsics": False,
            "camera_vectors_are_frustum_derived": True,
            "moge_inference_fov_x_degrees": inference_fov,
            "expected_full_input_fov_x_degrees_for_fixed_focal": expected_fixed_fovs
            if args.focal_source == "fixed"
            else None,
            "inference_projection_focal_condition_consistent": inference_projection_consistent,
            "depth_manifest": str(depth_manifest_path) if depth_manifest else None,
        },
        "baseline": {
            "positions": str(points_path),
            "points_used": int(len(points)),
            "camera_vectors": str(metadata_path),
            "max_camera_basis_orthogonality_error": max(basis_errors),
        },
        "fit": {
            "scale_m_per_vggt_unit": global_fit.scale,
            "accepted_frames_before_global_rejection": global_fit.input_count,
            "accepted_frames": global_fit.kept_count,
            "rejected_frame_scales": global_fit.input_count - global_fit.kept_count,
            "frame_scale_log_mad": global_fit.log_mad,
            "frame_bootstrap_95pct_ci_m_per_vggt_unit": bootstrap_frame_ci(accepted_array),
            "aggregation": "median in log space per frame, then equal-weight median across frames",
        },
        "withheld_measured_element_evaluation": {
            "reference_source": calibration.get("source"),
            "reference_real_length_m": real_length,
            "saved_length_vggt_units": measured_units,
            "predicted_length_m": predicted_length,
            "absolute_error_m": abs(predicted_length - real_length),
            "relative_error": relative_error,
            "passes_20_percent_threshold": bool(relative_error <= 0.2),
            "note": "Predicted length is derived from the saved scalar VGGT length; endpoints were not recovered.",
        },
        "same_geometry_scale_diagnostic": {
            "reference_scale_m_per_vggt_unit": expected_scale,
            "predicted_scale_m_per_vggt_unit": global_fit.scale,
            "relative_error": abs(global_fit.scale - expected_scale) / expected_scale,
            "valid_only_because_geometry_is_the_unchanged_published_vggt_baseline": True,
        },
        "frames": frame_records,
        "limitations": [
            "The fixed 812 px focal is a documented reprojection diagnostic, not exported VGGT intrinsics.",
            "A false inference/projection focal-condition flag means the run is diagnostic rather than the prespecified focal condition.",
            "Frame bootstrap spread is a sensitivity diagnostic; correlated frames are not independent scene samples.",
            "The single measured length validates scale only and cannot establish low geometric distortion.",
            "Do not compare this coefficient with a coefficient from a different reconstruction coordinate system.",
        ],
    }
    return result


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
