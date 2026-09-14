#!/usr/bin/env python3
"""Compare a predicted trajectory with the published VGGT trajectory.

Sim(3) agreement is a shape diagnostic, not independent pose ground truth. If
an independently predicted metric scale is supplied, the script also performs
a rigid-only alignment after converting each trajectory to metres.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fit_moge3_colmap_scale import qvec_to_rotation, read_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-meta", type=Path, required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument(
        "--format", choices=["scal3r-mat", "colmap-text", "vggt-cameras-json"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predicted-scale-m-per-unit",
        type=float,
        help="Scale for the prediction's own coordinates; enables rigid-only metric evaluation",
    )
    return parser.parse_args()


def load_prediction(path: Path, pose_format: str) -> dict[int, np.ndarray]:
    if pose_format == "scal3r-mat":
        matrix_rows = np.loadtxt(path, dtype=np.float64)
        matrix_rows = np.atleast_2d(matrix_rows)
        if matrix_rows.shape[1] != 16:
            raise ValueError("Scal3R mat.txt must contain flattened 4x4 camera-to-world matrices")
        return {index: row.reshape(4, 4)[:3, 3] for index, row in enumerate(matrix_rows)}
    if pose_format == "vggt-cameras-json":
        payload = json.loads(path.read_text())
        predictions = {}
        for position, frame in enumerate(payload["frames"]):
            name = str(frame["name"])
            try:
                profile_index = int(name.rsplit("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Expected VGGT frame name ending in a numeric profile index: {name}") from exc
            if int(frame["index"]) != position or int(frame["index"]) != profile_index:
                raise ValueError(f"VGGT frame order/index/name mismatch at position {position}: {name}")
            c2w = np.asarray(frame["c2w"], dtype=np.float64)
            w2c = np.asarray(frame["w2c"], dtype=np.float64)
            cam_view = np.asarray(frame["cam_view"], dtype=np.float64)
            intrinsics = np.asarray(frame["intrinsics"], dtype=np.float64)
            k_matrix = np.asarray(frame["K"], dtype=np.float64)
            if c2w.shape != (4, 4) or w2c.shape != (4, 4) or cam_view.shape != (4, 4):
                raise ValueError(f"Invalid VGGT camera matrix shape for {name}")
            if not np.allclose(c2w @ w2c, np.eye(4), atol=2e-5):
                raise ValueError(f"VGGT c2w/w2c are not inverses for {name}")
            if not np.allclose(cam_view, w2c.T, atol=2e-5):
                raise ValueError(f"VGGT cam_view is not transposed w2c for {name}")
            expected_k = np.array(
                [[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]],
                dtype=np.float64,
            )
            if not np.allclose(k_matrix, expected_k, atol=2e-5):
                raise ValueError(f"VGGT K/intrinsics mismatch for {name}")
            if profile_index in predictions:
                raise ValueError(f"Duplicate predicted profile index: {profile_index}")
            predictions[profile_index] = c2w[:3, 3]
        return predictions
    predictions = {}
    for image in read_images(path / "images.txt"):
        rotation = qvec_to_rotation(image.qvec)
        center = -(rotation.T @ image.tvec)
        stem = Path(image.name).stem
        try:
            profile_index = int(stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Expected image name ending in a numeric profile index: {image.name}") from exc
        if profile_index in predictions:
            raise ValueError(f"Duplicate predicted profile index: {profile_index}")
        predictions[profile_index] = center
    return predictions


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u_matrix, singular_values, v_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u_matrix @ v_transpose) < 0:
        correction[-1, -1] = -1
    rotation = u_matrix @ correction @ v_transpose
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("Predicted camera centers have zero variance; Sim(3) is undefined")
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares proper rigid alignment without a scale degree of freedom."""

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (target - target_mean).T @ (source - source_mean)
    u_matrix, _, v_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u_matrix @ v_transpose) < 0:
        correction[-1, -1] = -1
    rotation = u_matrix @ correction @ v_transpose
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def point_set_diagnostics(points: np.ndarray) -> dict[str, object]:
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular_values[0] if singular_values[0] else 0.0
    rank = int(np.count_nonzero(singular_values > tolerance))
    return {
        "centered_rank": rank,
        "centered_singular_values": [float(value) for value in singular_values],
        "degenerate_for_alignment": rank < 2,
        "near_collinear": bool(rank >= 2 and singular_values[1] / singular_values[0] < 1e-3),
    }


def trajectory_stats(points: np.ndarray, records: list[dict[str, object]]) -> dict[str, object]:
    """Summarize within-segment intervals and expose excluded edit/time gaps."""

    if len(points) != len(records):
        raise ValueError("Trajectory records must correspond one-to-one with points")
    intervals = []
    segment_transitions = 0
    nonpositive_time_intervals = 0
    for index in range(len(points) - 1):
        left, right = records[index], records[index + 1]
        if left.get("segment_id") != right.get("segment_id"):
            segment_transitions += 1
            continue
        elapsed = float(right["sequence_time_s"]) - float(left["sequence_time_s"])
        if elapsed <= 0:
            nonpositive_time_intervals += 1
            continue
        step = float(np.linalg.norm(points[index + 1] - points[index]))
        intervals.append(
            {
                "source_index_gap": int(right["source_index"]) - int(left["source_index"]),
                "elapsed_s": elapsed,
                "step": step,
                "speed": step / elapsed,
            }
        )
    base = {
        "within_segment_intervals": len(intervals),
        "segment_transitions_excluded": segment_transitions,
        "nonpositive_time_intervals_excluded": nonpositive_time_intervals,
        "source_index_gaps_gt_1": int(sum(item["source_index_gap"] > 1 for item in intervals)),
    }
    if not intervals:
        return {
            **base,
            "max_elapsed_s": None,
            "step_median": None,
            "step_cv": None,
            "speed_median": None,
            "speed_cv": None,
            "speed_jump_outliers": 0,
        }
    steps = np.asarray([item["step"] for item in intervals])
    speeds = np.asarray([item["speed"] for item in intervals])
    step_mean = float(np.mean(steps))
    speed_mean = float(np.mean(speeds))
    speed_median = float(np.median(speeds))
    speed_mad = float(np.median(np.abs(speeds - speed_median)))
    speed_threshold = speed_median + 6 * max(speed_mad, 1e-12)
    return {
        **base,
        "max_elapsed_s": float(max(item["elapsed_s"] for item in intervals)),
        "step_median": float(np.median(steps)),
        "step_cv": float(np.std(steps) / step_mean) if step_mean else None,
        "speed_median": speed_median,
        "speed_cv": float(np.std(speeds) / speed_mean) if speed_mean else None,
        "speed_jump_outliers": int(np.count_nonzero(speeds > speed_threshold)),
    }


def error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "max": float(np.max(errors)),
    }


def leave_one_out_scale_stability(source: np.ndarray, target: np.ndarray) -> dict[str, object]:
    """Measure how sensitive the Sim(3) scale is to any single matched pose."""

    if len(source) < 4:
        return {"samples": 0, "status": "unavailable_fewer_than_four_poses"}
    scales = np.asarray(
        [
            umeyama(source[np.arange(len(source)) != index], target[np.arange(len(target)) != index])[0]
            for index in range(len(source))
        ],
        dtype=np.float64,
    )
    mean = float(np.mean(scales))
    return {
        "samples": len(scales),
        "status": "evaluated",
        "min": float(np.min(scales)),
        "median": float(np.median(scales)),
        "max": float(np.max(scales)),
        "coefficient_of_variation": float(np.std(scales) / mean) if mean else None,
        "relative_max_min_range": float((np.max(scales) - np.min(scales)) / mean) if mean else None,
    }


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.scene_meta.read_text())
    with args.frames_csv.open(newline="") as handle:
        frame_rows = list(csv.DictReader(handle))
    prediction = load_prediction(args.poses, args.format)
    reference_by_profile = {}
    records_by_profile = {}
    for row in frame_rows:
        profile_index = int(row["profile_index"])
        source_index = int(row["source_index"])
        entry = metadata["path"][source_index]
        reference_by_profile[profile_index] = np.asarray(entry["position"], dtype=np.float64)
        records_by_profile[profile_index] = {
            "profile_index": profile_index,
            "source_index": source_index,
            "segment_id": row.get("segment_id") or entry.get("segment_id"),
            "video_time_s": float(row["video_time_s"]),
            "sequence_time_s": float(row["sequence_time_s"]),
        }
    common_indices = sorted(set(prediction) & set(reference_by_profile))
    if len(common_indices) < 3:
        raise SystemExit("At least three matched poses are required")
    predicted = np.vstack([prediction[index] for index in common_indices])
    reference = np.vstack([reference_by_profile[index] for index in common_indices])
    records = [records_by_profile[index] for index in common_indices]
    scale, rotation, translation = umeyama(predicted, reference)
    aligned = (scale * (rotation @ predicted.T)).T + translation
    errors = np.linalg.norm(aligned - reference, axis=1)
    reference_centered = reference - reference.mean(axis=0)
    reference_rms_radius = float(np.sqrt(np.mean(np.sum(reference_centered**2, axis=1))))
    per_segment = {}
    for segment in sorted({str(record["segment_id"]) for record in records}):
        mask = np.asarray([str(record["segment_id"]) == segment for record in records])
        per_segment[segment] = {"frames": int(mask.sum()), **error_summary(errors[mask])}
    summary = {
        "schema_version": 2,
        "pose_format": args.format,
        "expected_frames": len(frame_rows),
        "registered_frames": len(common_indices),
        "registered_fraction": len(common_indices) / len(frame_rows),
        "missing_profile_indices": sorted(set(reference_by_profile) - set(prediction)),
        "unexpected_prediction_indices": sorted(set(prediction) - set(reference_by_profile)),
        "sim3_scale_to_published_vggt": scale,
        "sim3_scale_leave_one_out_stability": leave_one_out_scale_stability(predicted, reference),
        "sim3_aligned_error_vggt_units": error_summary(errors),
        "published_path_rms_radius_vggt_units": reference_rms_radius,
        "sim3_rmse_fraction_published_path_rms_radius": (
            float(np.sqrt(np.mean(errors**2)) / reference_rms_radius) if reference_rms_radius else None
        ),
        "sim3_aligned_error_by_segment_vggt_units": per_segment,
        "prediction_geometry": point_set_diagnostics(predicted),
        "published_geometry": point_set_diagnostics(reference),
        "prediction_trajectory_units": trajectory_stats(predicted, records),
        "published_trajectory_vggt_units": trajectory_stats(reference, records),
        "matched_frames": [
            {**record, "sim3_error_vggt_units": float(error)} for record, error in zip(records, errors)
        ],
        "warning": "This measures agreement with the published VGGT trajectory, not independent pose accuracy.",
    }
    if args.predicted_scale_m_per_unit is not None:
        if args.predicted_scale_m_per_unit <= 0:
            raise ValueError("--predicted-scale-m-per-unit must be positive")
        calibration = metadata.get("calibration")
        if not calibration or float(calibration.get("scale_m_per_vggt_unit", 0)) <= 0:
            raise ValueError("Scene metadata needs a positive VGGT calibration for metric evaluation")
        published_scale = float(calibration["scale_m_per_vggt_unit"])
        predicted_m = predicted * args.predicted_scale_m_per_unit
        reference_m = reference * published_scale
        rigid_rotation, rigid_translation = rigid_alignment(predicted_m, reference_m)
        rigid_aligned = (rigid_rotation @ predicted_m.T).T + rigid_translation
        metric_errors = np.linalg.norm(rigid_aligned - reference_m, axis=1)
        metric_per_segment = {}
        for segment in sorted({str(record["segment_id"]) for record in records}):
            mask = np.asarray([str(record["segment_id"]) == segment for record in records])
            metric_per_segment[segment] = {"frames": int(mask.sum()), **error_summary(metric_errors[mask])}
        summary["metric_rigid_alignment"] = {
            "predicted_scale_m_per_prediction_unit": args.predicted_scale_m_per_unit,
            "published_scale_m_per_vggt_unit": published_scale,
            "scale_coefficients_compared_directly": False,
            "error_m": error_summary(metric_errors),
            "error_by_segment_m": metric_per_segment,
            "prediction_trajectory_m": trajectory_stats(predicted_m, records),
            "published_trajectory_m": trajectory_stats(reference_m, records),
            "interpretation": "Rigid-only error after each trajectory was converted with its own scale.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
