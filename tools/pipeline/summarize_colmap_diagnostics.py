#!/usr/bin/env python3
"""Summarize COLMAP/GLOMAP registration, residual, and depth-scale diagnostics.

The sparse-model residuals measure internal multi-view consistency.  The
trajectory comparison measures agreement with the published VGGT path.  Neither
is independent geometric ground truth, so those interpretations are written
into the output rather than left implicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fit_moge3_colmap_scale import observation_triplets, read_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="COLMAP text-model directory")
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--metric-scale", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def read_camera_records(path: Path) -> list[dict[str, object]]:
    cameras = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Invalid COLMAP camera line: {line}")
        cameras.append(
            {
                "camera_id": int(fields[0]),
                "model": fields[1],
                "width": int(fields[2]),
                "height": int(fields[3]),
                "parameters": [float(value) for value in fields[4:]],
            }
        )
    return cameras


def read_point_diagnostics(path: Path) -> tuple[list[float], list[float]]:
    errors, track_lengths = [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10 or (len(fields) - 8) % 2:
            raise ValueError(f"Invalid COLMAP point line: {line}")
        errors.append(float(fields[7]))
        track_lengths.append(float((len(fields) - 8) // 2))
    return errors, track_lengths


def database_diagnostics(path: Path) -> dict[str, object]:
    # URI read-only mode prevents a diagnostic command from modifying the bundle.
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        table_counts = {
            table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in ("images", "keypoints", "descriptors", "matches", "two_view_geometries")
        }
        keypoint_rows = [int(row[0]) for row in connection.execute("SELECT rows FROM keypoints")]
        raw_match_rows = [int(row[0]) for row in connection.execute("SELECT rows FROM matches")]
        geometries = [
            (int(rows), int(config))
            for rows, config in connection.execute("SELECT rows, config FROM two_view_geometries")
        ]
    return {
        "table_rows": table_counts,
        "extracted_keypoints": int(sum(keypoint_rows)),
        "raw_pair_matches": int(sum(raw_match_rows)),
        "image_pairs_with_raw_matches": int(sum(rows > 0 for rows in raw_match_rows)),
        "verified_inlier_correspondences": int(sum(rows for rows, _ in geometries)),
        "image_pairs_with_verified_geometry": int(sum(rows > 0 and config != 0 for rows, config in geometries)),
        "two_view_configuration_counts": {
            str(key): int(value) for key, value in sorted(Counter(config for _, config in geometries).items())
        },
    }


def segment_counts(names: list[str], segment_by_image: dict[str, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for name in names:
        counts[segment_by_image.get(name, "unknown")] += 1
    return {key: int(value) for key, value in sorted(counts.items())}


def summarize(args: argparse.Namespace) -> dict[str, object]:
    with args.frames_csv.open(newline="") as handle:
        frame_rows = list(csv.DictReader(handle))
    expected_names = [row["profile_file"] for row in frame_rows]
    segment_by_image = {row["profile_file"]: row["segment_id"] for row in frame_rows}

    images = read_images(args.model / "images.txt")
    registered_names = [image.name for image in images]
    registered_set = set(registered_names)
    expected_set = set(expected_names)
    if len(registered_set) != len(registered_names):
        raise ValueError("Duplicate image names in registered COLMAP model")

    feature_counts, triangulated_counts = [], []
    for image in images:
        triplets = list(observation_triplets(image.observations))
        feature_counts.append(float(len(triplets)))
        triangulated_counts.append(float(sum(point_id >= 0 for _, _, point_id in triplets)))

    reprojection_errors, track_lengths = read_point_diagnostics(args.model / "points3D.txt")
    metric = json.loads(args.metric_scale.read_text())
    trajectory = json.loads(args.trajectory.read_text())
    accepted = [frame for frame in metric["frames"] if frame.get("status") == "accepted"]
    kept = [frame for frame in accepted if frame.get("kept_for_global_fit")]
    scales = [float(frame["scale_m_per_colmap_unit"]) for frame in accepted]
    kept_scales = [float(frame["scale_m_per_colmap_unit"]) for frame in kept]

    scale_by_segment: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for frame in accepted:
        scale_by_segment[segment_by_image.get(frame["image"], "unknown")].append(frame)
    scale_segments = {}
    for segment, frames in sorted(scale_by_segment.items()):
        values = [float(frame["scale_m_per_colmap_unit"]) for frame in frames]
        scale_segments[segment] = {
            "accepted_before_global_rejection": len(frames),
            "kept_for_global_fit": int(sum(bool(frame.get("kept_for_global_fit")) for frame in frames)),
            "median_m_per_colmap_unit": float(np.median(values)),
            "min_m_per_colmap_unit": float(np.min(values)),
            "max_m_per_colmap_unit": float(np.max(values)),
        }

    result: dict[str, object] = {
        "schema_version": 1,
        "registration": {
            "expected_frames": len(expected_names),
            "registered_frames": len(registered_names),
            "registered_fraction": len(registered_names) / len(expected_names),
            "missing_images": sorted(expected_set - registered_set),
            "unexpected_images": sorted(registered_set - expected_set),
            "expected_by_segment": segment_counts(expected_names, segment_by_image),
            "registered_by_segment": segment_counts(registered_names, segment_by_image),
        },
        "cameras": {
            "records": read_camera_records(args.model / "cameras.txt"),
            "interpretation": "Intrinsics were estimated by SfM and are not independent camera calibration.",
        },
        "sparse_model": {
            "points_3d": len(reprojection_errors),
            "point_mean_reprojection_error_px": numeric_summary(reprojection_errors),
            "track_length_observations": numeric_summary(track_lengths),
            "features_listed_per_registered_image": numeric_summary(feature_counts),
            "triangulated_observations_per_registered_image": numeric_summary(triangulated_counts),
            "interpretation": (
                "COLMAP point errors and tracks measure internal multi-view consistency after optimization; "
                "they do not establish metric accuracy or completeness."
            ),
        },
        "depth_scale_fit": {
            "depth_definition": metric.get("depth_definition"),
            "registered_frames": metric.get("registered_frames"),
            "frames_accepted_before_global_rejection": len(accepted),
            "frames_kept_for_global_fit": len(kept),
            "missing_depth_frames": metric.get("missing_depth_frames"),
            "global_scale_m_per_colmap_unit": metric.get("global_scale_m_per_colmap_unit"),
            "kept_frame_log_scale_mad": metric.get("frame_scale_log_mad"),
            "accepted_frame_scale_summary_m_per_colmap_unit": numeric_summary(scales),
            "accepted_frame_scale_max_min_ratio": float(max(scales) / min(scales)) if scales else None,
            "kept_frame_scale_summary_m_per_colmap_unit": numeric_summary(kept_scales),
            "kept_images": [frame["image"] for frame in kept],
            "by_segment": scale_segments,
            "coefficient_comparison_to_published_vggt_performed": False,
            "interpretation": (
                "The coefficient applies only to this COLMAP coordinate system. Its cross-frame spread "
                "diagnoses depth-to-SfM scale stability; it is not compared directly with the VGGT coefficient."
            ),
        },
        "trajectory": trajectory,
    }
    if args.database:
        result["matching_database"] = database_diagnostics(args.database)
    return result


def main() -> int:
    args = parse_args()
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
