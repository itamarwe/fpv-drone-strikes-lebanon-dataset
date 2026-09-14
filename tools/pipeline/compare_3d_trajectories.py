#!/usr/bin/env python3
"""Sim(3)-compare two reconstructions on their matched profile-frame poses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_3d_trajectory import (
    error_summary,
    leave_one_out_scale_stability,
    load_prediction,
    point_set_diagnostics,
    trajectory_stats,
    umeyama,
)


FORMATS = ["scal3r-mat", "colmap-text", "vggt-cameras-json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-format", choices=FORMATS, required=True)
    parser.add_argument("--prediction-label", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-format", choices=FORMATS, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    with args.frames_csv.open(newline="") as handle:
        frame_rows = list(csv.DictReader(handle))
    expected_indices = {int(row["profile_index"]) for row in frame_rows}
    record_by_index = {
        int(row["profile_index"]): {
            "profile_index": int(row["profile_index"]),
            "source_index": int(row["source_index"]),
            "segment_id": row["segment_id"],
            "video_time_s": float(row["video_time_s"]),
            "sequence_time_s": float(row["sequence_time_s"]),
        }
        for row in frame_rows
    }
    prediction_by_index = load_prediction(args.prediction, args.prediction_format)
    reference_by_index = load_prediction(args.reference, args.reference_format)
    common_indices = sorted(expected_indices & set(prediction_by_index) & set(reference_by_index))
    if len(common_indices) < 3:
        raise ValueError("At least three common expected profile poses are required")
    predicted = np.vstack([prediction_by_index[index] for index in common_indices])
    reference = np.vstack([reference_by_index[index] for index in common_indices])
    records = [record_by_index[index] for index in common_indices]
    scale, rotation, translation = umeyama(predicted, reference)
    aligned = (scale * (rotation @ predicted.T)).T + translation
    errors = np.linalg.norm(aligned - reference, axis=1)
    reference_rms_radius = float(
        np.sqrt(np.mean(np.sum((reference - reference.mean(axis=0)) ** 2, axis=1)))
    )
    per_segment = {}
    for segment in sorted({record["segment_id"] for record in records}):
        mask = np.asarray([record["segment_id"] == segment for record in records])
        per_segment[segment] = {"frames": int(mask.sum()), **error_summary(errors[mask])}
    return {
        "schema_version": 1,
        "evaluation": "sim3_trajectory_agreement_on_matched_profile_frames",
        "prediction": {
            "label": args.prediction_label,
            "format": args.prediction_format,
            "available_expected_frames": len(expected_indices & set(prediction_by_index)),
            "missing_expected_indices": sorted(expected_indices - set(prediction_by_index)),
        },
        "reference": {
            "label": args.reference_label,
            "format": args.reference_format,
            "available_expected_frames": len(expected_indices & set(reference_by_index)),
            "missing_expected_indices": sorted(expected_indices - set(reference_by_index)),
        },
        "expected_profile_frames": len(expected_indices),
        "common_frames": len(common_indices),
        "common_profile_indices": common_indices,
        "sim3_scale_prediction_to_reference": scale,
        "sim3_scale_leave_one_out_stability": leave_one_out_scale_stability(predicted, reference),
        "sim3_aligned_error_reference_units": error_summary(errors),
        "reference_path_rms_radius_units": reference_rms_radius,
        "sim3_rmse_fraction_reference_path_rms_radius": (
            float(np.sqrt(np.mean(errors**2)) / reference_rms_radius) if reference_rms_radius else None
        ),
        "error_by_segment_reference_units": per_segment,
        "prediction_geometry": point_set_diagnostics(predicted),
        "reference_geometry": point_set_diagnostics(reference),
        "prediction_trajectory_units": trajectory_stats(predicted, records),
        "reference_trajectory_units": trajectory_stats(reference, records),
        "matched_frames": [
            {**record, "sim3_error_reference_units": float(error)}
            for record, error in zip(records, errors)
        ],
        "warning": (
            "This is relative trajectory agreement after a free similarity alignment. The reference is not "
            "assumed to be survey ground truth, and raw scale coefficients are not compared."
        ),
    }


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
