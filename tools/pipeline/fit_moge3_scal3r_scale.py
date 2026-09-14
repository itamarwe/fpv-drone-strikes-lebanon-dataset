#!/usr/bin/env python3
"""Fit metric scale to Scal3R using pixel-aligned MoGe-3 camera-z depths.

The pinned Scal3R export writes camera-z depth in its native reconstruction
gauge after deterministic resize-and-center-crop preprocessing.  This tool maps
MoGe depths onto that raster and estimates metres per Scal3R unit.  It does not
score the saved 3 m/7 m element because its endpoints are unavailable in the
new Scal3R coordinate system.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from metric_scale import occupied_grid_cells, robust_log_scale


@dataclass(frozen=True)
class Scal3RImageTransform:
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    output_width: int
    output_height: int


def scal3r_image_transform(
    source_width: int,
    source_height: int,
    proc_max_size: int = 518,
    proc_align_size: int = 14,
) -> Scal3RImageTransform:
    """Mirror Scal3R dc557e7 image_utils.py sizing and strict-center crop."""

    if min(source_width, source_height, proc_max_size, proc_align_size) <= 0:
        raise ValueError("Image and preprocessing dimensions must be positive")
    target_height = int((source_height / source_width) * proc_max_size)
    target_width = proc_max_size
    target_height = max(proc_align_size, target_height // proc_align_size * proc_align_size)
    target_width = max(proc_align_size, target_width // proc_align_size * proc_align_size)
    ratio = max(target_height / source_height, target_width / source_width)
    resized_height = max(proc_align_size, round(source_height * ratio / proc_align_size) * proc_align_size)
    resized_width = max(proc_align_size, round(source_width * ratio / proc_align_size) * proc_align_size)

    # Scal3R starts with cx=W/2, cy=H/2, scales K with the resize, then
    # strict-center-crops around the scaled principal point.
    principal_x = source_width / 2.0 * resized_width / source_width
    principal_y = source_height / 2.0 * resized_height / source_height
    crop_left = int(round(principal_x)) - target_width // 2
    crop_top = int(round(principal_y)) - target_height // 2
    if crop_left < 0 or crop_top < 0 or crop_left + target_width > resized_width or crop_top + target_height > resized_height:
        raise ValueError("Pinned Scal3R preprocessing would require padding; this adapter does not silently pad")
    return Scal3RImageTransform(
        source_width,
        source_height,
        resized_width,
        resized_height,
        crop_left,
        crop_top,
        target_width,
        target_height,
    )


def remap_moge_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    transform: Scal3RImageTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize and crop MoGe output onto Scal3R's processed image raster."""

    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    expected = (transform.source_height, transform.source_width)
    if depth.shape != expected or mask.shape != expected:
        raise ValueError(f"MoGe depth/mask must match source shape {expected}, got {depth.shape}/{mask.shape}")
    resized_depth = np.asarray(
        Image.fromarray(depth, mode="F").resize(
            (transform.resized_width, transform.resized_height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    resized_mask = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
            (transform.resized_width, transform.resized_height), Image.Resampling.NEAREST
        )
    ) > 0
    ys = slice(transform.crop_top, transform.crop_top + transform.output_height)
    xs = slice(transform.crop_left, transform.crop_left + transform.output_width)
    return resized_depth[ys, xs], resized_mask[ys, xs]


def read_scal3r_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        try:
            import cv2
        except (ImportError, OSError):
            # ffmpeg's OpenEXR decoder provides a dependency-light fallback and
            # preserves the single-channel float32 values without normalization.
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            width, height = int(stream["width"]), int(stream["height"])
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "grayf32le",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            ).stdout
            depth = np.frombuffer(decoded, dtype="<f4")
            if depth.size != width * height:
                raise ValueError(
                    f"ffmpeg decoded {depth.size} float samples, expected {width * height}: {path}"
                )
            depth = depth.reshape(height, width)
        else:
            depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise ValueError(f"Could not read Scal3R depth: {path}")
    depth = np.asarray(depth, dtype=np.float64).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected a single-channel Scal3R depth map, got {depth.shape}: {path}")
    return depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Exact image directory passed to Scal3R")
    parser.add_argument("--scal3r-result", type=Path, required=True)
    parser.add_argument("--moge-depths", type=Path, required=True, help="MoGe NPZs on the original input images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--votes-dir", type=Path)
    parser.add_argument("--proc-max-size", type=int, default=518)
    parser.add_argument("--proc-align-size", type=int, default=14)
    parser.add_argument("--min-votes", type=int, default=1000)
    parser.add_argument("--min-spatial-cells", type=int, default=8)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-columns", type=int, default=8)
    parser.add_argument("--max-log-deviation", type=float, default=0.7)
    parser.add_argument("--min-metric-depth-m", type=float, default=0.1)
    parser.add_argument("--max-metric-depth-m", type=float, default=200.0)
    return parser.parse_args()


def find_scal3r_depth(depth_dir: Path, index: int) -> Path:
    for suffix in (".exr", ".npy"):
        candidate = depth_dir / f"{index:06d}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No Scal3R depth for frame {index:06d} in {depth_dir}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    image_paths = sorted(
        path for path in args.images.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not image_paths:
        raise ValueError(f"No Scal3R input images found in {args.images}")
    moge_manifest_path = args.moge_depths / "manifest.json"
    moge_manifest = json.loads(moge_manifest_path.read_text()) if moge_manifest_path.exists() else None
    votes_dir = args.votes_dir or args.output.parent / f"{args.output.stem}_votes"
    votes_dir.mkdir(parents=True, exist_ok=True)
    frame_records = []
    frame_scales = []
    accepted_record_indices = []
    transforms_seen = set()
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            transform = scal3r_image_transform(
                image.width,
                image.height,
                proc_max_size=args.proc_max_size,
                proc_align_size=args.proc_align_size,
            )
        transforms_seen.add(tuple(transform.__dict__.values()))
        moge_path = args.moge_depths / f"{image_path.stem}.npz"
        if not moge_path.exists():
            raise FileNotFoundError(f"Missing MoGe depth for {image_path.name}: {moge_path}")
        with np.load(moge_path) as payload:
            moge_depth, moge_mask = remap_moge_depth(payload["depth"], payload["mask"], transform)
        scal3r_path = find_scal3r_depth(args.scal3r_result / "depths", index)
        scal3r_depth = read_scal3r_depth(scal3r_path)
        expected_shape = (transform.output_height, transform.output_width)
        if scal3r_depth.shape != expected_shape:
            raise ValueError(
                f"Scal3R depth {scal3r_path} is {scal3r_depth.shape}, expected {expected_shape} "
                "from the pinned preprocessing contract"
            )
        valid = (
            moge_mask
            & np.isfinite(moge_depth)
            & (moge_depth >= args.min_metric_depth_m)
            & (moge_depth <= args.max_metric_depth_m)
            & np.isfinite(scal3r_depth)
            & (scal3r_depth > 0)
        )
        ys, xs = np.nonzero(valid)
        votes = moge_depth[valid] / scal3r_depth[valid]
        raw_cells = occupied_grid_cells(valid, args.grid_rows, args.grid_columns)
        record: dict[str, Any] = {
            "index": index,
            "image": image_path.name,
            "moge_depth": moge_path.name,
            "scal3r_depth": scal3r_path.name,
            "raw_votes": int(len(votes)),
            "occupied_grid_cells": raw_cells,
        }
        keep = np.zeros(len(votes), dtype=bool)
        if len(votes) < args.min_votes:
            record["status"] = "insufficient_votes"
        elif raw_cells < args.min_spatial_cells:
            record["status"] = "insufficient_spatial_coverage"
        else:
            fit = robust_log_scale(votes, args.max_log_deviation)
            keep = fit.keep_mask
            kept_mask = np.zeros_like(valid)
            kept_mask[ys[keep], xs[keep]] = True
            kept_cells = occupied_grid_cells(kept_mask, args.grid_rows, args.grid_columns)
            record.update(
                {
                    "status": "accepted" if kept_cells >= args.min_spatial_cells else "insufficient_kept_spatial_coverage",
                    "kept_votes": fit.kept_count,
                    "rejected_votes": fit.input_count - fit.kept_count,
                    "kept_occupied_grid_cells": kept_cells,
                    "scale_m_per_scal3r_unit": fit.scale,
                    "vote_log_mad": fit.log_mad,
                }
            )
            if kept_cells >= args.min_spatial_cells:
                frame_scales.append(fit.scale)
                accepted_record_indices.append(len(frame_records))
        np.savez_compressed(
            votes_dir / f"{index:06d}_votes.npz",
            scale_m_per_scal3r_unit=votes.astype(np.float32),
            kept=keep,
            pixel_x=xs.astype(np.int32),
            pixel_y=ys.astype(np.int32),
        )
        frame_records.append(record)
    if not frame_scales:
        raise ValueError("No frame passed scale-vote and spatial-coverage checks")
    global_fit = robust_log_scale(np.asarray(frame_scales), args.max_log_deviation)
    for local_index, record_index in enumerate(accepted_record_indices):
        frame_records[record_index]["kept_for_global_fit"] = bool(global_fit.keep_mask[local_index])
    transform_records = [
        dict(zip(Scal3RImageTransform.__dataclass_fields__, values)) for values in sorted(transforms_seen)
    ]
    return {
        "schema_version": 1,
        "evaluation": "moge3_metric_camera_z_vs_scal3r_native_camera_z",
        "coordinate_system": "scal3r_native_gauge",
        "scale_m_per_scal3r_unit": global_fit.scale,
        "frames_before_global_rejection": global_fit.input_count,
        "frames_kept": global_fit.kept_count,
        "frame_scale_log_mad": global_fit.log_mad,
        "aggregation": "median in log space per frame, then equal-weight median across frames",
        "preprocessing_contract": {
            "scal3r_revision": "dc557e7be5ad821ed44b8ad37700311136061406",
            "proc_max_size": args.proc_max_size,
            "proc_align_size": args.proc_align_size,
            "center_crop": True,
            "transforms": transform_records,
            "input_order": "sorted unique file paths, matching pinned Scal3R collect_image_paths",
            "moge_inference_fov_x_degrees": moge_manifest.get("fov_x") if moge_manifest else None,
            "moge_depth_manifest": str(moge_manifest_path) if moge_manifest else None,
        },
        "frames": frame_records,
        "known_element_evaluation": {
            "status": "unavailable_missing_corresponding_endpoints_in_scal3r_geometry",
            "scale_coefficient_compared_with_published_vggt": False,
        },
        "limitations": [
            "MoGe is inferred on the source raster, then its depth is resampled onto Scal3R's exact image-space crop.",
            "This estimates the native Scal3R gauge only; it is not an independent geometric-accuracy score.",
            "The saved measured element cannot validate the new geometry until its corresponding endpoints are identified.",
        ],
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
