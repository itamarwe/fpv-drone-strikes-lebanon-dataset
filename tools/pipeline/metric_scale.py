#!/usr/bin/env python3
"""Numerical helpers for metric-scale evaluation.

The helpers in this module deliberately keep a reconstruction's coordinate
system separate from metric units.  A scale coefficient is only comparable to
another coefficient when both refer to the same stored geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustLogFit:
    scale: float
    input_count: int
    kept_count: int
    log_mad: float
    keep_mask: np.ndarray


def robust_log_scale(values: np.ndarray, max_log_deviation: float) -> RobustLogFit:
    """Fit a positive scale in log space and return a mask over the input."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        raise ValueError("No positive finite scale votes")
    logs = np.log(values[valid])
    initial_center = float(np.median(logs))
    local_keep = np.abs(logs - initial_center) <= max_log_deviation
    if not np.any(local_keep):
        raise ValueError("All scale votes rejected")
    kept_logs = logs[local_keep]
    center = float(np.median(kept_logs))
    keep_mask = np.zeros(len(values), dtype=bool)
    keep_mask[np.flatnonzero(valid)[local_keep]] = True
    return RobustLogFit(
        scale=float(np.exp(center)),
        input_count=int(len(values)),
        kept_count=int(np.count_nonzero(local_keep)),
        log_mad=float(np.median(np.abs(kept_logs - center))),
        keep_mask=keep_mask,
    )


def vggt_supported_crop(width: int, height: int) -> tuple[int, int, int, int]:
    """Return VGGT-Omega's centered aspect-ratio crop as left/top/width/height."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    aspect_ratio = height / width
    if aspect_ratio < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        return (max((width - crop_width) // 2, 0), 0, crop_width, height)
    if aspect_ratio > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        return (0, max((height - crop_height) // 2, 0), width, crop_height)
    return (0, 0, width, height)


def validate_camera_basis(
    right: np.ndarray,
    down: np.ndarray,
    forward: np.ndarray,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    """Validate and describe an OpenCV camera basis (x right, y down, z forward)."""

    matrix = np.column_stack((right, down, forward)).astype(np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Camera basis must contain three finite 3-vectors")
    gram = matrix.T @ matrix
    orthogonality_error = float(np.max(np.abs(gram - np.eye(3))))
    determinant = float(np.linalg.det(matrix))
    if orthogonality_error > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError(
            "Camera basis is not an orthonormal right/down/forward basis: "
            f"orthogonality_error={orthogonality_error:.3g}, determinant={determinant:.6g}"
        )
    return {"orthogonality_error": orthogonality_error, "determinant": determinant}


def project_camera_z(
    points: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    focal_x_px: float,
    focal_y_px: float | None = None,
    principal_x_px: float | None = None,
    principal_y_px: float | None = None,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer world points into a camera-z depth map."""

    if focal_x_px <= 0 or (focal_y_px is not None and focal_y_px <= 0):
        raise ValueError("Focal lengths must be positive")
    validate_camera_basis(right, down, forward)
    fy = focal_x_px if focal_y_px is None else focal_y_px
    cx = width / 2.0 if principal_x_px is None else principal_x_px
    cy = height / 2.0 if principal_y_px is None else principal_y_px
    relative = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    x = relative @ np.asarray(right, dtype=np.float64)
    y = relative @ np.asarray(down, dtype=np.float64)
    z = relative @ np.asarray(forward, dtype=np.float64)
    finite_front = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > min_depth)
    x, y, z = x[finite_front], y[finite_front], z[finite_front]
    u = np.rint(cx + focal_x_px * x / z).astype(np.int64)
    v = np.rint(cy + fy * y / z).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    depth = np.full((height, width), np.inf, dtype=np.float64)
    np.minimum.at(depth, (v, u), z)
    valid = np.isfinite(depth)
    depth[~valid] = np.nan
    return depth, valid


def crop_depth_to_vggt_view(
    depth: np.ndarray,
    mask: np.ndarray,
    source_size: tuple[int, int],
    crop: tuple[int, int, int, int],
    layout: str = "auto",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Normalize either a full-source or already-cropped depth map to the VGGT crop."""

    depth = np.asarray(depth)
    mask = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Depth and mask must be matching 2D arrays")
    source_width, source_height = source_size
    left, top, crop_width, crop_height = crop
    full_shape = (source_height, source_width)
    crop_shape = (crop_height, crop_width)
    if layout == "auto":
        matches_full = depth.shape == full_shape
        matches_crop = depth.shape == crop_shape
        if matches_full and not matches_crop:
            layout = "full"
        elif matches_crop and not matches_full:
            layout = "crop"
        elif matches_full and matches_crop:
            layout = "full"
        else:
            raise ValueError(
                f"Depth shape {depth.shape} matches neither source {full_shape} nor crop {crop_shape}"
            )
    if layout == "full":
        if depth.shape != full_shape:
            raise ValueError(f"Full depth shape must be {full_shape}, got {depth.shape}")
        row_slice = slice(top, top + crop_height)
        column_slice = slice(left, left + crop_width)
        return depth[row_slice, column_slice], mask[row_slice, column_slice], layout
    if layout == "crop":
        if depth.shape != crop_shape:
            raise ValueError(f"Cropped depth shape must be {crop_shape}, got {depth.shape}")
        return depth, mask, layout
    raise ValueError(f"Unknown depth layout: {layout}")


def occupied_grid_cells(mask: np.ndarray, rows: int = 4, columns: int = 8) -> int:
    """Count image-grid cells containing at least one valid correspondence."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or rows <= 0 or columns <= 0:
        raise ValueError("Expected a 2D mask and positive grid dimensions")
    occupied = 0
    for row_indices in np.array_split(np.arange(mask.shape[0]), rows):
        for column_indices in np.array_split(np.arange(mask.shape[1]), columns):
            if len(row_indices) and len(column_indices) and np.any(mask[np.ix_(row_indices, column_indices)]):
                occupied += 1
    return occupied


def intrinsics_pixels(intrinsics: np.ndarray, width: int, height: int) -> dict[str, float]:
    """Convert MoGe normalized intrinsics to pixels."""

    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite normalized 3x3 intrinsics")
    return {
        "fx_px": float(matrix[0, 0] * width),
        "fy_px": float(matrix[1, 1] * height),
        "cx_px": float(matrix[0, 2] * width),
        "cy_px": float(matrix[1, 2] * height),
    }
