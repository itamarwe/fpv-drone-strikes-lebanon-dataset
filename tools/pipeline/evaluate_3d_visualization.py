#!/usr/bin/env python3
"""Score explicitly matched reconstruction renders without camera or resize guessing.

The input is a JSON manifest. Files are never paired by directory order and
renders are never resized. An evaluation-view record is accepted only when it
states that the target RGB was not used for photometric camera fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


VIEW_ROLES = {"input", "evaluation", "showcase", "exploratory"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lpips-device",
        choices=("off", "auto", "cpu", "cuda"),
        default="off",
        help="Compute LPIPS when its package is installed (default: off).",
    )
    return parser.parse_args()


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context}: missing required field {key!r}")
    return mapping[key]


def _resolve(root: Path, value: str, context: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{context}: file does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if mask.shape != shape:
        raise ValueError(
            f"mask dimensions {mask.shape[::-1]} do not match image dimensions {shape[::-1]}: {path}"
        )
    return mask


def _finite_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _psnr(reference: np.ndarray, render: np.ndarray, mask: np.ndarray | None) -> tuple[float | None, bool]:
    diff2 = (reference - render) ** 2
    if mask is not None:
        diff2 = diff2[mask]
    mse = float(np.mean(diff2))
    return (None, True) if mse == 0 else (-10.0 * math.log10(mse), False)


def _ssim(
    reference: np.ndarray,
    render: np.ndarray,
    mask: np.ndarray | None,
    structural_similarity: Any,
) -> float | None:
    if structural_similarity is None:
        return None
    score, spatial = structural_similarity(
        reference,
        render,
        channel_axis=2,
        data_range=1.0,
        full=True,
    )
    if mask is None:
        return float(score)
    if spatial.ndim == 3:
        spatial = np.mean(spatial, axis=2)
    return float(np.mean(spatial[mask]))


@dataclass
class LpipsScorer:
    model: Any
    torch: Any
    device: str

    @classmethod
    def create(cls, requested: str) -> "LpipsScorer | None":
        if requested == "off":
            return None
        try:
            import lpips  # type: ignore
            import torch
        except ImportError as error:
            raise RuntimeError(
                "LPIPS was requested but torch/lpips is unavailable; install lpips or use --lpips-device off"
            ) from error
        device = requested
        if requested == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--lpips-device cuda requested but CUDA is unavailable")
        return cls(model=lpips.LPIPS(net="alex", spatial=True).to(device).eval(), torch=torch, device=device)

    def score(self, reference: np.ndarray, render: np.ndarray, mask: np.ndarray | None) -> float:
        torch = self.torch
        ref = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).to(self.device)
        pred = torch.from_numpy(render).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            spatial = self.model(ref * 2 - 1, pred * 2 - 1)
        if mask is None:
            return float(spatial.mean().item())
        weights = torch.from_numpy(mask.astype(np.float32))[None, None].to(self.device)
        if weights.shape[-2:] != spatial.shape[-2:]:
            weights = torch.nn.functional.interpolate(weights, size=spatial.shape[-2:], mode="nearest")
        denom = weights.sum()
        if float(denom.item()) == 0:
            raise ValueError("LPIPS mask contains no valid pixels")
        return float((spatial * weights).sum().div(denom).item())


def _validate_camera(record: dict[str, Any], frame_id: str, role: str, root: Path, context: str) -> dict[str, Any]:
    camera = _require(record, "camera", context)
    if not isinstance(camera, dict):
        raise ValueError(f"{context}.camera must be an object")
    required = (
        "camera_id",
        "target_frame_id",
        "artifact",
        "record_id",
        "pose_source",
        "photometric_fit_target_rgb",
    )
    for key in required:
        _require(camera, key, f"{context}.camera")
    if camera["target_frame_id"] != frame_id:
        raise ValueError(
            f"{context}.camera.target_frame_id={camera['target_frame_id']!r} does not match frame_id={frame_id!r}"
        )
    if not isinstance(camera["photometric_fit_target_rgb"], bool):
        raise ValueError(f"{context}.camera.photometric_fit_target_rgb must be boolean")
    if role == "evaluation" and camera["photometric_fit_target_rgb"]:
        raise ValueError(
            f"{context}: evaluation camera used target RGB for photometric fitting; this is not a held-out score"
        )
    artifact = _resolve(root, str(camera["artifact"]), f"{context}.camera")
    actual_hash = _sha256(artifact)
    expected_hash = camera.get("artifact_sha256")
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError(f"{context}.camera artifact SHA-256 does not match: {artifact}")
    return {
        "camera_id": str(camera["camera_id"]),
        "target_frame_id": frame_id,
        "artifact": str(artifact),
        "artifact_sha256": actual_hash,
        "record_id": str(camera["record_id"]),
        "pose_source": str(camera["pose_source"]),
        "intrinsics_source": camera.get("intrinsics_source"),
        "photometric_fit_target_rgb": camera["photometric_fit_target_rgb"],
    }


def _region_slices(region: dict[str, Any], width: int, height: int, context: str) -> tuple[slice, slice]:
    if region.get("coordinate_space", "reference_pixels") != "reference_pixels":
        raise ValueError(f"{context}: only coordinate_space='reference_pixels' is supported")
    x, y, crop_width, crop_height = [
        int(_require(region, key, context)) for key in ("x", "y", "width", "height")
    ]
    if x < 0 or y < 0 or crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"{context}: crop bounds must be non-negative with positive width/height")
    if x + crop_width > width or y + crop_height > height:
        raise ValueError(f"{context}: crop extends outside {width}x{height} image")
    if min(crop_width, crop_height) < 7:
        raise ValueError(f"{context}: crop must be at least 7x7 for SSIM")
    return slice(y, y + crop_height), slice(x, x + crop_width)


def _metric_block(
    reference: np.ndarray,
    render: np.ndarray,
    mask: np.ndarray | None,
    structural_similarity: Any,
    lpips_scorer: LpipsScorer | None,
) -> dict[str, Any]:
    valid_pixels = int(mask.sum()) if mask is not None else int(reference.shape[0] * reference.shape[1])
    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "coverage_fraction": 0.0,
            "psnr_db": None,
            "perfect_match": False,
            "ssim": None,
            "lpips": None,
            "unavailable_reason": "empty_valid_mask",
        }
    psnr, perfect = _psnr(reference, render, mask)
    return {
        "valid_pixels": valid_pixels,
        "coverage_fraction": valid_pixels / float(reference.shape[0] * reference.shape[1]),
        "psnr_db": psnr,
        "perfect_match": perfect,
        "ssim": _ssim(reference, render, mask, structural_similarity),
        "lpips": lpips_scorer.score(reference, render, mask) if lpips_scorer else None,
    }


def evaluate_record(
    record: dict[str, Any],
    root: Path,
    seen_ids: set[str],
    structural_similarity: Any,
    lpips_scorer: LpipsScorer | None,
    index: int,
) -> dict[str, Any]:
    context = f"records[{index}]"
    frame_id = str(_require(record, "frame_id", context)).strip()
    if not frame_id or frame_id in seen_ids:
        raise ValueError(f"{context}: frame_id must be non-empty and unique: {frame_id!r}")
    seen_ids.add(frame_id)
    role = str(_require(record, "view_role", context))
    if role not in VIEW_ROLES:
        raise ValueError(f"{context}: view_role must be one of {sorted(VIEW_ROLES)}")
    reference_path = _resolve(root, str(_require(record, "reference", context)), context)
    render_path = _resolve(root, str(_require(record, "render", context)), context)
    if reference_path == render_path:
        raise ValueError(f"{context}: reference and render resolve to the same file")
    mapping = _require(record, "pixel_mapping", context)
    if not isinstance(mapping, dict) or mapping.get("type") != "identity":
        raise ValueError(
            f"{context}: scorer only accepts explicit pixel_mapping={{'type':'identity'}}; pre-warp outside it"
        )
    reference = _load_rgb(reference_path)
    render = _load_rgb(render_path)
    if reference.shape != render.shape:
        raise ValueError(
            f"{context}: identity-mapped dimensions differ: reference={reference.shape[1]}x{reference.shape[0]}, "
            f"render={render.shape[1]}x{render.shape[0]}; renders are never resized"
        )
    height, width = reference.shape[:2]
    declared_size = mapping.get("size")
    if declared_size is not None and list(declared_size) != [width, height]:
        raise ValueError(f"{context}: pixel_mapping.size must be [width,height]=[{width},{height}]")
    mask = None
    mask_path = record.get("valid_mask")
    if mask_path is not None:
        mask = _load_mask(_resolve(root, str(mask_path), context), (height, width))
    camera = _validate_camera(record, frame_id, role, root, context)
    full_frame = _metric_block(reference, render, None, structural_similarity, lpips_scorer)
    masked = _metric_block(reference, render, mask, structural_similarity, lpips_scorer) if mask is not None else None
    crops = []
    crop_ids: set[str] = set()
    for crop_index, crop in enumerate(record.get("crops", [])):
        crop_context = f"{context}.crops[{crop_index}]"
        if not isinstance(crop, dict):
            raise ValueError(f"{crop_context} must be an object")
        crop_id = str(_require(crop, "id", crop_context)).strip()
        if not crop_id or crop_id in crop_ids:
            raise ValueError(f"{crop_context}: crop id must be non-empty and unique")
        crop_ids.add(crop_id)
        ys, xs = _region_slices(crop, width, height, crop_context)
        crop_mask = mask[ys, xs] if mask is not None else None
        crops.append(
            {
                "id": crop_id,
                "bounds": {key: int(crop[key]) for key in ("x", "y", "width", "height")},
                "full_crop": _metric_block(reference[ys, xs], render[ys, xs], None, structural_similarity, lpips_scorer),
                "masked_crop": (
                    _metric_block(reference[ys, xs], render[ys, xs], crop_mask, structural_similarity, lpips_scorer)
                    if crop_mask is not None else None
                ),
            }
        )
    return {
        "frame_id": frame_id,
        "scene": record.get("scene"),
        "segment_id": record.get("segment_id"),
        "view_role": role,
        "reference": str(reference_path),
        "render": str(render_path),
        "resolution": [width, height],
        "camera": camera,
        "full_frame": full_frame,
        "masked": masked,
        "crops": crops,
    }


def _aggregate(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    blocks = [record[field] for record in records if record[field] is not None]
    return {
        "image_count": len(blocks),
        "mean_psnr_db": _finite_mean(block["psnr_db"] for block in blocks),
        "perfect_match_count": sum(bool(block["perfect_match"]) for block in blocks),
        "mean_ssim": _finite_mean(block["ssim"] for block in blocks),
        "mean_lpips": _finite_mean(block["lpips"] for block in blocks),
        "mean_coverage_fraction": _finite_mean(block["coverage_fraction"] for block in blocks),
    }


def _grouped(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = record.get(key)
        label = "unlabelled" if value is None else str(value)
        values.setdefault(label, []).append(record)
    return {
        label: {"full_frame": _aggregate(group, "full_frame"), "masked": _aggregate(group, "masked")}
        for label, group in sorted(values.items())
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.pairs_manifest.resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1:
        raise SystemExit("pairs manifest schema_version must be 1")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise SystemExit("pairs manifest must contain a non-empty records array")
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        structural_similarity = None
    seen_ids: set[str] = set()
    try:
        lpips_scorer = LpipsScorer.create(args.lpips_device)
        records = [
            evaluate_record(
                record,
                manifest_path.parent,
                seen_ids,
                structural_similarity,
                lpips_scorer,
                index,
            )
            for index, record in enumerate(raw_records)
        ]
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    summary = {
        "schema_version": 2,
        "status": "complete",
        "dataset_id": payload.get("dataset_id"),
        "method_run_id": payload.get("method_run_id"),
        "pairing": "explicit_manifest_only",
        "pixel_mapping": "identity_only_no_resize",
        "lpips_device": lpips_scorer.device if lpips_scorer else None,
        "ssim_available": structural_similarity is not None,
        "view_roles": {role: sum(record["view_role"] == role for record in records) for role in sorted(VIEW_ROLES)},
        "full_frame": _aggregate(records, "full_frame"),
        "masked": _aggregate(records, "masked"),
        "by_scene": _grouped(records, "scene"),
        "by_segment": _grouped(records, "segment_id"),
        "by_view_role": _grouped(records, "view_role"),
        "images": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
