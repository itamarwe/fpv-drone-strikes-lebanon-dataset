#!/usr/bin/env python3
"""Render frozen QuerySplat Gaussians at proxy evaluation cameras.

The evaluation images are used only by a separate VGGT-Omega camera-head pass.
They are never sent to QuerySplat's scene encoder or TTO. Cameras from that
joint pose pass are aligned to the frozen training-camera frame using common
training cameras. The output records alignment residuals and remains a proxy
camera evaluation, not independent pose ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--querysplat-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gaussians-ply", type=Path, required=True)
    parser.add_argument(
        "--inference-timing",
        type=Path,
        help="Defaults to inference_timing.json next to the Gaussian PLY; must prove an opacity-0 export.",
    )
    parser.add_argument("--frozen-cameras", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-frames-csv", type=Path, required=True)
    parser.add_argument("--evaluation-images", type=Path, required=True)
    parser.add_argument("--evaluation-frames-csv", type=Path, required=True)
    parser.add_argument("--scene", choices=("scene_a", "scene_b"), required=True)
    parser.add_argument("--method-run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    parser.add_argument("--max-center-rmse-fraction", type=float, default=0.10)
    parser.add_argument("--max-orientation-median-deg", type=float, default=10.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_profile(csv_path: Path, images_dir: Path, scope: str) -> list[dict]:
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["profile_index"]))
    records = []
    for row in rows:
        image = images_dir / row["profile_file"]
        if not image.is_file():
            raise ValueError(f"Missing {scope} image: {image}")
        records.append(
            {
                "frame_id": f"{scope}:source_{int(row['source_index']):06d}",
                "profile_name": Path(row["profile_file"]).stem,
                "source_index": int(row["source_index"]),
                "source_file": row["source_file"],
                "segment_id": row["segment_id"],
                "image": image,
            }
        )
    if not records:
        raise ValueError(f"No rows in {csv_path}")
    return records


def preprocess(paths: list[Path], options, image_transform_class) -> torch.Tensor:
    transform = image_transform_class(crop_size=options.img_size, sample_size=options.img_size, max_crop=True)
    images = []
    for path in paths:
        with Image.open(path) as source:
            array = np.array(source.convert("RGB"), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        tensor, _, _, _ = transform.preprocess_images(tensor.unsqueeze(0))
        images.append(tensor[0])
    return torch.stack(images)


def c2w_from_cam_view(cam_view: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.swapaxes(cam_view, -2, -1))


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def align_cameras(joint_c2w: np.ndarray, frozen_c2w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    if len(joint_c2w) < 4 or joint_c2w.shape != frozen_c2w.shape:
        raise ValueError("Need at least four one-to-one common cameras for proxy alignment")
    # Camera orientations constrain the otherwise weak rotation about a nearly
    # linear flight path. Project their mean relative rotation back to SO(3).
    relative_rotations = [frozen[:3, :3] @ joint[:3, :3].T for joint, frozen in zip(joint_c2w, frozen_c2w)]
    rotation = project_rotation(np.sum(relative_rotations, axis=0))
    joint_centers = joint_c2w[:, :3, 3]
    frozen_centers = frozen_c2w[:, :3, 3]
    rotated = (rotation @ joint_centers.T).T
    source_center = rotated.mean(axis=0)
    target_center = frozen_centers.mean(axis=0)
    source_delta = rotated - source_center
    target_delta = frozen_centers - target_center
    denominator = float(np.sum(source_delta * source_delta))
    if denominator <= 1e-12:
        raise ValueError("Joint common camera centers have zero usable extent")
    scale = float(np.sum(source_delta * target_delta) / denominator)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid proxy alignment scale: {scale}")
    translation = target_center - scale * source_center
    predicted = scale * rotated + translation
    residuals = np.linalg.norm(predicted - frozen_centers, axis=1)
    path_extent = float(np.linalg.norm(frozen_centers.max(axis=0) - frozen_centers.min(axis=0)))
    if path_extent <= 1e-12:
        raise ValueError("Frozen common camera path has zero extent")
    orientation_errors = [
        rotation_angle_deg(frozen[:3, :3].T @ rotation @ joint[:3, :3])
        for joint, frozen in zip(joint_c2w, frozen_c2w)
    ]
    diagnostics = {
        "fit_camera_count": len(joint_c2w),
        "scale_joint_to_frozen": scale,
        "rotation_joint_to_frozen": rotation.tolist(),
        "translation_joint_to_frozen": translation.tolist(),
        "frozen_camera_path_extent": path_extent,
        "center_rmse": float(np.sqrt(np.mean(residuals**2))),
        "center_rmse_fraction_of_path_extent": float(np.sqrt(np.mean(residuals**2)) / path_extent),
        "center_residuals": residuals.tolist(),
        "orientation_error_degrees": orientation_errors,
        "orientation_median_degrees": float(np.median(orientation_errors)),
    }
    return scale, rotation, translation, diagnostics


def transform_c2w(c2w: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transformed = np.eye(4, dtype=np.float32)
    transformed[:3, :3] = rotation @ c2w[:3, :3]
    transformed[:3, 3] = scale * (rotation @ c2w[:3, 3]) + translation
    return transformed


def numeric_fields(names: tuple[str, ...], prefix: str) -> list[str]:
    fields = [name for name in names if name.startswith(prefix)]
    return sorted(fields, key=lambda value: int(value.rsplit("_", 1)[1]))


def load_gaussians(path: Path, device: torch.device) -> torch.Tensor:
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = vertex.dtype.names or ()
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    if not required.issubset(names):
        raise ValueError(f"Gaussian PLY missing fields: {sorted(required - set(names))}")
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")])
    opacity_logits = np.clip(np.asarray(vertex["opacity"], dtype=np.float32), -80, 80)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logits))
    scales = np.exp(np.column_stack([vertex[f"scale_{index}"] for index in range(3)]))
    rotations = np.column_stack([vertex[f"rot_{index}"] for index in range(4)])
    dc_names = numeric_fields(names, "f_dc_")
    rest_names = numeric_fields(names, "f_rest_")
    if len(dc_names) != 3 or len(rest_names) % 3:
        raise ValueError("Gaussian PLY has an unsupported spherical-harmonic layout")
    dc = np.column_stack([vertex[name] for name in dc_names])[:, None, :]
    rest_flat = np.column_stack([vertex[name] for name in rest_names])
    rest = rest_flat.reshape(len(vertex), 3, -1).transpose(0, 2, 1)
    colors = np.concatenate([dc, rest], axis=1).reshape(len(vertex), -1)
    activated = np.concatenate([xyz, opacity[:, None], scales, rotations, colors], axis=1).astype(np.float32)
    return torch.from_numpy(activated)[None].to(device)


def background(options, device: torch.device) -> torch.Tensor:
    if options.bg_color == "grey":
        return torch.full((3,), 0.5, device=device)
    if options.bg_color == "white":
        return torch.ones(3, device=device)
    if options.bg_color == "black":
        return torch.zeros(3, device=device)
    raise ValueError(f"Unsupported background: {options.bg_color}")


def save_rgb(tensor: torch.Tensor, path: Path):
    array = (tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def main() -> int:
    args = parse_args()
    if not 0 < args.alpha_threshold < 1:
        raise SystemExit("--alpha-threshold must be between 0 and 1")
    repo = args.querysplat_repo.resolve()
    sys.path.insert(0, str(repo))
    from scripts.models.vggt_encoder import VGGTEncoder
    from scripts.options import load_options_yaml
    from scripts.rendering.gs import GaussianRenderer
    from scripts.utils.data import ImageTransform

    train = read_profile(args.train_frames_csv.resolve(), args.train_images.resolve(), "train")
    evaluation = read_profile(args.evaluation_frames_csv.resolve(), args.evaluation_images.resolve(), "evaluation")
    frozen_payload = json.loads(args.frozen_cameras.read_text())
    gaussian_path = args.gaussians_ply.resolve()
    timing_path = (args.inference_timing or gaussian_path.parent / "inference_timing.json").resolve()
    if not timing_path.is_file():
        raise SystemExit(f"Missing inference timing/provenance export: {timing_path}")
    timing_payload = json.loads(timing_path.read_text())
    thresholds = timing_payload.get("gaussian_save_opacity_thresholds")
    if not isinstance(thresholds, list) or not any(float(value) == 0 for value in thresholds):
        raise SystemExit(
            "Evaluation rendering requires a lossless opacity-0 Gaussian export. "
            "Rerun QuerySplat with --gaussian_save_opacity_threshold 0 (or 0 0.05)."
        )
    frozen_by_name = {frame["name"]: frame for frame in frozen_payload.get("frames", [])}
    missing = [record["profile_name"] for record in train if record["profile_name"] not in frozen_by_name]
    if missing:
        raise SystemExit(f"Frozen camera export does not map all training frames: {missing}")
    options = load_options_yaml(args.config.resolve())
    all_records = train + evaluation
    images = preprocess([record["image"] for record in all_records], options, ImageTransform)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required by the pinned QuerySplat/VGGT-Omega release")
    encoder = VGGTEncoder(options).to(device).eval()
    with torch.inference_mode():
        outputs, _, _ = encoder._run_aggregator_with_images(images[None].to(device))
        final = outputs[encoder.aggregator.depth - 1]
        if final is None:
            raise RuntimeError("VGGT-Omega final camera layer is missing")
        joint_cam_view, joint_intrinsics, _ = encoder._decode_cameras(final, options.img_size)
    joint_cam_view_np = joint_cam_view[0].float().cpu().numpy()
    joint_c2w = c2w_from_cam_view(joint_cam_view_np)
    frozen_train_c2w = np.asarray([frozen_by_name[record["profile_name"]]["c2w"] for record in train], dtype=np.float64)
    scale, rotation, translation, diagnostics = align_cameras(joint_c2w[: len(train)], frozen_train_c2w)
    if diagnostics["center_rmse_fraction_of_path_extent"] > args.max_center_rmse_fraction:
        raise SystemExit(
            f"Proxy camera alignment center RMSE fraction {diagnostics['center_rmse_fraction_of_path_extent']:.4f} "
            f"exceeds {args.max_center_rmse_fraction:.4f}"
        )
    if diagnostics["orientation_median_degrees"] > args.max_orientation_median_deg:
        raise SystemExit(
            f"Proxy camera alignment median orientation error {diagnostics['orientation_median_degrees']:.3f}° "
            f"exceeds {args.max_orientation_median_deg:.3f}°"
        )
    output_dir = args.output_dir.resolve()
    reference_dir = output_dir / "reference_preprocessed"
    render_dir = output_dir / "rendered"
    mask_dir = output_dir / "valid_masks"
    depth_dir = output_dir / "depth"
    for directory in (reference_dir, render_dir, mask_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)
    gaussians = load_gaussians(gaussian_path, device)
    renderer = GaussianRenderer(options)
    evaluation_cameras = []
    pair_records = []
    eval_start = len(train)
    for local_index, record in enumerate(evaluation):
        joint_index = eval_start + local_index
        aligned_c2w = transform_c2w(joint_c2w[joint_index], scale, rotation, translation)
        aligned_w2c = np.linalg.inv(aligned_c2w)
        cam_view = torch.from_numpy(aligned_w2c.T.astype(np.float32))[None, None].to(device)
        intrinsics = joint_intrinsics[:, joint_index : joint_index + 1].float()
        with torch.inference_mode():
            rendered = renderer.render(gaussians, cam_view, background(options, device), intrinsics)
        safe_id = record["frame_id"].replace(":", "_")
        reference_path = reference_dir / f"{safe_id}.png"
        render_path = render_dir / f"{safe_id}.png"
        mask_path = mask_dir / f"{safe_id}.png"
        depth_path = depth_dir / f"{safe_id}.npz"
        save_rgb(images[joint_index], reference_path)
        save_rgb(rendered["images_pred"][0, 0], render_path)
        alpha = rendered["alphas_pred"][0, 0, 0].detach().cpu().numpy()
        Image.fromarray((alpha >= args.alpha_threshold).astype(np.uint8) * 255).save(mask_path)
        np.savez_compressed(depth_path, depth=rendered["depths_pred"][0, 0, 0].detach().cpu().numpy(), alpha=alpha)
        evaluation_cameras.append(
            {
                "frame_id": record["frame_id"],
                "source_index": record["source_index"],
                "source_file": record["source_file"],
                "segment_id": record["segment_id"],
                "joint_pose_index": joint_index,
                "joint_c2w": joint_c2w[joint_index].tolist(),
                "aligned_c2w": aligned_c2w.tolist(),
                "aligned_w2c": aligned_w2c.tolist(),
                "querysplat_cam_view": aligned_w2c.T.tolist(),
                "intrinsics_fx_fy_cx_cy": joint_intrinsics[0, joint_index].float().cpu().tolist(),
            }
        )
        pair_records.append(
            {
                "frame_id": record["frame_id"],
                "scene": args.scene,
                "segment_id": record["segment_id"],
                "view_role": "evaluation",
                "reference": os.path.relpath(reference_path, output_dir),
                "render": os.path.relpath(render_path, output_dir),
                "valid_mask": os.path.relpath(mask_path, output_dir),
                "pixel_mapping": {"type": "identity", "size": list(options.img_size[::-1])},
                "camera": {
                    "camera_id": f"proxy-evaluation:{record['frame_id']}",
                    "target_frame_id": record["frame_id"],
                    "artifact": "evaluation_cameras.json",
                    "record_id": f"evaluation_cameras[{local_index}]",
                    "pose_source": "joint_vggt_omega_pose_proxy_aligned_on_train16",
                    "intrinsics_source": "joint_vggt_omega_pose_proxy_512px_center_crop",
                    "photometric_fit_target_rgb": False,
                    "camera_localization_target_rgb": True,
                },
            }
        )
    camera_artifact = {
        "schema_version": 1,
        "status": "proxy_evaluation_cameras",
        "warning": "Evaluation RGB was used by the joint VGGT-Omega camera pass. It was not used by QuerySplat scene encoding or TTO. Poses are not independent ground truth.",
        "frozen_gaussians": str(gaussian_path),
        "frozen_gaussians_sha256": sha256(gaussian_path),
        "inference_timing": str(timing_path),
        "inference_timing_sha256": sha256(timing_path),
        "gaussian_export_opacity_thresholds": thresholds,
        "frozen_train_cameras": str(args.frozen_cameras.resolve()),
        "frozen_train_cameras_sha256": sha256(args.frozen_cameras.resolve()),
        "pose_pass": "VGGT-Omega aggregator + camera head on train16 and evaluation images",
        "alignment_fit_scope": "train16 camera correspondences only",
        "alignment_thresholds": {
            "max_center_rmse_fraction": args.max_center_rmse_fraction,
            "max_orientation_median_degrees": args.max_orientation_median_deg,
        },
        "alignment": diagnostics,
        "evaluation_cameras": evaluation_cameras,
    }
    camera_path = output_dir / "evaluation_cameras.json"
    camera_path.write_text(json.dumps(camera_artifact, indent=2, allow_nan=False) + "\n")
    camera_hash = sha256(camera_path)
    for record in pair_records:
        record["camera"]["artifact_sha256"] = camera_hash
    pairs = {
        "schema_version": 1,
        "dataset_id": "fpv_3d_pipeline_two_scene_2026_09",
        "method_run_id": args.method_run_id,
        "records": pair_records,
    }
    (output_dir / "image_pairs.evaluation_proxy.json").write_text(json.dumps(pairs, indent=2) + "\n")
    print(json.dumps({"evaluation_views": len(evaluation), "alignment": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
