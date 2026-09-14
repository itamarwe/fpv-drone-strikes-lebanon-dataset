#!/usr/bin/env python3
"""Inventory actual two-scene outputs, preserving immutable attempt history."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image


SCENES = ("scene_a", "scene_b")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
GEOMETRY_SUFFIXES = {".ply", ".obj", ".glb", ".gltf", ".stl"}


@dataclass(frozen=True)
class RunSpec:
    method: str
    mode: str
    profile: str
    primary_any: tuple[str, ...]
    contract_note: str


RUN_SPECS = (
    RunSpec("published_vggt", "current_product", "full", ("points_positions.bin",),
            "Published point buffers are a current-product baseline; camera-matched RGB renders need a separate export."),
    RunSpec("vggt_matched", "fresh_control", "common16", ("*.ply", "*.glb", "*.npz"),
            "Fresh identical-input VGGT control; preserve cameras, preprocessing, depths and geometry."),
    RunSpec("vggt_matched", "fresh_control", "dense_train", ("*.ply", "*.glb", "*.npz"),
            "Fresh dense-training VGGT control; evaluation views and temporal buffers must remain excluded."),
    RunSpec("moge3_baseline", "estimated_focal", "common16", ("metric_scale.json",),
            "Projection-derived scale sensitivity evidence, not independently surveyed metric accuracy."),
    RunSpec("moge3_baseline", "diagnostic_fixed_focal", "common16", ("metric_scale.json",),
            "Fixed-focal sensitivity condition; diagnostic evidence, not a visually comparable reconstruction."),
    RunSpec("scal3r_zju", "native", "common16", ("whole.ply", "mat.txt", "intri.yml"),
            "Common16 pilot: native RGB point cloud, cameras, intrinsics, masks and depth; no novel-view appearance renderer."),
    RunSpec("scal3r_zju", "native", "dense_train", ("whole.ply", "mat.txt", "intri.yml"),
            "Native RGB point cloud, cameras, intrinsics, masks and optional depth; no novel-view appearance renderer."),
    RunSpec("moge3_glomap", "native", "common16", ("points3D.txt", "points3D.bin", "*.ply"),
            "Common16 pilot: sparse scaffold and metric-depth scale evidence; not a photorealistic representation by itself."),
    RunSpec("moge3_glomap", "native", "dense_train", ("points3D.txt", "points3D.bin", "*.ply"),
            "Sparse scaffold and metric-depth scale evidence; not a photorealistic representation by itself."),
    RunSpec("surflo", "plain", "common16", ("final.ply",),
            "Oriented surface points only. Normal colours are a geometry diagnostic, not RGB appearance."),
    RunSpec("surflo", "guided_default", "common16", ("mesh.ply", "mesh_textured.ply", "mesh_textured.glb"),
            "Review neutral mesh and RGB/textured output separately; pinned CLI has no saved matched-view render set."),
    RunSpec("querysplat", "feed_forward", "common16", ("gaussians*.ply",),
            "Native renders are predicted input views only. vggt_depth_pointcloud.ply is the same-run VGGT-Omega geometry control, not QuerySplat output."),
    RunSpec("querysplat", "tto", "common16", ("gaussians*.ply",),
            "TTO fits the input RGB views; native renders remain in-sample, not held-out."),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(path: Path) -> str:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if name in {"points_positions.bin", "points_colors.bin"}:
        return "published_point_buffer"
    if path.suffix.lower() in GEOMETRY_SUFFIXES:
        if "gaussian" in name:
            return "gaussian_native"
        if "mesh" in name or path.suffix.lower() in {".obj", ".glb", ".gltf", ".stl"}:
            return "mesh_native"
        return "point_cloud_native"
    if path.suffix.lower() in IMAGE_SUFFIXES:
        if "rendered" in parts or "render" in name:
            return "rgb_render"
        if "depth" in name:
            return "depth_preview"
        if "normal" in name:
            return "normal_preview"
        if "input_frames" in parts:
            return "preprocessed_input"
        return "review_preview"
    if "camera" in name or name in {"images.txt", "cameras.txt", "mat.txt", "intri.yml", "extri.yml"}:
        return "camera_metadata"
    if "metric" in name or "score" in name or "evaluation" in name:
        return "evaluation"
    if "timing" in name or "summary" in name or path.suffix.lower() in {".json", ".npz", ".yaml", ".yml"}:
        return "run_metadata"
    if path.suffix.lower() in {".log", ".txt"}:
        return "log_or_text"
    return "other"


def artifact_record(path: Path, root: Path) -> dict:
    stat = path.stat()
    kind = classify(path)
    record = {
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(root.resolve())),
        "kind": kind,
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if path.suffix.lower() in IMAGE_SUFFIXES:
        try:
            with Image.open(path) as image:
                record["image_size"] = list(image.size)
        except OSError as error:
            record["read_error"] = str(error)
    if stat.st_size <= 64 * 1024 * 1024 and kind in {
        "rgb_render", "depth_preview", "normal_preview", "review_preview",
        "camera_metadata", "evaluation", "run_metadata",
    }:
        record["sha256"] = sha256(path)
    return record


def is_latest_alias(path: Path) -> bool:
    return "latest" in path.parts


def files_under(roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and not is_latest_alias(root):
            files.add(root.resolve())
        elif root.is_dir():
            files.update(path.resolve() for path in root.rglob("*") if path.is_file() and not is_latest_alias(path))
    return sorted(files)


def any_pattern(files: list[Path], patterns: tuple[str, ...]) -> bool:
    return any(any(fnmatch.fnmatch(path.name, pattern) or path.match(pattern) for pattern in patterns) for path in files)


def comparison_status(method: str, mode: str, files: list[Path]) -> tuple[str, list[str]]:
    names = {path.name for path in files}
    kinds = {classify(path) for path in files}
    warnings: list[str] = []
    if not files:
        return "not_reviewable", warnings
    if method == "moge3_baseline":
        warnings.append("metric-scale diagnostic only; do not interpret as independent survey accuracy")
        return ("metric_evidence_ready" if "metric_scale.json" in names else "not_reviewable"), warnings
    if method == "published_vggt":
        ready = {"points_positions.bin", "points_colors.bin"}.issubset(names)
        warnings.append("raw viewer point buffers need the published viewer or an explicit converter for review")
        return ("geometry_ready" if ready else "not_reviewable"), warnings
    if method == "querysplat":
        gaussian_names = {name for name in names if name.startswith("gaussians") and name.endswith(".ply")}
        required = {"predicted_input_cameras.json", "inference_timing.json"}
        has_native_pairs = bool(gaussian_names) and "predicted_input_cameras.json" in names and "rgb_render" in kinds and "preprocessed_input" in kinds
        if not required.issubset(names):
            warnings.append(f"missing recommended exports: {sorted(required - names)}")
        if not gaussian_names:
            warnings.append("missing Gaussian PLY export")
        if "vggt_depth_pointcloud.ply" in names:
            warnings.append("vggt_depth_pointcloud.ply is a VGGT-Omega geometry control and must be shown separately")
        warnings.append("pinned native renders use predicted input cameras; held-out appearance is unavailable")
        if mode == "tto":
            warnings.append("TTO uses input RGB photometric fitting")
        if has_native_pairs:
            return "input_view_appearance_ready", warnings
        if gaussian_names or "vggt_depth_pointcloud.ply" in names:
            return "geometry_only", warnings
        return "not_reviewable", warnings
    if method == "surflo":
        if mode == "plain":
            warnings.append("normal-coloured points are not an RGB appearance result")
            return ("geometry_ready" if "final.ply" in names else "not_reviewable"), warnings
        has_mesh = bool({"mesh.ply", "mesh_textured.ply", "mesh_textured.glb"} & names)
        has_texture = bool({"mesh_textured.ply", "mesh_textured.glb"} & names)
        warnings.append("pinned inference does not export matched-view renders or camera metadata")
        for summary_path in (path for path in files if path.name == "_infer_summary.json"):
            try:
                summary = json.loads(summary_path.read_text())
                texture = summary.get("scene", summary)
                if texture.get("texture_n_iterations") == 0:
                    warnings.append("vertex colours use TSDF initialization with 0 optimization iterations")
                init = str(texture.get("texture_init_color", ""))
                if "fallback=1.0" in init:
                    warnings.append("texture initialization used the relaxed TSDF fallback (factor 1.0)")
            except (OSError, json.JSONDecodeError):
                pass
        for log_path in (path for path in files if path.name == "run.log"):
            try:
                empty_masks = log_path.read_text(errors="replace").count("VGGT-conf mask is empty")
                if empty_masks:
                    warnings.append(f"normal guidance fell back to full-image depth alignment for {empty_masks} view(s)")
            except OSError:
                pass
        return ("geometry_and_texture_ready" if has_mesh and has_texture else "geometry_ready" if has_mesh else "not_reviewable"), warnings
    if method == "moge3_glomap":
        ready = bool({"points3D.txt", "points3D.bin"} & names)
        warnings.append("sparse geometry is a scaffold/registration diagnostic, not dense appearance reconstruction")
        return ("sparse_geometry_ready" if ready else "not_reviewable"), warnings
    if "rgb_render" in kinds and "camera_metadata" in kinds:
        return "camera_linked_renders_present", warnings
    if kinds & {"mesh_native", "point_cloud_native", "gaussian_native"}:
        return "geometry_ready", warnings
    return "not_reviewable", warnings


def infer_mode(payload: dict, method: str, run_json: Path) -> str:
    if payload.get("mode"):
        return str(payload["mode"])
    if method == "surflo" and payload.get("surflo_mode"):
        value = str(payload["surflo_mode"])
        return "guided_default" if value == "guided" and (run_json.parent / "guided_default").is_dir() else value
    if method == "querysplat" and payload.get("querysplat_mode"):
        return str(payload["querysplat_mode"])
    if method == "moge3_baseline":
        return run_json.parent.name
    return "native"


def discover_attempts(results_root: Path) -> list[dict]:
    attempts = []
    if not results_root.exists():
        return attempts
    for run_json in sorted(results_root.rglob("run.json")):
        if is_latest_alias(run_json):
            continue
        relative = run_json.relative_to(results_root)
        parts = relative.parts
        attempt_indexes = [index for index, part in enumerate(parts) if part.startswith("attempt-")]
        if len(parts) < 5 or not attempt_indexes:
            continue
        attempt_index = attempt_indexes[0]
        if attempt_index < 3:
            continue
        scene, profile, method = parts[0:3]
        try:
            payload = json.loads(run_json.read_text())
        except (OSError, json.JSONDecodeError) as error:
            payload = {"status": "invalid_metadata", "metadata_error": str(error)}
        mode = infer_mode(payload, method, run_json)
        scope = run_json.parent
        files = files_under([scope])
        attempts.append({
            "scene": str(payload.get("scene", scene)),
            "profile": str(payload.get("profile", profile)),
            "method": str(payload.get("method", method)),
            "mode": mode,
            "attempt_id": parts[attempt_index],
            "attempt_scope": str(scope.resolve()),
            "metadata_path": str(run_json.resolve()),
            "status": str(payload.get("status", "unknown")),
            "exit_code": payload.get("exit_code"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "started_at_utc": payload.get("started_at_utc"),
            "ended_at_utc": payload.get("ended_at_utc"),
            "input_image_count": payload.get("input_image_count"),
            "upstream_revision": payload.get("upstream_revision") or payload.get("revision"),
            "error": payload.get("error") or payload.get("failure_reason"),
            "metadata": payload,
            "_files": files,
        })
    return attempts


def public_attempt(attempt: dict, results_root: Path) -> dict:
    result = {key: value for key, value in attempt.items() if key != "_files"}
    result["artifacts"] = [artifact_record(path, results_root) for path in attempt["_files"]]
    return result


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    bundle_root = args.bundle_root.resolve() if args.bundle_root else None
    discovered = discover_attempts(results_root)
    assigned: set[Path] = set()
    runs = []
    for scene in SCENES:
        for spec in RUN_SPECS:
            matched = [attempt for attempt in discovered if (
                attempt["scene"], attempt["profile"], attempt["method"], attempt["mode"]
            ) == (scene, spec.profile, spec.method, spec.mode)]
            external_files: list[Path] = []
            searched_roots = [results_root / scene / spec.profile / spec.method]
            if spec.method == "published_vggt" and bundle_root is not None:
                published_root = bundle_root / "scenes" / scene / "baseline"
                external_files = files_under([published_root])
                searched_roots.append(published_root)
            files = sorted({path for attempt in matched for path in attempt["_files"]} | set(external_files))
            assigned.update(path for attempt in matched for path in attempt["_files"])
            completed_primary = any(
                attempt["status"] == "complete" and any_pattern(attempt["_files"], spec.primary_any)
                for attempt in matched
            ) or (not matched and bool(external_files) and any_pattern(external_files, spec.primary_any))
            if completed_primary:
                status = "outputs_available"
            elif matched and all(attempt["status"] in {"failed", "error", "oom", "invalid_metadata"} for attempt in matched):
                status = "failed"
            elif matched or external_files:
                status = "partial"
            else:
                status = "not_run"
            review_status, warnings = comparison_status(spec.method, spec.mode, files)
            runs.append({
                "scene": scene,
                "profile": spec.profile,
                "method": spec.method,
                "mode": spec.mode,
                "run_status": status,
                "review_status": review_status,
                "contract_note": spec.contract_note,
                "searched_roots": [str(root.resolve()) for root in searched_roots],
                "warnings": warnings,
                "attempts": [public_attempt(attempt, results_root) for attempt in matched],
                "artifacts": [artifact_record(path, results_root if results_root in path.parents else bundle_root) for path in files],
            })
    all_result_files = files_under([results_root]) if results_root.exists() else []
    unassigned = [artifact_record(path, results_root) for path in all_result_files if path not in assigned]
    status_names = ("outputs_available", "partial", "failed", "not_run")
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "bundle_root": str(bundle_root) if bundle_root else None,
        "discovery_policy": {
            "run_metadata": "Every immutable attempt-*/**/run.json is retained.",
            "latest_aliases": "Directories named latest are ignored to avoid double-counting copied aliases.",
        },
        "summary": {
            "expected_runs": len(runs),
            **{name: sum(run["run_status"] == name for run in runs) for name in status_names},
            "discovered_attempts": len(discovered),
            "completed_attempts": sum(attempt["status"] == "complete" for attempt in discovered),
            "failed_attempts": sum(attempt["status"] in {"failed", "error", "oom"} for attempt in discovered),
            "unassigned_file_count": len(unassigned),
        },
        "runs": runs,
        "unassigned_artifacts": unassigned,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
