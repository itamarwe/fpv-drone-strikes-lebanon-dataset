#!/usr/bin/env python3
"""Create an explicit input-view scoring manifest from real QuerySplat exports.

This intentionally emits ``view_role=input``.  The pinned QuerySplat inference
command renders its predicted input cameras; those images are not held-out
views, even when they happen to share names with another benchmark split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", choices=("feed_forward", "tto"), required=True)
    parser.add_argument("--output", type=Path, help="Defaults to RUN_DIR/image_pairs.input_views.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_manifest(path: Path, manifest: Path) -> str:
    return os.path.relpath(path.resolve(), manifest.parent.resolve())


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "image_pairs.input_views.json").resolve()
    camera_path = run_dir / "predicted_input_cameras.json"
    if not camera_path.is_file():
        raise SystemExit(f"Missing QuerySplat camera export: {camera_path}")
    cameras = json.loads(camera_path.read_text())
    frames = cameras.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SystemExit(f"No camera frames in: {camera_path}")
    camera_hash = sha256(camera_path)
    records = []
    seen_names: set[str] = set()
    for camera_index, frame in enumerate(frames):
        name = str(frame.get("name", "")).strip()
        index = frame.get("index")
        source = frame.get("source")
        if not name or name in seen_names or not isinstance(index, int) or not isinstance(source, str):
            raise SystemExit(f"Invalid/duplicate QuerySplat camera frame at index {camera_index}")
        seen_names.add(name)
        reference = run_dir / source
        render = run_dir / "rendered" / f"render_view{index}.png"
        if not reference.is_file() or not render.is_file():
            raise SystemExit(f"Missing input/render pair for camera {index}: {reference}, {render}")
        with Image.open(reference) as reference_image, Image.open(render) as render_image:
            if reference_image.size != render_image.size:
                raise SystemExit(
                    f"Identity mapping invalid for {name}: reference={reference_image.size}, render={render_image.size}"
                )
            width, height = reference_image.size
        frame_id = f"{args.scene}:{name}"
        records.append(
            {
                "frame_id": frame_id,
                "scene": args.scene,
                "view_role": "input",
                "reference": relative_to_manifest(reference, output),
                "render": relative_to_manifest(render, output),
                "pixel_mapping": {"type": "identity", "size": [width, height]},
                "camera": {
                    "camera_id": f"querysplat:{args.scene}:{args.mode}:input:{index}",
                    "target_frame_id": frame_id,
                    "artifact": relative_to_manifest(camera_path, output),
                    "artifact_sha256": camera_hash,
                    "record_id": f"frames[{camera_index}]",
                    "pose_source": "querysplat_vggt_omega_native_input_prediction",
                    "intrinsics_source": "querysplat_vggt_omega_native_input_prediction",
                    "photometric_fit_target_rgb": args.mode == "tto",
                },
            }
        )
    payload = {
        "schema_version": 1,
        "dataset_id": "fpv_3d_pipeline_two_scene_2026_09",
        "method_run_id": f"querysplat:{args.scene}:common16:{args.mode}",
        "claim_scope": "input_view_reconstruction_only_not_heldout",
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
