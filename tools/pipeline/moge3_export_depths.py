#!/usr/bin/env python3
"""Export MoGe-3 metric depth maps as portable NPZ files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--fov-x", type=float)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import cv2
    import numpy as np
    import torch
    from moge.model.v3 import MoGeModel

    image_paths = sorted(
        path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = MoGeModel.from_pretrained(str(args.checkpoint)).to(device).eval()
    records = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
        kwargs = {"refine_steps": args.refine_steps}
        if args.fov_x is not None:
            kwargs["fov_x"] = args.fov_x
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.infer(tensor, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        destination = args.output / f"{image_path.stem}.npz"
        np.savez_compressed(
            destination,
            depth=output["depth"].detach().cpu().numpy().astype(np.float32),
            mask=output["mask"].detach().cpu().numpy().astype(np.uint8),
            intrinsics=output["intrinsics"].detach().cpu().numpy().astype(np.float32),
            source_width=np.int32(image_rgb.shape[1]),
            source_height=np.int32(image_rgb.shape[0]),
        )
        records.append(
            {
                "image": image_path.name,
                "output": destination.name,
                "width": int(image_rgb.shape[1]),
                "height": int(image_rgb.shape[0]),
                "elapsed_s": elapsed,
                "timing_scope": "synchronized_model_infer_excludes_model_load_and_npz_export",
                "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "valid_depth_fraction": float(output["mask"].float().mean().item()),
            }
        )
        print(f"[moge3] {image_path.name} {elapsed:.3f}s")
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": str(args.checkpoint),
                "refine_steps": args.refine_steps,
                "fov_x": args.fov_x,
                "images": records,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
