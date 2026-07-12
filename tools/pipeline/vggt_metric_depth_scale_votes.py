#!/usr/bin/env python3
"""Compare VGGT scene-unit depth to monocular metric depth for scale votes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path, help="VGGT reconstruction dir with frames/, vggt_scene.glb, point_cloud.npz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", default="1,20,40,60,80,100")
    parser.add_argument("--backend", default="transformers_depth", choices=["transformers_depth", "moge2"])
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    parser.add_argument("--focal-px", type=float, default=812.0)
    parser.add_argument(
        "--model-estimates-fov",
        action="store_true",
        help="For backends that support it, do not pass the focal-derived FOV to the model",
    )
    parser.add_argument("--flip-y", action="store_true", default=True)
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    parser.add_argument("--crop-left", type=int, default=50)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=560)
    parser.add_argument("--crop-height", type=int, default=280)
    parser.add_argument("--max-depth-percentile", type=float, default=99.5)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--min-vggt-depth", type=float, default=1e-4)
    parser.add_argument("--min-metric-depth-m", type=float, default=0.5)
    parser.add_argument("--max-metric-depth-m", type=float, default=80.0)
    parser.add_argument("--scale-min", type=float, default=0.1)
    parser.add_argument("--scale-max", type=float, default=500.0)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--cache-depth", action="store_true", default=True)
    parser.add_argument("--no-cache-depth", dest="cache_depth", action="store_false")
    return parser.parse_args()


def transform_points(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return trimesh.transformations.transform_points(np.asarray(vertices), transform)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


class VggtGlbCameras:
    def __init__(self, glb_path: Path):
        self.scene = trimesh.load(glb_path)
        self.transforms: dict[str, np.ndarray] = {}
        for node in self.scene.graph.nodes:
            try:
                transform, geom_name = self.scene.graph[node]
            except Exception:
                continue
            if geom_name is not None:
                self.transforms[geom_name] = np.asarray(transform)

    def camera_basis(self, frame_number: int, flip_y: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        name = f"geometry_{frame_number}"
        geom = self.scene.geometry[name]
        transform = self.transforms.get(name, np.eye(4))
        vertices = transform_points(geom.vertices, transform)
        counts = np.bincount(geom.faces.reshape(-1), minlength=len(vertices))
        center = vertices[int(np.argmax(counts))]
        corners = vertices[[0, 2, 3, 4]]
        forward = unit(corners.mean(axis=0) - center)
        right = unit(((corners[0] + corners[3]) * 0.5) - ((corners[1] + corners[2]) * 0.5))
        down = unit(((corners[0] + corners[1]) * 0.5) - ((corners[2] + corners[3]) * 0.5))
        right = unit(right - np.dot(right, forward) * forward)
        down = unit(down - np.dot(down, forward) * forward - np.dot(down, right) * right)
        if flip_y:
            down = -down
        return center, right, down, forward


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def load_depth_model(backend: str, model_name: str, device: torch.device):
    if backend == "transformers_depth":
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
        return {"processor": processor, "model": model}
    if backend == "moge2":
        from moge.model.v2 import MoGeModel

        model = MoGeModel.from_pretrained(model_name).to(device).eval()
        return {"model": model}
    raise ValueError(f"Unknown backend: {backend}")


def run_metric_depth(
    image: Image.Image,
    backend: str,
    model_bundle: dict,
    device: torch.device,
    fov_x_degrees: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if backend == "transformers_depth":
        processor = model_bundle["processor"]
        model = model_bundle["model"]
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            predicted = outputs.predicted_depth.unsqueeze(1)
            resized = torch.nn.functional.interpolate(
                predicted,
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )
        depth = resized.squeeze().float().cpu().numpy()
        return depth, np.ones(depth.shape, dtype=bool), {}

    if backend == "moge2":
        model = model_bundle["model"]
        image_arr = np.asarray(image).astype(np.float32) / 255.0
        image_tensor = torch.tensor(image_arr, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.inference_mode():
            if fov_x_degrees is None:
                output = model.infer(image_tensor)
            else:
                output = model.infer(image_tensor, fov_x=fov_x_degrees)
        depth = output["depth"].float().cpu().numpy()
        mask = output["mask"].bool().cpu().numpy()
        meta = {}
        if "intrinsics" in output:
            intr = output["intrinsics"].float().cpu().numpy()
            meta["intrinsics"] = intr.tolist()
        return depth, mask, meta

    raise ValueError(f"Unknown backend: {backend}")


def project_vggt_depth(
    points: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    focal_px: float,
    max_depth_percentile: float,
    min_vggt_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    rel = points - center
    x = rel @ right
    y = rel @ down
    z = rel @ forward
    mask = z > min_vggt_depth
    if not np.any(mask):
        return np.full((height, width), np.nan, dtype=np.float32), np.zeros((height, width), dtype=bool)
    zmax = np.percentile(z[mask], max_depth_percentile)
    mask &= z <= zmax

    x = x[mask]
    y = y[mask]
    z = z[mask]
    u = np.rint(width / 2.0 + focal_px * x / z).astype(np.int32)
    v = np.rint(height / 2.0 + focal_px * y / z).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inside]
    v = v[inside]
    z = z[inside].astype(np.float32)

    depth = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v, u), z)
    valid = np.isfinite(depth)
    depth[~valid] = np.nan
    return depth, valid


def read_frame_rows(video_dir: Path) -> dict[int, dict[str, str]]:
    with (video_dir / "frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["frame_index"]): row for row in rows}


def summarize_scale(scales: np.ndarray) -> dict[str, float]:
    if len(scales) == 0:
        return {"count": 0}
    return {
        "count": int(len(scales)),
        "median": float(np.median(scales)),
        "mean": float(np.mean(scales)),
        "p10": float(np.percentile(scales, 10)),
        "p25": float(np.percentile(scales, 25)),
        "p75": float(np.percentile(scales, 75)),
        "p90": float(np.percentile(scales, 90)),
        "mad": float(np.median(np.abs(scales - np.median(scales)))),
    }


def plot_depth_triplet(
    frame_crop: Image.Image,
    metric_depth: np.ndarray,
    vggt_depth: np.ndarray,
    scale_map: np.ndarray,
    frame_number: int,
    out_path: Path,
    scale_min: float,
    scale_max: float,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), facecolor="white")
    axes[0].imshow(frame_crop)
    axes[0].set_title(f"frame {frame_number}")
    m = axes[1].imshow(metric_depth, cmap="magma")
    axes[1].set_title("metric depth m")
    fig.colorbar(m, ax=axes[1], fraction=0.046)
    v = axes[2].imshow(vggt_depth, cmap="viridis")
    axes[2].set_title("VGGT depth units")
    fig.colorbar(v, ax=axes[2], fraction=0.046)
    display_scale = np.ma.masked_invalid(scale_map)
    s = axes[3].imshow(
        display_scale,
        cmap="turbo",
        norm=mcolors.LogNorm(vmin=scale_min, vmax=scale_max),
    )
    axes[3].set_title("scale m/unit")
    fig.colorbar(s, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_scale_votes(frame_summaries: list[dict[str, float]], scale_sets: list[np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor="white")
    labels = [str(int(row["frame"])) for row in frame_summaries]
    medians = np.array([row.get("median", np.nan) for row in frame_summaries], dtype=float)
    p25 = np.array([row.get("p25", np.nan) for row in frame_summaries], dtype=float)
    p75 = np.array([row.get("p75", np.nan) for row in frame_summaries], dtype=float)
    x = np.arange(len(labels))
    axes[0].plot(x, medians, marker="o", color="#d64b26", label="median")
    axes[0].fill_between(x, p25, p75, color="#d64b26", alpha=0.2, label="IQR")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("m per VGGT unit")
    axes[0].set_title("Per-frame metric scale votes")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    all_scales = np.concatenate([s for s in scale_sets if len(s)]) if any(len(s) for s in scale_sets) else np.array([])
    if len(all_scales):
        bins = np.geomspace(max(np.nanmin(all_scales), 1e-3), np.nanmax(all_scales), 120)
        axes[1].hist(all_scales, bins=bins, color="#1677b9", alpha=0.85)
        axes[1].axvline(np.median(all_scales), color="#d64b26", lw=1.8, label=f"pooled median {np.median(all_scales):.3g}")
        axes[1].set_xscale("log")
        axes[1].legend()
    axes[1].set_title("Pooled pixel-level scale votes")
    axes[1].set_xlabel("m per VGGT unit")
    axes[1].set_ylabel("pixels")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_equal_frame_vote_heatmap(
    frame_summaries: list[dict[str, float]],
    scale_sets: list[np.ndarray],
    out_path: Path,
    scale_min: float,
    scale_max: float,
    bins: int,
) -> dict[str, float]:
    edges = np.geomspace(scale_min, scale_max, bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    rows = []
    for scales in scale_sets:
        hist, _ = np.histogram(scales, bins=edges)
        hist = hist.astype(float)
        rows.append(hist / hist.sum() if hist.sum() else hist)
    heat = np.vstack(rows) if rows else np.empty((0, bins))
    equal_vote = heat.mean(axis=0) if len(heat) else np.zeros(bins)
    peak_scale = float(centers[int(np.argmax(equal_vote))]) if len(equal_vote) else float("nan")
    medians = np.array([row.get("median", np.nan) for row in frame_summaries], dtype=float)
    equal_median = float(np.nanmedian(medians))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), facecolor="white", height_ratios=[1.1, 1.0])
    im = axes[0].imshow(
        heat,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        extent=[np.log10(edges[0]), np.log10(edges[-1]), len(frame_summaries) - 0.5, -0.5],
    )
    axes[0].set_yticks(np.arange(len(frame_summaries)), [str(int(row["frame"])) for row in frame_summaries])
    xticks = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    xticks = xticks[(xticks >= scale_min) & (xticks <= scale_max)]
    axes[0].set_xticks(np.log10(xticks), [f"{x:g}" for x in xticks])
    axes[0].set_xlabel("m per VGGT unit")
    axes[0].set_ylabel("frame")
    axes[0].set_title("Per-frame normalized scale histograms")
    for i, median in enumerate(medians):
        if np.isfinite(median):
            axes[0].plot(np.log10(median), i, marker="o", color="#67e8f9", markersize=4)
    fig.colorbar(im, ax=axes[0], fraction=0.035, label="within-frame vote share")

    axes[1].plot(centers, equal_vote, color="#1677b9", lw=2.0)
    axes[1].axvline(peak_scale, color="#d64b26", lw=1.8, label=f"equal-frame peak {peak_scale:.3g}")
    axes[1].axvline(equal_median, color="#333333", lw=1.2, ls="--", label=f"median(frame medians) {equal_median:.3g}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("m per VGGT unit")
    axes[1].set_ylabel("mean normalized vote")
    axes[1].set_title("Majority vote with each frame weighted equally")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return {"equal_frame_peak": peak_scale, "median_of_frame_medians": equal_median}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = [int(part.strip()) for part in args.samples.split(",") if part.strip()]

    cloud = np.load(args.video_dir / "point_cloud.npz")
    points = cloud["pts"].astype(np.float32)
    if len(points) > args.max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), args.max_points, replace=False)]

    cameras = VggtGlbCameras(args.video_dir / "vggt_scene.glb")
    frame_rows = read_frame_rows(args.video_dir)
    device = choose_device(args.device)
    model_bundle = load_depth_model(args.backend, args.model, device)
    fov_x_degrees = float(np.degrees(2.0 * np.arctan(args.crop_width / (2.0 * args.focal_px))))

    crop_box = (
        args.crop_left,
        args.crop_top,
        args.crop_left + args.crop_width,
        args.crop_top + args.crop_height,
    )
    summaries: list[dict[str, float]] = []
    all_scale_sets: list[np.ndarray] = []

    for frame_number in samples:
        frame_file = args.video_dir / "frames" / f"f_{frame_number:06d}.jpg"
        image_full = Image.open(frame_file).convert("RGB")
        image_crop = image_full.crop(crop_box)
        depth_cache = args.out_dir / f"f_{frame_number:06d}_metric_depth.npy"
        mask_cache = args.out_dir / f"f_{frame_number:06d}_metric_mask.npy"
        meta_cache = args.out_dir / f"f_{frame_number:06d}_metric_meta.json"
        if args.cache_depth and depth_cache.exists() and (args.backend == "transformers_depth" or mask_cache.exists()):
            metric_depth = np.load(depth_cache)
            model_mask = np.load(mask_cache).astype(bool) if mask_cache.exists() else np.ones(metric_depth.shape, dtype=bool)
            model_meta = json.loads(meta_cache.read_text()) if meta_cache.exists() else {}
        else:
            metric_depth, model_mask, model_meta = run_metric_depth(
                image_crop,
                backend=args.backend,
                model_bundle=model_bundle,
                device=device,
                fov_x_degrees=None if args.model_estimates_fov else fov_x_degrees,
            )
            if args.cache_depth:
                np.save(depth_cache, metric_depth.astype(np.float32))
                np.save(mask_cache, model_mask.astype(bool))
                meta_cache.write_text(json.dumps(model_meta, indent=2))

        center, right, down, forward = cameras.camera_basis(frame_number, flip_y=args.flip_y)
        vggt_depth, vggt_valid = project_vggt_depth(
            points=points,
            center=center,
            right=right,
            down=down,
            forward=forward,
            width=args.crop_width,
            height=args.crop_height,
            focal_px=args.focal_px,
            max_depth_percentile=args.max_depth_percentile,
            min_vggt_depth=args.min_vggt_depth,
        )
        metric_valid = (
            np.isfinite(metric_depth)
            & model_mask
            & (metric_depth >= args.min_metric_depth_m)
            & (metric_depth <= args.max_metric_depth_m)
        )
        valid = vggt_valid & metric_valid
        scale_map = np.full(metric_depth.shape, np.nan, dtype=np.float32)
        scale_map[valid] = metric_depth[valid] / vggt_depth[valid]
        scale_valid = (
            np.isfinite(scale_map)
            & (scale_map >= args.scale_min)
            & (scale_map <= args.scale_max)
        )
        scale_values = scale_map[scale_valid]
        scale_map[~scale_valid] = np.nan
        all_scale_sets.append(scale_values.astype(np.float32))
        np.save(args.out_dir / f"f_{frame_number:06d}_scale_votes.npy", scale_values.astype(np.float32))
        np.save(args.out_dir / f"f_{frame_number:06d}_scale_map.npy", scale_map.astype(np.float32))
        summary = summarize_scale(scale_values)
        row = frame_rows.get(frame_number, {})
        summary.update(
            {
                "frame": int(frame_number),
                "video_time_s": float(row["video_time_s"]) if row.get("video_time_s") else None,
                "metric_depth_median_m": float(np.median(metric_depth[metric_valid])),
                "vggt_depth_median_units": float(np.nanmedian(vggt_depth[vggt_valid])),
                "model_valid_pixels": int(model_mask.sum()),
            }
        )
        if model_meta:
            summary["model_intrinsics"] = json.dumps(model_meta.get("intrinsics"))
        summaries.append(summary)

        plot_depth_triplet(
            frame_crop=image_crop,
            metric_depth=metric_depth,
            vggt_depth=vggt_depth,
            scale_map=scale_map,
            frame_number=frame_number,
            out_path=args.out_dir / f"f_{frame_number:06d}_depth_scale_triplet.png",
            scale_min=args.scale_min,
            scale_max=args.scale_max,
        )

    pooled = np.concatenate([s for s in all_scale_sets if len(s)]) if any(len(s) for s in all_scale_sets) else np.array([])
    equal_frame_vote = plot_equal_frame_vote_heatmap(
        summaries,
        all_scale_sets,
        args.out_dir / "equal_frame_scale_vote_heatmap.png",
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        bins=args.bins,
    )
    result = {
        "video_dir": str(args.video_dir),
        "backend": args.backend,
        "model": args.model,
        "device": str(device),
        "focal_px": args.focal_px,
        "fov_x_degrees": fov_x_degrees,
        "crop_left_top_width_height": [args.crop_left, args.crop_top, args.crop_width, args.crop_height],
        "samples": samples,
        "frames": summaries,
        "pooled": summarize_scale(pooled),
        "equal_frame_vote": equal_frame_vote,
    }
    (args.out_dir / "scale_vote_summary.json").write_text(json.dumps(result, indent=2))
    with (args.out_dir / "scale_vote_summary.csv").open("w", newline="") as handle:
        keys = sorted({key for row in summaries for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    plot_scale_votes(summaries, all_scale_sets, args.out_dir / "scale_vote_histograms.png")
    print(args.out_dir / "scale_vote_summary.json")
    print(args.out_dir / "scale_vote_histograms.png")


if __name__ == "__main__":
    main()
