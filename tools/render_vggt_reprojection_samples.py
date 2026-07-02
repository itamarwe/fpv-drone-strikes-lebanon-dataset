#!/usr/bin/env python3
"""Render a VGGT point cloud from recovered camera-frustum viewpoints.

This is a qualitative diagnostic for checking whether the reconstructed scene
looks coherent from the original frame viewpoints. VGGT's exported GLB stores
visualization frustums, not the full predicted intrinsics, so the render uses a
user-provided focal length and the frustum orientation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path, help="Directory containing frames/, vggt_scene.glb, and point_cloud.npz")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--samples", default="1,20,40,60,80,100,121")
    parser.add_argument("--focal-px", type=float, default=400.0)
    parser.add_argument(
        "--auto-focal",
        action="store_true",
        help="Pick focal length by minimizing color error on projected points for sampled frames",
    )
    parser.add_argument("--focal-min", type=float, default=160.0)
    parser.add_argument("--focal-max", type=float, default=900.0)
    parser.add_argument("--focal-steps", type=int, default=38)
    parser.add_argument("--flip-y", action="store_true", help="Flip the inferred image vertical axis")
    parser.add_argument("--splat", type=int, default=1, help="Point radius in pixels")
    parser.add_argument("--max-depth-percentile", type=float, default=99.5)
    parser.add_argument(
        "--no-vggt-preprocess-crop",
        action="store_true",
        help="Disable VGGT-Omega's center crop for extreme aspect ratios when rendering overlays",
    )
    parser.add_argument(
        "--view",
        choices=["crop", "full"],
        default="crop",
        help="Output the VGGT input crop or the full source frame with the rendered crop pasted into it",
    )
    parser.add_argument("--quality", type=int, default=92)
    return parser.parse_args()


def transform_points(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return trimesh.transformations.transform_points(np.asarray(vertices), transform)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector
    return vector / norm


def vggt_supported_crop(width: int, height: int) -> tuple[int, int, int, int]:
    """Return the center crop used by VGGT-Omega before resizing.

    VGGT-Omega crops source images to keep height / width in [0.5, 2.0].
    For our 660x280 FPV crop, this means the model saw a 560x280 central crop.
    """
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2.0
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, crop_width, height
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, crop_height
    return 0, 0, width, height


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
        if name not in self.scene.geometry:
            raise KeyError(f"{name} is not present in GLB")
        geom = self.scene.geometry[name]
        transform = self.transforms.get(name, np.eye(4))
        vertices = transform_points(geom.vertices, transform)
        counts = np.bincount(geom.faces.reshape(-1), minlength=len(vertices))
        center = vertices[int(np.argmax(counts))]

        # The VGGT-Omega visualizer builds a cone and keeps the first cone's
        # four base corners at indices 0, 2, 3, and 4 after transformation.
        corners = vertices[[0, 2, 3, 4]]
        base_center = corners.mean(axis=0)
        forward = unit(base_center - center)
        right = unit(((corners[0] + corners[3]) * 0.5) - ((corners[1] + corners[2]) * 0.5))
        down = unit(((corners[0] + corners[1]) * 0.5) - ((corners[2] + corners[3]) * 0.5))

        right = unit(right - np.dot(right, forward) * forward)
        down = unit(down - np.dot(down, forward) * forward - np.dot(down, right) * right)
        if flip_y:
            down = -down
        return center, right, down, forward


def project_points(
    points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    forward: np.ndarray,
    focal_px: float,
    max_depth_percentile: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rel = points - center
    x = rel @ right
    y = rel @ down
    z = rel @ forward
    mask = z > 1e-4
    if not np.any(mask):
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i, empty_f, np.empty((0, 3), dtype=np.uint8)

    zmax = np.percentile(z[mask], max_depth_percentile)
    mask &= z <= zmax
    x = x[mask]
    y = y[mask]
    z = z[mask]
    colors = colors[mask]

    u = np.rint(width / 2.0 + focal_px * x / z).astype(np.int32)
    v = np.rint(height / 2.0 + focal_px * y / z).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inside]
    v = v[inside]
    z = z[inside]
    colors = colors[inside]
    return u, v, z, colors


def draw_projection(
    width: int,
    height: int,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    colors: np.ndarray,
    splat: int,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if not len(u):
        return canvas

    # Draw far-to-near so nearer points overwrite farther points.
    order = np.argsort(z)[::-1]
    u = u[order]
    v = v[order]
    colors = colors[order]
    for dy in range(-splat, splat + 1):
        vv = v + dy
        valid_v = (vv >= 0) & (vv < height)
        for dx in range(-splat, splat + 1):
            uu = u + dx
            ok = valid_v & (uu >= 0) & (uu < width)
            canvas[vv[ok], uu[ok]] = colors[ok]
    return canvas


def render_points(
    image_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    forward: np.ndarray,
    focal_px: float,
    max_depth_percentile: float,
    splat: int,
    use_vggt_preprocess_crop: bool,
) -> tuple[Image.Image, Image.Image, Image.Image, int, tuple[int, int, int, int]]:
    actual = Image.open(image_path).convert("RGB")
    full_width, full_height = actual.size
    if use_vggt_preprocess_crop:
        crop = vggt_supported_crop(full_width, full_height)
    else:
        crop = (0, 0, full_width, full_height)
    left, top, width, height = crop

    u, v, z, projected_colors = project_points(
        points=points,
        colors=colors,
        center=center,
        right=right,
        down=down,
        forward=forward,
        focal_px=focal_px,
        max_depth_percentile=max_depth_percentile,
        width=width,
        height=height,
    )
    crop_canvas = draw_projection(width, height, u, v, z, projected_colors, splat=splat)

    canvas = np.zeros((full_height, full_width, 3), dtype=np.uint8)
    canvas[top : top + height, left : left + width] = crop_canvas
    render = Image.fromarray(canvas)
    actual_arr = np.asarray(actual)
    overlay_arr = np.asarray(Image.blend(actual, render, 0.5)).copy()
    hit = canvas.sum(axis=2) > 0
    overlay_arr[hit] = (0.45 * actual_arr[hit] + 0.75 * canvas[hit]).clip(0, 255).astype(np.uint8)
    return actual, render, Image.fromarray(overlay_arr), len(u), crop


def focal_score_for_frame(
    image_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    forward: np.ndarray,
    focal_px: float,
    max_depth_percentile: float,
    use_vggt_preprocess_crop: bool,
) -> tuple[float, int]:
    actual = Image.open(image_path).convert("RGB")
    full_width, full_height = actual.size
    if use_vggt_preprocess_crop:
        left, top, width, height = vggt_supported_crop(full_width, full_height)
    else:
        left, top, width, height = (0, 0, full_width, full_height)

    u, v, _z, projected_colors = project_points(
        points=points,
        colors=colors,
        center=center,
        right=right,
        down=down,
        forward=forward,
        focal_px=focal_px,
        max_depth_percentile=max_depth_percentile,
        width=width,
        height=height,
    )
    if len(u) < 500:
        return float("inf"), len(u)
    image_crop = np.asarray(actual)[top : top + height, left : left + width]
    sampled = image_crop[v, u].astype(np.float32)
    projected = projected_colors.astype(np.float32)
    # Median absolute color distance is intentionally robust to missing geometry,
    # compression artifacts, blur, and points that belong to other viewpoints.
    error = np.median(np.mean(np.abs(sampled - projected), axis=1))
    coverage_penalty = 7500.0 / max(len(u), 7500)
    return float(error * coverage_penalty), len(u)


def choose_focal_px(
    video_dir: Path,
    samples: list[int],
    points: np.ndarray,
    colors: np.ndarray,
    cameras: VggtGlbCameras,
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, float]]]:
    use_crop = not args.no_vggt_preprocess_crop
    coarse = np.linspace(args.focal_min, args.focal_max, max(args.focal_steps, 2))
    all_candidates = [coarse]
    if len(coarse) > 1:
        # Add a local refinement around the best coarse value later.
        pass

    results: list[dict[str, float]] = []

    def score_focal(focal: float) -> float:
        frame_scores = []
        projected_counts = []
        for frame_number in samples:
            image_path = video_dir / "frames" / f"f_{frame_number:06d}.jpg"
            center, right, down, forward = cameras.camera_basis(frame_number, flip_y=args.flip_y)
            score, count = focal_score_for_frame(
                image_path=image_path,
                points=points,
                colors=colors,
                center=center,
                right=right,
                down=down,
                forward=forward,
                focal_px=float(focal),
                max_depth_percentile=args.max_depth_percentile,
                use_vggt_preprocess_crop=use_crop,
            )
            if np.isfinite(score):
                frame_scores.append(score)
                projected_counts.append(count)
        if not frame_scores:
            aggregate = float("inf")
            median_count = 0.0
        else:
            aggregate = float(np.median(frame_scores))
            median_count = float(np.median(projected_counts))
        results.append({"focal_px": float(focal), "score": aggregate, "median_projected_points": median_count})
        return aggregate

    for focal in all_candidates[0]:
        score_focal(float(focal))

    best = min(results, key=lambda row: row["score"])
    step = (args.focal_max - args.focal_min) / max(args.focal_steps - 1, 1)
    refine_min = max(args.focal_min, best["focal_px"] - step)
    refine_max = min(args.focal_max, best["focal_px"] + step)
    for focal in np.linspace(refine_min, refine_max, 21):
        score_focal(float(focal))

    best = min(results, key=lambda row: row["score"])
    return float(best["focal_px"]), results


def label(image: Image.Image, text: str) -> Image.Image:
    pad = 26
    out = Image.new("RGB", (image.width, image.height + pad), (18, 18, 18))
    out.paste(image, (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), text, fill=(245, 245, 245))
    return out


def load_frame_rows(video_dir: Path) -> list[dict[str, str]]:
    frames_csv = video_dir / "frames.csv"
    if not frames_csv.exists():
        return []
    with frames_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir
    out_dir = args.out_dir or video_dir / "reprojection_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    point_data = np.load(video_dir / "point_cloud.npz")
    points = point_data["pts"].astype(np.float32)
    colors = point_data["cols"][:, :3].astype(np.uint8)
    cameras = VggtGlbCameras(video_dir / "vggt_scene.glb")
    frame_rows = load_frame_rows(video_dir)
    times_by_file = {row["file"]: row.get("video_time_s", "") for row in frame_rows}

    samples = [int(part.strip()) for part in args.samples.split(",") if part.strip()]
    focal_px = float(args.focal_px)
    focal_scores: list[dict[str, float]] = []
    if args.auto_focal:
        focal_px, focal_scores = choose_focal_px(video_dir, samples, points, colors, cameras, args)

    rows = []
    crop = None
    for frame_number in samples:
        image_path = video_dir / "frames" / f"f_{frame_number:06d}.jpg"
        center, right, down, forward = cameras.camera_basis(frame_number, flip_y=args.flip_y)
        actual, render, overlay, count, crop = render_points(
            image_path=image_path,
            points=points,
            colors=colors,
            center=center,
            right=right,
            down=down,
            forward=forward,
            focal_px=focal_px,
            max_depth_percentile=args.max_depth_percentile,
            splat=args.splat,
            use_vggt_preprocess_crop=not args.no_vggt_preprocess_crop,
        )
        if args.view == "crop":
            left, top, width, height = crop
            crop_box = (left, top, left + width, top + height)
            actual = actual.crop(crop_box)
            render = render.crop(crop_box)
            overlay = overlay.crop(crop_box)
        file_name = image_path.name
        video_time = times_by_file.get(file_name, "")
        suffix = f" t={float(video_time):.1f}s" if video_time else ""
        row = Image.new("RGB", (actual.width * 3, actual.height + 26), (0, 0, 0))
        source_label = "actual VGGT crop" if args.view == "crop" else "actual full"
        row.paste(label(actual, f"{source_label} {file_name}{suffix}"), (0, 0))
        row.paste(label(render, f"VGGT render, f={focal_px:.0f}, {count} pts"), (actual.width, 0))
        row.paste(label(overlay, "overlay"), (actual.width * 2, 0))
        rows.append(row)

        stem = f"f_{frame_number:06d}"
        render.save(out_dir / f"{stem}_vggt_render.jpg", quality=args.quality)
        overlay.save(out_dir / f"{stem}_overlay.jpg", quality=args.quality)

    if rows:
        sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), (0, 0, 0))
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet_path = out_dir / "reprojection_contact_sheet.jpg"
        sheet.save(sheet_path, quality=args.quality)
        print(sheet_path)

    metadata = {
        "video_dir": str(video_dir),
        "samples": samples,
        "focal_px": focal_px,
        "auto_focal": bool(args.auto_focal),
        "flip_y": bool(args.flip_y),
        "used_vggt_preprocess_crop": not args.no_vggt_preprocess_crop,
        "view": args.view,
        "crop_left_top_width_height": list(crop) if crop is not None else None,
        "focal_scores": sorted(focal_scores, key=lambda row: row["score"])[:12],
    }
    (out_dir / "reprojection_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
