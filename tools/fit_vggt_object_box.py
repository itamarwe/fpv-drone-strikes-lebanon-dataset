#!/usr/bin/env python3
"""Fit a 3D object box in VGGT space from 2D object masks.

The intended workflow is deliberately simple:
1. Mark the object in several VGGT crop-frame images as polygons.
2. Project the VGGT point cloud into each marked frame.
3. Keep visible z-buffer points that land inside the polygon.
4. Vote across frames, fit a robust PCA-oriented box, and derive scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageDraw


BOX_EDGES = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path, help="VGGT reconstruction dir with frames/, vggt_scene.glb, point_cloud.npz")
    parser.add_argument("--boxes-json", type=Path, required=True, help="Manual polygons in VGGT crop coordinates")
    parser.add_argument("--mask-manifest", type=Path, default=None, help="Optional SAM mask manifest from segment_object_with_sam.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--known-length-m", type=float, required=True)
    parser.add_argument("--focal-px", type=float, default=812.0)
    parser.add_argument("--crop-left", type=int, default=50)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=560)
    parser.add_argument("--crop-height", type=int, default=280)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--max-luma", type=float, default=None, help="Optional RGB luma cutoff for vehicle-like dark points")
    parser.add_argument("--clip-low", type=float, default=1.0)
    parser.add_argument("--clip-high", type=float, default=99.0)
    parser.add_argument("--clip-iters", type=int, default=4)
    parser.add_argument("--flip-y", action="store_true", default=True)
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    return parser.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def transform_points(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return trimesh.transformations.transform_points(np.asarray(vertices), transform)


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


def load_polygons(path: Path) -> dict[int, list[tuple[float, float]]]:
    payload = json.loads(path.read_text())
    frames = payload.get("frames", payload)
    polygons: dict[int, list[tuple[float, float]]] = {}
    for key, value in frames.items():
        polygon = value.get("polygon", value) if isinstance(value, dict) else value
        polygons[int(key)] = [(float(x), float(y)) for x, y in polygon]
    return polygons


def load_mask_manifest(path: Path, width: int, height: int) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text())
    masks: dict[int, np.ndarray] = {}
    for key, value in payload.get("frames", {}).items():
        mask_path = Path(value["mask"])
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        if mask.shape != (height, width):
            raise ValueError(f"Mask for frame {key} has shape {mask.shape}, expected {(height, width)}")
        masks[int(key)] = mask
    return masks


def project_visible_points_in_polygon(
    points: np.ndarray,
    cameras: VggtGlbCameras,
    frame_number: int,
    polygon: list[tuple[float, float]],
    selection_mask: np.ndarray | None,
    width: int,
    height: int,
    focal_px: float,
    flip_y: bool,
) -> np.ndarray:
    center, right, down, forward = cameras.camera_basis(frame_number, flip_y=flip_y)
    rel = points - center
    x = rel @ right
    y = rel @ down
    z = rel @ forward
    front = z > 1e-4
    indices = np.nonzero(front)[0]
    x = x[front]
    y = y[front]
    z = z[front]
    u = np.rint(width / 2.0 + focal_px * x / z).astype(np.int32)
    v = np.rint(height / 2.0 + focal_px * y / z).astype(np.int32)
    inside_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inside_image]
    v = v[inside_image]
    z = z[inside_image]
    indices = indices[inside_image]

    if selection_mask is None:
        mask_image = Image.new("1", (width, height), 0)
        ImageDraw.Draw(mask_image).polygon(polygon, fill=1)
        polygon_mask = np.asarray(mask_image, dtype=bool)
    else:
        polygon_mask = selection_mask
    inside_polygon = polygon_mask[v, u]
    u = u[inside_polygon]
    v = v[inside_polygon]
    z = z[inside_polygon]
    indices = indices[inside_polygon]

    pixel = v * width + u
    order = np.lexsort((z, pixel))
    sorted_pixel = pixel[order]
    sorted_indices = indices[order]
    keep = np.r_[True, sorted_pixel[1:] != sorted_pixel[:-1]] if len(sorted_pixel) else np.array([], dtype=bool)
    return sorted_indices[keep]


def robust_oriented_box(
    points: np.ndarray,
    clip_low: float,
    clip_high: float,
    clip_iters: int,
) -> dict[str, np.ndarray]:
    working = points.copy()
    axes = np.eye(3)
    center = working.mean(axis=0)
    for _ in range(clip_iters):
        center = working.mean(axis=0)
        centered = working - center
        vals, vecs = np.linalg.eigh(centered.T @ centered / len(working))
        axes = vecs[:, np.argsort(vals)[::-1]]
        projected = (points - center) @ axes
        lo = np.percentile(projected, clip_low, axis=0)
        hi = np.percentile(projected, clip_high, axis=0)
        keep = ((projected >= lo) & (projected <= hi)).all(axis=1)
        working = points[keep]

    center = working.mean(axis=0)
    centered = working - center
    vals, vecs = np.linalg.eigh(centered.T @ centered / len(working))
    axes = vecs[:, np.argsort(vals)[::-1]]
    projected = (working - center) @ axes
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    dims = maxs - mins
    box_center = center + axes @ ((mins + maxs) * 0.5)
    corners_local = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=float,
    )
    corners = center + corners_local @ axes.T
    return {
        "center": box_center,
        "pca_center": center,
        "axes": axes,
        "mins": mins,
        "maxs": maxs,
        "dims": dims,
        "corners": corners,
        "kept_points": working,
    }


def project_world_points(
    world_points: np.ndarray,
    cameras: VggtGlbCameras,
    frame_number: int,
    width: int,
    height: int,
    focal_px: float,
    flip_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    center, right, down, forward = cameras.camera_basis(frame_number, flip_y=flip_y)
    rel = world_points - center
    x = rel @ right
    y = rel @ down
    z = rel @ forward
    u = width / 2.0 + focal_px * x / z
    v = height / 2.0 + focal_px * y / z
    return np.column_stack([u, v]), z


def draw_mask_sheet(
    video_dir: Path,
    polygons: dict[int, list[tuple[float, float]]],
    out_path: Path,
    crop_box: tuple[int, int, int, int],
) -> None:
    frames = sorted(polygons)
    width = crop_box[2] - crop_box[0]
    height = crop_box[3] - crop_box[1]
    sheet = Image.new("RGB", (width * 2, (height + 24) * ((len(frames) + 1) // 2)), (18, 18, 18))
    draw_sheet = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        image = Image.open(video_dir / "frames" / f"f_{frame:06d}.jpg").convert("RGB").crop(crop_box)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.polygon(polygons[frame], outline=(255, 70, 40, 255), fill=(255, 150, 20, 70), width=3)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        x = (i % 2) * width
        y = (i // 2) * (height + 24)
        draw_sheet.text((x + 6, y + 4), f"f_{frame:06d}", fill=(235, 235, 235))
        sheet.paste(image, (x, y + 24))
    sheet.save(out_path, quality=95)


def draw_box_projection_sheet(
    video_dir: Path,
    polygons: dict[int, list[tuple[float, float]]],
    cameras: VggtGlbCameras,
    corners: np.ndarray,
    out_path: Path,
    crop_box: tuple[int, int, int, int],
    focal_px: float,
    flip_y: bool,
) -> None:
    frames = sorted(polygons)
    width = crop_box[2] - crop_box[0]
    height = crop_box[3] - crop_box[1]
    sheet = Image.new("RGB", (width * 2, (height + 24) * ((len(frames) + 1) // 2)), (18, 18, 18))
    draw_sheet = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        image = Image.open(video_dir / "frames" / f"f_{frame:06d}.jpg").convert("RGB").crop(crop_box)
        projected, depth = project_world_points(corners, cameras, frame, width, height, focal_px, flip_y)
        draw = ImageDraw.Draw(image)
        for a, b in BOX_EDGES:
            if depth[a] > 0 and depth[b] > 0:
                draw.line([tuple(projected[a]), tuple(projected[b])], fill=(0, 255, 255), width=2)
        draw.polygon(polygons[frame], outline=(255, 70, 40), width=2)
        x = (i % 2) * width
        y = (i // 2) * (height + 24)
        draw_sheet.text((x + 6, y + 4), f"f_{frame:06d}", fill=(235, 235, 235))
        sheet.paste(image, (x, y + 24))
    sheet.save(out_path, quality=95)


def draw_axis_projection_sheet(
    video_dir: Path,
    polygons: dict[int, list[tuple[float, float]]],
    cameras: VggtGlbCameras,
    kept_points: np.ndarray,
    axis_endpoints: np.ndarray,
    out_path: Path,
    crop_box: tuple[int, int, int, int],
    focal_px: float,
    flip_y: bool,
) -> None:
    frames = sorted(polygons)
    width = crop_box[2] - crop_box[0]
    height = crop_box[3] - crop_box[1]
    sheet = Image.new("RGB", (width * 2, (height + 24) * ((len(frames) + 1) // 2)), (18, 18, 18))
    draw_sheet = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        image = Image.open(video_dir / "frames" / f"f_{frame:06d}.jpg").convert("RGB").crop(crop_box)
        draw = ImageDraw.Draw(image)
        projected_points, point_depth = project_world_points(kept_points, cameras, frame, width, height, focal_px, flip_y)
        in_front = point_depth > 0
        projected_points = projected_points[in_front]
        in_image = (
            (projected_points[:, 0] >= 0)
            & (projected_points[:, 0] < width)
            & (projected_points[:, 1] >= 0)
            & (projected_points[:, 1] < height)
        )
        for x, y in projected_points[in_image]:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 176, 0))

        projected_axis, axis_depth = project_world_points(axis_endpoints, cameras, frame, width, height, focal_px, flip_y)
        if np.all(axis_depth > 0):
            draw.line([tuple(projected_axis[0]), tuple(projected_axis[1])], fill=(0, 255, 255), width=4)
            for x, y in projected_axis:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 255, 255))
        draw.polygon(polygons[frame], outline=(255, 70, 40), width=2)

        x = (i % 2) * width
        y = (i // 2) * (height + 24)
        draw_sheet.text((x + 6, y + 4), f"f_{frame:06d}", fill=(235, 235, 235))
        sheet.paste(image, (x, y + 24))
    sheet.save(out_path, quality=95)


def plot_3d_box(
    points: np.ndarray,
    colors: np.ndarray,
    kept_points: np.ndarray,
    corners: np.ndarray,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(4)
    if len(points) > 70_000:
        sample_idx = rng.choice(len(points), 70_000, replace=False)
        bg = points[sample_idx]
        bg_colors = colors[sample_idx] / 255.0
    else:
        bg = points
        bg_colors = colors / 255.0

    fig = plt.figure(figsize=(9, 7), facecolor="#101214")
    ax = fig.add_subplot(111, projection="3d", facecolor="#101214")
    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], c=bg_colors, s=0.4, alpha=0.22, linewidths=0)
    ax.scatter(kept_points[:, 0], kept_points[:, 1], kept_points[:, 2], c="#ffb000", s=5, alpha=0.9, linewidths=0)
    for a, b in BOX_EDGES:
        xs, ys, zs = zip(corners[a], corners[b])
        ax.plot(xs, ys, zs, color="#36e4ff", lw=2.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.tick_params(colors="#d7dde2")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("#d7dde2")
    ax.view_init(elev=22, azim=-72)
    ax.set_box_aspect((1.1, 0.7, 1.2))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop_box = (
        args.crop_left,
        args.crop_top,
        args.crop_left + args.crop_width,
        args.crop_top + args.crop_height,
    )

    cloud = np.load(args.video_dir / "point_cloud.npz")
    points = cloud["pts"].astype(np.float64)
    colors = cloud["cols"]
    polygons = load_polygons(args.boxes_json)
    mask_overrides = (
        load_mask_manifest(args.mask_manifest, args.crop_width, args.crop_height)
        if args.mask_manifest is not None
        else {}
    )
    cameras = VggtGlbCameras(args.video_dir / "vggt_scene.glb")

    votes = np.zeros(len(points), dtype=np.uint16)
    selected_counts: dict[str, int] = {}
    for frame_number, polygon in sorted(polygons.items()):
        visible_indices = project_visible_points_in_polygon(
            points=points,
            cameras=cameras,
            frame_number=frame_number,
            polygon=polygon,
            selection_mask=mask_overrides.get(frame_number),
            width=args.crop_width,
            height=args.crop_height,
            focal_px=args.focal_px,
            flip_y=args.flip_y,
        )
        votes[visible_indices] += 1
        selected_counts[str(frame_number)] = int(len(visible_indices))

    candidate_indices = np.nonzero(votes >= args.min_votes)[0]
    if args.max_luma is not None:
        candidate_colors = colors[candidate_indices].astype(float)
        luma = 0.2126 * candidate_colors[:, 0] + 0.7152 * candidate_colors[:, 1] + 0.0722 * candidate_colors[:, 2]
        candidate_indices = candidate_indices[luma <= args.max_luma]
    if len(candidate_indices) < 20:
        raise SystemExit(f"Only {len(candidate_indices)} candidate points survived min-votes={args.min_votes}")

    object_points = points[candidate_indices]
    box = robust_oriented_box(object_points, args.clip_low, args.clip_high, args.clip_iters)
    dims = box["dims"]
    axis_midpoint = (box["mins"] + box["maxs"]) * 0.5
    axis_start = axis_midpoint.copy()
    axis_end = axis_midpoint.copy()
    axis_start[0] = box["mins"][0]
    axis_end[0] = box["maxs"][0]
    axis_endpoints = box["pca_center"] + np.vstack([axis_start, axis_end]) @ box["axes"].T
    length_units = float(dims[0])
    scale_m_per_unit = float(args.known_length_m / length_units)
    dims_m = dims * scale_m_per_unit

    np.savez_compressed(
        args.out_dir / "object_box_fit.npz",
        candidate_indices=candidate_indices,
        candidate_points=object_points.astype(np.float32),
        kept_points=box["kept_points"].astype(np.float32),
        corners=box["corners"].astype(np.float32),
        axis_endpoints=axis_endpoints.astype(np.float32),
        axes=box["axes"].astype(np.float32),
        dims=dims.astype(np.float32),
        votes=votes,
    )
    draw_mask_sheet(args.video_dir, polygons, args.out_dir / "manual_polygons_contact_sheet.jpg", crop_box)
    draw_box_projection_sheet(
        args.video_dir,
        polygons,
        cameras,
        box["corners"],
        args.out_dir / "box_projection_contact_sheet.jpg",
        crop_box,
        args.focal_px,
        args.flip_y,
    )
    draw_axis_projection_sheet(
        args.video_dir,
        polygons,
        cameras,
        box["kept_points"],
        axis_endpoints,
        args.out_dir / "axis_projection_contact_sheet.jpg",
        crop_box,
        args.focal_px,
        args.flip_y,
    )
    plot_3d_box(points, colors, box["kept_points"], box["corners"], args.out_dir / "box_3d.png")

    summary = {
        "video_dir": str(args.video_dir),
        "boxes_json": str(args.boxes_json),
        "mask_manifest": str(args.mask_manifest) if args.mask_manifest is not None else None,
        "frames": sorted(polygons),
        "visible_points_by_frame": selected_counts,
        "candidate_points_min_votes": args.min_votes,
        "max_luma": args.max_luma,
        "candidate_count": int(len(candidate_indices)),
        "kept_after_robust_clip": int(len(box["kept_points"])),
        "known_length_m": float(args.known_length_m),
        "length_vggt_units": length_units,
        "scale_m_per_vggt_unit": scale_m_per_unit,
        "box_dims_vggt_units": dims.tolist(),
        "box_dims_m_if_length_known": dims_m.tolist(),
        "box_center": box["center"].tolist(),
        "box_axes_columns": box["axes"].tolist(),
        "outputs": {
            "manual_polygons": str(args.out_dir / "manual_polygons_contact_sheet.jpg"),
            "box_projection": str(args.out_dir / "box_projection_contact_sheet.jpg"),
            "axis_projection": str(args.out_dir / "axis_projection_contact_sheet.jpg"),
            "box_3d": str(args.out_dir / "box_3d.png"),
            "fit_npz": str(args.out_dir / "object_box_fit.npz"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
