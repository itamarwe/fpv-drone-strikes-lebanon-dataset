#!/usr/bin/env python3
"""Create CPU-only, explicitly diagnostic previews of a native point cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--color-source", choices=("normals", "rgb", "neutral"), required=True)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--point-size", type=float, default=1.1)
    parser.add_argument("--clip-quantile", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    import trimesh

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if len(geometries) != 1:
            raise ValueError(f"Expected one point geometry, found {len(geometries)}")
        loaded = geometries[0]
    points = np.asarray(loaded.vertices, dtype=np.float64)
    normals = None
    if hasattr(loaded, "vertex_normals"):
        candidate = np.asarray(loaded.vertex_normals, dtype=np.float64)
        if candidate.shape == points.shape:
            normals = candidate
    if normals is None:
        raw = loaded.metadata.get("_ply_raw", {}).get("vertex", {}).get("data")
        names = getattr(getattr(raw, "dtype", None), "names", ()) or ()
        if raw is not None and {"nx", "ny", "nz"}.issubset(names):
            normals = np.column_stack([raw["nx"], raw["ny"], raw["nz"]]).astype(np.float64)
    colors = None
    if hasattr(loaded.visual, "vertex_colors"):
        candidate = np.asarray(loaded.visual.vertex_colors)
        if candidate.ndim == 2 and candidate.shape[0] == len(points) and candidate.shape[1] >= 3:
            colors = candidate[:, :3].astype(np.uint8)
    return points, normals, colors


def percentiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {}
    return {f"p{label}": float(np.quantile(finite, q)) for label, q in (("01", .01), ("50", .5), ("95", .95), ("99", .99))}


def robust_mask(points: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = float(np.quantile(distances, quantile))
    return distances <= radius, center, radius


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    covariance = np.cov((points - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes, eigenvalues[order]


def render_view(
    points: np.ndarray,
    colors: np.ndarray,
    title: str,
    elev: float,
    azim: float,
    point_size: float,
    output: Path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 8), dpi=180, facecolor="#0b0e12")
    axis = figure.add_subplot(111, projection="3d", facecolor="#0b0e12")
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size, linewidths=0, depthshade=False)
    low = np.quantile(points, 0.002, axis=0)
    high = np.quantile(points, 0.998, axis=0)
    span = np.maximum(high - low, 1e-6)
    middle = (low + high) / 2
    axis.set_xlim(middle[0] - span[0] / 2, middle[0] + span[0] / 2)
    axis.set_ylim(middle[1] - span[1] / 2, middle[1] + span[1] / 2)
    axis.set_zlim(middle[2] - span[2] / 2, middle[2] + span[2] / 2)
    axis.set_box_aspect(span)
    axis.view_init(elev=elev, azim=azim)
    axis.set_axis_off()
    axis.set_title(title, color="#f0f4f8", pad=10, fontsize=11)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=.96)
    figure.savefig(output, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def make_sheet(paths: list[Path], output: Path, label: str):
    cell = 720
    sheet = Image.new("RGB", (cell * 2, cell * 2 + 48), "#0b0e12")
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = ImageOps.contain(source.convert("RGB"), (cell, cell))
        x = (index % 2) * cell + (cell - image.width) // 2
        y = (index // 2) * cell + (cell - image.height) // 2
        sheet.paste(image, (x, y))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, cell * 2 + 15), label, fill="#f0f4f8", font=ImageFont.load_default())
    sheet.save(output, quality=94)


def main() -> int:
    args = parse_args()
    if args.max_points <= 0 or args.point_size <= 0 or not 0.5 <= args.clip_quantile <= 1:
        raise SystemExit("--max-points/--point-size must be positive and --clip-quantile must be in [0.5,1]")
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"Point cloud does not exist: {source}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    points, normals, stored_colors = load_cloud(source)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    normals = normals[finite] if normals is not None else None
    stored_colors = stored_colors[finite] if stored_colors is not None else None
    keep, center, clip_radius = robust_mask(points, args.clip_quantile)
    visible_points = points[keep]
    visible_normals = normals[keep] if normals is not None else None
    visible_colors = stored_colors[keep] if stored_colors is not None else None
    if args.color_source == "normals" and visible_normals is None:
        raise SystemExit("--color-source normals requested but PLY has no vertex normals")
    if args.color_source == "rgb" and visible_colors is None:
        raise SystemExit("--color-source rgb requested but PLY has no stored colours")
    if args.color_source == "neutral":
        colors = np.broadcast_to(np.array([[165, 174, 183]], dtype=np.uint8), (len(visible_points), 3))
    elif args.color_source == "normals":
        lengths = np.linalg.norm(visible_normals, axis=1, keepdims=True)
        unit_normals = np.divide(visible_normals, lengths, out=np.zeros_like(visible_normals), where=lengths > 1e-12)
        colors = np.clip((unit_normals * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    else:
        colors = visible_colors
    rng = np.random.default_rng(args.seed)
    if len(visible_points) > args.max_points:
        indices = np.sort(rng.choice(len(visible_points), args.max_points, replace=False))
        rendered_points = visible_points[indices]
        rendered_colors = colors[indices]
    else:
        rendered_points, rendered_colors = visible_points, colors
    axes, eigenvalues = pca_frame(visible_points)
    pca_points = (rendered_points - center) @ axes
    label = "NORMAL-DERIVED COLOURS — GEOMETRY DIAGNOSTIC, NOT RGB" if args.color_source == "normals" else f"COLOUR SOURCE: {args.color_source.upper()}"
    view_specs = [
        ("isometric", 24, -52),
        ("principal front", 8, -90),
        ("principal side", 8, 0),
        ("principal top", 90, -90),
    ]
    paths = []
    for name, elev, azim in view_specs:
        path = output_dir / f"preview_{name.replace(' ', '_')}.png"
        render_view(pca_points, rendered_colors / 255.0, f"{name} · {label}", elev, azim, args.point_size, path)
        paths.append(path)
    make_sheet(paths, output_dir / "preview_contact_sheet.jpg", label)
    nearest = {}
    local_normal_consistency = {}
    try:
        from scipy.spatial import cKDTree

        sample_count = min(len(visible_points), 50000)
        sample_indices = np.sort(rng.choice(len(visible_points), sample_count, replace=False))
        tree = cKDTree(visible_points)
        distances, neighbors = tree.query(visible_points[sample_indices], k=2, workers=-1)
        nearest = percentiles(distances[:, 1])
        if visible_normals is not None:
            source_normals = visible_normals[sample_indices]
            neighbor_normals = visible_normals[neighbors[:, 1]]
            denominator = np.linalg.norm(source_normals, axis=1) * np.linalg.norm(neighbor_normals, axis=1)
            valid_normals = denominator > 1e-12
            cosine = np.abs(np.sum(source_normals[valid_normals] * neighbor_normals[valid_normals], axis=1) / denominator[valid_normals])
            local_normal_consistency = percentiles(cosine)
    except ImportError:
        nearest = {"unavailable": "scipy is not installed"}
    normal_lengths = np.linalg.norm(visible_normals, axis=1) if visible_normals is not None else np.array([])
    metadata = {
        "schema_version": 1,
        "view_role": "exploratory_geometry_diagnostic",
        "photometric_scoring_allowed": False,
        "input": str(source),
        "input_sha256": sha256(source),
        "color_source": args.color_source,
        "warning": "Stored colours are treated exactly as declared; normals mode is never RGB appearance evidence.",
        "vertex_count_declared_or_loaded": int(len(finite)),
        "finite_vertex_count": int(len(points)),
        "visible_vertex_count": int(len(visible_points)),
        "visible_fraction_of_finite": float(len(visible_points) / len(points)),
        "rendered_sample_count": int(len(rendered_points)),
        "render_point_size": args.point_size,
        "clip_quantile": args.clip_quantile,
        "clip_radius": clip_radius,
        "coordinate_median": center.tolist(),
        "full_bounds": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
        "visible_bounds": {"min": visible_points.min(axis=0).tolist(), "max": visible_points.max(axis=0).tolist()},
        "pca_eigenvalues": eigenvalues.tolist(),
        "normal_length": percentiles(normal_lengths),
        "nearest_neighbor_absolute_normal_cosine": local_normal_consistency,
        "nearest_neighbor_distance": nearest,
        "previews": [path.name for path in paths],
        "contact_sheet": "preview_contact_sheet.jpg",
    }
    (output_dir / "preview_manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metadata, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
