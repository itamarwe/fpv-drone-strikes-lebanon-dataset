#!/usr/bin/env python3
"""Export two-scene benchmark reconstructions into the scene-viewer format.

Every reconstruction is written as ``scenes/<collection>/<scene>__<method>__<profile>/viewer/``
so the existing scene viewer and ``tools/local_scene_viewer_server.py`` can
show it. Candidate reconstructions are aligned into the published VGGT frame
with the same Umeyama Sim(3) fit (camera centres only) that the trajectory
evaluator uses, so the published ground grid, alignment quaternion and
metre-per-unit calibration apply to all of them. That alignment is a
visualisation convenience: it is not an independent metric calibration of the
candidate reconstruction, and the meta records this.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_3d_trajectory import umeyama  # noqa: E402
from fit_moge3_colmap_scale import qvec_to_rotation, read_images  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VIEWER_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "benchmarks" / "3d_pipeline_two_scene")
    parser.add_argument("--output-collection", default="benchmark_3d_two_scene",
                        help="Directory name under scenes/ that holds the exported viewers")
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--crop-path-radii", type=float, default=4.5,
                        help="Drop points farther than this many path RMS radii from the path centre (visual crop)")
    parser.add_argument("--scenes", nargs="*", default=["scene_a", "scene_b"])
    return parser.parse_args()


# ----------------------------------------------------------------------------- PLY

def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read xyz + rgb from ascii or binary_little_endian PLY (vertex element only)."""
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = next(line.split()[1] for line in header if line.startswith("format"))
        counts = [int(line.split()[2]) for line in header if line.startswith("element")]
        vertex_count = counts[0]
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header:
            if line.startswith("element"):
                in_vertex = line.split()[1] == "vertex"
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError("list property inside vertex element is unsupported")
                properties.append((parts[2], parts[1]))
        type_map = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
                    "int": "<i4", "int32": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, type_map[ptype]) for name, ptype in properties])
            data = np.fromfile(handle, dtype=dtype, count=vertex_count)
        elif fmt == "ascii":
            names = [name for name, _ in properties]
            raw = np.loadtxt(handle, dtype=np.float64, max_rows=vertex_count, ndmin=2)
            data = {name: raw[:, index] for index, name in enumerate(names)}
        else:
            raise ValueError(f"unsupported PLY format {fmt}")
    xyz = np.stack([np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["z"])], axis=1).astype(np.float64)
    if all(key in (name for name, _ in properties) for key in ("red", "green", "blue")):
        rgb = np.stack([np.asarray(data["red"]), np.asarray(data["green"]), np.asarray(data["blue"])], axis=1)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], rgb[finite]


# ----------------------------------------------------------------------------- cameras

def cameras_from_scal3r(run_dir: Path) -> dict[int, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(run_dir / "mat.txt", dtype=np.float64))
    return {index: row.reshape(4, 4) for index, row in enumerate(rows)}


def cameras_from_vggt_json(path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text())
    cameras = {}
    for frame in payload["frames"]:
        profile_index = int(str(frame["name"]).rsplit("_", 1)[1])
        cameras[profile_index] = np.asarray(frame["c2w"], dtype=np.float64)
    return cameras


def cameras_from_colmap(model_dir: Path) -> dict[int, np.ndarray]:
    cameras = {}
    for image in read_images(model_dir / "images.txt"):
        rotation = qvec_to_rotation(image.qvec)
        c2w = np.eye(4)
        c2w[:3, :3] = rotation.T
        c2w[:3, 3] = -(rotation.T @ image.tvec)
        profile_index = int(Path(image.name).stem.rsplit("_", 1)[1])
        cameras[profile_index] = c2w
    return cameras


def read_colmap_points(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with (model_dir / "points3D.txt").open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


# ----------------------------------------------------------------------------- export

def load_frames_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def subsample(xyz: np.ndarray, rgb: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) <= limit:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(xyz), size=limit, replace=False)
    keep.sort()
    return xyz[keep], rgb[keep]


def path_entry(profile_row: dict[str, str], published_entry: dict, c2w: np.ndarray, image_url: str) -> dict:
    rotation = c2w[:3, :3]
    return {
        "frame": int(profile_row["frame"]),
        "file": profile_row["source_file"],
        "video_file": published_entry.get("video_file"),
        "segment_id": profile_row.get("segment_id") or published_entry.get("segment_id"),
        "segment_index": published_entry.get("segment_index"),
        "is_attack": bool(published_entry.get("is_attack", False)),
        "video_time_s": float(profile_row["video_time_s"]),
        "segment_time_s": float(published_entry.get("segment_time_s", profile_row["sequence_time_s"])),
        "sequence_time_s": float(profile_row["sequence_time_s"]),
        "position": c2w[:3, 3].tolist(),
        "right": rotation[:, 0].tolist(),
        "down": rotation[:, 1].tolist(),
        "forward": rotation[:, 2].tolist(),
        "frame_image": image_url,
        "actual_image": image_url,
    }


def finish_path(path: list[dict]) -> None:
    end = np.asarray(path[-1]["position"])
    for entry in path:
        entry["distance_to_end_units"] = float(np.linalg.norm(np.asarray(entry["position"]) - end))


def write_viewer(out_dir: Path, xyz: np.ndarray, rgb: np.ndarray, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz.astype("<f4").tofile(out_dir / "points_positions.bin")
    rgb.astype(np.uint8).tofile(out_dir / "points_colors.bin")
    meta.update({
        "point_count": int(len(xyz)),
        "bbox_min": xyz.min(axis=0).astype(float).tolist(),
        "bbox_max": xyz.max(axis=0).astype(float).tolist(),
        "assets": {"positions": "points_positions.bin", "colors": "points_colors.bin"},
        "viewer_version": VIEWER_VERSION,
    })
    (out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def base_meta(published: dict, title: str, scene_id: str) -> dict:
    return {
        "title": title,
        "source_label": scene_id,
        "video_dir": "",
        "scene_id": scene_id,
        "scene_state_url": "",
        "default_scale_m_per_unit": published["default_scale_m_per_unit"],
        "sample_fps": published.get("sample_fps", 10),
        "calibration": published.get("calibration"),
        "ground_grid": published.get("ground_grid"),
        "scene_alignment_quaternion": published.get("scene_alignment_quaternion"),
    }


def latest_attempt(base: Path, prefix: str = "attempt-") -> Path | None:
    if not base.is_dir():
        return None
    attempts = sorted(p for p in base.iterdir() if p.name.startswith(prefix) and (p / "run.json").exists())
    complete = [p for p in attempts if json.loads((p / "run.json").read_text()).get("status") == "complete"]
    return complete[-1] if complete else None


def load_metric(path: Path, keys: list[str]) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {key: data.get(key) for key in keys if key in data}


def main() -> int:
    args = parse_args()
    bench = args.benchmark_dir
    bundle = bench / "work" / "bundle" / "scenes"
    results = bench / "results"
    scenes_root = ROOT / "scenes" / args.output_collection
    shared_root = scenes_root / "_shared"
    index: list[dict] = []

    for scene in args.scenes:
        source_meta = json.loads((bundle / scene / "source" / "scene_meta.json").read_text())
        published_path = source_meta["path"]
        # Shared source frames, referenced by absolute URL so every export reuses one copy.
        image_dir = shared_root / scene / "images"
        if not image_dir.exists():
            shutil.copytree(bundle / scene / "source" / "images", image_dir)
        image_url = lambda name: f"/scenes/{args.output_collection}/_shared/{scene}/images/{name}"  # noqa: E731

        def export(method: str, profile: str, title: str, xyz: np.ndarray, rgb: np.ndarray,
                   cameras: dict[int, np.ndarray] | None, frames_csv: Path | None, extra: dict) -> None:
            key = f"{scene}__{method}__{profile}"
            out_dir = scenes_root / key / "viewer"
            meta = base_meta(source_meta, title, key)
            alignment: dict = {"mode": "published_vggt_frame"}
            if cameras is None:
                path = []
                for entry in published_path:
                    row = {"frame": entry["frame"], "source_file": entry["file"], "segment_id": entry.get("segment_id"),
                           "video_time_s": entry["video_time_s"], "sequence_time_s": entry.get("sequence_time_s", entry["segment_time_s"])}
                    c2w = np.eye(4)
                    c2w[:3, 0] = entry["right"]; c2w[:3, 1] = entry["down"]; c2w[:3, 2] = entry["forward"]; c2w[:3, 3] = entry["position"]
                    path.append(path_entry(row, entry, c2w, image_url(entry["file"])))
                alignment["note"] = "Published reconstruction in its own frame; no alignment applied."
            else:
                rows = load_frames_csv(frames_csv)
                matched = [r for r in rows if int(r["profile_index"]) in cameras]
                pred = np.vstack([cameras[int(r["profile_index"])][:3, 3] for r in matched])
                ref = np.vstack([np.asarray(published_path[int(r["source_index"])]["position"]) for r in matched])
                scale, rotation, translation = umeyama(pred, ref)
                errors = np.linalg.norm((scale * (rotation @ pred.T)).T + translation - ref, axis=1)
                alignment.update({
                    "method": "umeyama_sim3_on_camera_centres",
                    "matched_cameras": int(len(matched)),
                    "expected_cameras": int(len(rows)),
                    "scale_prediction_to_published": float(scale),
                    "camera_centre_rmse_vggt_units": float(np.sqrt(np.mean(errors**2))),
                    "note": "Aligned to the published VGGT camera path for visual comparison. The published metre-per-unit "
                            "calibration is inherited through this alignment; it is not an independent metric calibration.",
                })
                xyz = (scale * (rotation @ xyz.T)).T + translation
                path = []
                for row in matched:
                    c2w = cameras[int(row["profile_index"])].copy()
                    aligned = np.eye(4)
                    aligned[:3, :3] = rotation @ c2w[:3, :3]
                    aligned[:3, 3] = scale * (rotation @ c2w[:3, 3]) + translation
                    published_entry = published_path[int(row["source_index"])]
                    path.append(path_entry(row, published_entry, aligned, image_url(row["source_file"])))
            finish_path(path)
            # Visual crop: drop far sky/outlier points so auto-fit frames the scene. The published
            # product cloud stays within ~3.4 path radii of the path centre on both scenes.
            centres = np.asarray([entry["position"] for entry in path])
            path_centre = centres.mean(axis=0)
            path_radius = float(np.sqrt(np.mean(np.sum((centres - path_centre) ** 2, axis=1)))) or 1.0
            distances = np.linalg.norm(xyz - path_centre, axis=1) / path_radius
            keep = distances <= args.crop_path_radii
            crop = {"rule": "distance_from_path_centre_over_path_rms_radius", "limit": args.crop_path_radii,
                    "points_before": int(len(xyz)), "points_dropped": int((~keep).sum())}
            xyz, rgb = xyz[keep], rgb[keep]
            xyz, rgb = subsample(xyz, rgb, args.max_points, args.seed)
            extra = {**extra, "visual_crop": crop}
            meta["path"] = path
            meta["benchmark"] = {"scene": scene, "method": method, "profile": profile, "alignment": alignment, **extra}
            write_viewer(out_dir, xyz, rgb, meta)
            index.append({"scene": scene, "method": method, "profile": profile, "title": title, "key": key,
                          "viewer_url": f"/scenes/{args.output_collection}/{key}/viewer/",
                          "point_count": int(len(xyz)), "cameras": len(path), "alignment": alignment, **extra})
            print(f"wrote {out_dir} points={len(xyz)} cameras={len(path)}")

        # Published VGGT baseline (current product).
        positions = np.fromfile(bundle / scene / "baseline" / "points_positions.bin", dtype="<f4").reshape(-1, 3).astype(np.float64)
        colors = np.fromfile(bundle / scene / "baseline" / "points_colors.bin", dtype=np.uint8).reshape(-1, 3)
        export("published_vggt", "full", f"{scene}: published VGGT (current product, {len(published_path)} frames)",
               positions, colors, None, None, {"attempt": None, "input_views": len(published_path)})

        for profile in ("common16", "dense_train"):
            frames_csv = bundle / scene / "profiles" / profile / "frames.csv"
            n_views = len(load_frames_csv(frames_csv))
            # QuerySplat feed-forward: VGGT-Omega depth point cloud + predicted cameras.
            attempt = latest_attempt(results / scene / profile / "querysplat")
            ff_dirs = sorted((results / scene / profile / "querysplat").glob("attempt-*/feed_forward")) if attempt else []
            if ff_dirs:
                ff = ff_dirs[-1]
                xyz, rgb = read_ply(ff / "vggt_depth_pointcloud.ply")
                cams = cameras_from_vggt_json(ff / "predicted_input_cameras.json")
                metrics = load_metric(ff / "evaluation_proxy" / "metrics.json", ["full_frame"])
                traj = load_metric(ff / "trajectory_vs_published.json", ["sim3_rmse_fraction_published_path_rms_radius"])
                export("vggt_omega_querysplat_ff", profile,
                       f"{scene}: VGGT-Omega / QuerySplat feed-forward depth cloud ({n_views} views)",
                       xyz, rgb, cams, frames_csv,
                       {"attempt": ff.parent.name, "input_views": n_views,
                        "heldout_full_frame": metrics.get("full_frame"), **traj})
            # Scal3R
            attempt = latest_attempt(results / scene / profile / "scal3r_zju")
            if attempt and (attempt / "points" / "whole.ply").exists():
                xyz, rgb = read_ply(attempt / "points" / "whole.ply")
                cams = cameras_from_scal3r(attempt)
                traj = load_metric(attempt / "trajectory.json", ["sim3_rmse_fraction_published_path_rms_radius", "registered_frames"])
                export("scal3r_zju", profile, f"{scene}: Scal3R-ZJU ({n_views} views)", xyz, rgb, cams, frames_csv,
                       {"attempt": attempt.name, "input_views": n_views, **traj})
            # GLOMAP sparse
            attempt = latest_attempt(results / scene / profile / "moge3_glomap")
            if attempt and (attempt / "sparse-txt" / "points3D.txt").exists():
                xyz, rgb = read_colmap_points(attempt / "sparse-txt")
                cams = cameras_from_colmap(attempt / "sparse-txt")
                traj = load_metric(attempt / "trajectory.json", ["sim3_rmse_fraction_published_path_rms_radius", "registered_frames"])
                export("glomap_sparse", profile, f"{scene}: GLOMAP sparse SfM ({n_views} views)", xyz, rgb, cams, frames_csv,
                       {"attempt": attempt.name, "input_views": n_views, **traj})

    (scenes_root / "index.json").write_text(json.dumps({"schema_version": 1, "entries": index}, indent=2) + "\n")
    print(f"wrote {scenes_root / 'index.json'} ({len(index)} viewers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
