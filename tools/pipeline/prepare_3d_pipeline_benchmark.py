#!/usr/bin/env python3
"""Prepare deterministic published-frame inputs for the two-scene 3D benchmark."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageStat


DEFAULT_SPEC = Path("benchmarks/3d_pipeline_two_scene/benchmark.json")
DEFAULT_WORK = Path("benchmarks/3d_pipeline_two_scene/work")
DEFAULT_CDN = "https://d2fioemadmrru3.cloudfront.net/"
ORCHESTRATION_FILES = (
    "tools/pipeline/benchmark_remote_job.py",
    "tools/pipeline/check_3d_pipeline_benchmark.py",
    "tools/pipeline/compare_3d_trajectories.py",
    "tools/pipeline/evaluate_3d_trajectory.py",
    "tools/pipeline/evaluate_3d_visualization.py",
    "tools/pipeline/fit_moge3_colmap_scale.py",
    "tools/pipeline/fit_moge3_scal3r_scale.py",
    "tools/pipeline/metric_scale.py",
    "tools/pipeline/evaluate_moge3_vggt_scale.py",
    "tools/pipeline/moge3_export_depths.py",
    "tools/pipeline/runpod_3d_pipeline_benchmark.sh",
    "tools/pipeline/summarize_colmap_diagnostics.py",
    "tools/pipeline/summarize_scal3r_diagnostics.py",
    "tools/pipeline/summarize_vggt_control.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, refresh: bool = False) -> None:
    if destination.exists() and not refresh:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "fpv-3d-benchmark/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
            shutil.copyfileobj(response, temp)
            temp_path = Path(temp.name)
    os.replace(temp_path, destination)


def expand_gzip_in_place(path: Path) -> None:
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic != b"\x1f\x8b":
        return
    with gzip.open(path, "rb") as source:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            shutil.copyfileobj(source, temp)
            temp_path = Path(temp.name)
    os.replace(temp_path, path)


def allocate_segment_counts(groups: list[list[int]], total: int) -> list[int]:
    if total < len(groups):
        raise ValueError("selection count must cover every segment")
    if sum(len(group) for group in groups) < total:
        raise ValueError("selection count exceeds available frames")
    minimum = 2 if total >= 2 * len(groups) and all(len(group) >= 2 for group in groups) else 1
    counts = [minimum for _ in groups]
    remaining = total - sum(counts)
    capacities = [len(group) - minimum for group in groups]
    while remaining:
        best = max(
            range(len(groups)),
            key=lambda idx: ((capacities[idx] - (counts[idx] - minimum)) / max(1, len(groups[idx])), -idx),
        )
        if counts[best] >= len(groups[best]):
            raise ValueError("could not allocate selection across segments")
        counts[best] += 1
        remaining -= 1
    return counts


def uniform_indices(indices: list[int], count: int) -> list[int]:
    if count == 1:
        return [indices[len(indices) // 2]]
    selected = []
    for position in range(count):
        offset = round(position * (len(indices) - 1) / (count - 1))
        selected.append(indices[offset])
    return selected


def select_common(path_entries: list[dict[str, Any]], count: int) -> list[int]:
    by_segment: dict[str, list[int]] = defaultdict(list)
    order: list[str] = []
    for index, entry in enumerate(path_entries):
        segment = str(entry.get("segment_id", "unknown"))
        if segment not in by_segment:
            order.append(segment)
        by_segment[segment].append(index)
    groups = [by_segment[segment] for segment in order]
    allocations = allocate_segment_counts(groups, count)
    return sorted(index for group, amount in zip(groups, allocations) for index in uniform_indices(group, amount))


def select_heldout(path_entries: list[dict[str, Any]], common: list[int]) -> list[int]:
    common_set = set(common)
    candidates: list[int] = []
    for left, right in zip(common, common[1:]):
        if path_entries[left].get("segment_id") != path_entries[right].get("segment_id"):
            continue
        midpoint = (left + right) // 2
        if midpoint not in common_set and midpoint not in candidates:
            candidates.append(midpoint)
    return candidates


def safe_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent))


def evaluation_split(entries: list[dict[str, Any]], scene_dir: Path, common: list[int]) -> tuple[list[int], list[int], dict[str, Any]]:
    """Freeze a conservative, reproducible image-content and temporal holdout split.

    Thumbnail MAE is a screening heuristic, not a guarantee of independent views.
    Full remains an explicitly in-sample diagnostic profile.
    """
    thumbnails, hashes = [], []
    for entry in entries:
        image_path = scene_dir / "source" / "images" / Path(entry["frame_image"]).name
        hashes.append(sha256_file(image_path))
        with Image.open(image_path) as image:
            thumbnails.append(image.convert("L").resize((64, 32), Image.Resampling.BOX))

    def near(left: int, right: int) -> bool:
        return hashes[left] == hashes[right] or ImageStat.Stat(
            ImageChops.difference(thumbnails[left], thumbnails[right])
        ).mean[0] <= 2.0

    candidates = select_heldout(entries, common)
    heldout = [index for index in candidates if not any(near(index, train) for train in common)]
    if not heldout:
        raise ValueError("No content-distinct heldout frames remain")
    excluded = set(heldout)
    for index in range(len(entries)):
        if any(near(index, hold) or (
            entries[index].get("segment_id") == entries[hold].get("segment_id")
            and abs(index - hold) <= 1
        ) for hold in heldout):
            excluded.add(index)
    dense, seen = [], set()
    for index, digest in enumerate(hashes):
        if index not in excluded and digest not in seen:
            dense.append(index)
            seen.add(digest)
    return heldout, dense, {
        "heldout_candidate_indices": candidates,
        "rejected_near_common_indices": sorted(set(candidates) - set(heldout)),
        "dense_excluded_indices": sorted(excluded),
        "temporal_buffer_frames_each_side_same_segment": 1,
        "near_duplicate_screen": "grayscale_64x32_box_thumbnail_mean_absolute_difference_le_2_of_255",
        "limitations": "Conservative heuristic; no guarantee of independent viewpoints. Heldout scoring requires independently estimated cameras and valid crop/mask mappings.",
    }


def write_profile(
    scene_dir: Path,
    profile: str,
    indices: Iterable[int],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile_dir = scene_dir / "profiles" / profile
    image_dir = profile_dir / "images"
    rows: list[dict[str, Any]] = []
    for profile_index, source_index in enumerate(indices):
        entry = entries[source_index]
        suffix = Path(entry["frame_image"]).suffix.lower() or ".jpg"
        target = image_dir / f"frame_{profile_index:06d}{suffix}"
        source = scene_dir / "source" / "images" / Path(entry["frame_image"]).name
        safe_link(source, target)
        rows.append(
            {
                "profile_index": profile_index,
                "source_index": source_index,
                "profile_file": target.name,
                "source_file": source.name,
                "frame": entry.get("frame"),
                "segment_id": entry.get("segment_id"),
                "video_time_s": entry.get("video_time_s"),
                "sequence_time_s": entry.get("sequence_time_s"),
            }
        )
    expected_names = {row["profile_file"] for row in rows}
    for existing in image_dir.glob("frame_*"):
        if existing.name not in expected_names:
            if not existing.is_symlink():
                raise ValueError(f"Refusing to remove non-generated profile file: {existing}")
            existing.unlink()
    with (profile_dir / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def prepare_scene(
    scene: dict[str, Any],
    bundle_dir: Path,
    cdn_base: str,
    refresh: bool,
) -> dict[str, Any]:
    scene_dir = bundle_dir / "scenes" / scene["key"]
    source_dir = scene_dir / "source"
    viewer_url = urllib.parse.urljoin(cdn_base, scene["viewer_key"])
    meta_path = source_dir / "scene_meta.json"
    download(viewer_url, meta_path, refresh)
    metadata = read_json(meta_path)
    entries = metadata["path"]
    if metadata["scene_id"] != scene["scene_id"]:
        raise ValueError(f"scene ID mismatch for {scene['key']}")
    if len(entries) != scene["expected_frame_count"]:
        raise ValueError(
            f"frame count mismatch for {scene['key']}: expected {scene['expected_frame_count']}, got {len(entries)}"
        )
    viewer_base = viewer_url.rsplit("/", 1)[0] + "/"
    file_records: list[dict[str, Any]] = []
    for entry in entries:
        relative = entry["frame_image"]
        destination = source_dir / "images" / Path(relative).name
        download(urllib.parse.urljoin(viewer_base, relative), destination, refresh)
        file_records.append(
            {
                "path": str(destination.relative_to(bundle_dir)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    baseline_dir = scene_dir / "baseline"
    for asset_name in ("positions", "colors"):
        relative = metadata["assets"][asset_name]
        destination = baseline_dir / Path(relative).name
        download(urllib.parse.urljoin(viewer_base, relative), destination, refresh)
        expand_gzip_in_place(destination)
        file_records.append(
            {
                "path": str(destination.relative_to(bundle_dir)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    common = select_common(entries, 16)
    heldout, dense_train, split_audit = evaluation_split(entries, scene_dir, common)
    profile_rows = {
        "full": write_profile(scene_dir, "full", range(len(entries)), entries),
        "common16": write_profile(scene_dir, "common16", common, entries),
        "dense_train": write_profile(scene_dir, "dense_train", dense_train, entries),
        "heldout": write_profile(scene_dir, "heldout", heldout, entries),
    }
    summary = {
        "key": scene["key"],
        "scene_id": scene["scene_id"],
        "viewer_url": viewer_url,
        "frame_count": len(entries),
        "segments": sorted({str(entry.get("segment_id")) for entry in entries}),
        "profiles": {name: len(rows) for name, rows in profile_rows.items()},
        "common16_source_indices": common,
        "heldout_source_indices": heldout,
        "dense_train_source_indices": dense_train,
        "split_audit": split_audit,
        "reference": scene["reference"],
        "source_files": file_records,
    }
    (scene_dir / "scene_bundle.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def create_archive(bundle_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    def portable_owner(member: tarfile.TarInfo) -> tarfile.TarInfo:
        member.uid = member.gid = 0
        member.uname = member.gname = "root"
        return member
    with tarfile.open(archive_path, "w:gz", dereference=False) as archive:
        archive.add(bundle_dir, arcname=bundle_dir.name, filter=portable_owner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--cdn-base", default=DEFAULT_CDN)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = read_json(args.spec)
    bundle_dir = args.work_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.spec, bundle_dir / "benchmark.json")
    for name in ("README.md", "REVISED_PLAN.md", "EXECUTION_STATUS.md", "calibration_references.json"):
        source_document = args.spec.with_name(name)
        if source_document.exists():
            shutil.copy2(source_document, bundle_dir / name)
    orchestration_dir = bundle_dir / "orchestration"
    orchestration_dir.mkdir(parents=True, exist_ok=True)
    for source_name in ORCHESTRATION_FILES:
        source = Path(source_name)
        if not source.exists():
            raise SystemExit(f"Missing orchestration file: {source}")
        destination = orchestration_dir / source.name
        shutil.copy2(source, destination)
        if source.suffix == ".sh":
            destination.chmod(0o755)
    scenes = [prepare_scene(scene, bundle_dir, args.cdn_base, args.refresh) for scene in spec["scenes"]]
    manifest = {
        "schema_version": 1,
        "benchmark_id": spec["benchmark_id"],
        "source_spec": str(args.spec),
        "scene_count": len(scenes),
        "orchestration_files": [Path(name).name for name in ORCHESTRATION_FILES],
        "scenes": scenes,
    }
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.archive:
        create_archive(bundle_dir, args.archive)
    print(
        json.dumps(
            {
                "bundle": str(bundle_dir),
                "archive": str(args.archive) if args.archive else None,
                "scenes": [
                    {
                        "key": scene["key"],
                        "frame_count": scene["frame_count"],
                        "profiles": scene["profiles"],
                        "source_file_count": len(scene["source_files"]),
                    }
                    for scene in scenes
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
