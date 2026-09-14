#!/usr/bin/env python3
"""Validate a prepared 3D benchmark bundle without running models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.bundle / "bundle_manifest.json"
    spec_path = args.bundle / "benchmark.json"
    if not manifest_path.exists() or not spec_path.exists():
        raise SystemExit("bundle_manifest.json and benchmark.json are required")
    manifest = json.loads(manifest_path.read_text())
    errors = []
    for filename in manifest.get("orchestration_files", []):
        path = args.bundle / "orchestration" / filename
        if not path.exists():
            errors.append(f"missing orchestration file: {path}")
    scene_results = []
    for scene in manifest["scenes"]:
        scene_dir = args.bundle / "scenes" / scene["key"]
        profiles = {}
        for profile, expected_count in scene["profiles"].items():
            csv_path = scene_dir / "profiles" / profile / "frames.csv"
            if not csv_path.exists():
                errors.append(f"missing {csv_path}")
                continue
            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))
            images = sorted((scene_dir / "profiles" / profile / "images").glob("*"))
            profiles[profile] = len(images)
            if len(rows) != expected_count or len(images) != expected_count:
                errors.append(f"{scene['key']}/{profile}: expected {expected_count}, csv={len(rows)}, images={len(images)}")
            for image in images:
                if not image.exists():
                    errors.append(f"broken image link: {image}")
        if not args.skip_hashes:
            for record in scene["source_files"]:
                path = args.bundle / record["path"]
                if not path.exists():
                    errors.append(f"missing {path}")
                elif path.stat().st_size != record["bytes"]:
                    errors.append(f"size mismatch: {path}")
                elif sha256_file(path) != record["sha256"]:
                    errors.append(f"sha256 mismatch: {path}")
        scene_results.append({"scene": scene["key"], "profiles": profiles})
    report = {"ok": not errors, "scenes": scene_results, "errors": errors}
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
