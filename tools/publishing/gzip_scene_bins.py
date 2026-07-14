#!/usr/bin/env python3
"""Migrate existing S3 scene binaries to transparent gzip encoding.

The command is resumable: objects already carrying ``Content-Encoding: gzip``
are skipped. Each remaining object is downloaded, deterministically compressed,
uploaded atomically to the same key with corrected metadata, and verified.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any


PRINT_LOCK = threading.Lock()


def aws_command(profile: str, *args: str) -> list[str]:
    command = ["aws", *args]
    if profile:
        command.extend(["--profile", profile])
    return command


def run_aws(profile: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        aws_command(profile, *args),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "AWS_PAGER": ""},
    )


def json_aws(profile: str, *args: str) -> Any:
    result = run_aws(profile, *args, "--output", "json")
    return json.loads(result.stdout or "null")


def is_scene_binary(key: str) -> bool:
    return key.startswith("scenes/") and "/viewer/" in key and key.endswith(".bin")


def list_scene_binaries(bucket: str, prefix: str, profile: str) -> list[dict[str, Any]]:
    payload = json_aws(profile, "s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix)
    return [row for row in payload.get("Contents", []) if is_scene_binary(str(row.get("Key", "")))]


def head_object(bucket: str, key: str, profile: str) -> dict[str, Any]:
    return json_aws(profile, "s3api", "head-object", "--bucket", bucket, "--key", key)


def gzip_file(source: Path, target: Path) -> None:
    with source.open("rb") as src, target.open("wb") as raw_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, compresslevel=9, mtime=0) as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)


def migrate_object(
    row: dict[str, Any],
    *,
    bucket: str,
    profile: str,
    force: bool,
) -> dict[str, Any]:
    key = str(row["Key"])
    before = head_object(bucket, key, profile)
    encoding = str(before.get("ContentEncoding") or "").lower()
    if "gzip" in {part.strip() for part in encoding.split(",")} and not force:
        return {"key": key, "status": "skipped", "compressed_bytes": int(before.get("ContentLength", 0))}

    with tempfile.TemporaryDirectory(prefix="fpv-scene-gzip-") as tmp:
        raw_path = Path(tmp) / "scene.bin"
        gzip_path = Path(tmp) / "scene.bin.gz"
        run_aws(
            profile,
            "s3",
            "cp",
            f"s3://{bucket}/{key}",
            str(raw_path),
            "--only-show-errors",
        )
        gzip_file(raw_path, gzip_path)
        run_aws(
            profile,
            "s3",
            "cp",
            str(gzip_path),
            f"s3://{bucket}/{key}",
            "--only-show-errors",
            "--content-type",
            "application/octet-stream",
            "--content-encoding",
            "gzip",
            "--cache-control",
            "public,max-age=31536000",
        )
        after = head_object(bucket, key, profile)
        if str(after.get("ContentEncoding") or "").lower() != "gzip":
            raise RuntimeError(f"{key}: uploaded object is missing Content-Encoding: gzip")
        if int(after.get("ContentLength", -1)) != gzip_path.stat().st_size:
            raise RuntimeError(f"{key}: uploaded compressed size does not match local result")
        return {
            "key": key,
            "status": "migrated",
            "raw_bytes": raw_path.stat().st_size,
            "compressed_bytes": gzip_path.stat().st_size,
        }


def invalidate(distribution_id: str, profile: str) -> str:
    payload = json_aws(
        profile,
        "cloudfront",
        "create-invalidation",
        "--distribution-id",
        distribution_id,
        "--paths",
        "/scenes/*",
    )
    return str(payload["Invalidation"]["Id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.environ.get("FPV_BUCKET_NAME", "fpv-drone-strikes-lebanon-dataset"))
    parser.add_argument("--prefix", default="scenes/")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "admin"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="recompress objects that already have gzip metadata")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--invalidate-distribution")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    objects = list_scene_binaries(args.bucket, args.prefix, args.profile)
    if args.limit:
        objects = objects[: args.limit]
    print(f"[gzip-scenes] found {len(objects)} scene binaries", flush=True)
    if args.dry_run:
        for row in objects:
            print(row["Key"])
        return 0

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                migrate_object,
                row,
                bucket=args.bucket,
                profile=args.profile,
                force=args.force,
            ): str(row["Key"])
            for row in objects
        }
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            key = futures[future]
            try:
                result = future.result()
                results.append(result)
                with PRINT_LOCK:
                    print(f"[gzip-scenes {completed}/{len(objects)}] {result['status']} {key}", flush=True)
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                with PRINT_LOCK:
                    print(f"[gzip-scenes {completed}/{len(objects)}] ERROR {key}: {exc}", flush=True)

    migrated = [row for row in results if row["status"] == "migrated"]
    raw_bytes = sum(int(row.get("raw_bytes", 0)) for row in migrated)
    compressed_bytes = sum(int(row.get("compressed_bytes", 0)) for row in migrated)
    print(
        f"[gzip-scenes] migrated={len(migrated)} skipped={len(results) - len(migrated)} "
        f"errors={len(errors)} raw_bytes={raw_bytes} compressed_bytes={compressed_bytes}",
        flush=True,
    )
    if errors:
        for error in errors:
            print(f"[gzip-scenes] {error}", flush=True)
        return 1
    if migrated and args.invalidate_distribution:
        invalidation_id = invalidate(args.invalidate_distribution, args.profile)
        print(f"[gzip-scenes] invalidation={invalidation_id} path=/scenes/*", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
