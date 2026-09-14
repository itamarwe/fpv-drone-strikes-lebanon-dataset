#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "model_backups/huggingface/vggt-omega"
MODEL_REPO = "facebook/VGGT-Omega"
SPACE_REPO = "facebook/vggt-omega"
MODEL_FILE = "vggt_omega_1b_512.pt"
SKYSEG_FILE = "skyseg.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_hf_file(repo_id: str, filename: str, out_dir: Path, repo_type: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"repo_id": repo_id, "filename": filename, "local_dir": str(out_dir)}
    if repo_type:
        kwargs["repo_type"] = repo_type
    path = Path(hf_hub_download(**kwargs))
    return path


def upload_to_s3(local_dir: Path, s3_uri: str, profile: str, region: str) -> None:
    cmd = [
        "aws",
        "s3",
        "sync",
        str(local_dir),
        s3_uri.rstrip("/") + "/",
        "--only-show-errors",
        "--sse",
        "AES256",
    ]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VGGT-Omega gated assets locally and optionally sync them to S3.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--s3-uri", default=os.environ.get("VGGT_OMEGA_BACKUP_S3", ""))
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""))
    parser.add_argument("--aws-region", default=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "")))
    parser.add_argument("--skip-s3", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token_file = Path.home() / ".cache/huggingface/token"
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or token_file.exists()):
        raise SystemExit("Missing HF token. Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or save ~/.cache/huggingface/token.")

    model_dir = args.out_dir / "model-facebook-VGGT-Omega"
    space_dir = args.out_dir / "space-facebook-vggt-omega"
    model_path = copy_hf_file(MODEL_REPO, MODEL_FILE, model_dir)
    skyseg_path = copy_hf_file(SPACE_REPO, SKYSEG_FILE, space_dir, repo_type="space")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assets": [
            {
                "repo_id": MODEL_REPO,
                "filename": MODEL_FILE,
                "local_path": str(model_path),
                "size_bytes": model_path.stat().st_size,
                "sha256": sha256(model_path),
            },
            {
                "repo_id": SPACE_REPO,
                "repo_type": "space",
                "filename": SKYSEG_FILE,
                "local_path": str(skyseg_path),
                "size_bytes": skyseg_path.stat().st_size,
                "sha256": sha256(skyseg_path),
            },
        ],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"out_dir": str(args.out_dir), "manifest": str(manifest_path), "assets": manifest["assets"]}, indent=2))

    if args.s3_uri and not args.skip_s3:
        upload_to_s3(args.out_dir, args.s3_uri, args.aws_profile, args.aws_region)
        print(json.dumps({"s3_uri": args.s3_uri.rstrip("/") + "/", "uploaded": True}, indent=2))
    elif args.s3_uri:
        print(json.dumps({"s3_uri": args.s3_uri.rstrip("/") + "/", "uploaded": False, "reason": "skip-s3"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
