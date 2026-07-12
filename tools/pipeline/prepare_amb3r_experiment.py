#!/usr/bin/env python3
"""Prepare a small AMB3R input folder from a VGGT reconstruction frame set."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_RECON_DIR = Path(
    "/tmp/fpv-flight-paths/attack_reconstructions/"
    "2026-06-06_sholef_howitzer_adaissah_attack_overlap"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon-dir", type=Path, default=DEFAULT_RECON_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/fpv-flight-paths/amb3r_inputs/sholef_smoke"))
    parser.add_argument("--frame-step", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames_csv = args.recon_dir / "frames.csv"
    frames_dir = args.recon_dir / "frames"
    if not frames_csv.exists():
        raise FileNotFoundError(f"Missing {frames_csv}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Missing {frames_dir}")

    rows = list(csv.DictReader(frames_csv.open()))
    selected = rows[:: max(args.frame_step, 1)][: args.max_frames]
    if not selected:
        raise SystemExit("No frames selected.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["amb3r_file", "source_file", "source_frame_index", "segment_id", "is_attack"],
        )
        writer.writeheader()
        for out_idx, row in enumerate(selected, start=1):
            src = frames_dir / row["file"]
            dst_name = f"{out_idx:06d}.jpg"
            dst = args.out_dir / dst_name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)
            writer.writerow(
                {
                    "amb3r_file": dst_name,
                    "source_file": str(src),
                    "source_frame_index": row["frame_index"],
                    "segment_id": row.get("segment_id", ""),
                    "is_attack": row.get("is_attack", ""),
                }
            )

    print(f"[amb3r] selected {len(selected)} frames from {args.recon_dir}")
    print(f"[amb3r] wrote image folder {args.out_dir}")
    print(f"[amb3r] wrote manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
