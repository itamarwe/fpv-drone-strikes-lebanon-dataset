#!/usr/bin/env python3
"""Build an aspect-preserving reference/render sheet from explicit pairs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=2, help="Number of frame pairs per row.")
    parser.add_argument("--cell-width", type=int, default=520)
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def main() -> int:
    args = parse_args()
    if args.columns <= 0 or args.cell_width < 240:
        raise SystemExit("--columns must be positive and --cell-width must be at least 240")
    manifest_path = args.pairs_manifest.resolve()
    payload = json.loads(manifest_path.read_text())
    records = payload.get("records")
    if payload.get("schema_version") != 1 or not isinstance(records, list) or not records:
        raise SystemExit("Expected a schema_version=1 manifest with non-empty records")
    root = manifest_path.parent
    pair_images = []
    for index, record in enumerate(records):
        mapping = record.get("pixel_mapping", {})
        if mapping.get("type") != "identity":
            raise SystemExit(f"records[{index}] is not explicitly identity-mapped")
        reference_path = resolve(root, str(record.get("reference", "")))
        render_path = resolve(root, str(record.get("render", "")))
        if not reference_path.is_file() or not render_path.is_file():
            raise SystemExit(f"records[{index}] references a missing image")
        with Image.open(reference_path) as source:
            reference = source.convert("RGB")
        with Image.open(render_path) as source:
            render = source.convert("RGB")
        if reference.size != render.size:
            raise SystemExit(f"records[{index}] identity pair has unequal sizes")
        half_width = args.cell_width // 2
        image_height = max(180, round(half_width * reference.height / reference.width))
        reference = ImageOps.contain(reference, (half_width, image_height))
        render = ImageOps.contain(render, (half_width, image_height))
        cell = Image.new("RGB", (args.cell_width, image_height + 54), "#0b0e12")
        cell.paste(reference, ((half_width - reference.width) // 2, (image_height - reference.height) // 2))
        cell.paste(render, (half_width + (half_width - render.width) // 2, (image_height - render.height) // 2))
        draw = ImageDraw.Draw(cell)
        font = ImageFont.load_default()
        draw.text((8, image_height + 7), f"{record.get('frame_id', index)} · {record.get('view_role', 'unknown')} view", fill="#f1f4f7", font=font)
        draw.text((8, image_height + 28), "INPUT / REFERENCE", fill="#6ed0ff", font=font)
        draw.text((half_width + 8, image_height + 28), "NATIVE RENDER", fill="#f3ba62", font=font)
        pair_images.append(cell)
    rows = (len(pair_images) + args.columns - 1) // args.columns
    cell_height = max(image.height for image in pair_images)
    sheet = Image.new("RGB", (args.columns * args.cell_width, rows * cell_height + 44), "#0b0e12")
    for index, pair in enumerate(pair_images):
        x = (index % args.columns) * args.cell_width
        y = (index // args.columns) * cell_height
        sheet.paste(pair, (x, y))
    draw = ImageDraw.Draw(sheet)
    roles = sorted({str(record.get("view_role", "unknown")) for record in records})
    if roles == ["evaluation"]:
        footer = "Explicit camera-linked evaluation views — proxy held-out cameras, not independent pose truth"
    elif roles == ["input"]:
        footer = "Explicit camera-linked input views — not held-out evaluation"
    else:
        footer = f"Explicit camera-linked views — roles: {', '.join(roles)}"
    draw.text((10, rows * cell_height + 15), footer, fill="#f1f4f7", font=ImageFont.load_default())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=94)
    metadata = {
        "schema_version": 1,
        "pairs_manifest": str(manifest_path),
        "pair_count": len(pair_images),
        "pairing": "explicit_manifest_only",
        "pixel_mapping": "identity_only_no_resize",
        "view_roles": sorted({str(record.get("view_role")) for record in records}),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
