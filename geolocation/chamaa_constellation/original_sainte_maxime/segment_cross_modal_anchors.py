#!/usr/bin/env python3
"""Segment comparable semantic anchors in a ground image and an orthophoto.

The output is deliberately image-space only: masks, boxes, centroids, scores,
and overlays.  Cross-view association is a later stage so segmentation quality
can be inspected without circularly assuming a location.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ortho_matching"))
from extract_scene_geometry import Sam3Segmenter  # noqa: E402


GROUPS = {
    "building": ["building", "house", "villa", "apartment building", "building roof",
                 "roof", "rooftop", "house roof", "building footprint",
                 "buildings seen from above", "house seen from above",
                 "villa seen from above", "aerial view of a house",
                 "aerial photograph of houses", "satellite view of buildings",
                 "satellite image of a house", "residential rooftops",
                 "red tiled roof", "white building roof", "orthophoto building",
                 "house in an orthophoto", "housing estate", "apartment complex"],
    "road": ["road", "street", "driveway", "golf cart path"],
    "fairway": ["golf course", "golf fairway", "green lawn"],
    "water": ["pond", "lake", "water"],
}

COLORS = {
    "building": (255, 85, 45),
    "road": (255, 190, 30),
    "fairway": (35, 205, 90),
    "water": (35, 145, 255),
}


def resize(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale == 1.0:
        return image, scale
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def deduplicate(items: list, threshold: float = 0.72) -> list:
    kept = []
    for item in sorted(items, key=lambda x: x.score, reverse=True):
        if not any(iou(item.mask, prior.mask) >= threshold for prior in kept):
            kept.append(item)
    return kept


def feature_record(item, semantic: str, scale: float, index: int) -> dict:
    ys, xs = np.where(item.mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "id": f"{semantic}_{index:03d}",
        "semantic": semantic,
        "prompt": item.prompt,
        "score": float(item.score),
        "area_px": int(item.mask.sum()),
        "centroid_xy": [float(xs.mean() / scale), float(ys.mean() / scale)],
        "bbox_xyxy": [float(x0 / scale), float(y0 / scale),
                       float(x1 / scale), float(y1 / scale)],
    }


def process(path: Path, name: str, out_dir: Path, model: Sam3Segmenter,
            max_side: int, threshold: float, groups: dict[str, list[str]]) -> dict:
    bgr_full = cv2.imread(str(path))
    if bgr_full is None:
        raise FileNotFoundError(path)
    bgr, scale = resize(bgr_full, max_side)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prompts = [p for values in groups.values() for p in values]
    # Run one semantic family at a time.  High-resolution orthophotos can
    # produce hundreds of masks per prompt; retaining every family at once is
    # unnecessarily memory hungry and can prevent the useful roof results
    # from ever being written.
    raw_by_semantic = {
        semantic: model.segment(rgb, group_prompts, threshold)
        for semantic, group_prompts in groups.items()
    }
    overlay = bgr.copy()
    records = []
    mask_dir = out_dir / f"{name}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for semantic, group_prompts in groups.items():
        instances = deduplicate(raw_by_semantic[semantic])
        for index, item in enumerate(instances):
            if item.mask.sum() < 30:
                continue
            record = feature_record(item, semantic, scale, index)
            records.append(record)
            color = np.asarray(COLORS[semantic], np.uint8)
            overlay[item.mask] = (0.58 * overlay[item.mask] + 0.42 * color).astype(np.uint8)
            cv2.imwrite(str(mask_dir / f"{record['id']}.png"), item.mask.astype(np.uint8) * 255)
            x0, y0, x1, y1 = [int(round(v * scale)) for v in record["bbox_xyxy"]]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), COLORS[semantic], 1)
            cv2.putText(overlay, record["id"], (x0, max(12, y0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLORS[semantic], 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{name}_overlay.jpg"), overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 94])
    prompt_counts = {
        prompt: sum(item.prompt == prompt for items in raw_by_semantic.values() for item in items)
        for prompt in prompts
    }
    return {"name": name, "image": str(path), "original_size": list(bgr_full.shape[1::-1]),
            "processed_size": list(bgr.shape[1::-1]), "scale": scale,
            "threshold": threshold, "prompt_counts": prompt_counts, "features": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground", type=Path, required=True)
    parser.add_argument("--ortho", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--only", choices=["both", "ground", "ortho"], default="both")
    parser.add_argument("--semantics", default="building,road,fairway,water",
                        help="comma-separated semantic families to run")
    parser.add_argument("--building-prompts", default=None,
                        help="comma-separated override for the building prompt family")
    args = parser.parse_args()
    selected = [name.strip() for name in args.semantics.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(GROUPS))
    if unknown:
        parser.error(f"unknown semantics: {', '.join(unknown)}")
    groups = {name: list(GROUPS[name]) for name in selected}
    if args.building_prompts is not None and "building" in groups:
        groups["building"] = [p.strip() for p in args.building_prompts.split(",") if p.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = Sam3Segmenter(args.model_id, args.device)
    output_path = args.out_dir / "segments.json"
    payload = json.loads(output_path.read_text()) if output_path.exists() else {}
    if args.only in {"both", "ground"}:
        payload["ground"] = process(args.ground, "ground", args.out_dir, model,
                                    args.max_side, args.threshold, groups)
    if args.only in {"both", "ortho"}:
        payload["ortho"] = process(args.ortho, "ortho", args.out_dir, model,
                                   args.max_side, args.threshold, groups)
    output_path.write_text(json.dumps(payload, indent=2))
    for key, item in payload.items():
        counts = {semantic: sum(f["semantic"] == semantic for f in item["features"])
                  for semantic in groups}
        print(key, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
