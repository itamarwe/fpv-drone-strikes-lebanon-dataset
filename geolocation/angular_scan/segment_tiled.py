#!/usr/bin/env python3
"""SAM 3 building detections on a large photo, in overlapping full-resolution tiles.

Uses the Sainte-Maxime segmentation code unchanged (tools/geolocation/segment_cross_modal_anchors.py
in the main repository: prompts building/house/roof, threshold 0.3, mask-IoU dedup 0.72) on each
tile, shifts the records to full-image pixels and removes cross-tile duplicates (box IoU >= 0.5,
keeping the higher score). Output: <out>/segments.json in the same format as a single-image run.

  python3 segment_tiled.py scenes/saint_paul/photo.jpg scenes/saint_paul/sam --tile 1000 --overlap 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

MAIN = Path.home() / "Documents/code/fpv-drone-strikes-lebanon-dataset"
sys.path.insert(0, str(MAIN / "tools/geolocation"))
T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:7.1f} s] {msg}", flush=True)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--tile", type=int, default=1000); ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--upscale", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.3); ap.add_argument("--prompts", default="building,house,roof")
    args = ap.parse_args()
    import segment_cross_modal_anchors as seg
    img = cv2.imread(str(args.image)); H0, W0 = img.shape[:2]
    if args.upscale != 1.0:                     # small video frames: segment an enlarged copy
        img = cv2.resize(img, None, fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    step = args.tile - args.overlap
    xs = sorted(set(list(range(0, max(1, W - args.tile), step)) + [max(0, W - args.tile)]))
    ys = sorted(set(list(range(0, max(1, H - args.tile), step)) + [max(0, H - args.tile)]))
    args.out.mkdir(parents=True, exist_ok=True); tiles_dir = args.out / "tiles"; tiles_dir.mkdir(exist_ok=True)
    log(f"loading SAM 3; {len(xs) * len(ys)} tiles of {args.tile} px over {W}x{H}")
    model = seg.Sam3Segmenter("facebook/sam3", "mps")
    groups = {"building": [p.strip() for p in args.prompts.split(",")]}
    feats = []; counts = {}
    for ty in ys:
        for tx in xs:
            name = f"tile_{tx}_{ty}"; path = tiles_dir / f"{name}.jpg"
            cv2.imwrite(str(path), img[ty:ty + args.tile, tx:tx + args.tile], [cv2.IMWRITE_JPEG_QUALITY, 97])
            rec = seg.process(path, name, tiles_dir, model, args.tile, args.threshold, groups)
            for f in rec["features"]:
                f = dict(f); f["tile"] = name
                f["centroid_xy"] = [f["centroid_xy"][0] + tx, f["centroid_xy"][1] + ty]
                x0, y0, x1, y1 = f["bbox_xyxy"]
                f["bbox_xyxy"] = [x0 + tx, y0 + ty, x1 + tx, y1 + ty]
                # a box touching an inner tile edge is truncated: drop it if a neighbour tile covers that edge
                cut = ((x0 <= 2 and tx > 0) or (y0 <= 2 and ty > 0) or (x1 >= args.tile - 3 and tx + args.tile < W)
                       or (y1 >= args.tile - 3 and ty + args.tile < H))
                if not cut:
                    feats.append(f)
            for k, v in rec["prompt_counts"].items():
                counts[k] = counts.get(k, 0) + v
            log(f"{name}: {len(rec['features'])} masks (running total {len(feats)})")
    feats.sort(key=lambda f: -f["score"]); kept = []
    for f in feats:
        if all(box_iou(f["bbox_xyxy"], g["bbox_xyxy"]) < .5 for g in kept):
            kept.append(f)
    for i, f in enumerate(kept):
        f["id"] = f"building_{i:03d}"
        if args.upscale != 1.0:
            s = 1 / args.upscale
            f["centroid_xy"] = [v * s for v in f["centroid_xy"]]; f["bbox_xyxy"] = [v * s for v in f["bbox_xyxy"]]
            f["area_px"] = f["area_px"] * s * s
    out = dict(ground=dict(name="ground", image=str(args.image), original_size=[W0, H0], processed_size=[W, H], scale=args.upscale,
                           threshold=args.threshold, prompt_counts=counts, tiles=dict(size=args.tile, overlap=args.overlap),
                           features=kept))
    (args.out / "segments.json").write_text(json.dumps(out, indent=2))
    ov = img.copy()
    for f in kept:
        x0, y0, x1, y1 = [int(v * args.upscale) for v in f["bbox_xyxy"]]; cv2.rectangle(ov, (x0, y0), (x1, y1), (45, 85, 255), 2)
    cv2.imwrite(str(args.out / "overlay.jpg"), cv2.resize(ov, (W // 2, H // 2)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    log(f"{len(kept)} buildings after cross-tile dedup -> {args.out / 'segments.json'}")


if __name__ == "__main__":
    main()
