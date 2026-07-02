#!/usr/bin/env python3
"""Generate prompted SAM masks for an object in VGGT crop frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path, help="VGGT reconstruction dir with frames/")
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/sam-vit-base")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--crop-left", type=int, default=50)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=560)
    parser.add_argument("--crop-height", type=int, default=280)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=2.5, help="Penalty starts above this mask/polygon area ratio")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_prompts(path: Path) -> dict[int, dict[str, object]]:
    payload = json.loads(path.read_text())
    frames = payload.get("frames", payload)
    prompts: dict[int, dict[str, object]] = {}
    for key, value in frames.items():
        if isinstance(value, dict):
            polygon = value.get("polygon", value.get("points"))
            positive = value.get("positive_points", [])
            negative = value.get("negative_points", [])
        else:
            polygon = value
            positive = []
            negative = []
        if polygon is None:
            raise ValueError(f"Missing polygon for frame {key}")
        prompts[int(key)] = {
            "polygon": [(float(x), float(y)) for x, y in polygon],
            "positive_points": [(float(x), float(y)) for x, y in positive],
            "negative_points": [(float(x), float(y)) for x, y in negative],
        }
    return prompts


def polygon_bbox(polygon: list[tuple[float, float]]) -> list[float]:
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def polygon_mask(polygon: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).polygon(polygon, fill=1)
    return np.asarray(image, dtype=bool)


def centroid_point(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    arr = np.asarray(polygon, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def score_mask(mask: np.ndarray, rough: np.ndarray, iou_score: float, max_area_ratio: float) -> float:
    mask_area = float(mask.sum())
    rough_area = float(rough.sum())
    if mask_area <= 0 or rough_area <= 0:
        return -1e9
    overlap = float((mask & rough).sum())
    overlap_recall = overlap / rough_area
    overlap_precision = overlap / mask_area
    area_ratio = mask_area / rough_area
    area_penalty = max(0.0, area_ratio - max_area_ratio) * 0.35
    return float(iou_score) + 0.6 * overlap_precision + 0.35 * overlap_recall - area_penalty


def make_overlay(image: Image.Image, rough: np.ndarray, mask: np.ndarray) -> Image.Image:
    overlay = image.convert("RGBA")
    arr = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    arr[rough] = [255, 80, 30, 70]
    arr[mask] = [0, 255, 255, 100]
    return Image.alpha_composite(overlay, Image.fromarray(arr, "RGBA")).convert("RGB")


def paste_with_label(sheet: Image.Image, image: Image.Image, x: int, y: int, label: str) -> None:
    draw = ImageDraw.Draw(sheet)
    draw.text((x + 6, y + 4), label, fill=(235, 235, 235))
    sheet.paste(image, (x, y + 24))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts_json)
    device = choose_device(args.device)
    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to(device).eval()
    crop_box = (
        args.crop_left,
        args.crop_top,
        args.crop_left + args.crop_width,
        args.crop_top + args.crop_height,
    )

    manifest: dict[str, object] = {
        "video_dir": str(args.video_dir),
        "prompts_json": str(args.prompts_json),
        "model": args.model,
        "device": str(device),
        "crop_left_top_width_height": [args.crop_left, args.crop_top, args.crop_width, args.crop_height],
        "mask_threshold": args.mask_threshold,
        "frames": {},
    }

    sheet_rows = []
    for frame_number, prompt in sorted(prompts.items()):
        image = Image.open(args.video_dir / "frames" / f"f_{frame_number:06d}.jpg").convert("RGB").crop(crop_box)
        polygon = prompt["polygon"]
        rough_mask = polygon_mask(polygon, args.crop_width, args.crop_height)
        box = polygon_bbox(polygon)
        positive = list(prompt["positive_points"])
        negative = list(prompt["negative_points"])
        if not positive:
            positive = [centroid_point(polygon)]
        point_list = positive + negative
        labels = [1] * len(positive) + [0] * len(negative)

        inputs = processor(
            image,
            input_boxes=[[box]],
            input_points=[point_list],
            input_labels=[labels],
            return_tensors="pt",
        )
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and key not in {"original_sizes", "reshaped_input_sizes"}:
                if torch.is_floating_point(value):
                    value = value.float()
                inputs[key] = value.to(device)
        with torch.inference_mode():
            outputs = model(**inputs, multimask_output=True)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            binarize=False,
        )[0][0]
        masks_np = masks.float().numpy()
        iou_scores = outputs.iou_scores.detach().cpu().numpy()[0, 0]
        candidate_masks = masks_np > args.mask_threshold
        scores = [
            score_mask(candidate_masks[i], rough_mask, float(iou_scores[i]), args.max_area_ratio)
            for i in range(candidate_masks.shape[0])
        ]
        best = int(np.argmax(scores))
        selected_mask = candidate_masks[best]

        mask_path = args.out_dir / f"f_{frame_number:06d}_sam_mask.png"
        Image.fromarray((selected_mask.astype(np.uint8) * 255), mode="L").save(mask_path)
        overlay = make_overlay(image, rough_mask, selected_mask)
        overlay_path = args.out_dir / f"f_{frame_number:06d}_sam_overlay.jpg"
        overlay.save(overlay_path, quality=95)

        manifest["frames"][str(frame_number)] = {
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "box": box,
            "positive_points": positive,
            "negative_points": negative,
            "rough_area_px": int(rough_mask.sum()),
            "mask_area_px": int(selected_mask.sum()),
            "sam_iou_scores": [float(x) for x in iou_scores],
            "selection_scores": [float(x) for x in scores],
            "selected_index": best,
        }
        sheet_rows.append((frame_number, image, overlay, selected_mask, rough_mask, iou_scores, scores, best))

    width = args.crop_width
    height = args.crop_height
    sheet = Image.new("RGB", (width * 2, (height + 24) * len(sheet_rows)), (18, 18, 18))
    for row_index, (frame_number, image, overlay, mask, rough, ious, scores, best) in enumerate(sheet_rows):
        y = row_index * (height + 24)
        label = (
            f"f_{frame_number:06d} mask={int(mask.sum())} rough={int(rough.sum())} "
            f"sam={ious[best]:.3f} score={scores[best]:.3f}"
        )
        paste_with_label(sheet, image, 0, y, label)
        paste_with_label(sheet, overlay, width, y, "rough=orange, SAM=cyan")
    contact_sheet = args.out_dir / "sam_mask_contact_sheet.jpg"
    sheet.save(contact_sheet, quality=95)
    manifest["contact_sheet"] = str(contact_sheet)
    (args.out_dir / "sam_masks_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(args.out_dir / "sam_masks_manifest.json")
    print(contact_sheet)


if __name__ == "__main__":
    main()
