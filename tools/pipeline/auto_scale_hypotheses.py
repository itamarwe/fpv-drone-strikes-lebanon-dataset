#!/usr/bin/env python3
"""Automatic scale hypothesis generation and robust voting.

This script turns many weak automatic visual cues into scale votes:

- open-vocabulary detections: tank, Humvee, Namer, D9, person, window, etc.
- object-size priors: known width/height/length ranges in meters
- optional metric depth estimates per detection
- the VGGT relative camera path

The core assumption for the first automatic pass is:

    if a known-size object is visible in an attack/overlap frame, its apparent
    size or estimated metric depth gives camera-to-target distance in meters.
    VGGT gives camera-to-terminal distance in scene units. Their ratio is a
    meters-per-VGGT-unit scale vote.

This assumption is imperfect. The script therefore emits every vote with a
weight and then uses log-space consensus clustering rather than trusting any
single object or frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_OUT_DIR = Path("/tmp/fpv-flight-paths")


OBJECT_PRIORS = {
    # Values are priors, not exact variant identification. They should be
    # replaced by measured object constraints when a specific target is known.
    "merkava": {
        "aliases": ["merkava", "tank", "main battle tank", "armored tank"],
        "length_m": 7.60,
        "width_m": 3.72,
        "height_m": 2.66,
        "prior_confidence": 0.75,
        "source": "Army Recognition / WeaponSystems dimensions for Merkava 4 class.",
    },
    "humvee": {
        "aliases": ["humvee", "hmmwv", "military jeep", "military vehicle"],
        "length_m": 4.57,
        "width_m": 2.18,
        "height_m": 1.83,
        "prior_confidence": 0.70,
        "source": "AM General HMMWV common dimensions; variant spread is significant.",
    },
    "namer": {
        "aliases": ["namer", "namer apc", "armored personnel carrier", "apc"],
        "length_m": 7.50,
        "width_m": 3.80,
        "height_m": 2.00,
        "prior_confidence": 0.70,
        "source": "Army Recognition / WeaponSystems Namer dimensions.",
    },
    "d9": {
        "aliases": ["d9", "d9 bulldozer", "bulldozer", "armored bulldozer", "engineering vehicle"],
        "length_m": 6.60,
        "width_m": 4.35,
        "height_m": 3.88,
        "prior_confidence": 0.65,
        "source": "Caterpillar D9 transport dimensions / blade-width specs; armor kit may change silhouette.",
    },
    "soldier": {
        "aliases": ["soldier", "person", "human", "man"],
        "height_m": 1.72,
        "width_m": 0.48,
        "prior_confidence": 0.45,
        "source": "Human-height prior; pose and partial visibility make this noisy.",
    },
    "window": {
        "aliases": ["window", "building window"],
        "height_m": 1.20,
        "width_m": 1.00,
        "prior_confidence": 0.35,
        "source": "Generic residential window prior; use only as weak vote.",
    },
    "door": {
        "aliases": ["door", "building door"],
        "height_m": 2.05,
        "width_m": 0.90,
        "prior_confidence": 0.45,
        "source": "Generic door prior; use only as weak vote.",
    },
    "story": {
        "aliases": ["building story", "floor", "storey"],
        "height_m": 3.0,
        "prior_confidence": 0.35,
        "source": "Typical floor-to-floor height prior.",
    },
    "tree": {
        "aliases": ["tree"],
        "height_m": 6.0,
        "prior_confidence": 0.20,
        "source": "Very weak generic tree-height prior; mostly useful as a sanity vote.",
    },
}


@dataclass
class Vote:
    video_id: str
    frame_index: int
    label: str
    prior_key: str
    method: str
    scale_m_per_unit: float
    weight: float
    known_meters: float
    observed_pixels: float | str
    metric_distance_m: float
    relative_distance_units: float
    detection_score: float
    notes: str


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def prior_for_label(label: str) -> tuple[str, dict] | tuple[None, None]:
    norm = normalize_label(label)
    for key, prior in OBJECT_PRIORS.items():
        for alias in prior["aliases"]:
            alias_norm = normalize_label(alias)
            if alias_norm == norm or alias_norm in norm:
                return key, prior
    return None, None


def load_path(path_csv: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows = list(csv.DictReader(path_csv.open()))
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    return rows, xyz


def terminal_index(rows: list[dict[str, str]]) -> int:
    attack_indexes = [idx for idx, row in enumerate(rows) if row.get("is_attack") == "true"]
    return attack_indexes[-1] if attack_indexes else len(rows) - 1


def focal_pixels(width: int, height: int, hfov_deg: float | None, vfov_deg: float | None) -> tuple[float, float]:
    if hfov_deg is None and vfov_deg is None:
        raise ValueError("Need --hfov-deg or --vfov-deg for pixel-size distance cues.")
    if hfov_deg is None:
        fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        return fy, fy
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    if vfov_deg is None:
        fy = fx
    else:
        fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return fx, fy


def read_detections(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def frame_image_path(recon_dir: Path, row_by_frame: dict[int, dict[str, str]], frame_index: int) -> Path:
    file_name = row_by_frame[frame_index]["file"]
    return recon_dir / "frames" / file_name


def center_weight(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> float:
    cx = ((x1 + x2) / 2.0 - width / 2.0) / (width / 2.0)
    cy = ((y1 + y2) / 2.0 - height / 2.0) / (height / 2.0)
    r2 = cx * cx + cy * cy
    return float(math.exp(-1.35 * r2))


def box_area_weight(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> float:
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frac = area / max(width * height, 1)
    # Very tiny boxes are weak; huge boxes are likely partial/close-up.
    return float(np.clip(math.sqrt(max(frac, 0.0)) * 8.0, 0.15, 1.0))


def is_artifact_filter_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "disable_artifact_filter", False))


def detection_artifact_reason(det: dict[str, str], image: Image.Image, args: argparse.Namespace) -> str | None:
    """Reject common FPV overlay detections before they become scale votes."""
    width, height = image.size
    x1, y1, x2, y2 = [float(det[k]) for k in ["x1", "y1", "x2", "y2"]]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    aspect = box_w / box_h
    center_x = ((x1 + x2) / 2.0) / max(width, 1)
    center_y = ((y1 + y2) / 2.0) / max(height, 1)
    min_aspect = float(getattr(args, "hud_min_aspect", 3.8))
    top_fraction = float(getattr(args, "hud_top_fraction", 0.30))

    # Propeller / HUD strokes often live in the top band and are long, flat,
    # dark shapes. They are visually object-like to open-vocabulary detectors.
    if y1 <= 0.03 * height and center_y < 0.38:
        return "touches_top_overlay"
    if aspect >= min_aspect and center_y < top_fraction:
        return "top_horizontal_hud_stroke"

    # The central reticle produces repeated horizontal glyphs near the horizon.
    if (
        aspect >= min_aspect
        and 0.30 <= center_y <= 0.53
        and 0.25 <= center_x <= 0.75
        and box_h <= 0.14 * height
    ):
        return "central_reticle_horizontal_stroke"
    if (
        0.32 <= center_x <= 0.68
        and 0.38 <= center_y <= 0.50
        and box_w <= 0.12 * width
        and box_h <= 0.08 * height
    ):
        return "central_reticle_glyph"

    x1i = max(0, min(width, int(round(x1))))
    y1i = max(0, min(height, int(round(y1))))
    x2i = max(0, min(width, int(round(x2))))
    y2i = max(0, min(height, int(round(y2))))
    if x2i <= x1i or y2i <= y1i:
        return "empty_box"
    crop = np.asarray(image.crop((x1i, y1i, x2i, y2i)).convert("RGB"))
    gray = crop.mean(axis=2)
    dark = gray < 55
    dark_total = int(dark.sum())
    if dark_total:
        row_counts = dark.sum(axis=1)
        row_concentration = float(row_counts.max() / dark_total)
        dense_row_fraction = float((row_counts > 0.35 * dark.shape[1]).sum() / max(dark.shape[0], 1))
        if aspect >= 4.5 and center_y < 0.55 and (row_concentration >= 0.18 or dense_row_fraction >= 0.18):
            return "horizontal_hud_stroke"

    # Some clips have a fixed yellow/red HUD marker in the lower-left overlay.
    red = ((crop[:, :, 0] > 170) & (crop[:, :, 1] < 100) & (crop[:, :, 2] < 100)).mean()
    yellow = ((crop[:, :, 0] > 190) & (crop[:, :, 1] > 150) & (crop[:, :, 2] < 110)).mean()
    if center_x < 0.18 and 0.62 <= center_y <= 0.86 and aspect <= 1.2 and (red >= 0.015 or yellow >= 0.55):
        return "lower_left_hud_marker"
    return None


def generate_votes(args: argparse.Namespace) -> list[Vote]:
    recon_dir = args.recon_dir
    rows, xyz = load_path(recon_dir / "relative_path.csv")
    row_by_frame = {int(r["frame_index"]): r for r in rows}
    term_idx = terminal_index(rows)
    terminal = xyz[term_idx]
    detections = read_detections(args.detections)
    video_id = recon_dir.name
    votes: list[Vote] = []
    size_cache: dict[int, tuple[int, int]] = {}
    image_cache: dict[int, Image.Image] = {}
    filter_counts: Counter[str] = Counter()

    for det in detections:
        frame_index = int(det["frame_index"])
        if frame_index not in row_by_frame or frame_index - 1 >= len(xyz):
            continue
        key, prior = prior_for_label(det["label"])
        if prior is None:
            continue
        x1, y1, x2, y2 = [float(det[k]) for k in ["x1", "y1", "x2", "y2"]]
        score = float(det.get("score") or 0.5)
        if frame_index not in image_cache:
            image_cache[frame_index] = Image.open(
                frame_image_path(recon_dir, row_by_frame, frame_index)
            ).convert("RGB")
            size_cache[frame_index] = image_cache[frame_index].size
        img = image_cache[frame_index]
        if is_artifact_filter_enabled(args):
            reason = detection_artifact_reason(det, img, args)
            if reason:
                filter_counts[reason] += 1
                continue
        width, height = size_cache[frame_index]
        fx, fy = focal_pixels(width, height, args.hfov_deg, args.vfov_deg)
        rel_distance = float(np.linalg.norm(xyz[frame_index - 1] - terminal))
        if rel_distance < args.min_relative_distance:
            continue
        base_weight = (
            float(prior["prior_confidence"])
            * np.clip(score, 0.0, 1.0)
            * center_weight(x1, y1, x2, y2, width, height)
            * box_area_weight(x1, y1, x2, y2, width, height)
        )

        # Pixel angular-size cues.
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        for dim_name, observed_px, focal in [
            ("width_m", box_w, fx),
            ("height_m", box_h, fy),
        ]:
            if dim_name not in prior:
                continue
            known = float(prior[dim_name])
            metric_distance = known * focal / observed_px
            scale = metric_distance / rel_distance
            if not np.isfinite(scale) or scale <= 0:
                continue
            dim_weight = base_weight * (0.85 if dim_name == "height_m" else 0.70)
            votes.append(
                Vote(
                    video_id=video_id,
                    frame_index=frame_index,
                    label=det["label"],
                    prior_key=key,
                    method=f"apparent_{dim_name}",
                    scale_m_per_unit=scale,
                    weight=float(dim_weight),
                    known_meters=known,
                    observed_pixels=float(observed_px),
                    metric_distance_m=float(metric_distance),
                    relative_distance_units=rel_distance,
                    detection_score=score,
                    notes=prior["source"],
                )
            )

        # Optional metric-depth cue, e.g. from Depth Anything V2 metric model.
        if det.get("depth_m"):
            depth_m = float(det["depth_m"])
            if depth_m > 0:
                scale = depth_m / rel_distance
                votes.append(
                    Vote(
                        video_id=video_id,
                        frame_index=frame_index,
                        label=det["label"],
                        prior_key=key,
                        method="metric_depth_to_terminal",
                        scale_m_per_unit=scale,
                        weight=float(base_weight * 1.15),
                        known_meters="",
                        observed_pixels="",
                        metric_distance_m=depth_m,
                        relative_distance_units=rel_distance,
                        detection_score=score,
                        notes="Metric depth cue from detection row depth_m.",
                    )
                )
    if filter_counts:
        print(
            "[artifact-filter] skipped "
            f"{sum(filter_counts.values())} detections before voting: {dict(filter_counts)}"
        )
    return votes


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cdf, 0.5)])


def winning_cluster(
    scales: np.ndarray, weights: np.ndarray, log_window: float
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    logs = np.log(scales)
    order = np.argsort(logs)
    best = (0.0, 0, 0)
    for left in range(len(order)):
        right = left
        while right < len(order) and logs[order[right]] - logs[order[left]] <= log_window:
            right += 1
        weight_sum = float(weights[order[left:right]].sum())
        if weight_sum > best[0]:
            best = (weight_sum, left, right)
    _, left, right = best
    inlier_idx = order[left:right]
    outlier_idx = np.setdiff1d(np.arange(len(scales)), inlier_idx)
    med_log = weighted_median(logs[inlier_idx], weights[inlier_idx])
    bounds = (float(np.min(logs[inlier_idx])), float(np.max(logs[inlier_idx])))
    return inlier_idx, outlier_idx, float(np.exp(med_log)), bounds


def consensus(votes: list[Vote], log_window: float) -> dict:
    if not votes:
        return {"status": "no_votes"}
    scales = np.array([v.scale_m_per_unit for v in votes], dtype=float)
    weights = np.array([max(v.weight, 1e-6) for v in votes], dtype=float)
    inlier_idx, outlier_idx, scale, _ = winning_cluster(scales, weights, log_window)
    inlier_scales = scales[inlier_idx]
    spread = float(np.exp(np.std(np.log(inlier_scales)))) if len(inlier_scales) > 1 else 1.0
    inlier_weight_fraction = float(weights[inlier_idx].sum() / weights.sum())
    represented_priors = sorted({votes[int(i)].prior_key for i in inlier_idx})
    represented_methods = sorted({votes[int(i)].method for i in inlier_idx})
    quality = "weak"
    if len(inlier_idx) >= 12 and inlier_weight_fraction >= 0.60 and len(represented_priors) >= 2:
        quality = "strong"
    elif len(inlier_idx) >= 8 and inlier_weight_fraction >= 0.40:
        quality = "tentative"
    return {
        "status": "ok",
        "quality": quality,
        "scale_m_per_unit": scale,
        "vote_count": len(votes),
        "inlier_count": int(len(inlier_idx)),
        "outlier_count": int(len(outlier_idx)),
        "inlier_weight_fraction": inlier_weight_fraction,
        "multiplicative_std": spread,
        "inlier_methods": represented_methods,
        "inlier_priors": represented_priors,
    }


def write_votes(votes: list[Vote], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Vote.__dataclass_fields__)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for vote in votes:
            writer.writerow(vote.__dict__)


def write_summary(summary: dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))


def write_priors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(OBJECT_PRIORS, indent=2))


def write_detection_template(path: Path) -> None:
    rows = [
        {
            "frame_index": 12,
            "label": "tank",
            "score": 0.72,
            "x1": 420,
            "y1": 260,
            "x2": 650,
            "y2": 360,
            "depth_m": "",
        }
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def detect_zeroshot(args: argparse.Namespace) -> None:
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Missing transformers. Install in the active environment with: "
            "pip install transformers accelerate"
        ) from exc

    frame_rows = list(csv.DictReader((args.recon_dir / "frames.csv").open()))
    selected = frame_rows[:: max(args.frame_step, 1)]
    labels = args.labels or default_detector_labels()
    detector = pipeline(
        task="zero-shot-object-detection",
        model=args.model,
        device=args.device,
    )
    out_rows = []
    for idx, row in enumerate(selected, start=1):
        image_path = args.recon_dir / "frames" / row["file"]
        image = Image.open(image_path).convert("RGB")
        results = detector(image, candidate_labels=labels, threshold=args.threshold)
        print(f"[detect {idx}/{len(selected)}] {row['file']}: {len(results)} detections")
        for result in results:
            box = result["box"]
            out_rows.append(
                {
                    "frame_index": row["frame_index"],
                    "label": result["label"],
                    "score": f"{float(result['score']):.4f}",
                    "x1": f"{float(box['xmin']):.2f}",
                    "y1": f"{float(box['ymin']):.2f}",
                    "x2": f"{float(box['xmax']):.2f}",
                    "y2": f"{float(box['ymax']):.2f}",
                    "depth_m": "",
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "label", "score", "x1", "y1", "x2", "y2", "depth_m"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[detect] wrote {len(out_rows)} rows -> {args.out}")


def default_detector_labels() -> list[str]:
    labels = []
    for prior in OBJECT_PRIORS.values():
        labels.extend(prior["aliases"][:2])
    # Add scene anchors that are useful for future map/georegistration stages.
    labels.extend(["road", "building", "wall", "roof", "utility pole", "vehicle", "tree line"])
    return sorted(set(labels))


def read_vote_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def cluster_from_vote_rows(
    vote_rows: list[dict[str, str]], log_window: float
) -> tuple[set[int], tuple[float, float], float]:
    scales = np.array([float(row["scale_m_per_unit"]) for row in vote_rows], dtype=float)
    weights = np.array([max(float(row["weight"]), 1e-6) for row in vote_rows], dtype=float)
    inlier_idx, _, scale, bounds = winning_cluster(scales, weights, log_window)
    return {int(idx) for idx in inlier_idx}, bounds, scale


def detection_scale_candidates(
    det: dict[str, str],
    recon_dir: Path,
    row_by_frame: dict[int, dict[str, str]],
    xyz: np.ndarray,
    terminal: np.ndarray,
    hfov_deg: float | None,
    vfov_deg: float | None,
    min_relative_distance: float,
    size_cache: dict[int, tuple[int, int]],
    artifact_args: argparse.Namespace | None = None,
) -> tuple[str | None, list[dict[str, float | str]]]:
    frame_index = int(det["frame_index"])
    if frame_index not in row_by_frame or frame_index - 1 >= len(xyz):
        return None, []
    prior_key, prior = prior_for_label(det["label"])
    if prior is None:
        return None, []
    img = Image.open(frame_image_path(recon_dir, row_by_frame, frame_index)).convert("RGB")
    if frame_index not in size_cache:
        size_cache[frame_index] = img.size
    if artifact_args is not None and is_artifact_filter_enabled(artifact_args):
        if detection_artifact_reason(det, img, artifact_args):
            return prior_key, []
    width, height = size_cache[frame_index]
    fx, fy = focal_pixels(width, height, hfov_deg, vfov_deg)
    x1, y1, x2, y2 = [float(det[k]) for k in ["x1", "y1", "x2", "y2"]]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    rel_distance = float(np.linalg.norm(xyz[frame_index - 1] - terminal))
    if rel_distance < min_relative_distance:
        return prior_key, []
    candidates: list[dict[str, float | str]] = []
    for dim_name, observed_px, focal in [
        ("width_m", box_w, fx),
        ("height_m", box_h, fy),
    ]:
        if dim_name not in prior:
            continue
        known = float(prior[dim_name])
        metric_distance = known * focal / observed_px
        scale = metric_distance / rel_distance
        if np.isfinite(scale) and scale > 0:
            candidates.append(
                {
                    "method": f"apparent_{dim_name}",
                    "known_meters": known,
                    "observed_pixels": observed_px,
                    "metric_distance_m": metric_distance,
                    "relative_distance_units": rel_distance,
                    "scale_m_per_unit": scale,
                }
            )
    if det.get("depth_m"):
        depth_m = float(det["depth_m"])
        if depth_m > 0:
            candidates.append(
                {
                    "method": "metric_depth_to_terminal",
                    "known_meters": "",
                    "observed_pixels": "",
                    "metric_distance_m": depth_m,
                    "relative_distance_units": rel_distance,
                    "scale_m_per_unit": depth_m / rel_distance,
                }
            )
    return prior_key, candidates


def draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    y_text = max(0, y1 - text_h - 7)
    draw.rectangle([x1, y_text, x1 + text_w + 8, y_text + text_h + 6], fill=(0, 0, 0))
    draw.text((x1 + 4, y_text + 3), label, fill=color, font=font)


def load_overlay_font(size: int = 17) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def visualize_scale(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    recon_dir = args.recon_dir
    votes_path = args.votes or recon_dir / "auto_scale_votes.csv"
    summary_path = args.summary or recon_dir / "auto_scale_summary.json"
    out_dir = args.out_dir or recon_dir / "scale_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    path_rows, xyz = load_path(recon_dir / "relative_path.csv")
    row_by_frame = {int(row["frame_index"]): row for row in path_rows}
    terminal = xyz[terminal_index(path_rows)]
    detections = read_detections(args.detections)
    vote_rows = read_vote_rows(votes_path)
    inlier_indices, bounds, cluster_scale = cluster_from_vote_rows(vote_rows, args.log_window)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    prior_counts = Counter(row["prior_key"] for row in vote_rows)
    inlier_prior_counts = Counter(
        vote_rows[idx]["prior_key"] for idx in inlier_indices if idx < len(vote_rows)
    )
    print(f"[visualize] votes={len(vote_rows)} cluster={cluster_scale:.3f} m/unit")
    print(f"[visualize] priors={dict(prior_counts)}")
    print(f"[visualize] inlier_priors={dict(inlier_prior_counts)}")

    scales = np.array([float(row["scale_m_per_unit"]) for row in vote_rows], dtype=float)
    weights = np.array([max(float(row["weight"]), 1e-6) for row in vote_rows], dtype=float)
    priors = sorted(set(row["prior_key"] for row in vote_rows))
    bins = np.logspace(np.log10(scales.min() * 0.8), np.log10(scales.max() * 1.2), 48)
    colors = {
        "d9": "#e5a84a",
        "humvee": "#78c6a3",
        "merkava": "#7aa2ff",
        "namer": "#cf8bf3",
        "story": "#f2d36b",
        "tree": "#5ba36b",
        "window": "#9cc9ff",
        "soldier": "#ff9a76",
        "door": "#d4b08c",
    }

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    for prior in priors:
        idx = [i for i, row in enumerate(vote_rows) if row["prior_key"] == prior]
        ax.hist(
            scales[idx],
            bins=bins,
            weights=weights[idx],
            alpha=0.58,
            label=prior,
            color=colors.get(prior, "#aaaaaa"),
        )
    ax.axvspan(math.exp(bounds[0]), math.exp(bounds[1]), color="#f7c948", alpha=0.18)
    ax.axvline(cluster_scale, color="#ffd166", linewidth=2.2, label="cluster median")
    ax.set_xscale("log")
    ax.set_xlabel("scale vote: meters per VGGT unit")
    ax.set_ylabel("weighted vote mass")
    ax.set_title(
        "Automatic scale vote histogram"
        f" | quality={summary.get('quality', 'unknown')}"
        f" | inliers={summary.get('inlier_count', len(inlier_indices))}/{len(vote_rows)}"
    )
    ax.grid(color="#333333", linewidth=0.8)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    hist_path = out_dir / "scale_vote_histogram.png"
    fig.savefig(hist_path, facecolor="#000000")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    y_positions = {prior: i for i, prior in enumerate(priors)}
    for i, row in enumerate(vote_rows):
        prior = row["prior_key"]
        jitter = ((i * 37) % 100) / 500.0 - 0.1
        is_inlier = i in inlier_indices
        ax.scatter(
            float(row["scale_m_per_unit"]),
            y_positions[prior] + jitter,
            s=24 + 360 * float(row["weight"]),
            color="#ffd166" if is_inlier else colors.get(prior, "#888888"),
            alpha=0.85 if is_inlier else 0.32,
            edgecolors="none",
        )
    ax.axvspan(math.exp(bounds[0]), math.exp(bounds[1]), color="#f7c948", alpha=0.18)
    ax.axvline(cluster_scale, color="#ffd166", linewidth=2.2)
    ax.set_xscale("log")
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(priors)
    ax.set_xlabel("scale vote: meters per VGGT unit")
    ax.set_title("Scale votes by object prior; amber points are cluster inliers")
    ax.grid(color="#333333", linewidth=0.8)
    fig.tight_layout()
    scatter_path = out_dir / "scale_vote_cluster.png"
    fig.savefig(scatter_path, facecolor="#000000")
    plt.close(fig)

    frame_dets: dict[int, list[dict]] = {}
    size_cache: dict[int, tuple[int, int]] = {}
    for det in detections:
        if float(det.get("score") or 0) < args.score_threshold:
            continue
        frame_index = int(det["frame_index"])
        prior_key, candidates = detection_scale_candidates(
            det,
            recon_dir,
            row_by_frame,
            xyz,
            terminal,
            args.hfov_deg,
            args.vfov_deg,
            args.min_relative_distance,
            size_cache,
            args,
        )
        if not candidates:
            continue
        candidate_scales = [float(candidate["scale_m_per_unit"]) for candidate in candidates]
        logs = [math.log(scale) for scale in candidate_scales]
        is_inlier = any(bounds[0] <= log_scale <= bounds[1] for log_scale in logs)
        best_scale = min(candidate_scales, key=lambda scale: abs(math.log(scale / cluster_scale)))
        frame_dets.setdefault(frame_index, []).append(
            {
                "det": det,
                "prior_key": prior_key,
                "is_inlier": is_inlier,
                "best_scale": best_scale,
                "area": (float(det["x2"]) - float(det["x1"])) * (float(det["y2"]) - float(det["y1"])),
            }
        )

    selected_frames = sorted(frame_dets)[: args.top_frames]
    font = load_overlay_font(args.font_size)
    rendered_frames = []
    for frame_index in selected_frames:
        image_path = frame_image_path(recon_dir, row_by_frame, frame_index)
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        candidates = sorted(
            frame_dets[frame_index],
            key=lambda row: (row["is_inlier"], float(row["det"].get("score") or 0), row["area"]),
            reverse=True,
        )[: args.max_boxes_per_frame]
        for row in candidates:
            det = row["det"]
            color = (255, 209, 102) if row["is_inlier"] else (145, 154, 164)
            label = (
                f"{row['prior_key']} {float(det.get('score') or 0):.2f} "
                f"{row['best_scale']:.0f}m/u"
            )
            draw_labeled_box(
                draw,
                (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])),
                label,
                color,
                font,
            )
        title = (
            f"frame {frame_index} | amber=in cluster | gray=rejected "
            f"| cluster {cluster_scale:.0f} m/unit"
        )
        draw.rectangle([0, 0, image.width, 24], fill=(0, 0, 0))
        draw.text((8, 7), title, fill=(255, 255, 255), font=font)
        frame_path = out_dir / f"scale_examples_frame_{frame_index:06d}.png"
        image.save(frame_path)
        rendered_frames.append(image)

    if rendered_frames:
        thumb_width = min(640, max(image.width for image in rendered_frames))
        thumbs = []
        for image in rendered_frames:
            scale = thumb_width / image.width
            thumbs.append(image.resize((thumb_width, int(image.height * scale))))
        cols = 2 if len(thumbs) > 1 else 1
        rows_count = math.ceil(len(thumbs) / cols)
        thumb_height = max(image.height for image in thumbs)
        contact = Image.new("RGB", (cols * thumb_width, rows_count * thumb_height), (0, 0, 0))
        for idx, image in enumerate(thumbs):
            x = (idx % cols) * thumb_width
            y = (idx // cols) * thumb_height
            contact.paste(image, (x, y))
        contact_path = out_dir / "scale_detection_examples.png"
        contact.save(contact_path)
        print(f"[visualize] wrote {contact_path}")
    print(f"[visualize] wrote {hist_path}")
    print(f"[visualize] wrote {scatter_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-priors")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "auto_scale_object_priors.json")

    p = sub.add_parser("write-detection-template")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "detections_template.csv")

    p = sub.add_parser("detect-zeroshot")
    p.add_argument("--recon-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--labels", nargs="*")
    p.add_argument("--threshold", type=float, default=0.12)
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("estimate")
    p.add_argument("--recon-dir", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--hfov-deg", type=float, default=90.0)
    p.add_argument("--vfov-deg", type=float)
    p.add_argument("--min-relative-distance", type=float, default=0.004)
    p.add_argument("--log-window", type=float, default=math.log(1.7))
    p.add_argument("--votes-out", type=Path)
    p.add_argument("--summary-out", type=Path)
    p.add_argument("--disable-artifact-filter", action="store_true")
    p.add_argument("--hud-min-aspect", type=float, default=3.8)
    p.add_argument("--hud-top-fraction", type=float, default=0.30)

    p = sub.add_parser("visualize")
    p.add_argument("--recon-dir", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--votes", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--hfov-deg", type=float, default=90.0)
    p.add_argument("--vfov-deg", type=float)
    p.add_argument("--min-relative-distance", type=float, default=0.004)
    p.add_argument("--log-window", type=float, default=math.log(1.7))
    p.add_argument("--top-frames", type=int, default=8)
    p.add_argument("--max-boxes-per-frame", type=int, default=9)
    p.add_argument("--score-threshold", type=float, default=0.08)
    p.add_argument("--font-size", type=int, default=17)
    p.add_argument("--disable-artifact-filter", action="store_true")
    p.add_argument("--hud-min-aspect", type=float, default=3.8)
    p.add_argument("--hud-top-fraction", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "write-priors":
        write_priors(args.out)
        print(f"[priors] wrote {args.out}")
    elif args.command == "write-detection-template":
        write_detection_template(args.out)
        print(f"[template] wrote {args.out}")
    elif args.command == "detect-zeroshot":
        detect_zeroshot(args)
    elif args.command == "estimate":
        votes = generate_votes(args)
        votes_out = args.votes_out or args.recon_dir / "auto_scale_votes.csv"
        summary_out = args.summary_out or args.recon_dir / "auto_scale_summary.json"
        write_votes(votes, votes_out)
        summary = consensus(votes, args.log_window)
        write_summary(summary, summary_out)
        print(f"[auto-scale] votes={len(votes)} summary={summary}")
        print(f"[auto-scale] wrote {votes_out}")
        print(f"[auto-scale] wrote {summary_out}")
    elif args.command == "visualize":
        visualize_scale(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
