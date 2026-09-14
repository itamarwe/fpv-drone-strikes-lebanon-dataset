#!/usr/bin/env python3
"""Before/after video: each reconstruction re-projected through the TRUE camera of every
frame and faded in and out over the original synthetic footage.

Left: reconstruction from raw fisheye frames. Right: reconstruction from the same frames
undistorted to pinhole. Both clouds live in Blender metres (aligned via the camera Sim(3)),
so a correct reconstruction lands exactly on the footage; a wrong one drifts off it.
Projection uses the scene's equisolid fisheye model so the original frames can be the background.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scenes" / "synthetic_zofa_al_bayada"


def font(sz):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()


def load(name):
    m = json.loads((S / name / "gt_overlay" / "meta.json").read_text())
    P = np.fromfile(S / name / "gt_overlay" / "vggt_points.bin", dtype="<f4").reshape(-1, 3).astype(np.float64) + np.asarray(m["origin_world_m"])
    C = np.fromfile(S / name / "gt_overlay" / "vggt_colors.bin", dtype=np.uint8).reshape(-1, 3)
    return P, C


def project_equisolid(Pc, f_mm, sw_mm, W, H):
    r = np.linalg.norm(Pc, axis=1); d = Pc / np.maximum(r, 1e-9)[:, None]
    theta = np.arccos(np.clip(d[:, 2], -1, 1)); phi = np.arctan2(d[:, 1], d[:, 0])
    rad = 2 * f_mm * np.sin(theta / 2) * (W / sw_mm)
    return W / 2 + rad * np.cos(phi), H / 2 + rad * np.sin(phi), r, theta


def render_overlay(P, C, w2c, intr, W, H, size=3):
    Pc = (w2c[:3, :3] @ P.T).T + w2c[:3, 3]
    u, v, r, theta = project_equisolid(Pc, intr["focal_length_mm"], intr["sensor_width_mm"], W, H)
    ok = (theta < np.radians(95)) & (r > 0.5) & (u >= 0) & (u < W - size) & (v >= 0) & (v < H - size)
    u, v, r, col = u[ok].astype(int), v[ok].astype(int), r[ok], C[ok]
    order = np.argsort(-r); u, v, r, col = u[order], v[order], r[order], col[order]
    img = np.zeros((H, W, 3), np.uint8); mask = np.zeros((H, W), bool); zb = np.full((H, W), np.inf)
    for dy in range(size):
        for dx in range(size):
            uu, vv = u + dx, v + dy; better = r < zb[vv, uu]
            zb[vv[better], uu[better]] = r[better]; img[vv[better], uu[better]] = col[better]; mask[vv[better], uu[better]] = True
    return img, mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="synthetic_zofa_fpv_roof_flyover_120f")
    ap.add_argument("--after", default="synthetic_zofa_fpv_roof_flyover_120f_pinhole")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "lens_undistortion_before_after" / "synthetic_fade_before_after.mp4")
    ap.add_argument("--fade-period-s", type=float, default=4.0)
    ap.add_argument("--fps-out", type=int, default=24)
    args = ap.parse_args()
    gt = json.loads((S / args.scene / "ground_truth" / "gt_cameras.json").read_text()); intr = gt["intrinsics"]; W, H = intr["resolution_px"]
    fps_src = json.loads((S / args.scene / "metadata.json").read_text()).get("sample_fps") or 5.411716445929171
    before, after = load(args.scene), load(args.after)
    frames_dir = args.out.parent / "fade_frames"; frames_dir.mkdir(parents=True, exist_ok=True)
    fnt, fnt_s = font(30), font(22)
    for i, fr in enumerate(gt["frames"]):
        bg = np.asarray(Image.open(S / args.scene / "frames" / f"f_{fr['frame']:06d}.jpg").convert("RGB")).astype(np.float32)
        w2c = np.linalg.inv(np.asarray(fr["c2w_opencv"], float))
        t = i / fps_src; alpha = 0.5 * (1 + np.sin(2 * np.pi * t / args.fade_period_s - np.pi / 2))  # 0 -> 1 -> 0
        panels = []
        for (P, C), tag, col in ((before, "BEFORE  raw fisheye frames", (255, 77, 109)), (after, "AFTER  undistorted to pinhole", (54, 228, 255))):
            ov, mask = render_overlay(P, C, w2c, intr, W, H)
            comp = bg.copy(); a = alpha * mask[..., None]
            comp = comp * (1 - a) + ov.astype(np.float32) * a
            im = Image.fromarray(comp.clip(0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
            d.rounded_rectangle([14, 14, 14 + 24 + d.textlength(tag, font=fnt), 60], radius=8, fill=(12, 13, 15))
            d.text((26, 20), tag, fill=col, font=fnt)
            panels.append(im)
        canvas = Image.new("RGB", (2 * W + 8, H + 92), (12, 13, 15)); canvas.paste(panels[0], (0, 0)); canvas.paste(panels[1], (W + 8, 0))
        d = ImageDraw.Draw(canvas)
        d.text((16, H + 12), f"3D reconstruction re-projected through the true camera and faded over the original footage    frame {i + 1}/120    overlay {int(alpha * 100):3d}%", fill=(200, 208, 216), font=fnt_s)
        d.text((16, H + 50), "same VGGT-Omega model, same 120 frames; the only change is undistorting the fisheye frames to a pinhole before inference", fill=(139, 149, 161), font=fnt_s)
        canvas.save(frames_dir / f"{i + 1:05d}.png")
        if i % 20 == 0: print("frame", i + 1, "alpha", round(float(alpha), 2), flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps_src:.6f}", "-i", str(frames_dir / "%05d.png"), "-r", str(args.fps_out),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)], check=True)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
