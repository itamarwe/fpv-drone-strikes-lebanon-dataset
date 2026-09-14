#!/usr/bin/env python3
"""Before/after transition video over the Blender ground-truth model (synthetic flyover).

Fixed camera. Three states, all with the class-coloured Blender model drawn underneath:
  BEFORE     raw-fisheye VGGT-Omega reconstruction on top
  REFERENCE  Blender model only
  AFTER      undistorted-input VGGT-Omega reconstruction on top
Cycle: BEFORE -> REFERENCE -> AFTER -> REFERENCE -> (next cycle).

Numbered rings mark ghost buildings in the BEFORE run. When BEFORE fades out the rings
slide to the real buildings those pixels belong to (traced through the undistortion map
into the AFTER run), and slide back when BEFORE returns.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_before_after import look_at, project, splat, draw_polyline, font  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "scenes" / "synthetic_zofa_al_bayada"
Z2Y = lambda A: np.stack([A[:, 0], A[:, 2], -A[:, 1]], axis=1)  # Blender Z-up -> renderer Y-up
BG = (10, 11, 13); WHITE = (255, 255, 255); RED = (255, 77, 109); CYAN = (54, 228, 255); GREY = (170, 178, 188); MARK = (255, 214, 10)

# Centres in the recentred overlay frame (Y-up).
# "before": bright points > 4 m from the Blender surface in the raw-fisheye run (the ghost smear).
# "after":  the pixels of that smear traced through the fisheye->pinhole remap into the undistorted run,
#           snapped to the centroid of the Blender building they land on (ring 6: ground, no snap).
MARKERS = [
    {"n": 1, "before": [144.7, -17.8, -70.7], "after": [43.5, -36.1, -107.9], "ra": 13, "rb": 13},
    {"n": 2, "before": [147.3, -16.3, -51.7], "after": [75.8, -27.7, -84.4], "ra": 13, "rb": 13},
    {"n": 3, "before": [152.9, -8.5, -5.1], "after": [140.0, -17.7, -29.2], "ra": 13, "rb": 13},
    {"n": 4, "before": [101.6, 0.1, 64.3], "after": [61.3, -7.1, 75.6], "ra": 13, "rb": 13},
    {"n": 5, "before": [118.1, 10.1, 110.1], "after": [27.9, 3.0, 115.8], "ra": 13, "rb": 13},
    {"n": 6, "before": [-24.9, -38.8, -64.3], "after": [-68.6, -36.7, -58.2], "ra": 30, "rb": 12, "ground": True},
]
LEGEND = ["1-5  ghost buildings in the raw run; the rings slide to the real buildings those pixels belong to",
          "6  ground edge folded upward  ->  flat"]
TAGS = {"before": ("BEFORE  raw fisheye frames", RED), "ref": ("REFERENCE  Blender model only", GREY), "after": ("AFTER  undistorted to pinhole", CYAN)}


def load(name, gt=False):
    d = S / name / "gt_overlay"; m = json.loads((d / "meta.json").read_text())
    P = np.fromfile(d / ("gt_points.bin" if gt else "vggt_points.bin"), dtype="<f4").reshape(-1, 3).astype(float)
    C = np.fromfile(d / ("gt_colors.bin" if gt else "vggt_colors.bin"), dtype=np.uint8).reshape(-1, 3)
    return m, Z2Y(P), C


def ring(centre_y, ra, rb, d, left, n=72):
    a = np.linspace(0, 2 * np.pi, n)
    return centre_y[None] + d[None] * (ra * np.cos(a))[:, None] + left[None] * (rb * np.sin(a))[:, None]


def render_state(W, H, R, eye, fpx, gt, recon, paths, gt_size, rec_size, gt_dim):
    img = np.full((H, W, 3), BG, np.uint8); zb = np.full((H, W), np.inf)
    gP, gC = gt
    splat(img, zb, gP, (gC.astype(np.float32) * gt_dim).astype(np.uint8), R, eye, fpx, gt_size)
    if recon is not None: splat(img, zb, recon[0], recon[1], R, eye, fpx, rec_size)
    im = Image.fromarray(img)
    for P, col, w in paths: draw_polyline(im, P, R, eye, fpx, col, w)
    return im


def draw_markers(im, R, eye, fpx, markers, s, d, left):
    """s in [0,1]: 0 = rings at the ghost (before) positions, 1 = at the real (after) positions."""
    im = im.copy(); dr = ImageDraw.Draw(im); W, H = im.size
    for mk in markers:
        c = (1 - s) * np.asarray(mk["before"]) + s * np.asarray(mk["after"]); c = c.copy(); c[1] -= 0 if mk.get("ground") else 4
        u, v, z, ok = project(ring(c, mk["ra"], mk["rb"], d, left), R, eye, fpx, W, H)
        pts = [(float(a), float(b)) for a, b, o in zip(u, v, ok) if o]
        if len(pts) > 3:
            dr.line(pts + [pts[0]], fill=MARK, width=4, joint="curve")
            cx, cy = np.mean([p[0] for p in pts]), min(p[1] for p in pts)
            dr.ellipse([cx - 22, cy - 58, cx + 22, cy - 14], fill=MARK); dr.text((cx - 9, cy - 54), str(mk["n"]), fill=BG, font=font(30))
    return im


def compose(state_im, W, H, header, footer, title, sub, tag, tag_col, alpha_tag=1.0):
    cv = Image.new("RGB", (W, H + header + footer), BG); cv.paste(state_im, (0, header)); d = ImageDraw.Draw(cv)
    d.text((28, 18), title, fill=(235, 238, 242), font=font(38)); d.text((28, 66), sub, fill=(150, 160, 172), font=font(22))
    tw = d.textlength(tag, font=font(46)); box = Image.new("RGBA", (int(tw) + 56, 78), (0, 0, 0, 0)); bd = ImageDraw.Draw(box)
    bd.rounded_rectangle([0, 0, box.width - 1, 77], radius=14, fill=(*tag_col, int(255 * alpha_tag))); bd.text((28, 12), tag, fill=(10, 11, 13, int(255 * alpha_tag)), font=font(46))
    cv.paste(box, (W - box.width - 24, 16), box)
    y = header + H + 14
    d.text((28, y), "white: true camera path    red: reconstructed camera path    grey model: Blender ground truth    colour: VGGT-Omega reconstruction", fill=(150, 160, 172), font=font(22)); y += 34
    for line in LEGEND:
        d.ellipse([28, y + 2, 54, y + 28], fill=MARK); d.text((36, y + 3), line.split()[0][0], fill=BG, font=font(20)); d.text((66, y + 1), line, fill=(215, 220, 226), font=font(24)); y += 34
    return cv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", choices=["oblique", "topdown"], default="oblique")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--hold-s", type=float, default=1.8)
    ap.add_argument("--fade-s", type=float, default=1.2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1600x1150")
    args = ap.parse_args()
    W, H = map(int, args.size.split("x")); header, footer = 110, 120
    out = args.out or ROOT / "reports" / "lens_undistortion_before_after" / f"synthetic_transition_{args.view}.mp4"
    m0, gP, gC = load("synthetic_zofa_fpv_roof_flyover_120f", gt=True)
    mb, bP, bC = load("synthetic_zofa_fpv_roof_flyover_120f"); ma, aP, aC = load("synthetic_zofa_fpv_roof_flyover_120f_pinhole")
    G = Z2Y(np.array([c["position"] for c in m0["gt_cameras"]])); Vb = Z2Y(np.array([c["position"] for c in mb["vggt_cameras"]])); Va = Z2Y(np.array([c["position"] for c in ma["vggt_cameras"]]))
    ctr = G.mean(0); d = G[-1] - G[0]; d[1] = 0; d /= np.linalg.norm(d); left = np.cross([0, 1, 0], d); ext = np.linalg.norm(G[-1] - G[0])
    if args.view == "oblique":
        eye, tgt, fpx = ctr - d * 0.55 * ext + np.array([0, 0.9 * ext, 0]), ctr + d * 0.05 * ext, 1060
    else:
        eye, tgt, fpx = ctr + np.array([0, 1.6 * ext, 0]), ctr, 1150
    R, eye = look_at(eye, tgt, up=tuple(d) if args.view == "topdown" else (0, 1, 0))
    base = {
        "before": render_state(W, H, R, eye, fpx, (gP, gC), (bP, bC), [(G, WHITE, 5), (Vb, RED, 4)], 1, 2, 0.62),
        "ref": render_state(W, H, R, eye, fpx, (gP, gC), None, [(G, WHITE, 5)], 1, 2, 0.62),
        "after": render_state(W, H, R, eye, fpx, (gP, gC), (aP, aC), [(G, WHITE, 5), (Va, RED, 4)], 1, 2, 0.62),
    }
    title = "Synthetic FPV flyover vs Blender ground truth"
    sub = "VGGT-Omega, same model, same 120 frames. Only change: undistort first."
    rings = lambda im, s: draw_markers(im, R, eye, fpx, MARKERS, s, d, left)
    compose(rings(base["before"], 0), W, H, header, footer, title, sub, *TAGS["before"]).save(out.parent / f"synthetic_transition_{args.view}_before.jpg", quality=94)
    compose(rings(base["ref"], 1), W, H, header, footer, title, sub, *TAGS["ref"]).save(out.parent / f"synthetic_transition_{args.view}_reference.jpg", quality=94)
    compose(rings(base["after"], 1), W, H, header, footer, title, sub, *TAGS["after"]).save(out.parent / f"synthetic_transition_{args.view}_after.jpg", quality=94)
    if args.cycles <= 0:
        print("stills only"); return 0
    frames_dir = out.parent / f"transition_frames_{args.view}"; frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"): f.unlink()
    fps = args.fps; hold, fade = int(args.hold_s * fps), int(args.fade_s * fps); idx = 0
    ring_pos = {"before": 0.0, "ref": 1.0, "after": 1.0}
    ease = lambda x: 0.5 - 0.5 * np.cos(np.pi * x)

    def emit(im, s, tag_key, alpha_tag):
        nonlocal idx
        compose(rings(im, s), W, H, header, footer, title, sub, *TAGS[tag_key], alpha_tag=alpha_tag).save(frames_dir / f"{idx:05d}.png"); idx += 1

    def hold_state(k):
        for _ in range(hold): emit(base[k], ring_pos[k], k, 1.0)

    def fade_states(a, b):
        for k in range(fade):
            t = ease((k + 1) / fade)
            im = Image.blend(base[a], base[b], t)
            s = ring_pos[a] + (ring_pos[b] - ring_pos[a]) * t
            tag_key = b if t >= 0.5 else a
            emit(im, s, tag_key, min(1.0, 0.35 + 1.3 * abs(t - 0.5) * 2))

    seq = ["before", "ref", "after", "ref"]
    for _ in range(args.cycles):
        for i, k in enumerate(seq):
            hold_state(k); fade_states(k, seq[(i + 1) % len(seq)])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    print("wrote", out, "frames", idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
