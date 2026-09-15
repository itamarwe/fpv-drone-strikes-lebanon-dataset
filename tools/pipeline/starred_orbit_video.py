#!/usr/bin/env python3
"""Orbiting before/after clip per scene: BEFORE -> AFTER -> BEFORE -> AFTER while the camera circles the scene.

Same content as the transition video (published product vs lens-corrected run over the viewer's ground grid,
white independent SfM path, cyan reconstructed path) but the viewpoint sweeps around the scene during the
two cycles, so every frame is rendered. Output: reports/starred_undistort/<video_id>/transition_orbit.mp4,
and with --montage a concatenation of the selected scenes: reports/starred_undistort/best/best_scenes_orbit.mp4.

  starred_orbit_video.py --only <video_id> ... [--cycles 2] [--sweep-deg 240] [--montage]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_before_after import quat_to_R, look_at  # noqa: E402
from make_viewer_style_before_after import grid_in_aligned_frame  # noqa: E402
from evaluate_3d_trajectory import umeyama  # noqa: E402
from starred_postprocess import load_viewer, sim3_apply, colmap_centres, path_vs_reference, render_panel, compose, TAGS, SCENES, REPORTS, SPEC  # noqa: E402


def orbit_view(path: np.ndarray, cloud_ctr: np.ndarray, theta: float):
    """Camera on a circle around the scene, looking at the midpoint between the reconstruction and the flight path;
    theta in radians (0 = the transition video's view side)."""
    ext = float(np.linalg.norm(path[-1] - path[0]))
    ctr = 0.5 * (path.mean(0) + cloud_ctr); ctr[1] = path.mean(0)[1] - 0.12 * ext
    d = path[-1] - path[0]; d[1] = 0; d /= max(np.linalg.norm(d), 1e-9); left = np.cross([0, 1, 0], d)
    base = -d * 0.45 + left * 0.62  # the fixed view's horizontal offset
    r = 0.95 * ext; a0 = np.arctan2(base @ left, base @ d)
    a = a0 + theta
    eye = ctr + (np.cos(a) * d + np.sin(a) * left) * r + np.array([0, 0.7 * ext, 0])
    return look_at(eye, ctr)


def make(scene: dict, cycles: int, sweep_deg: float, hold_s: float, fade_s: float, fps: int, size=(1600, 1000), max_pts=1_000_000) -> Path | None:
    vid = scene["video_id"]; base = SCENES / vid
    pub_dir, pin_dir, cal_dir = base / "published", base / "pinhole" / "viewer", base / "calib" / "glomap_fisheye"
    if not (pin_dir / "scene_meta.json").exists():
        return None
    rng = np.random.default_rng(3)
    before = load_viewer(pub_dir, max_pts, rng); after_raw = load_viewer(pin_dir, max_pts, rng)
    n = min(len(before["cams"]), len(after_raw["cams"]))
    A = np.array([c["position"] for c in after_raw["cams"][:n]]); B = np.array([c["position"] for c in before["cams"][:n]])
    s, R, t = umeyama(A, B); after = sim3_apply(s, R, t, after_raw)
    metrics = json.loads((base / "metrics.json").read_text())
    ref_path = None
    bv = metrics.get("before_vs_reference", {}).get("sim3_to_reference")
    if bv and (cal_dir / "images.txt").exists():
        ref = colmap_centres(cal_dir); sr, Rr, tr = bv["scale"], np.array(bv["R"]), np.array(bv["t"])
        ref_path = ((Rr.T @ (np.vstack([ref[i] for i in sorted(ref)]) - tr).T).T) / sr
    Rq = quat_to_R(before["meta"]["scene_alignment_quaternion"]); rot = lambda run: sim3_apply(1.0, Rq, np.zeros(3), run)
    grid = grid_in_aligned_frame(before["meta"]["ground_grid"], Rq) if before["meta"].get("ground_grid") else None
    b, a = rot(before), rot(after); ref_rot = (ref_path @ Rq.T) if ref_path is not None else None
    bvr, avr = metrics.get("before_vs_reference", {}), metrics.get("after_vs_reference", {})
    lines = []
    if "rmse_fraction_of_path_length" in bvr and "rmse_fraction_of_path_length" in avr:
        lines.append(f"camera path disagreement vs independent SfM:  {100 * bvr['rmse_fraction_of_path_length']:.2f}%  ->  {100 * avr['rmse_fraction_of_path_length']:.2f}%")
        lines.append(f"local scale drift (std over 20-frame windows):  {bvr['local_scale_std']:.3f}  ->  {avr['local_scale_std']:.3f}")
    W, H = size; header, footer = 100, 150
    title = scene["title"]; sub = "VGGT-Omega, same model, same frames. Only change: undistort first."
    out_dir = REPORTS / vid; out_dir.mkdir(parents=True, exist_ok=True)
    fr_dir = out_dir / "orbit_frames"; fr_dir.mkdir(exist_ok=True)
    for f in fr_dir.glob("*.png"):
        f.unlink()
    hold, fade = int(hold_s * fps), int(fade_s * fps)
    # alpha schedule: cycles x (hold before, fade to after, hold after, fade to before)
    ease = lambda x: 0.5 - 0.5 * np.cos(np.pi * x)
    alphas = []
    for c in range(cycles):
        alphas += [0.0] * hold + [ease((k + 1) / fade) for k in range(fade)] + [1.0] * hold
        if c < cycles - 1:  # end the clip on AFTER: no fade back before the cut to the next scene
            alphas += [1 - ease((k + 1) / fade) for k in range(fade)]
    total = len(alphas); path_b = np.array([c["position"] for c in b["cams"]]); cloud_ctr = np.median(b["P"], axis=0)
    for i, alpha in enumerate(alphas):
        theta = np.radians(sweep_deg) * i / max(1, total - 1)
        view = orbit_view(path_b, cloud_ctr, theta)
        b_im = render_panel(b["P"], b["C"], b["cams"], grid, ref_rot, view, W, H)
        a_im = render_panel(a["P"], a["C"], a["cams"], grid, ref_rot, view, W, H)
        im = b_im if alpha <= 0 else a_im if alpha >= 1 else Image.blend(b_im, a_im, alpha)
        key = "after" if alpha >= 0.5 else "before"; aa = alpha if alpha >= 0.5 else 1 - alpha
        compose(im, W, H, header, footer, title, sub, *TAGS[key], lines, alpha_tag=min(1.0, 0.35 + 1.3 * abs(aa - 0.5) * 2)).save(fr_dir / f"{i:05d}.png")
    out = out_dir / "transition_orbit.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(fr_dir / "%05d.png"), "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    for f in fr_dir.glob("*.png"):
        f.unlink()
    fr_dir.rmdir()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="video ids (default: the visually picked scenes)")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--sweep-deg", type=float, default=240.0, help="total orbit over the clip")
    ap.add_argument("--hold-s", type=float, default=1.8)
    ap.add_argument("--fade-s", type=float, default=1.2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--montage", action="store_true", help="also concatenate the clips into best/best_scenes_orbit.mp4")
    args = ap.parse_args()
    spec = {s["video_id"]: s for s in json.loads(SPEC.read_text())["scenes"]}
    ids = args.only or json.loads((SPEC.parent / "publish_selection.json").read_text())["user_picked"]
    outs = []
    for vid in ids:
        try:
            out = make(spec[vid], args.cycles, args.sweep_deg, args.hold_s, args.fade_s, args.fps)
            print(f"{vid}: {out}", flush=True); outs.append(out)
        except Exception as exc:
            print(f"{vid}: FAILED {exc}", flush=True)
    if args.montage and outs:
        best = REPORTS / "best"; best.mkdir(exist_ok=True)
        lst = best / "concat_orbit.txt"; lst.write_text("".join(f"file '{o.resolve()}'\n" for o in outs if o))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(best / "best_scenes_orbit.mp4")], check=True)
        print("montage:", best / "best_scenes_orbit.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
