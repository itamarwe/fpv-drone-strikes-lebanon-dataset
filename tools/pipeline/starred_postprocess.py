#!/usr/bin/env python3
"""Post-process the starred undistort re-runs: metrics, overlay viewers, before/after videos, index.

Per scene (scenes/starred_undistort/<video_id>/):
  published/   the published product: VGGT-Omega on the raw clean-crop frames  (BEFORE)
  pinhole/     the re-run on the same frames undistorted with the self-calibrated lens (AFTER)
  calib/       GLOMAP OPENCV_FISHEYE self-calibration (independent camera path used as the reference)

Outputs:
  scenes/starred_undistort/<video_id>/metrics.json
  scenes/starred_undistort/<video_id>/overlay/index.html   (both runs in the published viewer frame)
  reports/starred_undistort/<video_id>/transition.mp4 + before.jpg / after.jpg
  reports/starred_undistort/index.html + SUMMARY.md
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
from render_before_after import quat_to_R, look_at, project, splat, draw_polyline, font  # noqa: E402
from make_viewer_style_before_after import frustum_pts, draw_grid, grid_in_aligned_frame  # noqa: E402
from evaluate_3d_trajectory import umeyama  # noqa: E402
from fit_moge3_colmap_scale import qvec_to_rotation, read_images  # noqa: E402
from build_recon_overlay_viewer import HTML as OVERLAY_HTML, COLOURS  # noqa: E402
from starred_overlay_video import make as make_overlay_video  # noqa: E402
from starred_camera_overlays import published_viewer, pinhole_overlays  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BATCH = __import__("os").environ.get("FPV_UNDISTORT_BATCH", "starred_undistort")  # batch name: benchmarks/<BATCH>, scenes/<BATCH>, reports/<BATCH>
SPEC = ROOT / "benchmarks" / BATCH / "starred_scenes.json"
SCENES = ROOT / "scenes" / BATCH
REPORTS = ROOT / "reports" / BATCH
BG = (10, 11, 13); WHITE = (255, 255, 255); RED = (255, 77, 109); CYAN = (54, 228, 255); ORANGE = (255, 176, 0)
GATE = __import__("os").environ.get("FPV_UNDISTORT_GATE", "0") == "1"  # only build media for scenes that improved
GATE_MAX_FX_RATIO = 1.45
GROUND_WARN_DEG = 10.0  # ground normals of the two runs further apart than this get a "check ground" flag (not a rejection)
TAGS = {"before": ("BEFORE  raw frames", RED), "after": ("AFTER  undistorted to pinhole", CYAN)}


# ---------------------------------------------------------------- loading

def load_viewer(vdir: Path, max_pts: int, rng) -> dict:
    meta = json.loads((vdir / "scene_meta.json").read_text())
    P = np.fromfile(vdir / Path(meta["assets"]["positions"]).name, dtype="<f4").reshape(-1, 3).astype(float)
    C = np.fromfile(vdir / Path(meta["assets"]["colors"]).name, dtype=np.uint8).reshape(-1, 3)
    if len(P) > max_pts:
        k = rng.choice(len(P), size=max_pts, replace=False); P, C = P[k], C[k]
    cams = [{"frame": e["frame"], "position": np.asarray(e["position"], float), "right": np.asarray(e["right"], float),
             "down": np.asarray(e["down"], float), "forward": np.asarray(e["forward"], float)} for e in meta["path"]]
    return {"meta": meta, "P": P, "C": C, "cams": cams}


def sim3_apply(s, R, t, run: dict) -> dict:
    out = dict(run)
    out["P"] = (s * (R @ run["P"].T)).T + t
    out["cams"] = [{"frame": c["frame"], "position": s * (R @ c["position"]) + t, "right": R @ c["right"], "down": R @ c["down"], "forward": R @ c["forward"]} for c in run["cams"]]
    return out


def colmap_centres(model_dir: Path) -> dict[int, np.ndarray]:
    out = {}
    for im in read_images(model_dir / "images.txt"):
        R = qvec_to_rotation(im.qvec)
        out[int(Path(im.name).stem.rsplit("_", 1)[1])] = -(R.T @ im.tvec)
    return out


# ---------------------------------------------------------------- metrics

def path_vs_reference(path: np.ndarray, ref: dict[int, np.ndarray], window: int = 20) -> dict:
    common = sorted(i for i in ref if 1 <= i <= len(path))
    if len(common) < 8:
        return {"matched": len(common)}
    A = path[[i - 1 for i in common]]; B = np.vstack([ref[i] for i in common])
    s, R, t = umeyama(A, B)
    err = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    ref_len = float(np.sum(np.linalg.norm(np.diff(B, axis=0), axis=1)))
    rel = []
    for st in range(0, len(common) - window + 1, max(1, window // 2)):
        rel.append(umeyama(A[st:st + window], B[st:st + window])[0] / s)
    rel = np.asarray(rel) if rel else np.array([1.0])
    return {"matched": len(common), "rmse_fraction_of_path_length": float(np.sqrt(np.mean(err ** 2)) / ref_len),
            "local_scale_range": [float(rel.min()), float(rel.max())], "local_scale_std": float(rel.std()),
            "sim3_to_reference": {"scale": float(s), "R": R.tolist(), "t": t.tolist()}}


def focal_ratio(pin: Path) -> dict:
    z = np.load(pin / "runpod_artifacts" / "predictions.npz", allow_pickle=True)
    K = np.asarray(z["intrinsic"], float); W = int(np.asarray(z["depth"]).shape[2])
    u = json.loads((pin / "frames" / "undistortion.json").read_text())["output_pinhole"]
    ref_fx = u["fx"] * W / u["width"]
    return {"vggt_fx_px_median": float(np.median(K[:, 0, 0])), "reference_fx_px_at_vggt_size": float(ref_fx),
            "fx_ratio_vggt_over_reference": float(np.median(K[:, 0, 0]) / ref_fx), "vggt_input_width": W,
            "depth_conf_median": float(np.median(np.asarray(z["depth_conf"], np.float32)))}


# ---------------------------------------------------------------- rendering

def render_panel(P, C, cams, grid, ref_path, view, W, H, fpx=1100, size=2):
    R, eye = view
    img = np.full((H, W, 3), BG, np.uint8); zb = np.full((H, W), np.inf)
    splat(img, zb, P, C, R, eye, fpx, size)
    im = Image.fromarray(img)
    if grid:
        draw_grid(im, R, eye, fpx, grid["origin"], grid["u"], grid["v"], grid["size_units"], grid["minor_step_units"], grid["major_step_units"])
    path = np.array([c["position"] for c in cams]); ext = float(np.linalg.norm(path[-1] - path[0]))
    if ref_path is not None:
        draw_polyline(im, ref_path, R, eye, fpx, WHITE, 4)
    draw_polyline(im, path, R, eye, fpx, CYAN, 3)
    fl = ext * 0.045
    for i in range(0, len(cams), 10):
        draw_polyline(im, frustum_pts(cams[i], fl), R, eye, fpx, CYAN, 1)
    draw_polyline(im, frustum_pts(cams[-1], fl * 1.4), R, eye, fpx, RED, 2)
    draw_polyline(im, frustum_pts(cams[0], fl * 1.6), R, eye, fpx, ORANGE, 3)
    dr = ImageDraw.Draw(im)
    for pt, col, rad in ((path[0], ORANGE, 9), (path[-1], RED, 7)):
        u, v, z, ok = project(pt[None], R, eye, fpx, W, H)
        if ok[0]:
            dr.ellipse([u[0] - rad, v[0] - rad, u[0] + rad, v[0] + rad], fill=col)
    return im


def shared_view(path: np.ndarray):
    ext = float(np.linalg.norm(path[-1] - path[0]))
    ctr = path.mean(0) + np.array([0, -0.12 * ext, 0])
    d = path[-1] - path[0]; d[1] = 0; d /= max(np.linalg.norm(d), 1e-9); left = np.cross([0, 1, 0], d)
    eye, tgt = ctr - d * 0.45 * ext + left * 0.62 * ext + np.array([0, 0.62 * ext, 0]), ctr
    return look_at(eye, tgt)


def compose(state_im, W, H, header, footer, title, sub, tag, tag_col, lines, alpha_tag=1.0):
    cv = Image.new("RGB", (W, H + header + footer), BG); cv.paste(state_im, (0, header)); d = ImageDraw.Draw(cv)
    d.text((28, 18), title[:70], fill=(235, 238, 242), font=font(34)); d.text((28, 64), sub, fill=(150, 160, 172), font=font(21))
    tw = d.textlength(tag, font=font(40)); box = Image.new("RGBA", (int(tw) + 48, 66), (0, 0, 0, 0)); bd = ImageDraw.Draw(box)
    bd.rounded_rectangle([0, 0, box.width - 1, 65], radius=12, fill=(*tag_col, int(255 * alpha_tag))); bd.text((24, 10), tag, fill=(10, 11, 13, int(255 * alpha_tag)), font=font(40))
    cv.paste(box, (W - box.width - 24, 18), box)
    y = header + H + 12
    d.text((28, y), "white: independent SfM camera path (self-calibrated lens)    cyan: VGGT-Omega camera path and frusta    grid: 2 m / 10 m", fill=(150, 160, 172), font=font(21)); y += 32
    for line in lines:
        d.text((28, y), line, fill=(215, 220, 226), font=font(24)); y += 32
    return cv


def make_video(vid: str, before: dict, after: dict, grid, ref_path, title, lines, out_dir: Path, cycles=3, hold_s=1.8, fade_s=1.2, fps=30, size=(1600, 1000)) -> None:
    W, H = size; header, footer = 100, 150
    view = shared_view(np.array([c["position"] for c in before["cams"]]))
    b_im = render_panel(before["P"], before["C"], before["cams"], grid, ref_path, view, W, H)
    a_im = render_panel(after["P"], after["C"], after["cams"], grid, ref_path, view, W, H)
    sub = "VGGT-Omega, same model, same frames. Only change: undistort first."
    out_dir.mkdir(parents=True, exist_ok=True)
    compose(b_im, W, H, header, footer, title, sub, *TAGS["before"], lines).save(out_dir / "before.jpg", quality=93)
    compose(a_im, W, H, header, footer, title, sub, *TAGS["after"], lines).save(out_dir / "after.jpg", quality=93)
    frames_dir = out_dir / "frames"; frames_dir.mkdir(exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()
    hold, fade = int(hold_s * fps), int(fade_s * fps); idx = 0
    ease = lambda x: 0.5 - 0.5 * np.cos(np.pi * x)

    def emit(alpha):
        nonlocal idx
        im = Image.blend(b_im, a_im, alpha)
        key = "after" if alpha >= 0.5 else "before"
        a = alpha if alpha >= 0.5 else 1 - alpha
        compose(im, W, H, header, footer, title, sub, *TAGS[key], lines, alpha_tag=min(1.0, 0.35 + 1.3 * abs(a - 0.5) * 2)).save(frames_dir / f"{idx:05d}.png"); idx += 1

    for _ in range(cycles):
        for _ in range(hold): emit(0.0)
        for k in range(fade): emit(ease((k + 1) / fade))
        for _ in range(hold): emit(1.0)
        for k in range(fade): emit(1 - ease((k + 1) / fade))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_dir / "transition.mp4")], check=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()
    frames_dir.rmdir()


# ---------------------------------------------------------------- overlay viewer

def write_overlay(vid: str, before: dict, after: dict, meta_pub: dict, out: Path, title: str, align: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    layers = []
    for i, (label, run, img_url) in enumerate((("published product (raw frames)", before, "../published/images/{name}"), ("undistorted to pinhole", after, "../pinhole/frames/{name}"))):
        cva = "../published/viewer/camera_view_assets/" if i == 0 else "../pinhole/viewer/camera_view_assets/"
        image_urls = {"actual": cva + "{stem}.jpg", "render": cva + "{stem}_vggt_render.jpg", "overlay": cva + "{stem}_overlay.jpg"}
        run["P"].astype("<f4").tofile(out / f"layer{i}_points.bin"); run["C"].astype(np.uint8).tofile(out / f"layer{i}_colors.bin")
        cams = [{"frame": c["frame"], "position": c["position"].tolist(), "right": c["right"].tolist(), "down": c["down"].tolist(), "forward": c["forward"].tolist()} for c in run["cams"]]
        layers.append({"key": f"layer{i}", "label": label, "colour": COLOURS[i % len(COLOURS)], "points": int(len(run["P"])), "cameras": cams,
                       "alignment": {"reference": i == 0, **({} if i == 0 else align)}, "frame_image_url": img_url, "image_urls": image_urls,
                       "frame_names": [f"f_{c['frame']:06d}.jpg" for c in run["cams"]]})
    meta = {"title": title, "scale_m_per_unit": meta_pub.get("default_scale_m_per_unit"), "layers": layers,
            "ground_grid": meta_pub.get("ground_grid"), "scene_alignment_quaternion": meta_pub.get("scene_alignment_quaternion"),
            "note": "Undistorted run Sim(3)-aligned to the published product on camera centres; units are the published run's VGGT units."}
    (out / "meta.json").write_text(json.dumps(meta) + "\n")
    (out / "index.html").write_text(OVERLAY_HTML)


# ---------------------------------------------------------------- per scene

def process(scene: dict, rng, skip_video: bool) -> dict | None:
    vid = scene["video_id"]; base = SCENES / vid
    pub_dir, pin_dir, cal_dir = base / "published", base / "pinhole" / "viewer", base / "calib" / "glomap_fisheye"
    if not (pin_dir / "scene_meta.json").exists():
        return None
    before = load_viewer(pub_dir, 1_500_000, rng)
    after_raw = load_viewer(pin_dir, 1_500_000, rng)
    n = min(len(before["cams"]), len(after_raw["cams"]))
    A = np.array([c["position"] for c in after_raw["cams"][:n]]); B = np.array([c["position"] for c in before["cams"][:n]])
    s, R, t = umeyama(A, B)
    after = sim3_apply(s, R, t, after_raw)
    resid = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    align = {"matched_cameras": int(n), "scale_to_reference": float(s), "centre_rmse_ref_units": float(np.sqrt(np.mean(resid ** 2)))}
    metrics = {"video_id": vid, "title": scene["title"], "frames_published": len(before["cams"]), "frames_pinhole": len(after_raw["cams"]),
               "after_to_before_alignment": align, "published_scale_m_per_unit": before["meta"].get("default_scale_m_per_unit")}
    ref_path = None
    if (cal_dir / "images.txt").exists():
        ref = colmap_centres(cal_dir)
        metrics["reference_registered"] = len(ref)
        metrics["before_vs_reference"] = path_vs_reference(np.array([c["position"] for c in before["cams"]]), ref)
        metrics["after_vs_reference"] = path_vs_reference(np.array([c["position"] for c in after_raw["cams"]]), ref)
        cam_line = next(l for l in (cal_dir / "cameras.txt").read_text().splitlines() if not l.startswith("#"))
        metrics["calibration_camera"] = cam_line
        # reference path drawn in the published frame: invert the before->ref Sim(3)
        bv = metrics["before_vs_reference"].get("sim3_to_reference")
        if bv:
            sr, Rr, tr = bv["scale"], np.array(bv["R"]), np.array(bv["t"])
            idx = sorted(ref)
            ref_pts = np.vstack([ref[i] for i in idx])
            ref_path = ((Rr.T @ (ref_pts - tr).T).T) / sr
    try:
        metrics["after_focal"] = focal_ratio(base / "pinhole")
    except Exception as exc:
        metrics["after_focal_error"] = str(exc)
    # ground agreement: each run fits its own ground plane; map the new run's normal into the published frame through
    # the camera-path alignment and measure the angle between the two. A large angle means the two reconstructions
    # disagree about which way is down, and one of the fits (not necessarily the new one) should be inspected.
    gn, gp = after_raw["meta"].get("ground_grid"), before["meta"].get("ground_grid")
    if gn and gp:
        n_new = R @ np.asarray(gn["normal"], float); n_new /= np.linalg.norm(n_new)
        n_pub = np.asarray(gp["normal"], float) / np.linalg.norm(gp["normal"])
        metrics["ground"] = {"angle_to_published_deg": float(np.degrees(np.arccos(np.clip(abs(n_new @ n_pub), -1, 1)))),
                             "inliers_new": gn.get("inlier_count"), "inliers_published": gp.get("inlier_count"),
                             "warn": bool(np.degrees(np.arccos(np.clip(abs(n_new @ n_pub), -1, 1))) > GROUND_WARN_DEG)}
    # improvement gate: the expensive media is only produced when the lens correction demonstrably helped
    bvr0, avr0 = metrics.get("before_vs_reference", {}), metrics.get("after_vs_reference", {})
    fx0 = metrics.get("after_focal", {}).get("fx_ratio_vggt_over_reference")
    reasons = []
    if "rmse_fraction_of_path_length" not in bvr0 or "rmse_fraction_of_path_length" not in avr0:
        reasons.append("no independent SfM reference to compare against")
    else:
        if not avr0["rmse_fraction_of_path_length"] < bvr0["rmse_fraction_of_path_length"]:
            reasons.append("camera path disagreement did not decrease")
        if avr0["local_scale_std"] > 1.2 * bvr0["local_scale_std"]:
            reasons.append("local scale drift got more than 20% worse")
    if fx0 is not None and fx0 > GATE_MAX_FX_RATIO:
        reasons.append(f"VGGT focal estimate {fx0:.2f}x the calibrated lens (calibration suspect)")
    try:
        u = json.loads((base / "pinhole" / "frames" / "undistortion.json").read_text())
        metrics["calibration"] = {"model": u["source_camera"]["model"], "fx_px": u["output_pinhole"]["fx"], "hfov_deg": u["output_pinhole"]["hfov_deg"],
                                  "distortion_params": u["source_camera"]["params"][4:], "image_size": [u["output_pinhole"]["width"], u["output_pinhole"]["height"]]}
    except Exception:
        pass
    metrics["improved"] = not reasons; metrics["gate_reasons"] = reasons
    (base / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if GATE and reasons:
        skip_video = True  # no transition / reprojection videos or camera-view overlays for scenes that did not improve
    # everything below lives in the published viewer frame rotated by its alignment quaternion (grounded like the viewer)
    Rq = quat_to_R(before["meta"]["scene_alignment_quaternion"])
    rot = lambda run: sim3_apply(1.0, Rq, np.zeros(3), run)
    grid = grid_in_aligned_frame(before["meta"]["ground_grid"], Rq) if before["meta"].get("ground_grid") else None
    write_overlay(vid, before, after, before["meta"], base / "overlay", f"{scene['title']}: published vs undistorted", align)
    if not skip_video:
        bvr, avr = metrics.get("before_vs_reference", {}), metrics.get("after_vs_reference", {})
        lines = []
        if "rmse_fraction_of_path_length" in bvr and "rmse_fraction_of_path_length" in avr:
            lines.append(f"camera path disagreement vs independent SfM:  {100 * bvr['rmse_fraction_of_path_length']:.2f}%  ->  {100 * avr['rmse_fraction_of_path_length']:.2f}%")
            lines.append(f"local scale drift (std over 20-frame windows):  {bvr['local_scale_std']:.3f}  ->  {avr['local_scale_std']:.3f}")
        if "after_focal" in metrics:
            lines.append(f"VGGT focal estimate vs calibrated lens (after):  {metrics['after_focal']['fx_ratio_vggt_over_reference']:.2f}x")
        ref_rot = (ref_path @ Rq.T) if ref_path is not None else None
        make_video(vid, rot(before), rot(after), grid, ref_rot, scene["title"], lines, REPORTS / vid)
        if not (REPORTS / vid / "overlay.mp4").exists():
            try:
                make_overlay_video(scene, json.loads(SPEC.read_text())["cdn_base"], 10.0, 4.0)
            except Exception as exc:
                print(f"{vid}: overlay video failed: {exc}", flush=True)
        try:  # camera-view mode of the standard viewer: real frames with the model drawn over them, both versions
            published_viewer(scene, json.loads(SPEC.read_text())["cdn_base"], False); pinhole_overlays(scene, False)
        except Exception as exc:
            print(f"{vid}: camera-view overlays failed: {exc}", flush=True)
    return metrics


def write_index(rows: list[dict]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = ["# Starred scenes: published product vs undistorted re-run", "",
          "| Scene | Frames | SfM registered | Path disagreement before -> after | Local scale std before -> after | Focal ratio after | Viewer | Video |", "|---|---:|---:|---|---|---:|---|---|"]
    html = ["<!doctype html><meta charset=utf-8><title>Starred undistort re-run</title><style>body{font:14px system-ui;background:#0c0d0f;color:#e8ebef;padding:20px}table{border-collapse:collapse}td,th{padding:6px 10px;border-bottom:1px solid #2a3037;text-align:left}a{color:#36e4ff}</style>",
            "<h1>Starred scenes: published product vs undistorted re-run</h1><p>Viewer links assume <code>python3 tools/local_scene_viewer_server.py --port 8766</code>.</p><table><tr><th>Scene</th><th>Frames</th><th>SfM reg.</th><th>Path disagreement</th><th>Local scale std</th><th>Focal ratio (after)</th><th>Viewer</th><th>Video</th></tr>"]
    for m in rows:
        b, a = m.get("before_vs_reference", {}), m.get("after_vs_reference", {})
        pd = f"{100 * b['rmse_fraction_of_path_length']:.2f}% -> {100 * a['rmse_fraction_of_path_length']:.2f}%" if "rmse_fraction_of_path_length" in a and "rmse_fraction_of_path_length" in b else "n/a"
        ls = f"{b['local_scale_std']:.3f} -> {a['local_scale_std']:.3f}" if "local_scale_std" in a and "local_scale_std" in b else "n/a"
        fr = f"{m['after_focal']['fx_ratio_vggt_over_reference']:.2f}" if "after_focal" in m else "n/a"
        vid = m["video_id"]
        viewer = f"/scenes/{BATCH}/{vid}/overlay/index.html"
        md.append(f"| {m['title']} | {m['frames_published']} | {m.get('reference_registered', 'n/a')} | {pd} | {ls} | {fr} | [overlay]({viewer}) | [transition]({vid}/transition.mp4) · [reprojection]({vid}/overlay.mp4) |")
        html.append(f"<tr><td>{m['title']}</td><td>{m['frames_published']}</td><td>{m.get('reference_registered', 'n/a')}</td><td>{pd}</td><td>{ls}</td><td>{fr}</td><td><a href='{viewer}'>overlay</a></td><td><a href='/scenes/{BATCH}/{vid}/published/viewer/'>camera view published</a> · <a href='/scenes/{BATCH}/{vid}/pinhole/viewer/'>camera view undistorted</a> · <a href='{vid}/transition.mp4'>transition</a> · <a href='{vid}/overlay.mp4'>reprojection</a> · <a href='{vid}/before.jpg'>before</a> · <a href='{vid}/after.jpg'>after</a></td></tr>")
    md += ["", "Path disagreement: Sim(3) residual of the VGGT camera centres against the GLOMAP self-calibration path, as a fraction of its length. ",
           "Local scale: 20-frame window scale over the global scale. Focal ratio: VGGT's estimated focal over the calibrated pinhole focal (1.0 = consistent)."]
    (REPORTS / "SUMMARY.md").write_text("\n".join(md) + "\n")
    (REPORTS / "index.html").write_text("\n".join(html) + "</table>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    rng = np.random.default_rng(3)
    rows = []
    for s in spec["scenes"]:
        if args.only and s["video_id"] not in args.only:
            m_path = SCENES / s["video_id"] / "metrics.json"
            if m_path.exists():
                rows.append(json.loads(m_path.read_text()))
            continue
        try:
            m = process(s, rng, args.skip_video)
        except Exception as exc:
            print(f"{s['video_id']}: FAILED {exc}", flush=True)
            continue
        if m:
            rows.append(m)
            b, a = m.get("before_vs_reference", {}), m.get("after_vs_reference", {})
            print(f"{s['video_id']}: path {100 * b.get('rmse_fraction_of_path_length', float('nan')):.2f}% -> {100 * a.get('rmse_fraction_of_path_length', float('nan')):.2f}%  "
                  f"scale std {b.get('local_scale_std', float('nan')):.3f} -> {a.get('local_scale_std', float('nan')):.3f}  fx ratio {m.get('after_focal', {}).get('fx_ratio_vggt_over_reference', float('nan')):.2f}", flush=True)
    write_index(rows)
    print("index:", REPORTS / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
