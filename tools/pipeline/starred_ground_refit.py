#!/usr/bin/env python3
"""Re-fit a re-run scene's ground plane to the terrain at the target (points within R m of the final camera, no colour
filter), write it into the lens-corrected viewer meta and the staged copy (originals backed up as *.orig.json), and render
old-vs-new grids under the corrected cloud.   usage: ground_refit.py <video_id> [radius_m] [--dry-run]"""
import json, os, sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pipeline")); sys.path.insert(0, str(ROOT / "tools"))
from build_vggt_scene_artifacts import scene_alignment_quaternion, unit
from render_before_after import quat_to_R, look_at, splat, draw_polyline, font
from make_viewer_style_before_after import draw_grid, grid_in_aligned_frame, frustum_pts
BATCH = os.environ.get("FPV_UNDISTORT_BATCH", "all_undistort"); SCENES = ROOT / "scenes" / BATCH
vid = sys.argv[1]; radius_m = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 25.0; dry = "--dry-run" in sys.argv
spec = {s["video_id"]: s for s in json.loads((ROOT / "benchmarks" / BATCH / "starred_scenes.json").read_text())["scenes"]}
vd = SCENES / vid / "pinhole" / "viewer"; meta = json.loads((vd / "scene_meta.json").read_text())
P = np.fromfile(vd / Path(meta["assets"]["positions"]).name, dtype="<f4").reshape(-1, 3).astype(float)
C = np.fromfile(vd / Path(meta["assets"]["colors"]).name, dtype=np.uint8).reshape(-1, 3)
cams = np.array([e["position"] for e in meta["path"]], float); scale = float(meta.get("default_scale_m_per_unit") or 117.6)
orig = vd / "scene_meta.orig.json"  # builder's fit, kept from a previous re-fit: always compare against that
old = (json.loads(orig.read_text()) if orig.exists() else meta)["ground_grid"]; thr = old["threshold_units"]
# --- terminal-region RANSAC (same threshold and refinement as the builder, candidates = terrain around the target)
near = P[np.linalg.norm(P - cams[-1], axis=1) < radius_m / scale]
rng = np.random.default_rng(42); cand = near if len(near) <= 80000 else near[rng.choice(len(near), 80000, replace=False)]
sub = cand[rng.choice(len(cand), min(14000, len(cand)), replace=False)]; best = (-1, None)
for tri in cand[rng.choice(len(cand), size=(4000, 3))]:
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); nn = np.linalg.norm(n)
    if nn < 1e-9: continue
    n /= nn; d = -float(n @ tri[0]); k = int((np.abs(sub @ n + d) < thr).sum())
    if k > best[0]: best = (k, (n, d))
n, d = best[1]
for _ in range(3):
    inl = cand[np.abs(cand @ n + d) < thr]; c = inl.mean(0); n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]; d = -float(n @ c)
if np.median(cams @ n + d) < 0: n, d = -n, -d          # cameras above the ground
inl = cand[np.abs(cand @ n + d) < thr]
vh = np.linalg.svd(inl - inl.mean(0), full_matrices=False)[2]; u = unit(vh[0] - float(vh[0] @ n) * n); v = unit(np.cross(n, u))
mid = (cams[0] + cams[-1]) * 0.5; origin = mid - (float(n @ mid) + d) * n
proj = np.column_stack([(cams - origin) @ u, (cams - origin) @ v]); span = proj.max(0) - proj.min(0)
major = old["major_step_units"]; size = max(float(np.ceil((max(span) + major * 2) / major) * major), major * 5)
new = dict(old, normal=n.tolist(), d=d, origin=origin.tolist(), u=u.tolist(), v=v.tolist(), inlier_count=int(len(inl)), candidate_count=int(len(cand)),
           size_units=size, fixed_size_units=size, fit="terminal_region", fit_radius_m=radius_m)
quat = scene_alignment_quaternion(n, u)
ho, hn = (cams @ np.array(old["normal"]) + old["d"]) * scale, (cams @ n + d) * scale
print(f"{vid}: candidates {len(cand)} within {radius_m:.0f} m of the final camera, inliers {len(inl)}")
print(f"  terminal camera height: old {ho[-1]:+.1f} m -> new {hn[-1]:+.1f} m | first camera: old {ho[0]:.1f} m -> new {hn[0]:.1f} m | cloud >0.5 m below plane: old {100*float((P@np.array(old['normal'])+old['d'] < -0.5/scale).mean()):.1f}% -> new {100*float((P@n+d < -0.5/scale).mean()):.1f}%")
print(f"  old vs new normal: {np.degrees(np.arccos(np.clip(abs(np.array(old['normal'])@n),-1,1))):.1f} deg")
# --- side-by-side render: corrected cloud under the old grid and under the new grid, both levelled by their own quaternion
def panel(grid, q, W=1400, H=900):
    Rq = quat_to_R(q); Pr = P @ Rq.T; cr = cams @ Rq.T; g = grid_in_aligned_frame(grid, Rq)
    path = cr; ext = float(np.linalg.norm(path[-1] - path[0])); ctr = path.mean(0) + np.array([0, -0.12 * ext, 0])
    dv = path[-1] - path[0]; dv[1] = 0; dv /= max(np.linalg.norm(dv), 1e-9); left = np.cross([0, 1, 0], dv)
    R, eye = look_at(ctr - dv * 0.45 * ext + left * 0.62 * ext + np.array([0, 0.62 * ext, 0]), ctr)
    img = np.full((H, W, 3), (10, 11, 13), np.uint8); zb = np.full((H, W), np.inf); splat(img, zb, Pr, C, R, eye, 1100, 2); im = Image.fromarray(img)
    draw_grid(im, R, eye, 1100, g["origin"], g["u"], g["v"], g["size_units"], g["minor_step_units"], g["major_step_units"])
    draw_polyline(im, path, R, eye, 1100, (54, 228, 255), 3)
    cams_r = [{"position": cr[i], "right": np.array(meta["path"][i]["right"]) @ Rq.T, "down": np.array(meta["path"][i]["down"]) @ Rq.T, "forward": np.array(meta["path"][i]["forward"]) @ Rq.T} for i in range(len(cr))]
    for i in range(0, len(cams_r), 10): draw_polyline(im, frustum_pts(cams_r[i], ext * 0.045), R, eye, 1100, (54, 228, 255), 1)
    draw_polyline(im, frustum_pts(cams_r[-1], ext * 0.06), R, eye, 1100, (255, 77, 109), 2)
    return im
out = ROOT / "reports" / BATCH / vid / "ground_refit.jpg"
a, b = panel(old, meta["scene_alignment_quaternion"]), panel(new, quat)
cv = Image.new("RGB", (a.width * 2 + 12, a.height + 90), (10, 11, 13)); cv.paste(a, (0, 90)); cv.paste(b, (a.width + 12, 90)); dr = ImageDraw.Draw(cv)
dr.text((24, 14), spec.get(vid, {}).get("title", vid)[:80], fill=(235, 238, 242), font=font(30))
dr.text((24, 52), f"ground plane fitted by the artifact builder (colour-filtered, whole cloud): final camera {ho[-1]:+.0f} m above it", fill=(255, 176, 0), font=font(22))
dr.text((a.width + 36, 52), f"ground re-fitted to the terrain within {radius_m:.0f} m of the target: final camera {hn[-1]:+.0f} m", fill=(54, 228, 255), font=font(22))
cv.save(out, quality=92); print("  wrote", out)
if dry: sys.exit(0)
targets = [vd / "scene_meta.json"] + [p for p in (ROOT / "scenes" / vid).glob("*/viewer/scene_meta.json")]
for p in targets:
    bak = p.with_name("scene_meta.orig.json")
    if not bak.exists(): shutil.copy2(p, bak)
    m = json.loads(p.read_text()); m["ground_grid"] = new; m["scene_alignment_quaternion"] = quat; m["ground_note"] = f"ground plane re-fitted to the terrain within {radius_m:.0f} m of the target (terminal-region fit); builder fit kept in scene_meta.orig.json"
    p.write_text(json.dumps(m, indent=2)); print("  updated", p.relative_to(ROOT))
