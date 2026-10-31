#!/usr/bin/env python3
"""Ground-plane diagnosis for a re-run scene: compares candidate fits (viewer / ransac / low / terminal)."""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pipeline"))
from evaluate_3d_trajectory import umeyama
BATCH = os.environ.get("FPV_UNDISTORT_BATCH", "all_undistort"); SCENES = ROOT / "scenes" / BATCH

def load(vd):
    m = json.loads((vd / "scene_meta.json").read_text())
    P = np.fromfile(vd / Path(m["assets"]["positions"]).name, dtype="<f4").reshape(-1, 3).astype(float)
    return m, P, np.array([e["position"] for e in m["path"]], float)

def ransac_plane(P, thr, iters=600, seed=0):
    rng = np.random.default_rng(seed)
    if len(P) > 60000: P = P[rng.choice(len(P), 60000, replace=False)]
    best = (-1, None, None)
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(s[1]-s[0], s[2]-s[0]); nn = np.linalg.norm(n)
        if nn < 1e-12: continue
        n /= nn; d = -n @ s[0]; k = int((np.abs(P @ n + d) < thr).sum())
        if k > best[0]: best = (k, n, d)
    n, d = best[1], best[2]; inl = np.abs(P @ n + d) < thr
    c = P[inl].mean(0); n = np.linalg.svd(P[inl]-c, full_matrices=False)[2][2]; d = -n @ c
    return n, d, int(inl.sum())

def orient(n, d, cams):
    return (-n, -d) if np.median(cams @ n + d) < 0 else (n, d)

def describe(label, n, d, inl, P, cams, scale, ref_n):
    h = cams @ n + d; below = float((P @ n + d < -0.5/scale).mean())
    ang = float(np.degrees(np.arccos(np.clip(abs(n @ ref_n), -1, 1))))
    print(f"   {label:9s} inliers {inl:6d}  angle to published {ang:5.1f} deg  terminal cam {h[-1]*scale:6.1f} m  first cam {h[0]*scale:6.1f} m  min cam {h.min()*scale:6.1f} m  cloud >0.5 m below plane {100*below:4.1f}%")
    return n, d

def profile(P, cams, n, d, scale, bins=9):
    fwd = cams[-1] - cams[max(0, len(cams)-15)]; fwd -= n*(fwd@n); fwd /= max(np.linalg.norm(fwd), 1e-9); lat = np.cross(n, fwd)
    rel = P - cams[-1]; along = rel @ fwd; across = rel @ lat
    sel = (np.abs(along) < 40/scale) & (np.abs(across) < 60/scale)
    if sel.sum() < 500: return
    e = np.linspace(-60/scale, 60/scale, bins+1); out = []
    for i in range(bins):
        s = sel & (across >= e[i]) & (across < e[i+1]); out.append(f"{np.median(P[s]@n+d)*scale:+5.1f}" if s.sum() > 50 else "  n/a")
    print(f"   cross-track profile at the target (median height above plane, m, over 120 m): {' '.join(out)}")

vid = sys.argv[1]; radius = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
base = SCENES / vid
mp, Pp, cp = load(base/"published"); mc, Pc, cc = load(base/"pinhole"/"viewer")
scale = float(mp.get("default_scale_m_per_unit") or 117.6)
gp = mp["ground_grid"]; npub = np.array(gp["normal"])/np.linalg.norm(gp["normal"]); npub, dp = orient(npub, gp["d"], cp)
n_ = min(len(cc), len(cp)); s, R, t = umeyama(cc[:n_], cp[:n_]); ref_c = R.T @ npub; thr = gp["threshold_units"]
print(f"== {vid}  ({scale:.1f} m/unit, alignment scale {s:.3f})"); print("published product:")
describe("viewer", npub, dp, gp["inlier_count"], Pp, cp, scale, npub); profile(Pp, cp, npub, dp, scale)
print("lens-corrected run:")
gc = mc["ground_grid"]; nc = np.array(gc["normal"])/np.linalg.norm(gc["normal"]); nc, dc = orient(nc, gc["d"], cc)
describe("viewer", nc, dc, gc["inlier_count"], Pc, cc, scale*s, ref_c); profile(Pc, cc, nc, dc, scale*s)
n, d, k = ransac_plane(Pc, thr); n, d = orient(n, d, cc); describe("ransac", n, d, k, Pc, cc, scale*s, ref_c)
hgt = Pc @ nc + dc; low = Pc[hgt < np.quantile(hgt, 0.35)]; n, d, k = ransac_plane(low, thr); n, d = orient(n, d, cc); describe("low", n, d, k, Pc, cc, scale*s, ref_c)
near = Pc[np.linalg.norm(Pc-cc[-1], axis=1) < radius/(scale*s)]
if len(near) > 200:
    n, d, k = ransac_plane(near, thr); n, d = orient(n, d, cc); describe("terminal", n, d, k, Pc, cc, scale*s, ref_c); profile(Pc, cc, n, d, scale*s)
else: print(f"   terminal: only {len(near)} points within {radius:.0f} m of the final camera")
