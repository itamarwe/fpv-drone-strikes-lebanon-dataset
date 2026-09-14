#!/usr/bin/env python3
"""Build a browser overlay of Blender ground truth and the VGGT-Omega reconstruction.

Writes <scene>/gt_overlay/{index.html, meta.json, *.bin}. Everything is in
Blender world metres: the VGGT cloud and cameras are mapped through the same
Sim(3) (camera centres) used by compare_reconstruction_to_blender_gt.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_3d_trajectory import umeyama  # noqa: E402
from compare_reconstruction_to_blender_gt import resize_nearest  # noqa: E402

VEG = ("olive", "pine", "cypress", "shrub", "grass", "meadow", "scrub", "flower", "wildflower", "mustard", "orchard")
GROUND = ("terrain", "road", "gravel", "driveway", "embankment", "limestone", "headland", "ravine", "field boundar")
MOVING = ("car", "pedestrian")


def gt_colour(name: str) -> tuple[int, int, int]:
    n = name.lower()
    if any(k in n for k in MOVING):
        return (230, 60, 60)
    if any(k in n for k in VEG):
        return (70, 140, 60)
    if any(k in n for k in GROUND):
        return (176, 150, 110)
    return (225, 215, 200)  # buildings, walls, roofs, fixtures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_dir", type=Path)
    p.add_argument("--gt-points", type=int, default=1_500_000)
    p.add_argument("--vggt-points", type=int, default=1_500_000)
    p.add_argument("--near-m", type=float, default=120.0, help="range from the true path that counts as near")
    args = p.parse_args()
    sd = args.scene_dir
    out = sd / "gt_overlay"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(5)

    gt = json.loads((sd / "ground_truth" / "gt_cameras.json").read_text())
    gt_c2w = np.array([f["c2w_opencv"] for f in gt["frames"]], dtype=np.float64)
    frame_ids = [int(f["frame"]) for f in gt["frames"]]

    z = np.load(sd / "runpod_artifacts" / "predictions.npz", allow_pickle=True)
    ext = np.asarray(z["extrinsic"], dtype=np.float64)
    pred_c2w = np.array([np.linalg.inv(np.vstack([e, [0, 0, 0, 1]])) for e in ext])
    s, R, t = umeyama(pred_c2w[:, :3, 3], gt_c2w[:, :3, 3])
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
    pred_al = np.array([T @ m for m in pred_c2w])
    # Fix the scale back into the rotation block for a proper camera basis.
    for m in pred_al:
        m[:3, :3] /= s

    # VGGT cloud: raw world points, confidence >= median, sky masked, aligned.
    wp = np.asarray(z["world_points_from_depth"]); S, H, W = wp.shape[:3]
    conf = np.asarray(z["depth_conf"]).reshape(S, H, W)
    keep = np.isfinite(wp).all(axis=-1) & (conf >= np.quantile(conf, 0.5))
    sky_files = sorted((sd / "runpod_artifacts" / "sky_masks").glob("*"))
    if len(sky_files) == S:
        from PIL import Image
        for i, f in enumerate(sky_files):
            keep[i] &= resize_nearest(np.asarray(Image.open(f).convert("L")), (H, W)) >= 128
    imgs = np.asarray(z["images"])  # (S,3,H,W) in 0..1
    cols = (np.transpose(imgs, (0, 2, 3, 1))[keep] * 255).clip(0, 255).astype(np.uint8)
    pts = wp[keep].astype(np.float64)
    if len(pts) > args.vggt_points:
        k = rng.choice(len(pts), size=args.vggt_points, replace=False); pts, cols = pts[k], cols[k]
    pts_al = (s * (R @ pts.T)).T + t

    gts = np.load(sd / "ground_truth" / "gt_surface_points.npz")
    gxyz = gts["xyz"]; gcls = gts["cls"]
    names = json.loads(str(gts["class_names"]))
    palette = np.array([gt_colour(names[str(i)]) for i in range(len(names))], dtype=np.uint8)
    from scipy.spatial import cKDTree
    path_tree = cKDTree(gt_c2w[:, :3, 3])
    g_range, _ = path_tree.query(gxyz, k=1, workers=-1)
    near = g_range < args.near_m
    far_idx = np.where(~near)[0]
    far_keep = rng.choice(far_idx, size=min(len(far_idx), max(0, args.gt_points - int(near.sum()))), replace=False)
    sel = np.concatenate([np.where(near)[0], far_keep])
    gxyz, gcls = gxyz[sel], gcls[sel]
    gcol = palette[gcls]
    # VGGT: split by range to the nearest true camera so the far background can be hidden.
    v_range, _ = path_tree.query(pts_al, k=1, workers=-1)
    v_near = v_range < args.near_m

    # Recentre everything on the GT path centroid so float32 stays precise.
    origin = gt_c2w[:, :3, 3].mean(axis=0)
    (gxyz - origin).astype("<f4").tofile(out / "gt_points.bin"); gcol.astype(np.uint8).tofile(out / "gt_colors.bin")
    (pts_al[v_near] - origin).astype("<f4").tofile(out / "vggt_points.bin"); cols[v_near].astype(np.uint8).tofile(out / "vggt_colors.bin")
    (pts_al[~v_near] - origin).astype("<f4").tofile(out / "vggt_points_far.bin"); cols[~v_near].astype(np.uint8).tofile(out / "vggt_colors_far.bin")

    def cam_list(c2w: np.ndarray) -> list[dict]:
        return [{"frame": fid, "position": (m[:3, 3] - origin).tolist(), "right": m[:3, 0].tolist(),
                 "down": m[:3, 1].tolist(), "forward": m[:3, 2].tolist()} for fid, m in zip(frame_ids, c2w)]

    comparison = json.loads((sd / "gt_comparison" / "comparison.json").read_text()) if (sd / "gt_comparison" / "comparison.json").exists() else {}
    meta = {
        "title": "Synthetic FPV roof flyover: Blender truth vs VGGT-Omega",
        "units": "metres (Blender world, recentred on the ground-truth path centroid)",
        "origin_world_m": origin.tolist(),
        "sim3_metres_per_vggt_unit": float(s),
        "gt_points": int(len(gxyz)), "gt_points_near": int(near.sum()), "vggt_points": int(len(pts_al)), "vggt_points_near": int(v_near.sum()), "near_m": args.near_m,
        "legend": {"ground": [176, 150, 110], "vegetation": [70, 140, 60], "buildings": [225, 215, 200], "vehicles_people": [230, 60, 60]},
        "gt_cameras": cam_list(gt_c2w), "vggt_cameras": cam_list(pred_al),
        "frame_image_url": "../frames/f_{frame:06d}.jpg",
        "summary": {
            "scale": {k: comparison.get("scale", {}).get(k) for k in ("metres_per_vggt_unit_sim3", "camera_centre_error_fraction_of_path_length")},
            "local_scale_range": comparison.get("local_scale", {}).get("relative_scale_range"),
            "surface_median_m": comparison.get("cloud_vs_surface", {}).get("distance_to_surface_m_after_rigid_refinement", {}).get("median"),
            "depth_ratio_full": comparison.get("depth_vs_gt", {}).get("global_median_ratio_pred_over_gt"),
            "depth_ratio_central": comparison.get("depth_vs_gt", {}).get("median_central_30deg_ratio_across_frames"),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta) + "\n")
    (out / "index.html").write_text(HTML)
    print(f"wrote {out}: gt={len(gxyz)} vggt={len(pts_al)} scale={s:.2f}")
    return 0


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Flyover: Blender truth vs VGGT-Omega</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0c0d0f; color: #e8ebef; font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; overflow: hidden; }
  canvas { display: block; width: 100vw; height: 100vh; }
  .panel { position: absolute; left: 12px; top: 12px; width: 330px; background: rgba(21,24,28,.92); border: 1px solid #2a3037; border-radius: 10px; padding: 12px 14px; }
  .panel h1 { font-size: 14px; margin: 0 0 6px; }
  .muted { color: #8b95a1; font-size: 12px; }
  .row { display: flex; gap: 8px; align-items: center; margin: 6px 0; flex-wrap: wrap; }
  label.t { display: inline-flex; gap: 5px; align-items: center; padding: 3px 8px; border: 1px solid #2a3037; border-radius: 6px; cursor: pointer; }
  input[type=range] { width: 100%; }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
  .stats { font-size: 12px; color: #c9d1d9; }
  .stats b { color: #fff; }
  #frameimg { width: 100%; border-radius: 6px; margin-top: 6px; border: 1px solid #2a3037; }
  button { background: #0f1216; color: #e8ebef; border: 1px solid #2a3037; border-radius: 6px; padding: 4px 10px; font: inherit; cursor: pointer; }
  .foot { position: absolute; right: 12px; bottom: 10px; color: #8b95a1; font-size: 12px; background: rgba(21,24,28,.8); padding: 6px 10px; border-radius: 8px; }
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}</script>
</head>
<body>
<canvas id="c"></canvas>
<div class="panel">
  <h1 id="title">Loading…</h1>
  <div class="muted" id="sub"></div>
  <div class="row">
    <label class="t"><input type="checkbox" id="showGT" checked /> <span class="sw" style="background:#b0966e"></span>Blender model</label>
    <label class="t"><input type="checkbox" id="showV" checked /> <span class="sw" style="background:#9fd3ff"></span>VGGT cloud</label>
  </div>
  <div class="row">
    <label class="t"><input type="checkbox" id="showGTPath" checked /> <span class="sw" style="background:#ffffff"></span>true path</label>
    <label class="t"><input type="checkbox" id="showVPath" checked /> <span class="sw" style="background:#ff4d6d"></span>VGGT path</label>
    <label class="t"><input type="checkbox" id="showGrid" checked /> 50 m grid</label>
    <label class="t"><input type="checkbox" id="showFar" /> VGGT far background</label>
  </div>
  <div class="row"><span class="muted">VGGT cloud tint</span>
    <label class="t"><input type="radio" name="tint" value="rgb" checked /> RGB</label>
    <label class="t"><input type="radio" name="tint" value="blue" /> blue</label>
    <span class="muted">point</span><input type="range" id="psize" min="0.2" max="3" step="0.1" value="0.9" style="width:80px" />
  </div>
  <div class="row"><input type="range" id="frame" min="0" max="119" value="0" /></div>
  <div class="row"><button id="play">▶</button><button id="fit">Fit</button><button id="follow">Follow camera</button><span id="frameText" class="muted"></span></div>
  <img id="frameimg" alt="" />
  <div class="stats" id="stats"></div>
</div>
<div class="foot">Drag to orbit · scroll to zoom · right-drag to pan. VGGT is Sim(3)-aligned on camera centres; its scale is the ground-truth fit.</div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const base = location.pathname.replace(/\/[^/]*$/, "/");
const meta = await (await fetch(base + "meta.json", {cache: "no-store"})).json();
const bin = async (n, T) => new T(await (await fetch(base + n)).arrayBuffer());
const [gp, gc, vp, vc, vpf, vcf] = await Promise.all([bin("gt_points.bin", Float32Array), bin("gt_colors.bin", Uint8Array), bin("vggt_points.bin", Float32Array), bin("vggt_colors.bin", Uint8Array), bin("vggt_points_far.bin", Float32Array), bin("vggt_colors_far.bin", Uint8Array)]);

const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x0c0d0f);
const scene = new THREE.Scene();
// Blender is Z-up; render with Z up by rotating the root so +Z maps to three's +Y.
const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; scene.add(root);
const camera = new THREE.PerspectiveCamera(50, 1, 0.5, 20000);
const controls = new OrbitControls(camera, canvas); controls.enableDamping = true;

function cloud(pos, col, size, tintBlue) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const f = new Float32Array(col.length);
  for (let i = 0; i < col.length; i += 3) {
    if (tintBlue) { f[i] = 0.62; f[i+1] = 0.83; f[i+2] = 1.0; }
    else { f[i] = col[i]/255; f[i+1] = col[i+1]/255; f[i+2] = col[i+2]/255; }
  }
  g.setAttribute("color", new THREE.BufferAttribute(f, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({size, vertexColors: true, sizeAttenuation: true}));
}
const gtCloud = cloud(gp, gc, 0.9, false); root.add(gtCloud);
const vRgb = cloud(vp, vc, 0.9, false); root.add(vRgb);
const vBlue = cloud(vp, vc, 0.9, true); vBlue.visible = false; root.add(vBlue);
const vFarRgb = cloud(vpf, vcf, 0.9, false); vFarRgb.visible = false; root.add(vFarRgb);
const vFarBlue = cloud(vpf, vcf, 0.9, true); vFarBlue.visible = false; root.add(vFarBlue);

const v3 = a => new THREE.Vector3(a[0], a[1], a[2]);
function pathLine(cams, color) {
  const g = new THREE.BufferGeometry().setFromPoints(cams.map(c => v3(c.position)));
  return new THREE.Line(g, new THREE.LineBasicMaterial({color, linewidth: 2}));
}
function frustum(cam, color, len) {
  const c = v3(cam.position), f = v3(cam.forward), r = v3(cam.right), d = v3(cam.down);
  const n = c.clone().add(f.clone().multiplyScalar(len)); const hw = len * 0.9, hh = len * 0.55;
  const k = [n.clone().add(r.clone().multiplyScalar(-hw)).add(d.clone().multiplyScalar(-hh)), n.clone().add(r.clone().multiplyScalar(hw)).add(d.clone().multiplyScalar(-hh)), n.clone().add(r.clone().multiplyScalar(hw)).add(d.clone().multiplyScalar(hh)), n.clone().add(r.clone().multiplyScalar(-hw)).add(d.clone().multiplyScalar(hh))];
  const pts = [c, k[0], k[1], c, k[2], k[1], k[2], k[3], c, k[0], k[3]];
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color}));
}
const gtPath = new THREE.Group(); gtPath.add(pathLine(meta.gt_cameras, 0xffffff)); root.add(gtPath);
const vPath = new THREE.Group(); vPath.add(pathLine(meta.vggt_cameras, 0xff4d6d)); root.add(vPath);
for (let i = 0; i < meta.gt_cameras.length; i += 10) { gtPath.add(frustum(meta.gt_cameras[i], 0xffffff, 6)); vPath.add(frustum(meta.vggt_cameras[i], 0xff4d6d, 6)); }
let curGT = null, curV = null;
const marker = new THREE.Mesh(new THREE.SphereGeometry(1.5, 12, 12), new THREE.MeshBasicMaterial({color: 0xffb000})); root.add(marker);

// Grid on the ground: use the lowest 5% of GT z as ground reference.
const zs = Array.from({length: gp.length / 3}, (_, i) => gp[i*3+2]).sort((a, b) => a - b);
const groundZ = zs[Math.floor(zs.length * 0.05)];
const grid = new THREE.GridHelper(1000, 20, 0x4b6472, 0x25313a); grid.rotation.x = Math.PI / 2; grid.position.z = groundZ; root.add(grid);

const $ = id => document.getElementById(id);
$("title").textContent = meta.title;
$("sub").textContent = `${meta.gt_points.toLocaleString()} Blender surface points (${meta.gt_points_near.toLocaleString()} within ${meta.near_m} m of the path, kept in full) · ${meta.vggt_points.toLocaleString()} VGGT points (${meta.vggt_points_near.toLocaleString()} near) · ${meta.units}`;
const s = meta.summary, f2 = (x, d=2) => x == null ? "n/a" : Number(x).toFixed(d);
$("stats").innerHTML = `Sim(3) scale <b>${f2(s.scale?.metres_per_vggt_unit_sim3)} m/unit</b> · path RMSE <b>${f2(100*(s.scale?.camera_centre_error_fraction_of_path_length ?? NaN),1)}%</b> of length<br>local scale range <b>${s.local_scale_range ? s.local_scale_range.map(x=>f2(x)).join(" – ") : "n/a"}</b> · near-surface median <b>${f2(s.surface_median_m)} m</b><br>depth ratio full-frame <b>${f2(s.depth_ratio_full)}</b> · central 30° <b>${f2(s.depth_ratio_central)}</b>`;

$("showGT").onchange = e => gtCloud.visible = e.target.checked;
$("showV").onchange = e => { const t = document.querySelector("input[name=tint]:checked").value; const far = $("showFar").checked; vRgb.visible = e.target.checked && t === "rgb"; vBlue.visible = e.target.checked && t === "blue"; vFarRgb.visible = far && vRgb.visible; vFarBlue.visible = far && vBlue.visible; };
$("showFar").onchange = () => $("showV").onchange({target: $("showV")});
document.querySelectorAll("input[name=tint]").forEach(r => r.onchange = () => $("showV").onchange({target: $("showV")}));
$("showGTPath").onchange = e => gtPath.visible = e.target.checked;
$("showVPath").onchange = e => vPath.visible = e.target.checked;
$("showGrid").onchange = e => grid.visible = e.target.checked;
$("psize").oninput = e => { const v = Number(e.target.value); for (const c of [gtCloud, vRgb, vBlue, vFarRgb, vFarBlue]) c.material.size = v; };

let following = false;
function setFrame(i) {
  const g = meta.gt_cameras[i], v = meta.vggt_cameras[i];
  if (curGT) curGT.removeFromParent(); if (curV) curV.removeFromParent();
  curGT = frustum(g, 0xffb000, 10); curV = frustum(v, 0xff4d6d, 10); gtPath.add(curGT); vPath.add(curV);
  marker.position.copy(v3(g.position)); marker.visible = !following;
  const err = v3(g.position).distanceTo(v3(v.position));
  $("frameText").textContent = `frame ${g.frame}/${meta.gt_cameras.length} · camera error ${err.toFixed(1)} m`;
  $("frameimg").src = base + meta.frame_image_url.replace("{frame:06d}", String(g.frame).padStart(6, "0"));
  if (following) {
    const c = v3(g.position), fw = v3(g.forward), up = v3(g.down).multiplyScalar(-1);
    const eye = c.clone().sub(fw.clone().multiplyScalar(45)).add(up.clone().multiplyScalar(18));
    const tgt = c.clone().add(fw.clone().multiplyScalar(40));
    camera.position.copy(eye.applyMatrix4(root.matrixWorld)); controls.target.copy(tgt.applyMatrix4(root.matrixWorld));
  }
}
$("frame").oninput = e => setFrame(Number(e.target.value));
let playing = false, acc = 0;
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "❚❚" : "▶"; };
$("follow").onclick = () => { following = !following; $("follow").style.borderColor = following ? "#36e4ff" : "#2a3037"; if (following) setFrame(Number($("frame").value)); };
function fit() {
  following = false; $("follow").style.borderColor = "#2a3037"; marker.visible = true;
  const box = new THREE.Box3(); for (const c of meta.gt_cameras) box.expandByPoint(v3(c.position).applyMatrix4(root.matrixWorld));
  const ctr = box.getCenter(new THREE.Vector3()); const size = Math.max(200, box.getSize(new THREE.Vector3()).length());
  camera.position.copy(ctr.clone().add(new THREE.Vector3(size * 0.55, size * 0.5, size * 0.8))); controls.target.copy(ctr);
}
$("fit").onclick = fit;
function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; if (canvas.width !== w * renderer.getPixelRatio()) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); } }
addEventListener("resize", resize);
root.updateMatrixWorld(true); fit(); setFrame(0);
let last = performance.now();
renderer.setAnimationLoop(now => {
  resize(); const dt = (now - last) / 1000; last = now;
  if (playing) { acc += dt; if (acc > 1 / 5.41) { acc = 0; const n = (Number($("frame").value) + 1) % meta.gt_cameras.length; $("frame").value = n; setFrame(n); } }
  controls.update(); renderer.render(scene, camera);
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
