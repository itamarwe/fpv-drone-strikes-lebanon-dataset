#!/usr/bin/env python3
"""Overlay several VGGT-Omega reconstructions of the same footage in one 3D page.

Each run directory must hold runpod_artifacts/predictions.npz (extrinsic, depth_conf,
world_points_from_depth, images), runpod_artifacts/sky_masks and frames/. The first
run is the reference frame; every other run is Sim(3)-aligned to it on camera
centres. Writes <out>/{index.html, meta.json, *.bin}. Layers can be shown one at a
time (solo) or overlaid.

  build_recon_overlay_viewer.py --out DIR --title T label=scene_dir [label=scene_dir ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_3d_trajectory import umeyama  # noqa: E402
from compare_reconstruction_to_blender_gt import resize_nearest  # noqa: E402

COLOURS = ["#ffffff", "#ff4d6d", "#36e4ff", "#ffb000", "#7cff6b", "#c58bff"]


def load_run(sd: Path, max_points: int, rng) -> dict:
    z = np.load(sd / "runpod_artifacts" / "predictions.npz", allow_pickle=True)
    ext = np.asarray(z["extrinsic"], float)
    c2w = np.array([np.linalg.inv(np.vstack([e, [0, 0, 0, 1]])) for e in ext])
    wp = np.asarray(z["world_points_from_depth"]); S, H, W = wp.shape[:3]
    conf = np.asarray(z["depth_conf"]).reshape(S, H, W)
    keep = np.isfinite(wp).all(axis=-1) & (conf >= np.quantile(conf, 0.5))
    sky = sorted((sd / "runpod_artifacts" / "sky_masks").glob("*"))
    if len(sky) == S:
        for i, f in enumerate(sky):
            keep[i] &= resize_nearest(np.asarray(Image.open(f).convert("L")), (H, W)) >= 128
    cols = (np.transpose(np.asarray(z["images"]), (0, 2, 3, 1))[keep] * 255).clip(0, 255).astype(np.uint8)
    pts = wp[keep].astype(np.float64)
    if len(pts) > max_points:
        k = rng.choice(len(pts), size=max_points, replace=False); pts, cols = pts[k], cols[k]
    return {"c2w": c2w, "pts": pts, "cols": cols, "frames": sorted(p.name for p in (sd / "frames").glob("*.jpg"))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="label=scene_dir; first is the reference")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--points-per-run", type=int, default=1_200_000)
    p.add_argument("--scale-m-per-unit", type=float, default=None, help="metres per reference unit (e.g. the published manual calibration)")
    args = p.parse_args()
    rng = np.random.default_rng(9)
    args.out.mkdir(parents=True, exist_ok=True)
    layers = []
    ref = None
    for i, item in enumerate(args.runs):
        label, path = item.split("=", 1); sd = Path(path)
        run = load_run(sd, args.points_per_run, rng)
        if ref is None:
            ref = run; s, R, t = 1.0, np.eye(3), np.zeros(3); align = {"reference": True}
        else:
            n = min(len(run["c2w"]), len(ref["c2w"]))
            s, R, t = umeyama(run["c2w"][:n, :3, 3], ref["c2w"][:n, :3, 3])
            err = np.linalg.norm((s * (R @ run["c2w"][:n, :3, 3].T)).T + t - ref["c2w"][:n, :3, 3], axis=1)
            align = {"reference": False, "matched_cameras": int(n), "scale_to_reference": float(s), "centre_rmse_ref_units": float(np.sqrt(np.mean(err**2)))}
        pts = (s * (R @ run["pts"].T)).T + t
        cams = []
        for fi, m in enumerate(run["c2w"]):
            Rm = R @ m[:3, :3]; c = s * (R @ m[:3, 3]) + t
            cams.append({"frame": fi + 1, "position": c.tolist(), "right": Rm[:, 0].tolist(), "down": Rm[:, 1].tolist(), "forward": Rm[:, 2].tolist()})
        pts.astype("<f4").tofile(args.out / f"layer{i}_points.bin"); run["cols"].tofile(args.out / f"layer{i}_colors.bin")
        layers.append({"key": f"layer{i}", "label": label, "colour": COLOURS[i % len(COLOURS)], "points": int(len(pts)), "cameras": cams, "alignment": align,
                       "frame_image_url": f"../{sd.name}/frames/" + "{name}", "frame_names": run["frames"]})
        print(f"{label}: {len(pts)} points, {len(cams)} cameras, align {align}")
    # Map everything into the reference run's viewer frame (scene_meta.json), so its ground grid,
    # alignment quaternion and metre-per-unit scale apply. The demo GLB frame differs from the raw
    # predictions frame by a rigid transform (fitted here on the reference cameras).
    ref_dir = Path(args.runs[0].split("=", 1)[1]); vm = ref_dir / "viewer" / "scene_meta.json"
    ground_grid = align_quat = None
    if vm.exists():
        vmeta = json.loads(vm.read_text())
        P_meta = np.array([e["position"] for e in vmeta["path"]], float); n = min(len(P_meta), len(ref["c2w"]))
        sg, Rg, tg = umeyama(ref["c2w"][:n, :3, 3], P_meta[:n])
        resid = np.linalg.norm((sg * (Rg @ ref["c2w"][:n, :3, 3].T)).T + tg - P_meta[:n], axis=1)
        print(f"raw->viewer frame: scale {sg:.4f}, residual max {resid.max():.5f}")
        for i, L in enumerate(layers):
            pts = np.fromfile(args.out / f"{L['key']}_points.bin", dtype="<f4").reshape(-1, 3).astype(np.float64)
            ((sg * (Rg @ pts.T)).T + tg).astype("<f4").tofile(args.out / f"{L['key']}_points.bin")
            for c in L["cameras"]:
                Rm = Rg @ np.array([c["right"], c["down"], c["forward"]]).T
                c["position"] = (sg * (Rg @ np.array(c["position"])) + tg).tolist()
                c["right"], c["down"], c["forward"] = Rm[:, 0].tolist(), Rm[:, 1].tolist(), Rm[:, 2].tolist()
        ground_grid = vmeta.get("ground_grid"); align_quat = vmeta.get("scene_alignment_quaternion")
        if args.scale_m_per_unit is None:
            args.scale_m_per_unit = vmeta.get("default_scale_m_per_unit")
    meta = {"title": args.title, "scale_m_per_unit": args.scale_m_per_unit, "layers": layers,
            "ground_grid": ground_grid, "scene_alignment_quaternion": align_quat,
            "note": "All runs Sim(3)-aligned to the first run on camera centres; units are the reference run's VGGT units."}
    (args.out / "meta.json").write_text(json.dumps(meta) + "\n")
    (args.out / "index.html").write_text(HTML)
    print("wrote", args.out)
    return 0


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Reconstruction overlay</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0c0d0f; color: #e8ebef; font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; overflow: hidden; }
  canvas { display: block; width: 100vw; height: 100vh; }
  .panel { position: absolute; left: 12px; top: 12px; width: 340px; background: rgba(21,24,28,.93); border: 1px solid #2a3037; border-radius: 10px; padding: 12px 14px; max-height: calc(100vh - 24px); overflow: auto; }
  .panel h1 { font-size: 14px; margin: 0 0 6px; }
  .muted { color: #8b95a1; font-size: 12px; }
  .row { display: flex; gap: 8px; align-items: center; margin: 6px 0; flex-wrap: wrap; }
  label.t { display: inline-flex; gap: 6px; align-items: center; padding: 3px 8px; border: 1px solid #2a3037; border-radius: 6px; cursor: pointer; }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
  input[type=range] { width: 100%; }
  button { background: #0f1216; color: #e8ebef; border: 1px solid #2a3037; border-radius: 6px; padding: 4px 10px; font: inherit; cursor: pointer; }
  button.on { border-color: #36e4ff; }
  #frameimg { width: 100%; border-radius: 6px; margin-top: 6px; border: 1px solid #2a3037; }
  .layer { display: grid; grid-template-columns: auto 1fr auto; gap: 6px; align-items: center; padding: 4px 0; border-bottom: 1px solid #1d2227; font-size: 12px; }
  .layer small { color: #8b95a1; }
  .foot { position: absolute; right: 12px; bottom: 10px; color: #8b95a1; font-size: 12px; background: rgba(21,24,28,.8); padding: 6px 10px; border-radius: 8px; }
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}</script>
</head>
<body>
<canvas id="c"></canvas>
<div class="panel">
  <h1 id="title">Loading…</h1>
  <div class="muted" id="sub"></div>
  <div class="row"><span class="muted">Mode</span><button id="modeSolo" class="on">one at a time</button><button id="modeOverlay">overlay</button>
    <span class="muted">tint</span><label class="t"><input type="checkbox" id="tint" /> by run</label></div>
  <div id="layers"></div>
  <div class="row"><span class="muted">point</span><input type="range" id="psize" min="0.001" max="0.02" step="0.001" value="0.004" style="width:120px" /><label class="t"><input type="checkbox" id="showPaths" checked /> paths</label></div>
  <div class="row"><input type="range" id="frame" min="0" max="1" value="0" /></div>
  <div class="row"><button id="play">▶</button><button id="fit">Fit</button><button id="follow">Follow camera</button><span id="frameText" class="muted"></span></div>
  <img id="frameimg" alt="" />
</div>
<div class="foot">Drag to orbit · scroll to zoom · right-drag to pan. Runs are Sim(3)-aligned to the reference run on camera centres.</div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const base = location.pathname.replace(/\/[^/]*$/, "/");
const meta = await (await fetch(base + "meta.json", {cache: "no-store"})).json();
const bin = async (n, T) => new T(await (await fetch(base + n)).arrayBuffer());
const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({canvas, antialias: true}); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setClearColor(0x0c0d0f);
const scene = new THREE.Scene(); const root = new THREE.Group(); scene.add(root);
function alignmentQuaternion(fromNormal, fromU) {
  const n = new THREE.Vector3(...fromNormal).normalize();
  const qAlign = new THREE.Quaternion().setFromUnitVectors(n, new THREE.Vector3(0, 1, 0));
  const uRot = new THREE.Vector3(...fromU).applyQuaternion(qAlign).normalize();
  const uXZ = new THREE.Vector3(uRot.x, 0, uRot.z); if (uXZ.lengthSq() < 1e-8) uXZ.set(1, 0, 0); else uXZ.normalize();
  return new THREE.Quaternion().setFromUnitVectors(uXZ, new THREE.Vector3(1, 0, 0)).multiply(qAlign);
}
if (meta.ground_grid) {
  const g = meta.ground_grid;
  const q = Array.isArray(meta.scene_alignment_quaternion) && meta.scene_alignment_quaternion.length === 4
    ? new THREE.Quaternion(...meta.scene_alignment_quaternion) : alignmentQuaternion(g.fitted_normal || g.normal, g.u);
  root.quaternion.copy(q);
  const o = new THREE.Vector3(...g.origin), u = new THREE.Vector3(...g.u).normalize(), v = new THREE.Vector3(...g.v).normalize();
  const half = g.size_units / 2, step = g.minor_step_units, majorEvery = Math.max(1, Math.round(g.major_step_units / g.minor_step_units));
  const gridGroup = new THREE.Group(); root.add(gridGroup);
  for (let i = -Math.ceil(half / step); i <= Math.ceil(half / step); i++) {
    const off = i * step, major = i % majorEvery === 0;
    const mat = new THREE.LineBasicMaterial({color: major ? 0x4b6472 : 0x25313a, transparent: true, opacity: major ? 0.86 : 0.46});
    gridGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([o.clone().add(u.clone().multiplyScalar(-half)).add(v.clone().multiplyScalar(off)), o.clone().add(u.clone().multiplyScalar(half)).add(v.clone().multiplyScalar(off))]), mat));
    gridGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([o.clone().add(v.clone().multiplyScalar(-half)).add(u.clone().multiplyScalar(off)), o.clone().add(v.clone().multiplyScalar(half)).add(u.clone().multiplyScalar(off))]), mat));
  }
}
const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1000); const controls = new OrbitControls(camera, canvas); controls.enableDamping = true;
const v3 = a => new THREE.Vector3(a[0], a[1], a[2]);
const $ = id => document.getElementById(id);
const layers = [];
for (const L of meta.layers) {
  const [pos, col] = await Promise.all([bin(L.key + "_points.bin", Float32Array), bin(L.key + "_colors.bin", Uint8Array)]);
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const rgb = new Float32Array(col.length); for (let i = 0; i < col.length; i++) rgb[i] = col[i] / 255;
  g.setAttribute("color", new THREE.BufferAttribute(rgb, 3));
  const tintc = new THREE.Color(L.colour);
  const mat = new THREE.PointsMaterial({size: 0.004, vertexColors: true, sizeAttenuation: true});
  const pts = new THREE.Points(g, mat); root.add(pts);
  const pathG = new THREE.Group(); root.add(pathG);
  pathG.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(L.cameras.map(c => v3(c.position))), new THREE.LineBasicMaterial({color: tintc})));
  layers.push({...L, pts, mat, pathG, tintc, rgb, cur: null});
}
function frustum(cam, color, len) {
  const c = v3(cam.position), f = v3(cam.forward), r = v3(cam.right), d = v3(cam.down);
  const n = c.clone().add(f.clone().multiplyScalar(len)); const hw = len * 0.9, hh = len * 0.4;
  const k = [n.clone().add(r.clone().multiplyScalar(-hw)).add(d.clone().multiplyScalar(-hh)), n.clone().add(r.clone().multiplyScalar(hw)).add(d.clone().multiplyScalar(-hh)), n.clone().add(r.clone().multiplyScalar(hw)).add(d.clone().multiplyScalar(hh)), n.clone().add(r.clone().multiplyScalar(-hw)).add(d.clone().multiplyScalar(hh))];
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints([c, k[0], k[1], c, k[2], k[1], k[2], k[3], c, k[0], k[3]]), new THREE.LineBasicMaterial({color}));
}
let solo = true, active = 0, following = false;
const box = new THREE.Box3(); for (const c of layers[0].cameras) box.expandByPoint(v3(c.position)); const extent = Math.max(0.2, box.getSize(new THREE.Vector3()).length());
$("title").textContent = meta.title;
$("sub").textContent = (meta.ground_grid ? `ground grid ${meta.ground_grid.minor_step_m}/${meta.ground_grid.major_step_m} m · ` : "") + `${layers.length} reconstructions · ${layers.reduce((a, l) => a + l.points, 0).toLocaleString()} points · reference units` + (meta.scale_m_per_unit ? ` · ${meta.scale_m_per_unit.toFixed(1)} m/unit (manual calibration of the reference)` : "");
const list = $("layers");
layers.forEach((L, i) => {
  const row = document.createElement("div"); row.className = "layer";
  const a = L.alignment; const info = a.reference ? "reference" : `scale ×${a.scale_to_reference.toFixed(3)} · cam RMSE ${a.centre_rmse_ref_units.toFixed(4)}`;
  row.innerHTML = `<input type="${solo ? "radio" : "checkbox"}" name="lay" data-i="${i}" ${i === 0 ? "checked" : ""}/><span><span class="sw" style="background:${L.colour}"></span> ${L.label}<br><small>${L.points.toLocaleString()} pts · ${L.cameras.length} cams · ${info}</small></span><span></span>`;
  list.appendChild(row);
});
function applyVisibility() {
  const inputs = [...list.querySelectorAll("input[name=lay]")];
  layers.forEach((L, i) => { const on = solo ? i === active : inputs[i].checked; L.pts.visible = on; L.pathG.visible = on && $("showPaths").checked; });
}
list.addEventListener("change", e => { const i = Number(e.target.dataset.i); if (solo) active = i; applyVisibility(); });
function setMode(s) { solo = s; $("modeSolo").classList.toggle("on", s); $("modeOverlay").classList.toggle("on", !s); list.querySelectorAll("input[name=lay]").forEach((el, i) => { el.type = s ? "radio" : "checkbox"; el.checked = s ? i === active : true; }); applyVisibility(); }
$("modeSolo").onclick = () => setMode(true); $("modeOverlay").onclick = () => setMode(false);
$("tint").onchange = e => { for (const L of layers) { const attr = L.pts.geometry.getAttribute("color"); if (e.target.checked) { for (let i = 0; i < attr.count; i++) attr.setXYZ(i, L.tintc.r, L.tintc.g, L.tintc.b); } else { attr.array.set(L.rgb); } attr.needsUpdate = true; } };
$("psize").oninput = e => { for (const L of layers) L.mat.size = Number(e.target.value); };
$("showPaths").onchange = applyVisibility;
const nframes = layers[0].cameras.length; $("frame").max = String(nframes - 1);
function setFrame(i) {
  for (const L of layers) { if (L.cur) L.cur.removeFromParent(); const c = L.cameras[Math.min(i, L.cameras.length - 1)]; L.cur = frustum(c, L.tintc, extent * 0.03); L.pathG.add(L.cur); }
  const ref = layers[0].cameras[i]; $("frameText").textContent = `frame ${i + 1}/${nframes}`;
  const name = layers[active].frame_names[i] || layers[0].frame_names[i]; $("frameimg").src = base + layers[active].frame_image_url.replace("{name}", name);
  if (following) { root.updateMatrixWorld(true); const c = v3(ref.position), fw = v3(ref.forward), up = v3(ref.down).multiplyScalar(-1);
    camera.position.copy(c.clone().sub(fw.clone().multiplyScalar(extent * 0.12)).add(up.clone().multiplyScalar(extent * 0.05)).applyMatrix4(root.matrixWorld)); controls.target.copy(c.clone().add(fw.clone().multiplyScalar(extent * 0.1)).applyMatrix4(root.matrixWorld)); }
}
$("frame").oninput = e => setFrame(Number(e.target.value));
let playing = false, acc = 0; $("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "❚❚" : "▶"; };
$("follow").onclick = () => { following = !following; $("follow").classList.toggle("on", following); setFrame(Number($("frame").value)); };
function fit() { following = false; $("follow").classList.remove("on"); root.updateMatrixWorld(true); const ctr = box.getCenter(new THREE.Vector3()).applyMatrix4(root.matrixWorld); camera.position.copy(ctr.clone().add(new THREE.Vector3(extent * 0.6, extent * 0.5, extent * 0.9))); controls.target.copy(ctr); camera.near = extent / 1000; camera.far = extent * 50; camera.updateProjectionMatrix(); }
$("fit").onclick = fit;
function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; if (canvas.width !== w * renderer.getPixelRatio()) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); } }
addEventListener("resize", resize);
applyVisibility(); fit(); setFrame(0);
let last = performance.now();
renderer.setAnimationLoop(now => { resize(); const dt = (now - last) / 1000; last = now; if (playing) { acc += dt; if (acc > 0.1) { acc = 0; const n = (Number($("frame").value) + 1) % nframes; $("frame").value = n; setFrame(n); } } controls.update(); renderer.render(scene, camera); });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
