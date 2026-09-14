"""Export ground truth from the synthetic FPV flyover Blender file.

Run inside Blender:
  Blender -b fpv_roof_flyover.blend --python-exit-code 1 --python export_blender_flyover_ground_truth.py -- --out DIR [--depth]

Writes into DIR:
  gt_cameras.json        per frame: c2w (Blender camera convention: -Z forward, +Y up), OpenCV-style c2w
                         (+Z forward, +Y down), position, intrinsics of the equisolid fisheye model.
  gt_surface_points.npz  area-weighted surface sample (xyz, class id) of every evaluated mesh and
                         geometry-node instance inside a box around the flight path.
  gt_depth/NNNNN.npy     (optional) per-frame Cycles depth at the render resolution, float32 metres,
                         Blender's Depth pass semantics for a panoramic camera (ray length).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--depth", action="store_true")
parser.add_argument("--depth-samples", type=int, default=1)
parser.add_argument("--points", type=int, default=3_000_000)
parser.add_argument("--margin-m", type=float, default=250.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--skip-surface", action="store_true")
args = parser.parse_args(argv)
args.out.mkdir(parents=True, exist_ok=True)

scene = bpy.context.scene
cam = scene.camera
assert cam is not None and cam.data.type in ("PANO", "PERSP")
res = (scene.render.resolution_x, scene.render.resolution_y)

# ---------------------------------------------------------------- cameras
# Blender camera looks down its local -Z with +Y up. OpenCV / VGGT convention is +Z forward, +Y down.
BLENDER_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
frames = []
for f in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(f)
    mw = np.array(cam.matrix_world, dtype=np.float64)
    c2w_cv = mw @ BLENDER_TO_CV
    frames.append({
        "frame": f,
        "time_seconds": (f - scene.frame_start) * scene.render.fps_base / scene.render.fps,
        "position_m": mw[:3, 3].tolist(),
        "c2w_blender": mw.tolist(),
        "c2w_opencv": c2w_cv.tolist(),
    })
scene.frame_set(scene.frame_start)
cd = cam.data
if cd.type == "PERSP":
    fx = cd.lens / cd.sensor_width * res[0]
    intrinsics = {"model": "pinhole", "fx": fx, "fy": fx, "cx": res[0] / 2, "cy": res[1] / 2, "resolution_px": list(res),
                  "focal_length_mm": float(cd.lens), "sensor_width_mm": float(cd.sensor_width),
                  "hfov_deg": float(math.degrees(2 * math.atan(cd.sensor_width / (2 * cd.lens)))),
                  "note": "Blender PERSP camera; the Z pass is camera-z for this camera type."}
else:
  intrinsics = {
    "model": "equisolid_fisheye",
    "focal_length_mm": float(cd.fisheye_lens),
    "sensor_width_mm": float(cd.sensor_width),
    "sensor_height_mm": float(cd.sensor_height),
    "sensor_fit": cd.sensor_fit,
    "fisheye_fov_radians": float(cd.fisheye_fov),
    "resolution_px": list(res),
    "note": "Equisolid: r = 2 f sin(theta/2) on the sensor plane; pixel scale = resolution_x / sensor_width.",
  }
(args.out / "gt_cameras.json").write_text(json.dumps({
    "convention": "c2w_opencv maps camera coordinates (+X right, +Y down, +Z forward) to Blender world metres",
    "world_units": "metres",
    "intrinsics": intrinsics,
    "frames": frames,
}, indent=2))
print(f"[gt] cameras: {len(frames)}", flush=True)

# ---------------------------------------------------------------- surface sample
positions = np.array([fr["position_m"] for fr in frames])
lo = positions.min(axis=0) - args.margin_m
hi = positions.max(axis=0) + args.margin_m
lo[2] = -1e9  # keep everything below the path (terrain), cap above
hi[2] = positions.max(axis=0)[2] + 60.0
print(f"[gt] sample box lo={lo.round(1).tolist()} hi={hi.round(1).tolist()}", flush=True)

if args.skip_surface:
    print('[gt] surface: skipped', flush=True)
dg = bpy.context.evaluated_depsgraph_get()
rng = np.random.default_rng(args.seed)
class_names: dict[str, int] = {}
mesh_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
DENSITY_PTS_PER_M2 = args.points / 4.0e5  # provisional density; final set is subsampled to --points
PER_INSTANCE_CAP = 60_000


def class_of(name: str) -> int:
    key = name.split(".")[0]
    if key not in class_names:
        class_names[key] = len(class_names)
    return class_names[key]


def mesh_triangles(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    key = mesh.name_full
    if key in mesh_cache:
        return mesh_cache[key]
    if len(mesh.polygons) == 0:
        mesh_cache[key] = None
        return None
    mesh.calc_loop_triangles()
    n_tri = len(mesh.loop_triangles)
    if n_tri == 0:
        mesh_cache[key] = None
        return None
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)
    tris = np.empty(n_tri * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area_local = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).astype(np.float64)
    mesh_cache[key] = (v0, v1, v2, area_local)
    return mesh_cache[key]


if not args.skip_surface:
    chunks_xyz, chunks_cls = [], []
    count_inst = 0
    total_area = 0.0
    total_tri = 0
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH":
            continue
        mw = np.array(inst.matrix_world, dtype=np.float64)
        bb = np.array([mw @ np.array([*corner, 1.0]) for corner in ob.bound_box])[:, :3]
        if (bb.max(axis=0) < lo).any() or (bb.min(axis=0) > hi).any():
            continue
        cached = mesh_triangles(ob.data)
        if cached is None:
            continue
        v0, v1, v2, area_local = cached
        # Uniform scale assumed for area scaling; instances here are rigid + uniform scale.
        scale3 = np.linalg.norm(mw[:3, :3], axis=0)
        area = area_local * float(scale3[0] * scale3[1])
        area_sum = float(area.sum())
        if area_sum <= 0:
            continue
        n_pts = int(min(PER_INSTANCE_CAP, max(8, round(area_sum * DENSITY_PTS_PER_M2))))
        pick = rng.choice(len(area), size=n_pts, p=area / area_sum)
        r1 = np.sqrt(rng.random(n_pts)); r2 = rng.random(n_pts)
        local = (1 - r1)[:, None] * v0[pick] + (r1 * (1 - r2))[:, None] * v1[pick] + (r1 * r2)[:, None] * v2[pick]
        world = (mw[:3, :3] @ local.astype(np.float64).T).T + mw[:3, 3]
        inside = ((world >= lo) & (world <= hi)).all(axis=1)
        if not inside.any():
            continue
        cls = class_of(inst.parent.name if inst.is_instance and inst.parent else ob.name)
        chunks_xyz.append(world[inside].astype(np.float32))
        chunks_cls.append(np.full(int(inside.sum()), cls, dtype=np.int32))
        count_inst += 1
        total_area += area_sum * float(inside.mean())
        total_tri += len(area)

    pts = np.concatenate(chunks_xyz); cls = np.concatenate(chunks_cls)
    if len(pts) > args.points:
        keep = rng.choice(len(pts), size=args.points, replace=False)
        pts, cls = pts[keep], cls[keep]
    np.savez_compressed(args.out / "gt_surface_points.npz", xyz=pts, cls=cls,
                        class_names=np.array(json.dumps({v: k for k, v in class_names.items()})),
                        box_lo=lo, box_hi=hi, triangle_count=total_tri, instance_count=count_inst, total_area_m2=float(total_area))
    print(f"[gt] surface: {count_inst} instances, {total_tri} triangles, area {total_area:.0f} m2, {len(pts)} points, {len(class_names)} classes", flush=True)


# ---------------------------------------------------------------- depth
if args.depth:
    depth_dir = args.out / "gt_depth"
    depth_dir.mkdir(exist_ok=True)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.depth_samples
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.render.use_motion_blur = False
    scene.render.film_transparent = False
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_combined = True
    if hasattr(scene, "compositing_node_group"):  # Blender 5.x
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new("gt_depth_compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
    else:
        scene.use_nodes = True
        tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    rl = tree.nodes.new("CompositorNodeRLayers")
    out = tree.nodes.new("CompositorNodeOutputFile")
    if hasattr(out, "directory"):  # Blender 5.x
        out.directory = str(depth_dir)
        items = out.file_output_items
        for item in list(items):
            items.remove(item)
        items.new("FLOAT", "depth_")
    else:
        out.base_path = str(depth_dir)
        out.file_slots[0].path = "depth_"
    out.format.file_format = "OPEN_EXR_MULTILAYER" if hasattr(out, "directory") else "OPEN_EXR"
    out.format.color_depth = "32"
    out.format.exr_codec = "NONE"
    tree.links.new(rl.outputs["Depth"], out.inputs[0])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    for f in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(f)
        scene.render.filepath = str(depth_dir / f"rgb_{f:05d}.png")
        bpy.ops.render.render(write_still=False)
        exrs = sorted(depth_dir.glob("*.exr"), key=lambda q: q.stat().st_mtime)
        assert exrs, f"no EXR written for frame {f} in {depth_dir}"
        exr = exrs[-1]
        import OpenImageIO as oiio  # bundled with Blender
        handle = oiio.ImageInput.open(str(exr))
        spec = handle.spec()
        arr = np.array(handle.read_image(oiio.FLOAT), dtype=np.float32).reshape(spec.height, spec.width, -1)
        handle.close()
        chan = [i for i, name in enumerate(spec.channelnames) if name.endswith(("Z", ".V", "V"))]
        px = arr[..., chan[0] if chan else 0]
        px = np.where(px > 1e9, np.nan, px)  # Cycles uses a huge sentinel for no-hit (sky)
        np.save(depth_dir / f"{f:05d}.npy", px)  # OIIO returns top-down rows
        exr.unlink()
        if f % 10 == 0 or f == scene.frame_start:
            ok = np.isfinite(px)
            print(f"[gt] depth {f}/{scene.frame_end} hit_frac={ok.mean():.3f} median={np.median(px[ok]):.1f} min={px[ok].min():.2f}", flush=True)
    print("[gt] depth done", flush=True)
print("[gt] DONE", flush=True)
