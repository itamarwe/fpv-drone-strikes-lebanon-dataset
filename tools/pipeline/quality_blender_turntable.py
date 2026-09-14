"""Render an explicitly exploratory turntable from a native mesh asset.

Run with Blender, for example:
  blender -b --python quality_blender_turntable.py -- \
    --input mesh_textured.glb --output-dir review/turntable --mode native

These renders are presentation/geometry-review views. They are deliberately
labelled exploratory and must not be used as camera-matched photometric scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("neutral", "native"), default="neutral")
    parser.add_argument("--engine", choices=("eevee", "cycles_cpu"), default="eevee")
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--denoise", action="store_true", help="Enable Cycles denoising only when the Blender build supports it.")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_asset(path: Path) -> list:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise ValueError(f"Unsupported input format: {suffix}")
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise ValueError(f"No mesh objects imported from {path}")
    if not any(len(obj.data.polygons) for obj in objects):
        raise ValueError(
            "Imported asset contains points but no faces; use CloudCompare/MeshLab for point review, "
            "and do not accept a blank Blender render"
        )
    return objects


def bounds(objects: list) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return low, high


def neutral_material():
    material = bpy.data.materials.new("Benchmark neutral clay")
    material.diffuse_color = (0.62, 0.65, 0.68, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.62, 0.65, 0.68, 1.0)
    principled.inputs["Roughness"].default_value = 0.72
    return material


def bind_native_vertex_colours(objects: list) -> list[dict]:
    """Bind imported PLY colour attributes to Principled Base Color explicitly."""
    bindings = []
    for obj in objects:
        attributes = list(getattr(obj.data, "color_attributes", []))
        if not attributes:
            bindings.append({"object": obj.name, "status": "no_vertex_colour_attribute"})
            continue
        active = getattr(obj.data.color_attributes, "active_color", None)
        attribute = active or attributes[0]
        material = bpy.data.materials.new(f"{obj.name} native vertex colours")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get("Principled BSDF")
        if principled is None:
            raise ValueError("Principled BSDF node is unavailable")
        try:
            colour_node = nodes.new("ShaderNodeVertexColor")
            colour_node.layer_name = attribute.name
        except RuntimeError:
            colour_node = nodes.new("ShaderNodeAttribute")
            colour_node.attribute_name = attribute.name
        links.new(colour_node.outputs["Color"], principled.inputs["Base Color"])
        principled.inputs["Roughness"].default_value = 0.72
        obj.data.materials.clear()
        obj.data.materials.append(material)
        bindings.append({
            "object": obj.name,
            "status": "bound_to_principled_base_color",
            "attribute": attribute.name,
            "domain": attribute.domain,
            "data_type": attribute.data_type,
        })
    return bindings


def look_at(obj, target: Vector):
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(name: str, location: Vector, target: Vector, energy: float, size: float):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, target)


def main() -> int:
    args = parse_args()
    if min(args.frames, args.width, args.height, args.samples, args.threads) <= 0:
        raise SystemExit("frames, dimensions, samples and threads must be positive")
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"Input does not exist: {source}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    objects = import_asset(source)
    colour_bindings = []
    if args.mode == "neutral":
        material = neutral_material()
        for obj in objects:
            if obj.type == "MESH":
                obj.data.materials.clear()
                obj.data.materials.append(material)
    elif source.suffix.lower() == ".ply":
        colour_bindings = bind_native_vertex_colours(objects)
    low, high = bounds(objects)
    center = (low + high) * 0.5
    extent = high - low
    radius = max(float(extent.length) * 1.15, 0.01)
    camera_data = bpy.data.cameras.new("Benchmark camera")
    camera_data.lens = 52
    camera = bpy.data.objects.new("Benchmark camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    add_area_light("Key", center + Vector((radius, -radius, radius)), center, 1600, radius)
    add_area_light("Fill", center + Vector((-radius, -radius * 0.5, radius * 0.4)), center, 900, radius)
    world = bpy.context.scene.world or bpy.data.worlds.new("Benchmark world")
    bpy.context.scene.world = world
    world.color = (0.025, 0.03, 0.04)
    scene = bpy.context.scene
    if args.engine == "cycles_cpu":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = args.denoise
        scene.render.threads_mode = "FIXED"
        scene.render.threads = args.threads
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 2, 0) else "BLENDER_EEVEE"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    camera_records = []
    for index in range(args.frames):
        theta = 2 * math.pi * index / args.frames
        camera.location = center + Vector((radius * math.cos(theta), radius * math.sin(theta), radius * 0.36))
        look_at(camera, center)
        output_path = output_dir / f"turntable_{index:04d}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        camera_records.append(
            {
                "frame": index,
                "output": output_path.name,
                "matrix_world": [list(row) for row in camera.matrix_world],
                "lens_mm": camera_data.lens,
            }
        )
    metadata = {
        "schema_version": 1,
        "view_role": "exploratory",
        "photometric_scoring_allowed": False,
        "reason": "object-centred turntable cameras are not matched to source footage",
        "input": str(source),
        "input_sha256": sha256(source),
        "mode": args.mode,
        "native_vertex_colour_bindings": colour_bindings,
        "engine": args.engine,
        "samples": args.samples if args.engine == "cycles_cpu" else None,
        "threads": args.threads if args.engine == "cycles_cpu" else None,
        "denoise": args.denoise if args.engine == "cycles_cpu" else None,
        "resolution": [args.width, args.height],
        "bounds": {"min": list(low), "max": list(high)},
        "cameras": camera_records,
    }
    (output_dir / "turntable_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
