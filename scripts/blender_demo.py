"""Headless Blender demo for the sfm2mesh output.

Imports the mesh PLY, PCA-aligns it, scales it to a fixed size, builds a
three-point light rig with a camera on a turntable, renders four stills and
saves a .blend the user can open interactively.

Run:
  blender -b --python scripts/blender_demo.py -- <mesh.ply> <out_dir>
"""

import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def args_after_dashes():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path):
    bpy.ops.wm.ply_import(filepath=path)
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError(f"no mesh imported from {path}")
    return objs[0]


def pca_align(obj):
    """Rotate so the principal axes match world axes, centre, uniform scale."""
    mesh = obj.data
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)

    centre = co.mean(axis=0)
    centred = co - centre
    cov = centred.T @ centred / max(n, 1)
    _, vecs = np.linalg.eigh(cov)  # columns are eigenvectors, ascending
    vecs = vecs[:, ::-1]  # descending variance
    if np.linalg.det(vecs) < 0:  # keep it a rotation
        vecs[:, 2] *= -1.0
    # Rows of `vecs.T` are the new basis vectors.
    rotated = centred @ vecs

    extent = rotated.max(axis=0) - rotated.min(axis=0)
    scale = 2.0 / max(extent.max(), 1e-9)
    rotated *= scale
    rotated -= (rotated.max(axis=0) + rotated.min(axis=0)) / 2.0

    mesh.vertices.foreach_set("co", rotated.reshape(-1))
    mesh.update()
    return extent * scale


def colour_attribute(obj):
    attrs = obj.data.color_attributes
    if not attrs:
        return None
    return attrs[0].name


def colour_material(attr_name):
    mat = bpy.data.materials.new("vertex_colour")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = attr_name
    bsdf.inputs["Roughness"].default_value = 0.55
    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def clay_material(name="clay", value=0.82):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (value, value, value, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def wire_material():
    mat = bpy.data.materials.new("wire")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    emit.inputs["Strength"].default_value = 1.0
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def add_lights():
    def area(name, loc, energy, size):
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = loc
        direction = Vector((0.0, 0.0, 0.0)) - Vector(loc)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return obj

    area("key", (3.5, -3.5, 4.0), 700.0, 4.0)
    area("fill", (-4.0, -1.5, 2.0), 220.0, 5.0)
    area("rim", (0.5, 4.0, 3.0), 500.0, 3.0)


def add_camera():
    data = bpy.data.cameras.new("camera")
    data.lens = 55.0
    cam = bpy.data.objects.new("camera", data)
    bpy.context.collection.objects.link(cam)

    target = bpy.data.objects.new("target", None)
    bpy.context.collection.objects.link(target)
    target.location = (0.0, 0.0, 0.0)

    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.context.scene.camera = cam
    return cam


def place_camera(cam, angle_deg, elevation=0.42, radius=4.6):
    theta = math.radians(angle_deg)
    cam.location = (
        radius * math.cos(theta),
        radius * math.sin(theta),
        radius * elevation,
    )


def setup_render(out_dir, res=(900, 750), samples=64):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.02, 0.022, 0.028, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    scene.render.filepath = os.path.join(out_dir, "render")


def render_to(cam, path, angle, elevation=0.42):
    place_camera(cam, angle, elevation)
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def add_wireframe_copy(obj, mat):
    copy = obj.copy()
    copy.data = obj.data.copy()
    bpy.context.collection.objects.link(copy)
    copy.data.materials.clear()
    copy.data.materials.append(mat)
    mod = copy.modifiers.new("wire", type="WIREFRAME")
    mod.thickness = 0.006
    mod.use_even_offset = True
    return copy


def main():
    raw = args_after_dashes()
    if len(raw) != 2:
        raise SystemExit("usage: blender -b --python blender_demo.py -- <mesh.ply> <out_dir>")
    mesh_path, out_dir = raw[0], raw[1]
    os.makedirs(out_dir, exist_ok=True)

    clear_scene()
    obj = import_mesh(mesh_path)
    extent = pca_align(obj)
    print(f"[demo] mesh: {len(obj.data.vertices)} vertices, {len(obj.data.polygons)} faces")
    print(f"[demo] aligned extent: {extent}")

    attr = colour_attribute(obj)
    print(f"[demo] colour attribute: {attr}")
    obj.data.materials.clear()
    obj.data.materials.append(
        colour_material(attr) if attr else clay_material()
    )

    add_lights()
    cam = add_camera()
    setup_render(out_dir)

    for angle, name in [(35, "view_1"), (140, "view_2"), (245, "view_3")]:
        print(f"[demo] rendering {name}")
        render_to(cam, os.path.join(out_dir, f"{name}.png"), angle)

    # Wireframe diagnostic: show the actual open patches honestly.
    print("[demo] rendering wireframe")
    obj.hide_render = True
    clay = clay_material("clay_white", 0.9)
    obj.data.materials.clear()
    obj.data.materials.append(clay)
    obj.hide_render = False
    add_wireframe_copy(obj, wire_material())
    render_to(cam, os.path.join(out_dir, "wireframe.png"), 35, 0.5)

    blend_path = os.path.join(out_dir, "swiss_knife_demo.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[demo] saved {blend_path}")


if __name__ == "__main__":
    main()
