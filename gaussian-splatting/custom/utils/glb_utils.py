import numpy as np

import trimesh

from pygltflib import GLTF2


def read_points(path):
    scene = trimesh.load(path)
    geom = scene.geometry["geometry_0"]
    return np.array(geom.vertices)

def read_colors(path):
    gltf = GLTF2().load(path)

    primitive = gltf.meshes[0].primitives[0]

    color_accessor_idx = primitive.attributes.COLOR_0
    color_accessor = gltf.accessors[color_accessor_idx]

    buffer_view_idx = color_accessor.bufferView
    buffer_view = gltf.bufferViews[buffer_view_idx]

    with open(path, "rb") as f:
        f.seek(buffer_view.byteOffset)
        raw = f.read(buffer_view.byteLength)

    rgba = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
    rgb = rgba[:, :3]

    return rgb

def read_from_monst3r_glb(path):
    return read_points(path), read_colors(path)

    


