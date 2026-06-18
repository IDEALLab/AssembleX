import os
from argparse import ArgumentParser
from collections import defaultdict

import trimesh


def print_graph(scene):
    parents = scene.graph.transforms.parents
    children = defaultdict(list)
    for node, parent in parents.items():
        children[parent].append(node)

    geometry_nodes = set(scene.graph.nodes_geometry)

    def walk(node, depth=0):
        indent = "  " * depth
        _, geom_name = scene.graph[node]
        geom_tag = f"  [geom: {geom_name}]" if node in geometry_nodes else ""
        print(f"{indent}{node}{geom_tag}")
        for child in sorted(children.get(node, [])):
            walk(child, depth + 1)

    roots = set(parents.values()) - set(parents.keys())
    for root in sorted(roots):
        walk(root)


def repair_mesh(mesh):
    # GLB sub-meshes store duplicate vertices at material seams. Merging them
    # (process=True) closes the open boundaries and makes the mesh watertight.
    merged = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    if merged.is_watertight:
        return merged
    # Fallback: pymeshfix for genuinely broken geometry
    import pymeshfix

    meshfix = pymeshfix.MeshFix(merged.vertices, merged.faces)
    meshfix.repair()
    repaired = trimesh.Trimesh(
        vertices=meshfix.points, faces=meshfix.faces, process=False
    )
    print(f"  watertight after repair: {repaired.is_watertight}")
    return repaired


def separate_parts(filepath, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    scene = trimesh.load(filepath, process=False, force="scene")

    if not isinstance(scene, trimesh.Scene):
        scene.export(os.path.join(output_dir, "single_part.obj"))
        print("Exported: single_part (no scene graph found)")
        return

    print("\n=== Scene graph ===")
    print_graph(scene)
    print("===================\n")

    parents_dict = scene.graph.transforms.parents

    def is_root(node):
        # A node is root-level if it has no recorded parent
        return parents_dict.get(node) is None

    group_meshes = defaultdict(list)
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        if geometry_name not in scene.geometry:
            continue

        parent = parents_dict.get(node_name)
        grandparent = parents_dict.get(parent) if parent is not None else None
        if parent is None or is_root(parent) or is_root(grandparent):
            # Direct child of world, or child of the top-level assembly node: standalone part
            group_key = node_name
        else:
            # Grandparent exists beyond assembly level: primitive → merge under parent group
            group_key = parent

        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        group_meshes[group_key].append(mesh)

    for part_name, meshes in group_meshes.items():
        if not meshes:
            continue
        combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        clean_name = str(part_name).replace(" ", "_").replace("/", "-")
        if not combined.is_watertight:
            combined = repair_mesh(combined)
        combined.export(os.path.join(output_dir, f"{clean_name}.obj"))
        print(f"Exported: {clean_name} ({len(meshes)} primitive(s) merged)")


def process_directory(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".glb", ".obj")):
            filepath = os.path.join(input_dir, filename)
            file_out_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            print(f"Processing {filepath}...")
            separate_parts(filepath, file_out_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Separate multi-part meshes (GLB or OBJ) in a directory into individual OBJ files."
    )
    parser.add_argument(
        "--input-dir", required=True, help="Input directory containing GLB or OBJ files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for individual OBJ files"
    )
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
