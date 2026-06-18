import contextlib
import itertools
import os
import threading
import time
from functools import cached_property

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import redmax_py as redmax
import trimesh

# from ATA.examples.test_multi_sim import get_xml_string
from ATA.examples.run_joint_plan import PhysicsPlanner, get_xml_string
from ATA.utils.renderer import SimRenderer
from tqdm import tqdm

# This file lives at <repo_root>/core/simulation.py, so the repo root
# (which holds the assets/, ATA/ and ASAPx/ trees) is one level up.
project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)


class ContactTree:
    def __init__(self, meshes):
        self.meshes = meshes
        self.G = nx.Graph()
        self.G.add_nodes_from(meshes.keys())
        self.update()

    def update(self, meshes=None):  # First approach: trimesh collision manager
        manager = trimesh.collision.CollisionManager()
        if meshes is not None:
            self.meshes = meshes
        for id, mesh in self.meshes.items():
            manager.add_object(id, mesh)

        _is_overlapping, names_overlapping, _collision_data = (
            manager.in_collision_internal(return_names=True, return_data=True)
        )
        for obj1, obj2 in names_overlapping:
            self.G.add_edge(obj1, obj2)

    def draw(self):
        print("Contact graph edges:", self.G.edges())
        nx.draw(self.G, with_labels=True)
        plt.show()


class Simulation:
    def __init__(self, assembly):
        self.assembly = assembly
        self.asset_folder = os.path.join(project_base_dir, "assets")
        self.assembly_dir = os.path.join(project_base_dir, self.assembly.assembly_dir)

    @cached_property
    def tree(self):
        objects = {}
        for obj in self.assembly.objects.values():
            objects[obj.id] = obj.tri_mesh
        return ContactTree(objects)

    # See how assembly behaves under gravity
    def test_gravity(self, show=False):

        print(f"Testing gravity for assembly in {self.assembly_dir} ...")
        # USING RUN_JOINT_PLAN VERSION OF XML STRING
        xml_string = get_xml_string(
            self.assembly_dir,
            [obj.id for obj in self.assembly.objects.values()],
            [],
            "translational",
            "sdf",
            0.05,
            0.01,
            False,
            color_scheme="default",
            gravity=-9.81,
            friction=0.5,
            ground=True,
        )
        """
        xml_string = get_xml_string(assembly_dir=self.assembly_dir,
                                    part_ids=[obj.id for obj in self.assembly.objects.values()],
                                    fixed=False,
                                    body_type='sdf',
                                    sdf_dx=0.05,
                                    sdf_res=20,
                                    gravity=9.81,
                                    friction=0.5,
                                    )
        """
        sim = redmax.Simulation(xml_string, asset_folder=self.asset_folder)
        sim.reset()
        # sim_renderer = SimRenderer(sim, self.assembly)
        # sim_renderer.render()

        for _i in tqdm(range(2000)):
            sim.forward(1, verbose=True)

        SimRenderer.replay(sim, record=False)

    def test_stable(self, show=False):
        print("\nTesting stability under gravity")

        self.tree.draw()

        threshold = 0.4

        obj_list = list(self.assembly.objects)

        ground = lower_bound(self.assembly.objects) - 0.01
        vector_expected = np.array([0, 0, -0.01])
        planner = PhysicsPlanner(
            self.asset_folder,
            self.assembly_dir,
            obj_list,
            [],
            rotation=True,
            gravity=-9.81,
            friction=0.5,
            ground=ground,
        )
        original_states = planner.get_state()
        original_positions = parse_state(original_states.q)

        body_names = [f"part{obj_id}" for obj_id in obj_list]

        def _forward_step(sim):
            sim.forward(1, verbose=False)

        planner.sim.reset()
        for i in tqdm(range(500)):
            # planner.sim.clear_contact_bodies()
            t0 = time.time()
            t = threading.Thread(target=_forward_step, args=(planner.sim,), daemon=True)
            t.start()
            while t.is_alive():
                t.join(timeout=0.1)  # re-enter Python every 100ms so Ctrl+C fires
            elapsed = time.time() - t0

            converged = planner.sim.is_converged()

            # Minimum signed distance between consecutive body pairs (negative = penetrating)
            pair_distances = {}
            for a, b in itertools.pairwise(body_names):
                with contextlib.suppress(Exception):
                    pair_distances[f"{a}-{b}"] = planner.sim.get_body_distance(a, b)

            # Bodies in contact this step (history cleared above so this is per-step only)
            in_contact = {
                n: planner.sim.get_contact_bodies(n)
                for n in body_names
                if planner.sim.get_contact_bodies(n)
            }

            print(
                f"step={i:3d}  t={elapsed:.4f}s  converged={converged}"
                + (f"  dist={pair_distances}" if pair_distances else "")
                + (f"  contacts={in_contact}" if in_contact else "")
            )

        # Cumulative time breakdown (resets on sim.reset())
        planner.sim.print_time_report()

        final_states = planner.get_state()
        final_positions = parse_state(final_states.q)

        translation_vectors = [
            final - original
            for final, original in zip(
                final_positions, original_positions, strict=False
            )
        ]
        translated_meshes = {}
        for obj_id, translation in zip(obj_list, translation_vectors, strict=False):
            original_mesh = self.assembly.objects[obj_id].tri_mesh
            translated_mesh = original_mesh.copy()
            translated_mesh.apply_translation(translation)
            translated_meshes[obj_id] = translated_mesh

        self.tree.update(translated_meshes)
        self.tree.draw()

        norms = [
            np.linalg.norm(final - original - vector_expected)
            for final, original in zip(
                final_positions, original_positions, strict=False
            )
        ]
        print(f"Original states: {original_positions}")
        print(f"Final states: {final_positions}")
        print(f"Differences: {norms}")

        unstable = [i for i, norm in enumerate(norms) if norm >= threshold]
        if all(norm < threshold for norm in norms):
            print("Assembly is stable under gravity.")
        else:
            print("Assembly is NOT stable under gravity.")
            for i in unstable:
                print(f"Part {obj_list[i]} is unstable with difference {norms[i]:.4f}")

        if show:
            planner.sim.replay()

        return unstable


def parse_state(states):
    positions = []
    for i in range(0, len(states), 6):
        pos = np.array(
            states[i : i + 6][:3]
        )  # For now we only consider translational movement
        positions.append(pos)
    return positions


def lower_bound(objects, axis="z"):
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    min_val = float("inf")
    for obj in objects.values():
        pos = min(obj.tri_mesh.vertices[:, axis_idx])
        min_val = min(min_val, pos)
    return min_val
