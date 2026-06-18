import os
import sys
from pathlib import Path

import pyvista as pv


class Renderer:
    def __init__(self, assembly):
        self.assembly = assembly

    def _render_transition(
        self, parts_rest, part_move, parts_removed, pose, output_path, n_frames=15
    ):
        """Render a brief static hold of the assembly at the current state using the
        same physics renderer (redmax / MultiPartPathPlanner) as the step GIFs.

        No path planning is run: the moving part is held at its assembled pose for
        n_frames frames by constructing the path manually.  This gives a visually
        consistent transition between reversed step GIFs.

        Returns the output Path on success, None if the render cannot be produced
        (missing ASAPx, no pose, or simulation error).
        """
        import numpy as np

        if pose is None:
            return None

        # renderer.py lives at <repo_root>/core/, so go up one level.
        repo_root = Path(__file__).resolve().parents[1]
        asap_dir = str(repo_root / "ASAPx")
        if asap_dir not in sys.path:
            sys.path.insert(0, asap_dir)

        try:
            from plan_sequence.physics_planner import MultiPartPathPlanner
        except ImportError:
            return None

        asset_folder = str(repo_root / "assets")
        assembly_dir = str(self.assembly.assembly_dir)

        try:
            planner = MultiPartPathPlanner(
                asset_folder,
                assembly_dir,
                parts_rest,
                part_move,
                parts_removed=parts_removed,
                pose=pose,
                save_sdf=False,
            )
            # render() iterates each path element as a `qm` global-coordinate
            # column vector (it does `sim.get_joint_q_from_qm(name, qm)` per
            # frame). A 4x4 SE3 matrix is the wrong shape — pull the qm
            # representation of the already-assembled pose from the sim and
            # replicate it n_frames times to hold the part stationary.
            qm0 = planner.sim.get_joint_qm(f"part{part_move}")
            path = [np.array(qm0, dtype=float, copy=True) for _ in range(n_frames)]
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            planner.render(
                path=path, reverse=False, record_path=str(output_path), make_video=False
            )
            return output_path if output_path.exists() else None
        except Exception as exc:
            print(f"[Renderer] transition render for part {part_move!r} failed: {exc}")
            return None

    def _poses_from_tree(self):
        """Load a {part_id: pose} mapping from tree.pkl.

        Used as a fallback when step.pose is None — which happens for ATA-planned
        assemblies and for ASAP cache files that predate the 'steps' JSON field.
        Returns an empty dict if the tree file does not exist or cannot be parsed.
        """
        import pickle

        tree_path = self.assembly.storage_dir / "log" / "tree.pkl"
        if not tree_path.exists():
            return {}
        try:
            with open(tree_path, "rb") as f:
                tree = pickle.load(f)
            poses = {}
            for _, _, data in tree.edges(data=True):
                si = data.get("sim_info") or {}
                part = si.get("part_move")
                pose = si.get("pose")
                if part is not None and pose is not None and si.get("feasible"):
                    poses[part] = pose
            return poses
        except Exception:
            return {}

    def stitch_reversed(self, angle="iso1", output_path=None, n_transition_frames=15):
        """Stitch per-step GIFs into a single reversed assembly GIF.

        Steps are processed in reverse sequence order and each GIF's frames are
        also reversed, so the output shows the full disassembly played backwards
        (i.e. the assembly motion).  Between consecutive reversed GIFs a brief
        transition is rendered via `_render_transition` (same physics renderer as
        the step GIFs), showing the current assembly state as a static hold.  If
        the transition render fails the GIFs are concatenated directly.

        `angle` selects which entry in step.gifs to use; every step uses the same
        angle.  The per-step `step.gifs` dict is not modified.

        Works correctly whether the sequence was produced by a fresh planning run
        or loaded from a cached sequence.json — poses missing from the Step objects
        (ATA planner, old ASAP cache without 'steps' field) are sourced from
        tree.pkl when available.

        Returns the Path of the saved output GIF.
        """
        from PIL import Image

        steps = self.assembly.sequence
        if not steps:
            raise ValueError("Assembly sequence is empty.")

        sequence = [s.obj_id for s in steps]
        all_part_ids = set(self.assembly.objects.keys())
        n = len(steps)

        # Fallback pose source for planners / cache paths that don't populate step.pose.
        _tree_poses = None

        transition_dir = self.assembly.storage_dir / "transitions"

        all_frames = []
        durations = []

        for j, step in enumerate(reversed(steps)):
            gif_path = step.gifs.get(angle)
            if not gif_path or not Path(gif_path).exists():
                raise FileNotFoundError(
                    f"No GIF found for step {step.obj_id!r} at angle {angle!r} "
                    f"(got: {gif_path})"
                )

            with Image.open(gif_path) as gif:
                frames = []
                frame_durations = []
                for i in range(gif.n_frames):
                    gif.seek(i)
                    frames.append(gif.copy().convert("RGBA"))
                    frame_durations.append(gif.info.get("duration", 100))

            for frame, dur in zip(
                reversed(frames), reversed(frame_durations), strict=False
            ):
                all_frames.append(frame)
                durations.append(dur)

            # Add a transition between this reversed GIF and the next one.
            if j < n - 1 and n_transition_frames > 0:
                # Forward-sequence index of the step we just placed.
                k = n - 1 - j
                part_move = sequence[k]
                parts_rest = sorted(all_part_ids - set(sequence[: k + 1]))
                parts_removed = list(sequence[:k])

                pose = steps[k].pose
                if pose is None:
                    # step.pose is not populated (ATA planner or pre-'steps' cache).
                    # Load lazily from tree.pkl the first time we need it.
                    if _tree_poses is None:
                        _tree_poses = self._poses_from_tree()
                    raw = _tree_poses.get(part_move)
                    pose = raw.tolist() if hasattr(raw, "tolist") else raw

                trans_gif_path = transition_dir / f"trans_{k}.gif"
                trans_gif = self._render_transition(
                    parts_rest,
                    part_move,
                    parts_removed,
                    pose,
                    trans_gif_path,
                    n_frames=n_transition_frames,
                )
                if trans_gif is not None and trans_gif.exists():
                    with Image.open(trans_gif) as tgif:
                        for i in range(tgif.n_frames):
                            tgif.seek(i)
                            all_frames.append(tgif.copy().convert("RGBA"))
                            durations.append(tgif.info.get("duration", 100))

        if output_path is None:
            output_path = self.assembly.storage_dir / f"assembly_reversed_{angle}.gif"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_frames[0].save(
            output_path,
            save_all=True,
            append_images=all_frames[1:],
            loop=0,
            duration=durations,
            disposal=2,
        )
        return output_path

    def _create_images(self, camera_angles=None):
        objs = self.assembly.objects
        for obj in objs.values():
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Creating images for object {obj.name}...")

            if camera_angles is None:
                camera_angles = {
                    "iso1": "iso",
                    "top": "xy",
                    "front": "xz",
                    "side": "yz",
                }

            images_dir = self.assembly.storage_dir / "images"
            images_dir.mkdir(exist_ok=True)

            for view_name, cam_pos in camera_angles.items():
                expected_img_path = (
                    images_dir / f"{self.assembly.id}_{obj.id}_{view_name}.png"
                )
                if expected_img_path.exists():
                    if self.assembly.evaluation and self.assembly.evaluation.verbose:
                        print(f"Loading image from {expected_img_path}...")
                    obj.image_paths[view_name] = expected_img_path
                    continue

                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(obj.mesh, color="lightgray", show_edges=False)
                plotter.camera_position = cam_pos
                plotter.screenshot(expected_img_path)
                plotter.close()

                obj.image_paths[view_name] = expected_img_path

    def view_collisions(self, names_overlapping, show=False, save=False):
        output_path = self.assembly.output_dir / "collision_visualizations"
        output_path.mkdir(parents=True, exist_ok=True)

        # 2x2 multi-view layout: iso (current), top (xy), side (yz), front (xz).
        views = [
            (0, 0, "iso", "Iso"),
            (0, 1, "xy", "Top"),
            (1, 0, "yz", "Side"),
            (1, 1, "xz", "Front"),
        ]

        for obj1, obj2 in names_overlapping:
            o1 = self.assembly.objects[obj1]
            o2 = self.assembly.objects[obj2]
            m1, m2 = o1.mesh, o2.mesh
            n1 = getattr(o1, "name", None) or str(obj1)
            n2 = getattr(o2, "name", None) or str(obj2)
            try:
                overlap = m1.triangulate().boolean_intersection(m2.triangulate())
                if overlap is None or overlap.n_points == 0:
                    overlap = None
            except Exception as e:
                print(
                    f"  failed to compute collision overlap mesh for {obj1}/{obj2}: {e}"
                )
                overlap = None

            legend_entries = [(n1, "red"), (n2, "blue")]
            if overlap is not None:
                legend_entries.append(("overlap", "purple"))

            plotter = pv.Plotter(
                off_screen=save, shape=(2, 2), window_size=(1200, 1200)
            )
            for row, col, cam, label in views:
                plotter.subplot(row, col)
                plotter.add_text(label, font_size=10)
                plotter.add_mesh(m1, color="red", opacity=0.5)
                plotter.add_mesh(m2, color="blue", opacity=0.5)
                if overlap is not None:
                    plotter.add_mesh(overlap, color="purple", opacity=0.3)
                plotter.add_legend(
                    legend_entries, bcolor="white", size=(0.3, 0.15), loc="lower right"
                )
                plotter.camera_position = cam

            if show:
                plotter.show()

            if save:
                screenshot_path = output_path / f"{obj1}_{obj2}.png"
                plotter.screenshot(screenshot_path)
                self.assembly.collisions[(obj1, obj2)] = screenshot_path

            plotter.close()
