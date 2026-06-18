"""Frozen, vendored copy of the data_gen branch's tool-evaluation code path.

This module pins the *exact* axis-picking, orientation, and chain-of-thought
tool-decision logic from the data_gen branch so the tool data-collection
experiments (test_convex_decomp, test_tool_needed, and their aggregators)
reproduce the data generated there. setup's evolved core/tool_analyzer.py
diverged in _vlm_pick, _pick_orientation (contact from the convex part vs the
whole mesh) and the _pick_axis/_pick_orientation return shapes, which would
change the collected data. ToolEvaluator subclasses ToolAnalyzer and overrides
those with the data_gen versions while inheriting the identical helpers
(_render_arrows, _make_messages, _rotation_between).

The tetrahedral 4-view part rendering used by the VLM tool decision is included
as _create_tetra_images and scoped to this feature only, so setup's default
part-image rendering is unchanged.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.patches import Patch
from openai import OpenAI

import settings
from core.models import (
    Tool,
    ToolDescription,
    ToolReasoningResponse,
    decompose_tool_convex,
    encode_image,
    get_PC_transform,
)
from core.tool_analyzer import ToolAnalyzer

_INV_SQRT3 = 1.0 / np.sqrt(3.0)
_TETRA_DIRECTIONS = {
    "iso1": np.array([1, 1, 1]) * _INV_SQRT3,
    "tetra2": np.array([1, -1, -1]) * _INV_SQRT3,
    "tetra3": np.array([-1, 1, -1]) * _INV_SQRT3,
    "tetra4": np.array([-1, -1, 1]) * _INV_SQRT3,
}

_SYSTEM_AXIS_SELECTOR = (
    "You are an expert mechanical engineer analyzing a CAD model of an assembly tool.\n"
    "You are shown three rendered views (isometric, top, side) of the tool with numbered "
    "axis arrows overlaid. Each arrow is the minor principal inertia axis of one convex "
    "segment of the tool and represents a candidate application direction.\n\n"
    "Select the axis that best matches the tool's PRIMARY application direction — "
    "the direction the tool travels or is inserted during use:\n"
    "  - Screwdriver / hex key: along the shaft (tip toward the fastener)\n"
    "  - Allen key: along the longer arm of the L-shape\n"
    "  - Wrench: perpendicular to the jaw-opening plane\n\n"
    "Respond with exactly ONE integer — the axis number. No explanation."
)

_SYSTEM_AXIS_SELECTOR_PART = (
    "You are an expert mechanical engineer analyzing a CAD model of an assembly part.\n"
    "You are shown three rendered views (isometric, top, side) of the part with numbered "
    "axis arrows overlaid. Each arrow is the minor principal inertia axis of one convex "
    "segment of the part and represents a candidate tool-approach direction.\n\n"
    "Select the axis that best matches the PRIMARY direction a tool would approach this part — "
    "the axis along which a tool inserts, grips, or drives this part:\n"
    "  - Screw / bolt: along the threading axis\n"
    "  - Nut / hex head: perpendicular to the hex faces (the drive axis)\n"
    "  - Bracket / plate with fastener hole: perpendicular to the hole opening\n\n"
    "Respond with exactly ONE integer — the axis number. No explanation."
)

_SYSTEM_ORIENTATION_PICKER = (
    "You are an expert mechanical engineer analyzing an assembly tool.\n"
    "You are shown three rendered views of the tool with two arrows: "
    "arrow 0 (red, at the positive end) and arrow 1 (blue, at the negative end). "
    "Each arrow is positioned at one extreme of the tool and points outward along its axis.\n\n"
    "Select the arrow that points in the direction the tool APPROACHES the workpiece — "
    "i.e., the tip/working end aimed at the fastener or part:\n"
    "  - Screwdriver: blade/tip end pointing toward the screw head\n"
    "  - Allen key: short insertion arm pointing into the hex socket\n"
    "  - Wrench: jaw end pointing toward the nut\n\n"
    "Respond with exactly 0 or 1. No explanation."
)

_SYSTEM_ORIENTATION_PICKER_PART = (
    "You are an expert mechanical engineer analyzing an assembly part.\n"
    "You are shown three rendered views of the part with two arrows: "
    "arrow 0 (red, at the positive end) and arrow 1 (blue, at the negative end). "
    "Each arrow is positioned at one extreme of the part and points outward along its axis.\n\n"
    "Select the arrow that points toward the feature a tool would engage with — "
    "i.e., the end or face where a tool makes contact:\n"
    "  - Screw / bolt: arrow pointing toward the head (where the driver inserts)\n"
    "  - Nut: arrow pointing toward the face a wrench grips\n"
    "  - Bracket / plate: arrow pointing toward the fastener hole opening\n\n"
    "Respond with exactly 0 or 1. No explanation."
)

_SYSTEM_TOOL_DESCRIBER = (
    "You are an expert mechanical engineer and assembly technician. "
    "You are shown a single rendered view of a physical assembly tool. "
    "Describe the tool concisely in three short fields:\n"
    "  - appearance: one sentence describing its visible geometry (shape, handle, working end).\n"
    "  - purpose:    one sentence describing what the tool mechanically does.\n"
    "  - typical_use: one sentence describing which part features it engages with "
    "(screw heads, hex sockets, nuts, threaded fasteners, etc.).\n\n"
    "Keep every field under 25 words. Use precise engineering terminology."
)

_SYSTEM_TOOL_ANALYST_COT = (
    "You are an expert mechanical engineer and assembly technician. "
    "Your task is to decide whether a specific tool is required to remove a single part "
    "from an otherwise rigid assembly, given four rendered views of the part and a catalog "
    "of available tools with textual descriptions.\n\n"
    "### Assembly Context\n"
    "- The overall assembly is rigid. During this step, ONLY the target part moves; "
    "all other parts remain stationary.\n"
    "- Disassembly is performed by a human hand.\n"
    "- A tool is 'required' only if moving the part by bare hand is unrealistic or "
    "impractical (e.g. overcoming threaded friction, unfastening tight joints, prying). "
    "If the part can be slid, lifted, or translated by hand, no tool is needed.\n\n"
    "Consider only the action required to remove this part and assume no other parts need to be removed to access it. "
    "### Reasoning Process (Chain of Thought)\n"
    "Work through these steps in order. Fill EACH field with concrete observations — do not "
    "skip any step.\n"
    "  1. part_observation: examine the four rendered views. Note distinctive features "
    "such as fastener heads, hex faces, slots, threads, knurling, or smooth surfaces. "
    "Refer to the part name to anchor your reading of the geometry.\n"
    "  2. motion_analysis: from the part's geometry and name, infer how it is removed "
    "(slide out, lift, unscrew, pry). Identify whether mechanical advantage is needed.\n"
    "  3. tool_evaluation: compare the motion requirement to each tool's appearance/purpose/"
    "typical_use. Argue for the best match, or argue that no tool is needed. Be explicit "
    "about why other tools are rejected.\n"
    "  4. final_tool: output the chosen tool name exactly as it appears in the allowed list, "
    "or 'none' if no tool is required.\n\n"
    "### Output\n"
    "Return a JSON object with the four fields above. final_tool MUST be one of the allowed responses."
)


class ToolEvaluator(ToolAnalyzer):
    """data_gen tool-evaluation methods, pinned for reproducibility."""

    def _vlm_pick(self, system_prompt, img_path, user_text):
        """Send one image to the VLM and return the first integer found in the reply."""
        user_content = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(img_path)}"},
            },
        ]
        client = OpenAI(api_key=self.assembly.openai_api_key)
        response = client.chat.completions.create(
            model=settings.LLM_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        if self.assembly.evaluation:
            self.assembly.evaluation.tokens_used += response.usage.total_tokens
        raw = response.choices[0].message.content.strip()
        print(f"  AI response: '{raw}'")
        match = re.search(r"\d+", raw)
        return int(match.group()) if match else None

    def _pick_axis(self, obj, obj_id, save_dir, minor_only, mode):
        """Decompose obj into convex parts, collect candidate axes, and pick one."""
        parts, _ = decompose_tool_convex(obj, save_dir / "parts", draw=False)

        cmap = plt.get_cmap("tab20")
        row_indices = [2] if minor_only else [0, 1, 2]
        axes = []  # list of (center, direction)
        for part in parts:
            transform = get_PC_transform(part.tri_mesh, draw=False)
            center = part.tri_mesh.center_mass
            for row in row_indices:
                direction = transform[row, :3]
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm
                    # Canonical hemisphere: flip so the first non-negligible
                    # component is positive, giving a stable half-space partition.
                    for component in direction:
                        if abs(component) > 1e-4:
                            if component < 0:
                                direction = -direction
                            break
                    axes.append((center, direction))

        def _axis_color(i):
            rgb = cmap(i / max(len(axes) - 1, 1))[:3]
            return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"

        arrows = [
            (origin, direction, _axis_color(i), str(i))
            for i, (origin, direction) in enumerate(axes)
        ]
        mode_label = "minor axis per part" if minor_only else "all axes per part"

        if mode == "user":
            img = self._render_arrows(obj.tri_mesh, arrows, "iso", bg="black")
            patches = [
                Patch(color=_axis_color(i), label=f"Axis {i}") for i in range(len(axes))
            ]

            fig, ax = plt.subplots(figsize=(13, 9))
            ax.imshow(img)
            ax.set_title(
                f"{obj_id}  —  {len(axes)} candidate axes ({mode_label})\n"
                f"Choose the application axis (0 – {len(axes) - 1})",
                fontsize=14,
            )
            ax.axis("off")
            ax.legend(handles=patches, loc="lower right", fontsize=11, framealpha=0.85)
            plt.tight_layout()
            plt.show()
            plt.close(fig)

            while True:
                raw = input(f"[{obj_id}] Axis (0 – {len(axes) - 1}): ").strip()
                if raw.isdigit() and 0 <= int(raw) < len(axes):
                    break
                print(f"  Invalid – enter a number between 0 and {len(axes) - 1}.")
            chosen = int(raw)

        else:  # ai — three views + numerical legend for reliable reasoning
            view_specs = [("iso", "Isometric"), ("xy", "Top (XY)"), ("yz", "Side (YZ)")]
            ai_patches = [
                Patch(
                    color=_axis_color(i),
                    label=f"Axis {i}  [{', '.join(f'{v:+.2f}' for v in d)}]",
                )
                for i, (_, d) in enumerate(axes)
            ]
            fig, ax_plots = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(
                f"{obj_id}  —  {len(axes)} candidate axes ({mode_label})", fontsize=13
            )
            for ax_plot, (cam, label) in zip(ax_plots, view_specs, strict=False):
                ax_plot.imshow(
                    self._render_arrows(obj.tri_mesh, arrows, cam, bg="white")
                )
                ax_plot.set_title(label, fontsize=11)
                ax_plot.axis("off")
            fig.legend(
                handles=ai_patches,
                loc="lower center",
                ncol=len(axes),
                fontsize=10,
                framealpha=0.9,
            )
            plt.tight_layout(rect=[0, 0.08, 1, 1])

            save_dir.mkdir(parents=True, exist_ok=True)
            img_path = save_dir / f"{obj_id}_axis.png"
            fig.savefig(img_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            axes_text = "\n".join(
                f"  Axis {i}: [{', '.join(f'{v:+.3f}' for v in d)}]"
                for i, (_, d) in enumerate(axes)
            )
            axis_prompt = (
                _SYSTEM_AXIS_SELECTOR
                if isinstance(obj, Tool)
                else _SYSTEM_AXIS_SELECTOR_PART
            )
            result = self._vlm_pick(
                axis_prompt,
                img_path,
                f"Object: {obj_id}\nCandidate axes (0–{len(axes) - 1}):\n{axes_text}\n\n"
                "Views: isometric, top (XY), side (YZ).\n"
                "Which axis is the primary application direction? Respond with one integer.",
            )
            chosen = result if result is not None and 0 <= result < len(axes) else 0
            if result is None:
                print("  Warning: could not parse axis — defaulting to 0.")

        print(f"  Axis {chosen} selected.")
        return axes[chosen][1]

    def _pick_orientation(self, obj, obj_id, axis, save_dir, mode):
        """Find the two extreme points along axis, show opposing arrows, and pick one.

        Returns the final signed direction vector and the chosen contact point.
        """
        g = obj.tri_mesh.center_mass
        axis_distances = np.dot(obj.tri_mesh.vertices, axis)
        v_pos = obj.tri_mesh.vertices[np.argmax(axis_distances)]
        v_neg = obj.tri_mesh.vertices[np.argmin(axis_distances)]
        pos_point = axis * np.dot(v_pos - g, axis) + g
        neg_point = axis * np.dot(v_neg - g, axis) + g

        # Arrow 0: red, at the + extreme, pointing outward in +axis
        # Arrow 1: blue, at the − extreme, pointing outward in −axis
        arrows = [
            (pos_point, axis, "#e63946", "0"),
            (neg_point, -axis, "#457b9d", "1"),
        ]
        patches = [
            Patch(color="#e63946", label="0  — positive end  (+direction)"),
            Patch(color="#457b9d", label="1  — negative end  (−direction)"),
        ]

        if mode == "user":
            img = self._render_arrows(obj.tri_mesh, arrows, "iso", bg="black")

            fig, ax = plt.subplots(figsize=(13, 9))
            ax.imshow(img)
            ax.set_title(
                f"{obj_id}  —  Select application orientation\n"
                "Pick the arrow pointing toward where the tool contacts the part (0 or 1)",
                fontsize=14,
            )
            ax.axis("off")
            ax.legend(handles=patches, loc="lower right", fontsize=11, framealpha=0.85)
            plt.tight_layout()
            plt.show()
            plt.close(fig)

            while True:
                raw = input(f"[{obj_id}] Orientation (0 or 1): ").strip()
                if raw in ("0", "1"):
                    break
                print("  Invalid – enter 0 or 1.")
            chosen = int(raw)

        else:  # ai
            view_specs = [("iso", "Isometric"), ("xy", "Top (XY)"), ("yz", "Side (YZ)")]
            fig, ax_plots = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(
                f"{obj_id}  —  Select application direction (0=red, 1=blue)",
                fontsize=13,
            )
            for ax_plot, (cam, label) in zip(ax_plots, view_specs, strict=False):
                ax_plot.imshow(
                    self._render_arrows(obj.tri_mesh, arrows, cam, bg="white")
                )
                ax_plot.set_title(label, fontsize=11)
                ax_plot.axis("off")
            fig.legend(
                handles=patches, loc="lower center", ncol=2, fontsize=10, framealpha=0.9
            )
            plt.tight_layout(rect=[0, 0.06, 1, 1])

            save_dir.mkdir(parents=True, exist_ok=True)
            img_path = save_dir / f"{obj_id}_orientation.png"
            fig.savefig(img_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            orientation_prompt = (
                _SYSTEM_ORIENTATION_PICKER
                if isinstance(obj, Tool)
                else _SYSTEM_ORIENTATION_PICKER_PART
            )
            result = self._vlm_pick(
                orientation_prompt,
                img_path,
                f"Object: {obj_id}\n"
                f"Arrow 0 (red):  direction [{', '.join(f'{v:+.3f}' for v in axis)}]\n"
                f"Arrow 1 (blue): direction [{', '.join(f'{v:+.3f}' for v in -axis)}]\n\n"
                "Views: isometric, top (XY), side (YZ).\n"
                "Which arrow points toward where the tool contacts the workpiece? Respond with 0 or 1.",
            )
            chosen = result if result in (0, 1) else 0
            if result not in (0, 1):
                print("  Warning: could not parse orientation — defaulting to 0.")

        final_direction = axis if chosen == 0 else -axis
        contact_point = pos_point if chosen == 0 else neg_point
        print(f"  Orientation {chosen} selected → {final_direction}")
        return final_direction, contact_point

    def select_axes(
        self, objects, output_dir, minor_only=True, mode="ai", allow_overwrite=False
    ):
        """For each object in `objects`: pick the best axis, then pick its orientation,
        and cache the result next to the .obj file as `{id}_axes.json`.

        Args:
            objects:         Dict of id → Object (or Tool). Pass scaled_tools for tools,
                             self.assembly.objects for assembly parts, or any mix.
            output_dir:      Directory for temporary render images.
            minor_only:      When True, only the minor principal axis per convex part is shown.
            mode:            "user" — interactive prompts.  "ai" — VLM auto-picks.
            allow_overwrite: When True, ask before re-using an existing cache entry.

        Returns dict: obj_id -> (direction, contact) tuple of np.ndarray, shape (3,).
        """
        output_dir = Path(output_dir)
        results = {}

        for obj_id, obj in objects.items():
            cache_path = obj.path.parent / f"{obj.id}_axes.json"

            if obj.direction is not None:
                print(
                    f"\n[{obj_id}] Cache found ({cache_path}), "
                    f"originally set by: {obj._axes_data.get('mode', '?')}\n"
                    f"  direction={obj.direction.tolist()}, contact={obj.contact_point.tolist()}"
                )
                use_cache = True
                if allow_overwrite:
                    use_cache = (
                        input(f"[{obj_id}] Overwrite cache? (y/N): ").strip().lower()
                        != "y"
                    )
                if use_cache:
                    results[obj_id] = (obj.direction, obj.contact_point)
                    continue

            obj_dir = output_dir / obj_id
            print(f"\n[{obj_id}] Step 1/2 — axis selection  (mode={mode})")
            axis = self._pick_axis(obj, obj_id, obj_dir, minor_only, mode)

            print(f"\n[{obj_id}] Step 2/2 — orientation selection  (mode={mode})")
            direction, contact = self._pick_orientation(
                obj, obj_id, axis, obj_dir, mode
            )

            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "obj_id": obj_id,
                        "mode": mode,
                        "minor_only": minor_only,
                        "direction": direction.tolist(),
                        "contact": contact.tolist(),
                    },
                    f,
                    indent=2,
                )
            print(f"\n[{obj_id}] Saved to cache: {cache_path}")

            # Invalidate cached_property so next access reloads fresh values
            for attr in ("_axes_data", "direction", "contact_point"):
                obj.__dict__.pop(attr, None)

            results[obj_id] = (direction, contact)
            print(f"\n[{obj_id}] Final direction: {direction}, contact: {contact}")

        return results

    def select_tool_axes(
        self, output_dir, minor_only=True, mode="ai", allow_overwrite=False
    ):
        return self.select_axes(
            self.assembly.scaled_tools, output_dir, minor_only, mode, allow_overwrite
        )

    def sample_axes(self, objects, output_dir, n_samples=1, minor_only=True, mode="ai"):
        """Run the axis + orientation picker `n_samples` times per object, bypassing the
        on-disk JSON cache and never writing to it. Useful for measuring AI variance.

        Returns {obj_id: [(direction, contact), ...]} with n_samples entries per object.
        """
        output_dir = Path(output_dir)
        results = {obj_id: [] for obj_id in objects}

        for sample_i in range(n_samples):
            print(f"\n=== Sample {sample_i + 1}/{n_samples} (mode={mode}) ===")
            for obj_id, obj in objects.items():
                obj_dir = output_dir / obj_id / f"sample_{sample_i}"
                print(f"\n[{obj_id}]  Step 1/2 — axis selection")
                axis = self._pick_axis(obj, obj_id, obj_dir, minor_only, mode)
                print(f"\n[{obj_id}]  Step 2/2 — orientation selection")
                direction, contact = self._pick_orientation(
                    obj, obj_id, axis, obj_dir, mode
                )
                results[obj_id].append((direction, contact))
        return results

    def sample_tool_axes(self, output_dir, n_samples=1, minor_only=True, mode="ai"):
        return self.sample_axes(
            self.assembly.scaled_tools, output_dir, n_samples, minor_only, mode
        )

    def _create_tetra_images(self, obj_idx=None):
        """Render the tetrahedral 4-view part images fed to the tool-decision VLM.

        Vendored from the data_gen Renderer.create_images tetrahedral branch and
        scoped to the tool-decision path. Populates obj.image_paths and caches
        PNGs under the assembly storage_dir; setup's default part-image
        rendering is left unchanged.
        """
        objs = (
            self.assembly.objects
            if obj_idx is None
            else {obj_idx: self.assembly.objects[obj_idx]}
        )
        for obj in objs.values():
            render_mesh = obj.mesh
            bounds = render_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            center = np.array(
                [
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2,
                ]
            )
            diag = float(
                np.linalg.norm(
                    [
                        bounds[1] - bounds[0],
                        bounds[3] - bounds[2],
                        bounds[5] - bounds[4],
                    ]
                )
            )
            distance = max(diag * 2, 1e-3)

            if obj.image_paths is None:
                obj.image_paths = {}

            for view_name, cam_dir in _TETRA_DIRECTIONS.items():
                expected_img_path = (
                    self.assembly.storage_dir
                    / f"{self.assembly.id}_{obj.id}_{view_name}.png"
                )
                if expected_img_path.exists():
                    obj.image_paths[view_name] = expected_img_path
                    continue

                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(render_mesh, color="lightgray", show_edges=False)
                direction = np.asarray(cam_dir, dtype=float)
                direction = direction / np.linalg.norm(direction)
                cam_xyz = center + direction * distance
                up = (0, 0, 1) if abs(direction[2]) < 0.95 else (0, 1, 0)
                plotter.camera_position = [tuple(cam_xyz), tuple(center), up]
                plotter.screenshot(expected_img_path)
                plotter.close()
                obj.image_paths[view_name] = expected_img_path

    def _render_tool_iso(self, tool):
        """Render an isometric view of a tool mesh, cached next to its .obj file."""
        img_path = tool.path.parent / f"{tool.id}_iso.png"
        if img_path.exists():
            return img_path

        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        plotter.background_color = "white"
        plotter.add_mesh(
            pv.wrap(tool.tri_mesh), color="lightsteelblue", show_edges=False
        )
        plotter.camera_position = "iso"
        plotter.screenshot(img_path)
        plotter.close()
        return img_path

    def _ensure_tool_descriptions(self):
        """For each available tool, generate (or load cached) name + description fields.

        Stores a per-tool cache as `{tool.id}_description.json` next to the .obj file.
        Returns a dict: tool_id -> {"name", "appearance", "purpose", "typical_use"}.
        """
        descriptions = {}
        client = OpenAI(api_key=self.assembly.openai_api_key)

        for tool_id, tool in self.assembly.evaluation.tools.items():
            cache_path = tool.path.parent / f"{tool.id}_description.json"

            if cache_path.exists():
                with open(cache_path) as f:
                    descriptions[tool_id] = json.load(f)
                continue

            print(f"  Generating description for tool '{tool.name}' ({tool_id})...")
            img_path = self._render_tool_iso(tool)

            user_content = [
                {
                    "type": "text",
                    "text": f"Tool catalog name: {tool.name}\nDescribe this tool.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(img_path)}"
                    },
                },
            ]

            try:
                response = client.beta.chat.completions.parse(
                    model=settings.LLM_model,
                    messages=self._make_messages(_SYSTEM_TOOL_DESCRIBER, user_content),
                    response_format=ToolDescription,
                )
                if self.assembly.evaluation:
                    self.assembly.evaluation.tokens_used += response.usage.total_tokens
                parsed = response.choices[0].message.parsed
                entry = {
                    "name": tool.name,
                    "appearance": parsed.appearance,
                    "purpose": parsed.purpose,
                    "typical_use": parsed.typical_use,
                }
            except Exception as e:
                print(f"  Warning: description generation failed for {tool_id}: {e}")
                entry = {
                    "name": tool.name,
                    "appearance": f"A {tool.name}.",
                    "purpose": f"Used as a {tool.name}.",
                    "typical_use": "General-purpose fasteners.",
                }

            with open(cache_path, "w") as f:
                json.dump(entry, f, indent=2)
            descriptions[tool_id] = entry

        return descriptions

    def _ensure_part_name(self, obj):
        """Make sure obj.name is set to a meaningful engineering name (not just the id)."""
        if obj.name and obj.name != str(obj.id):
            return obj.name

        # Try loading existing cache first
        names_path = self.assembly.storage_dir / "part_names.json"
        if names_path.exists():
            with open(names_path) as f:
                data = json.load(f)
            if obj.id in data:
                obj.name = data[obj.id]
                return obj.name

        # Fall back to generating all part names via the FeedbackGenerator
        print(f"  No name for part {obj.id} — generating names for assembly...")
        self.assembly.name_parts(iso_only=False, log_probs=False)
        return obj.name

    def check_tool_needed_cot(self, obj_idx):
        """Chain-of-thought tool-need detection.

        Uses the four static rendered views of the part (iso1, top, front, side) plus
        textual descriptions of every available tool. Asks the VLM to reason in four
        steps (observe → motion → evaluate → decide) and return a structured response.

        Returns (tool_name_or_None, reasoning_dict).
        """
        obj = self.assembly.objects[obj_idx]

        # Ensure the part has 4-angle renders
        if not obj.image_paths or not any(obj.image_paths.values()):
            print(f"  No images for part {obj_idx} — rendering...")
            self._create_tetra_images(obj_idx=obj_idx)

        # Ensure the part has an engineering name
        self._ensure_part_name(obj)

        # Ensure every tool has a description
        tool_descs = self._ensure_tool_descriptions()

        # Build the tool catalog block
        tool_lines = []
        for tool_id, d in tool_descs.items():
            tool_lines.append(
                f"- {tool_id} ('{d['name']}')\n"
                f"    appearance:  {d['appearance']}\n"
                f"    purpose:     {d['purpose']}\n"
                f"    typical_use: {d['typical_use']}"
            )
        catalog = "\n".join(tool_lines)
        allowed = ", ".join([*list(tool_descs.keys()), "none"])

        # Build the user message: text + 4 part images
        user_text = (
            f"Target part id: {obj.id}\n"
            f"Target part name: {obj.name}\n\n"
            f"Available tools:\n{catalog}\n\n"
            f"Allowed final_tool values: {allowed}\n\n"
            "Below are four rendered views of the part (isometric, top, front, side). "
            "Work through the four reasoning fields described in the system prompt, "
            "then output the chosen tool."
        )
        user_content = [{"type": "text", "text": user_text}]

        for view_name, img in (obj.image_paths or {}).items():
            if img is None or not Path(img).exists():
                continue
            user_content.append({"type": "text", "text": f"View: {view_name}"})
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"},
                }
            )

        client = OpenAI(api_key=self.assembly.openai_api_key)
        try:
            response = client.beta.chat.completions.parse(
                model=settings.LLM_model,
                messages=self._make_messages(_SYSTEM_TOOL_ANALYST_COT, user_content),
                response_format=ToolReasoningResponse,
            )
            if self.assembly.evaluation:
                self.assembly.evaluation.tokens_used += response.usage.total_tokens
        except Exception as e:
            print(f"  Error during OpenAI call in check_tool_needed_cot: {e}")
            return None, None

        parsed = response.choices[0].message.parsed
        reasoning = parsed.model_dump()
        choice = parsed.final_tool.strip().lower().strip("'\"")

        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(f"  Reasoning for {obj.name}:")
            for k in ("part_observation", "motion_analysis", "tool_evaluation"):
                print(f"    {k}: {reasoning[k]}")
            print(f"    final_tool: {reasoning['final_tool']}")

        if "none" in choice:
            return "none", reasoning
        if choice not in self.assembly.scaled_tools:
            print(f"  Warning: '{choice}' not in tool catalog. Returning 'error'.")
            return "error", reasoning
        return choice, reasoning
