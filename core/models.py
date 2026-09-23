import base64
import json
import os
from functools import cached_property
from pathlib import Path

import coacd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydantic
import pyvista as pv
import trimesh
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# Pydantic models for structured OpenAI API responses
class PartAnalysis(pydantic.BaseModel):
    obj_names: list[str]


class ToolInfo(pydantic.BaseModel):
    name: str = pydantic.Field(
        description="A 1-3 word standard engineering name for the tool "
        "(e.g. 'Phillips screwdriver', 'hex key', 'open-end wrench').",
    )
    description: str = pydantic.Field(
        description="One short sentence describing what this tool is and what it is generally used for. "
        "Do not speculate about which specific part of the assembly it is used on.",
    )


class ToolAnalysis(pydantic.BaseModel):
    tool_infos: list[ToolInfo]


class SingleWordResponse(pydantic.BaseModel):
    word: str = pydantic.Field(
        description="Exactly one single word. No spaces, no punctuation, no conversational text.",
        pattern=r"^[a-zA-Z]+$",
    )


class ShortStringResponse(pydantic.BaseModel):
    result: list[str] = pydantic.Field(
        description="A classification output consisting of distinct words. Do not use compound words.",
        min_length=1,
        max_length=3,
    )


class ToolDescription(pydantic.BaseModel):
    appearance: str = pydantic.Field(
        description="One sentence describing what the tool looks like geometrically."
    )
    purpose: str = pydantic.Field(
        description="One sentence describing what the tool is used for mechanically."
    )
    typical_use: str = pydantic.Field(
        description="One sentence describing the part features it engages with (e.g. screw heads, hex sockets, nuts)."
    )


class ToolReasoningResponse(pydantic.BaseModel):
    part_observation: str = pydantic.Field(
        description="What the part looks like and notable features (heads, threads, slots, hex faces, etc.)."
    )
    motion_analysis: str = pydantic.Field(
        description="How the part is removed — slide, lift, rotate, unscrew, pry — based on its geometry and name."
    )
    tool_evaluation: str = pydantic.Field(
        description="Which tool(s) (from the provided list) plausibly match, and why one is preferred or none is needed."
    )
    final_tool: str = pydantic.Field(
        description="The chosen tool name, exactly one of the allowed responses (a tool name or 'none')."
    )


class PartNameResponse(pydantic.BaseModel):
    """Chain-of-thought structured output for per-part naming.

    The reasoning fields force the model to think about the assembly's purpose
    and the part's role before committing to a name; the final `name` is what
    we actually use downstream.
    """

    assembly_purpose: str = pydantic.Field(
        description="One short sentence: what is the OVERALL ASSEMBLY likely to be — "
        "the finished product or mechanism — based on the parts you can see in the "
        "context render? Be concrete (e.g. 'a folding workshop stool', "
        "'a small geared hand-crank winch'), not generic ('an assembly')."
    )
    part_role: str = pydantic.Field(
        description="One short sentence: what FUNCTIONAL ROLE does this part play in "
        "that assembly? Why is it there mechanically — does it fasten, support, "
        "transmit, guide, seal, pivot, cover, brace, etc.?"
    )
    location: str = pydantic.Field(
        description="One short sentence: WHERE does this part sit in the assembly "
        "(top/bottom/inside/edge/between which other parts)? Use the assembly "
        "context render to ground this."
    )
    distinguishing_features: str = pydantic.Field(
        description="One short sentence naming the geometric features that uniquely "
        "identify this part by name (holes, threads, slots, flanges, taper, profile). "
        "These are the features the chosen name should evoke."
    )
    name: str = pydantic.Field(
        description="The final standard engineering name — 1 to 3 words, no compound "
        "words, no punctuation, no quotes. Examples: 'flange bearing', 'countersunk "
        "screw', 'L bracket', 'spur gear', 'woodruff key'. This name must let a "
        "technician pick the right part out of a pile just from the words.",
        pattern=r"^[A-Za-z][A-Za-z\- ]{1,40}$",
    )


# --- Selective DFA feedback schema (used by Feedback.generate_feedback_selective) ---
class FeedbackPoint(pydantic.BaseModel):
    category: str = pydantic.Field(
        description=(
            "One of: 'observation' (something specific you see in the images), "
            "'dfa_issue' (a concrete Design-for-Assembly pitfall present in this step), "
            "'recommendation' (a specific geometric change that would improve assemblability)."
        )
    )
    text: str = pydantic.Field(
        description=(
            "One concrete, actionable sentence. Reference visible part features or state numbers. "
            "Avoid generic DFA advice or restating definitions."
        )
    )


class SelectiveStepFeedback(pydantic.BaseModel):
    needs_feedback: bool = pydantic.Field(
        description=(
            "False if the step is straightforward and no genuinely actionable feedback "
            "would help. When False, leave 'points' empty."
        )
    )
    points: list[FeedbackPoint] = pydantic.Field(
        default_factory=list,
        description=(
            "Only include points that are non-trivial and specific to this step. "
            "Skip categories that don't apply rather than padding with weak content."
        ),
    )


# --- Planning-failure evidence (populated by ASAPx seq_plan when no full sequence is found) ---
from dataclasses import dataclass, field


@dataclass
class FailureEntry:
    child_part: str
    fail_reason: str  # 'stability' | 'tool' | 'assembly' | 'grasp'
    evidence_image: str | None = None  # path relative to log_dir
    # Optional, fail_reason-specific fields:
    unstable_parts: list[str] = field(default_factory=list)  # stability
    directions: list[dict] = field(
        default_factory=list
    )  # assembly: per-axis (success, path_len)
    subkind: str | None = None  # tool: 'collision' | 'access'
    colliding_parts: list = field(default_factory=list)  # tool
    tried_tools: list[str] = field(default_factory=list)  # tool
    all_evidence_images: list[str] = field(default_factory=list)


@dataclass
class PlanningFailures:
    log_dir: Path
    depth: int
    deepest_nodes: list[list[str]]
    failures: list[FailureEntry] = field(default_factory=list)

    @classmethod
    def from_json(cls, log_dir):
        log_dir = Path(log_dir)
        fp = log_dir / "failures.json"
        if not fp.exists():
            return None
        with open(fp) as f:
            payload = json.load(f)
        entries = []
        for e in payload.get("failures", []):
            entries.append(
                FailureEntry(
                    child_part=e.get("child_part"),
                    fail_reason=e.get("fail_reason"),
                    evidence_image=e.get("evidence_image"),
                    unstable_parts=e.get("unstable_parts", []) or [],
                    directions=e.get("directions", []) or [],
                    subkind=e.get("subkind"),
                    colliding_parts=e.get("colliding_parts", []) or [],
                    tried_tools=e.get("tried_tools", []) or [],
                    all_evidence_images=e.get("all_evidence_images", []) or [],
                )
            )
        return cls(
            log_dir=log_dir,
            depth=int(payload.get("depth", 0)),
            deepest_nodes=payload.get("deepest_nodes", []),
            failures=entries,
        )

    def for_part(self, part_id):
        for e in self.failures:
            if e.child_part == part_id:
                return e
        return None

    def resolve(self, rel_path):
        if rel_path is None:
            return None
        return self.log_dir / rel_path


# --- Geometric-manual pipeline schema (used by Feedback.generate_manual_iterative) ---
class AnnotationDecision(pydantic.BaseModel):
    label_text: str = pydantic.Field(
        description="Short label naming the moving part as it should appear in the manual."
    )
    callout_position: list[float] = pydantic.Field(
        description="[x, y] pixel for the label text. Avoid the part centroid by at least ~50 px.",
        min_length=2,
        max_length=2,
    )
    show_tool_icon: bool = pydantic.Field(
        description="Whether to draw a tool indicator on the page (only if a tool is required)."
    )
    tool_icon_position: list[float] = pydantic.Field(
        default_factory=lambda: [0.0, 0.0],
        description="[x, y] pixel for the tool icon if show_tool_icon is true; otherwise ignored.",
        min_length=2,
        max_length=2,
    )
    add_zoom_inset: bool = pydantic.Field(
        description="Whether the step needs a zoomed inset (true for small/fine-detail parts)."
    )
    step_note: str | None = pydantic.Field(
        default=None,
        description="One short instruction line. Null if no extra note is needed.",
    )


class Step:
    def __init__(self, obj_id, path_matrices=None, gif_path=None):
        self.obj_id = obj_id
        self.matrices = path_matrices
        self.images = {"iso1": [], "iso2": [], "iso3": [], "iso4": []}
        self.gifs = {"iso1": None, "iso2": None, "iso3": None, "iso4": None}
        self.flags = {
            "collision": False,
            "tool_collision": False,
            "unreachable": False,
            "success": None,
            "tool_orientation": False,
        }
        self.tool = None
        self.pose = None
        self.rotated = False
        self.parts_fix = []

        if gif_path:
            self.gifs["iso1"] = gif_path

    def rendered_angles(self):
        """Angle keys that currently have extracted frames -- the set
        rank_angles() scores, and the set a cached ranking has to cover for
        apply_angle_ranking() to consider it still valid."""
        return {angle for angle, img_paths in self.images.items() if img_paths}

    def _reorder_by_ranking(self, ranking):
        """Reorder images/gifs to the (score, angle) order in `ranking`.

        Angles absent from the ranking are dropped (they have no frames)."""
        temp_imgs = {}
        temp_gifs = {}
        for _score, angle in ranking:
            temp_imgs[angle] = self.images[angle]
            temp_gifs[angle] = self.gifs.get(angle)
        self.images = temp_imgs
        self.gifs = temp_gifs

    def apply_angle_ranking(self, ranking, verbose=True):
        """Reorder images/gifs from a ranking produced by an earlier
        rank_angles() call (e.g. one cached in sequence.json) instead of
        re-reading every frame and re-scoring it.

        Returns the normalised ranking when it was applied, or None when it is
        stale -- i.e. it does not cover exactly the angles rendered now -- in
        which case the step is left untouched and the caller should re-rank."""
        try:
            ranking = sorted((float(score), angle) for score, angle in ranking)
        except (TypeError, ValueError):
            return None
        if {angle for _score, angle in ranking} != self.rendered_angles():
            return None
        self._reorder_by_ranking(ranking)
        if verbose:
            print(f"Angle ranking for step {self.obj_id} (cached): {ranking}")
        return ranking

    def rank_angles(self, show=False, verbose=True):
        ranking = []
        diff_images = {}
        for angle, img_paths in self.images.items():
            if img_paths:
                first_frame = cv2.imread(img_paths[0])
                last_frame = cv2.imread(img_paths[-1])
                score, diff_image = ssim(
                    first_frame, last_frame, full=True, channel_axis=-1
                )
                ranking.append((float(score), angle))
                diff_images[angle] = diff_image

        if show:
            num_angles = len(diff_images)
            plt.figure(figsize=(5 * num_angles + 5, 5))
            plt.suptitle(f"Angle Ranking for Object {self.obj_id}")

            for idx, (angle, diff_image) in enumerate(diff_images.items()):
                plt.subplot(1, num_angles + 1, idx + 1)
                plt.imshow(diff_image, cmap="gray")
                plt.title(f"Diff - {angle}")
                plt.axis("off")

            plt.subplot(1, num_angles + 1, num_angles + 1)
            angles = [r[1] for r in ranking]
            scores = [1 - r[0] for r in ranking]
            plt.bar(angles, scores, color="blue")
            plt.title(
                "inverted SSIM Scores (Relative), \nhigher scores implies higher difference"
            )
            plt.ylabel("Percentage above minimum (%)")

            plt.tight_layout()
            plt.show()

        ranking.sort()
        self._reorder_by_ranking(ranking)

        if verbose:
            print(f"Angle ranking for step {self.obj_id}: {ranking}")

        return ranking


class Object:
    def __init__(self, id, name=None, image_paths=None, path=None):
        self.id = id
        self.name = name or str(id)
        self.image_paths = image_paths
        self.path = path
        self.step_nr = None

    @cached_property
    def mesh(self):
        return pv.read(self.path)

    @cached_property
    def tri_mesh(self):
        return trimesh.load_mesh(self.path)

    @cached_property
    def _axes_data(self):
        if not self.path:
            return {}
        cache_path = self.path.parent / f"{self.id}_axes.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return {}

    @cached_property
    def direction(self):
        data = self._axes_data
        return np.array(data["direction"]) if "direction" in data else None

    @cached_property
    def contact_point(self):
        data = self._axes_data
        return np.array(data["contact"]) if "contact" in data else None


def get_PC_transform(input_mesh, draw=False):
    mesh = input_mesh.copy()
    mesh = mesh.apply_translation(-mesh.center_mass)
    axes_transform = mesh.principal_inertia_transform[:3, :3]
    axes_transform = np.vstack([axes_transform, [0, 0, 0]])
    axes_transform = np.column_stack([axes_transform, [0, 0, 0, 1]])

    if draw:
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color="lightgray", show_edges=True)

        origin = mesh.center_mass
        bbox = mesh.bounding_box
        axis_length = np.linalg.norm(bbox.extents) / 4.0
        colors = ["red", "green", "blue"]
        labels = ["X", "Y", "Z"]

        for i, (color, label) in enumerate(zip(colors, labels, strict=False)):
            end_point = origin + axes_transform[:3, i] * axis_length
            line = pv.Line(origin, end_point)
            plotter.add_mesh(line, color=color, line_width=3, label=label)

        plotter.add_legend()
        plotter.show()
    return axes_transform


class Tool(Object):
    """An Object representing a physical assembly tool.

    direction and contact_point are lazy-loaded from the per-object JSON cache
    written by ToolAnalyzer.select_axes(), stored next to the .obj file in the
    tool's own normalized coordinate space (universal across assemblies).
    Per-assembly Tool instances carry `scaling_factor`, the relative scale used
    to express the tool in the assembly's normalized space; Assembly.init_tools
    applies it to both the mesh and the cached contact_point.
    """

    def __init__(
        self,
        id,
        name=None,
        image_paths=None,
        path=None,
        scaling_factor=1.0,
        description=None,
    ):
        super().__init__(id=id, name=name, image_paths=image_paths, path=path)
        self.scaling_factor = scaling_factor
        self.description = description


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_gif_frames(gif_path, output_folder, n_imgs=3):
    os.makedirs(output_folder, exist_ok=True)

    img_paths = []
    with Image.open(gif_path) as img:
        for frame_idx in np.linspace(0, img.n_frames - 1, num=n_imgs, dtype=int):
            img.seek(frame_idx)

            storage_path = os.path.join(output_folder, f"frame_{frame_idx:03d}.png")
            img.save(storage_path)

            img_paths.append(storage_path)
    return img_paths


def scale_mesh(mesh, scaling_factors):
    _center, scale = scaling_factors
    mesh.apply_scale(1 / scale)
    return mesh


def decompose_tool_convex(tool_obj, output_dir, draw=True):
    """Decompose a tool mesh into approximately convex parts using CoACD.

    Saves each part as an .obj file under output_dir and calls get_PC_transform(draw=draw)
    on each resulting Object.  Returns the list of convex-part Objects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = tool_obj.tri_mesh
    mesh = coacd.Mesh(
        vertices=src.vertices.astype(np.float64),
        indices=src.faces.astype(np.int32),
    )

    coacd.set_log_level("error")
    parts = coacd.run_coacd(mesh)

    objects = []
    PC_transforms = []
    for i, (vertices, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        obj_path = output_dir / f"{tool_obj.id}_convex_{i:03d}.obj"
        part_mesh.export(str(obj_path))

        obj = Object(id=f"{tool_obj.id}_convex_{i:03d}", path=obj_path)
        # cache the already-computed trimesh so get_PC_transform doesn't re-load from disk
        obj.__dict__["tri_mesh"] = part_mesh
        objects.append(obj)

        print(
            f"  Part {i}: {len(part_mesh.vertices)} vertices, {len(part_mesh.faces)} faces"
        )
        PC_transform = get_PC_transform(part_mesh, draw=draw)
        PC_transforms.append(PC_transform)

    print(
        f"Decomposed '{tool_obj.id}' into {len(objects)} convex part(s). Saved to {output_dir}"
    )
    return objects, PC_transforms
