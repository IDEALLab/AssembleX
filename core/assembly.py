import functools
import json
import os
import shutil
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from core.collision_checker import CollisionChecker
from core.feedback_generator import FeedbackGenerator
from core.manual_generator import ManualGenerator
from core.models import Object, Tool
from core.renderer import Renderer
from core.sequence_planner import SequencePlanner
from core.simulation import Simulation
from core.tool_analyzer import ToolAnalyzer


def init_openai():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


class Eval:
    def __init__(
        self,
        output_dir,
        dir="assets/",
        verbose=0,
        cache="read",
        cache_dir="assets/assembly_cache",
        token_limit=1_000_000,
    ):
        self.assemblies = []
        self.tools = {}
        self.openai_api_key = None
        self.output_dir = output_dir
        self.data_path = Path(dir)
        self.cache = cache
        self.verbose = verbose
        self.cache_dir = Path(cache_dir)
        self.tokens_used = 0
        self.token_limit = token_limit
        self._token_limit_announced = False
        self.skipped_llm_calls = 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.tools:
            self.init_tools()

    def tokens_exhausted(self, call_name=None):
        """Return True if the cumulative token budget has been reached.

        On the first crossing, emit a loud warning. Subsequent skips log a
        short per-call notice so the user can see which calls are bypassed.
        """
        if self.token_limit is None or self.tokens_used < self.token_limit:
            return False
        if not self._token_limit_announced:
            print(
                f"\n[token-limit] Reached {self.tokens_used:,} / {self.token_limit:,} tokens — "
                f"skipping all subsequent LLM calls."
            )
            self._token_limit_announced = True
        self.skipped_llm_calls += 1
        if call_name:
            print(f"[token-limit] skipped LLM call: {call_name}")
        return True

    def init_tools(self):
        tools_dir = self.data_path / "tools"
        for tool_subdir in sorted(tools_dir.iterdir()):
            if not tool_subdir.is_dir():
                continue
            obj_files = sorted(f for f in tool_subdir.iterdir() if f.suffix == ".obj")
            if not obj_files:
                continue
            tool_file = obj_files[0]
            tool = Tool(id=tool_file.stem.lower(), path=tool_file)
            self.tools[tool.id] = tool

    def add_assembly(self, id, dir, storage_dir=None):
        assert id is not None, (
            "add_assembly requires an assembly id; pass --id when running a subcommand that operates on assemblies."
        )
        if storage_dir is not None:
            cache_id = Path(storage_dir)
            cache_id.mkdir(parents=True, exist_ok=True)
            assembly = Assembly(
                id=id,
                dir=dir,
                evaluation=self,
                output_folder=self.output_dir,
                storage_dir=cache_id,
            )
        elif self.cache in {"read", "update"}:
            name = dir.strip("/").split("/")[-1]
            cache_id = self.get_cache_directory(id, name)
            if self.cache == "update":
                print(f"Updating cache for assembly {id}. Clearing {cache_id}...")
                if cache_id.exists():
                    while True:
                        user_input = input(
                            f"Cache directory {cache_id} already exists. Do you want to clear it and re-run the processing pipeline? (y/n): "
                        )
                        if user_input.lower() == "y":
                            shutil.rmtree(cache_id)
                            break
                        elif user_input.lower() == "n":
                            break
            cache_id.mkdir(parents=True, exist_ok=True)
            if cache_id.exists() and any(cache_id.iterdir()):
                print(
                    f"Cache found for assembly {id} at {cache_id}. Loading from cache..."
                )
            else:
                print(
                    f"No cache found for assembly {id}. Running full processing pipeline..."
                )
            assembly = Assembly(
                id=id,
                dir=dir,
                evaluation=self,
                output_folder=self.output_dir,
                storage_dir=cache_id,
            )
        else:
            assembly = Assembly(
                id=id, dir=dir, evaluation=self, output_folder=self.output_dir
            )

        self.assemblies.append(assembly)

    def get_cache_directory(self, assembly_id, name):
        cache_id_dir = self.cache_dir / name / str(assembly_id)
        cache_id_dir.mkdir(parents=True, exist_ok=True)
        return cache_id_dir


class Assembly:
    def __init__(self, dir, output_folder, storage_dir=None, id=None, evaluation=None):
        self.id = id
        if self.id is None:
            self.id = self.init_part_id()

        # TODO handle paths better
        self.output_dir = None
        self.storage_dir = storage_dir
        # self.data_path = Path("assets/" + dir) OBSOLETE
        # self.data_path_ATA = Path(dir) OBSOLETE
        self.assembly_dir = Path(dir) / str(self.id)

        self.openai_api_key = None
        self.evaluation = evaluation
        self.scaled_tools = None

        self.objects = None
        self.sequence = []
        self.remaining = set()
        self.collisions = {}
        self.planning_failures = None

        if self.openai_api_key is None:
            if evaluation and evaluation.openai_api_key:
                self.openai_api_key = evaluation.openai_api_key
            else:
                self.openai_api_key = init_openai()

        if self.output_dir is None:
            self.output_dir = self.get_output_dir(output_folder)

        if storage_dir is None:
            self.storage_dir = self.output_dir

        if self.scaled_tools is None:
            self.init_tools()

        if self.objects is None:
            self.objects = self.init_objects_()
            if self.evaluation and self.evaluation.verbose:
                print(f"Initialized part with {len(self.objects)} objects ...")

        # Component instances — each handles one area of responsibility
        self.renderer = Renderer(self)
        self.collision_checker = CollisionChecker(self)
        self.planner = SequencePlanner(self)
        self.tool_analyzer = ToolAnalyzer(self)
        self.instructions = {
            "Title": f"Assembly Manual for assembly {self.id}\n\n",
            "Steps": [],
            "Feedback": [],
            "Manual": [],
        }
        self.feedback = FeedbackGenerator(self)
        self.manual = ManualGenerator(self, self.feedback)
        self.simulation = Simulation(self)

    @staticmethod
    def _make_messages(system_text, user_content):
        """Build a standard [system, user] message list for chat-completion calls."""
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _to_canonical(step, matrix):
        """Undo the sim pose rotation so the matrix is in canonical assembly frame.

        ASAPx places every part at ``pose @ canonical_transform`` inside the sim, so
        export_replay_matrices returns transforms in that pose-rotated world frame.
        Multiplying by inv(pose) recovers the canonical frame that tri_mesh lives in.
        When pose is None (ATA planner or unrotated ASAP steps) the matrix is already
        canonical and is returned unchanged.
        """
        if step.pose is None:
            return matrix
        pose = np.asarray(step.pose)
        return np.linalg.inv(pose) @ matrix

    def create_doc(self):
        """Concatenate everything in self.instructions into a single text file."""
        output_file = self.output_dir / "full_instructions_document.txt"
        with open(output_file, "w") as f:
            for key, content in self.instructions.items():
                f.write(f"--- {key} ---\n")
                if isinstance(content, list):
                    f.writelines(f"{item}\n" for item in content)
                else:
                    f.write(f"{content}\n")
                f.write("\n")
        if self.evaluation and self.evaluation.verbose:
            print(f"Document saved to {output_file}")

    def get_output_dir(self, output_folder=None):
        output_dir = output_folder / self.id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def init_objects_(self):
        objects = {}
        obj_files = [
            f for f in sorted(os.listdir(self.assembly_dir)) if f.endswith(".obj")
        ]
        for obj_file in obj_files:
            obj_id = obj_file.replace(".obj", "")
            objects[obj_id] = Object(
                id=obj_id,
                image_paths={"iso1": None, "iso2": None},
                path=Path(self.assembly_dir / obj_file),
            )
        return objects

    def init_tools(self):
        self.scaled_tools = {}
        assembly_norm_file = self.assembly_dir / "normalization.json"
        assembly_scale = 1.0
        if assembly_norm_file.exists():
            with open(assembly_norm_file) as f:
                assembly_scale = json.load(f).get("scale", 1.0)

        for tool_id, tool in self.evaluation.tools.items():
            tool_norm_file = tool.path.parent / "normalization.json"
            tool_scale = 1.0
            if tool_norm_file.exists():
                with open(tool_norm_file) as f:
                    tool_scale = json.load(f).get("scale", 1.0)

            # Cancel out the tool's preprocessing scale, apply assembly scale
            relative_scale = assembly_scale / tool_scale
            scale_transform = np.eye(4)
            scale_transform[:3, :3] *= relative_scale

            print(
                f"Scaling tool {tool_id} by factor {relative_scale:.4f} to match assembly scale"
            )

            new_tool = Tool(
                id=tool.id,
                name=tool.name,
                image_paths=tool.image_paths,
                path=tool.path,
                scaling_factor=relative_scale,
                description=tool.description,
            )
            scaled_mesh = tool.tri_mesh.copy()
            scaled_mesh.apply_transform(scale_transform)
            new_tool.__dict__["tri_mesh"] = scaled_mesh
            # JSON cache stores contact_point in the tool's own normalized space; scale to this assembly.
            if new_tool.contact_point is not None:
                new_tool.__dict__["contact_point"] = (
                    new_tool.contact_point * relative_scale
                )
            self.scaled_tools[tool_id] = new_tool

    # --- Delegation methods — keep call sites in main.py unchanged ---

    def get_collisions(self, draw=False, mode="boolean", show=False):
        return self.collision_checker.get_collisions(draw=draw, mode=mode, show=show)

    def view_collisions(self, names_overlapping, show=False, save=False):
        return self.renderer.view_collisions(names_overlapping, show=show, save=save)

    @functools.cached_property
    def images(self):
        self.renderer._create_images()
        return {obj_id: obj.image_paths for obj_id, obj in self.objects.items()}

    def get_assembly_plans(self, args):
        return self.planner.get_assembly_plans(args)

    def update_sequence(self, sequence, assign_step_nr=True):
        return self.planner.update_sequence(sequence, assign_step_nr)

    def check_tool_needed(
        self,
        obj_idx,
        show=False,
        show_part=True,
        opposite=False,
        allow_unclear=False,
        logprobs=False,
        force_static=False,
    ):
        return self.tool_analyzer.check_tool_needed(
            obj_idx,
            show=show,
            show_part=show_part,
            opposite=opposite,
            allow_unclear=allow_unclear,
            logprobs=logprobs,
            force_static=force_static,
        )

    def analyze_assembly_tools(
        self, overwrite=False, allow_unclear=False, logprobs=False, force_static=True
    ):
        return self.tool_analyzer.analyze_assembly_tools(
            overwrite=overwrite,
            allow_unclear=allow_unclear,
            logprobs=logprobs,
            force_static=force_static,
        )

    def apply_tool(self, tool_name, obj_id, show=False, invert=False):
        return self.tool_analyzer.apply_tool(
            tool_name, obj_id, show=show, invert=invert
        )

    def check_tool_collision(
        self, tool_mesh, tool_name="tool", move_id=None, show=False, mode="depth"
    ):
        return self.tool_analyzer.check_tool_collision(
            tool_mesh, tool_name=tool_name, move_id=move_id, show=show, mode=mode
        )

    def check_tool_assemblable(self, tool_mesh, ass_obj, show=False):
        return self.tool_analyzer.check_tool_assemblable(tool_mesh, ass_obj, show=show)

    def name_parts(self, iso_only=False, log_probs=False):
        return self.feedback.name_parts(iso_only=iso_only, log_probs=log_probs)

    def generate_instructions(self):
        return self.feedback.generate_instructions()
