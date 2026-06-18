import json
import os
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh
from ATA.examples.run_joint_plan import BFSPlanner
from matplotlib.patches import Patch
from PIL import Image

import settings
from core.llm import image_part, llm_chat
from core.models import (
    Object,
    SingleWordResponse,
    Tool,
    ToolAnalysis,
    ToolInfo,
    decompose_tool_convex,
    encode_image,
    get_PC_transform,
)

# This file lives at <repo_root>/core/tool_analyzer.py, so the repo root
# (which holds the assets/, ATA/ and ASAPx/ trees) is one level up.
project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)


def _rotation_between(src, tgt):
    """Return a 4×4 rotation matrix that maps unit vector src onto tgt."""
    src = np.asarray(src, float)
    tgt = np.asarray(tgt, float)
    src = src / np.linalg.norm(src)
    tgt = tgt / np.linalg.norm(tgt)
    axis = np.cross(src, tgt)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(src, tgt))
    if sin_a < 1e-8:
        if cos_a > 0:
            return np.eye(4)
        perp = (
            np.array([1.0, 0.0, 0.0])
            if abs(src[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        return trimesh.transformations.rotation_matrix(np.pi, axis)
    return trimesh.transformations.rotation_matrix(
        np.arctan2(sin_a, cos_a), axis / sin_a
    )


# --- System prompts (role + task, identical for every call of that type) ---

_SYSTEM_TOOL_NAMING = (
    "You are an expert mechanical engineer, machinist, and assembly technician. "
    "Your task is to identify hand tools from their rendered CAD images using precise engineering terminology. "
    "For each tool, provide:\n"
    "  - name: 1 to 3 words, the standard industry tool name "
    "(e.g. 'Phillips screwdriver', 'hex key', 'open-end wrench', 'socket wrench', 'flathead screwdriver').\n"
    "  - description: one short sentence covering what the tool is and what it is generally used for. "
    "Base the description only on the tool itself; do not speculate about which specific assembly part it engages.\n"
    "Output ONLY the structured response. No explanations or conversational filler."
)

_SYSTEM_TOOL_ANALYST = (
    "You are an expert mechanical engineer and assembly technician. "
    "Your task is to analyze CAD images and determine whether a specific tool is required "
    "to perform a single disassembly step.\n\n"
    "### Assembly Context\n"
    "- The overall assembly is completely rigid. During this step, ONLY the target part moves; "
    "all other parts remain strictly stationary.\n"
    "- Disassembly is performed by a human hand.\n"
    "- A tool is 'required' only if moving the part by bare hand is unrealistic or highly impractical "
    "(e.g., overcoming threaded friction, unfastening tight joints). "
    "If the part can easily be slid, lifted, or translated without aid, no tool is needed.\n\n"
    "### Decision Criteria\n"
    "1. Part geometry: does it have features designed for a specific tool "
    "(screw head, hex socket, wrench flat, tool-insertion slot)?\n"
    "2. Removal motion: does the sequence show rotation or prying that requires mechanical advantage?\n"
    "3. Part name: is the part type commonly associated with a specific tool?\n\n"
    "### Output\n"
    "Respond with exactly ONE word. The allowed responses are listed in the user message."
)

_SYSTEM_TOOL_ANALYST_STATIC = (
    "You are an expert mechanical engineer and assembly technician. "
    "Your task is to analyze CAD images of an isolated assembly part and determine whether "
    "a specific tool is required to disassemble that part from a typical surrounding assembly.\n\n"
    "### Context\n"
    "- You are shown several rendered views of the SAME part from different camera angles "
    "(e.g. isometric, top, front, side). No disassembly motion is provided — base your "
    "decision on the static geometry alone.\n"
    "- Disassembly is performed by a human hand; the assembly itself is rigid.\n"
    "- A tool is 'required' only if moving the part by bare hand is unrealistic or highly impractical "
    "(e.g. overcoming threaded friction, unfastening tight joints). If a human can simply slide, "
    "lift, or translate the part without aid, no tool is needed.\n\n"
    "### Decision Criteria — geometry only\n"
    "1. Tool-specific features: screw heads (slotted, Phillips, hex), nut faces, wrench flats, "
    "hex sockets, drive splines, tool-insertion slots.\n"
    "2. Part type: is the shape a recognisable fastener (screw, bolt, nut) or a hand-removable "
    "component (bracket, panel, cover)?\n"
    "3. Accessibility: would a human finger reasonably fit and exert enough force on the part?\n\n"
    "### Output\n"
    "Respond with exactly ONE word. The allowed responses are listed in the user message."
)

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


class ToolAnalyzer:
    def __init__(self, assembly):
        self.assembly = assembly

    def _make_messages(self, system_text, user_content):
        """Build a standard [system, user] message list."""
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ]

    def name_tools(self, iso_only=False):
        """Assign a name and short description to each tool on the assembly.

        Mirrors Feedback.name_parts: results are cached in
        ``<storage_dir>/tool_names.json`` as ``{tool_id: {"name": ..., "description": ...}}``.
        If iso_only=True, all tools are identified in a single batched API call using
        their 'iso1' view; otherwise one call per tool is made using every available view.
        Names and descriptions are written back to both ``evaluation.tools`` (the
        canonical set used by check_tool_needed) and ``self.assembly.scaled_tools``.
        """
        prompt_args = {"model": "gpt-4o"}

        tools = self.assembly.evaluation.tools if self.assembly.evaluation else {}
        if not tools:
            return True

        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(f"\nNaming {len(tools)} tools...")

        def _propagate():
            for tool_id, tool in tools.items():
                scaled = (
                    self.assembly.scaled_tools.get(tool_id)
                    if self.assembly.scaled_tools
                    else None
                )
                if scaled is not None:
                    scaled.name = tool.name
                    scaled.description = tool.description

        names_file_path = self.assembly.storage_dir / "tool_names.json"
        if names_file_path.exists():
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Loading tool names from {names_file_path}...")
            with open(names_file_path) as f:
                data = json.load(f)
                for tool_id, info in data.items():
                    if tool_id in tools:
                        tools[tool_id].name = info.get("name")
                        tools[tool_id].description = info.get("description")
            _propagate()
            return True

        token_cost = 0

        if iso_only:
            if self.assembly.evaluation and self.assembly.evaluation.tokens_exhausted(
                "name_tools (iso_only)"
            ):
                return None

            tool_order = list(tools.values())
            user_content = [
                {
                    "type": "text",
                    "text": (
                        "Identify each of the following tools. For each, give a 1-3 word "
                        "engineering name and a short general description, in the order shown."
                    ),
                }
            ]
            for tool in tool_order:
                user_content.append({"type": "text", "text": f"Tool {tool.id}"})
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(tool.image_paths['iso1'])}",
                            "detail": "low",
                        },
                    }
                )

            try:
                response = llm_chat(
                    self.assembly.evaluation,
                    self.assembly.openai_api_key,
                    parse=True,
                    **prompt_args,
                    messages=self._make_messages(_SYSTEM_TOOL_NAMING, user_content),
                    response_format=ToolAnalysis,
                )
                if self.assembly.evaluation and self.assembly.evaluation.verbose:
                    print(response.choices[0].message.parsed)

                infos = response.choices[0].message.parsed.tool_infos
                names_dict = {}
                for tool, info in zip(tool_order, infos, strict=False):
                    tool.name = info.name
                    tool.description = info.description
                    names_dict[tool.id] = {
                        "name": info.name,
                        "description": info.description,
                    }

                with open(names_file_path, "w") as f:
                    json.dump(names_dict, f)

                _propagate()
                return response.choices[0].message.parsed
            except Exception as e:
                print(f"Error during OpenAI API call in name_tools: {e}")
                return None

        else:
            for tool in tools.values():
                if (
                    self.assembly.evaluation
                    and self.assembly.evaluation.tokens_exhausted(
                        f"name_tools (tool {tool.id})"
                    )
                ):
                    tool.name = f"Tool_{tool.id}"
                    tool.description = None
                    continue

                user_content = [
                    {
                        "type": "text",
                        "text": "Identify this tool from its rendered views.",
                    }
                ]
                for img in tool.image_paths.values():
                    if img is not None:
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(img)}"
                                },
                            }
                        )

                try:
                    response = llm_chat(
                        self.assembly.evaluation,
                        self.assembly.openai_api_key,
                        parse=True,
                        **prompt_args,
                        messages=self._make_messages(_SYSTEM_TOOL_NAMING, user_content),
                        response_format=ToolInfo,
                    )
                    info = response.choices[0].message.parsed

                    if self.assembly.evaluation and self.assembly.evaluation.verbose:
                        print(f"Received name '{info.name}' for tool {tool.id}")

                    tool.name = info.name
                    tool.description = info.description
                    token_cost += response.usage.total_tokens

                except Exception as e:
                    print(
                        f"Error during OpenAI API call in name_tools for tool {tool.id}: {e}"
                    )
                    tool.name = f"Tool_{tool.id}"
                    tool.description = None
                    print(f"Defaulting to name '{tool.name}' for tool {tool.id}")

            names_dict = {
                tool.id: {"name": tool.name, "description": tool.description}
                for tool in tools.values()
            }
            with open(names_file_path, "w") as f:
                json.dump(names_dict, f)

            _propagate()
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Total Token Cost for naming tools: {token_cost}")
            return True

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
        if logprobs:
            prompt_args = {
                "model": "gpt-4o",
                "max_tokens": 300,
                "response_format": SingleWordResponse,
                "logprobs": True,
                "top_logprobs": 3,
            }
        else:
            prompt_args = {"model": settings.LLM_model}

        obj = self.assembly.objects[obj_idx]

        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(
                f"\nChecking if tool is required for step {obj.step_nr}, obj {obj.name}"
            )
            available_tools = ", ".join(
                [tool.name for tool in self.assembly.scaled_tools.values()]
            )
            print(f"Available tools: {available_tools}")

        # Try to find the per-step disassembly time-lapse (motion mode). When no such
        # sequence/step is available, or `force_static` is set, fall back to scoring
        # the part from multi-angle static renders alone.
        img_paths = None
        if not force_static:
            if obj.step_nr is not None:
                step = self.assembly.sequence[obj.step_nr]
            else:
                step = next(
                    (s for s in self.assembly.sequence if s.obj_id == obj_idx), None
                )
            if step is not None and step.images:
                it = iter(step.images.keys())
                try:
                    angle = next(it)
                    if opposite:
                        angle = next(it)
                except StopIteration:
                    angle = None
                if angle is not None:
                    candidate = step.images.get(angle)
                    if candidate:
                        img_paths = candidate

        static_mode = img_paths is None

        # Ensure part renderings exist whenever they're going to be used in the prompt.
        needs_part_imgs = static_mode or show_part
        no_part_imgs = obj.image_paths is None or all(
            v is None for v in (obj.image_paths or {}).values()
        )
        if needs_part_imgs and no_part_imgs:
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"No part images found for object {obj_idx}. Creating images...")
            self.assembly.images

        if static_mode:
            part_imgs = [
                (name, path)
                for name, path in (obj.image_paths or {}).items()
                if path is not None
            ]
            if not part_imgs:
                if self.assembly.evaluation and self.assembly.evaluation.verbose:
                    print(
                        f"No part images available for object {obj_idx} in static mode."
                    )
                return None, None

        if show:
            preview_paths = img_paths if not static_mode else [p for _, p in part_imgs]
            _fig, axes = plt.subplots(1, len(preview_paths), figsize=(15, 5))
            if len(preview_paths) == 1:
                axes = [axes]
            for ax, img_path in zip(axes, preview_paths, strict=False):
                im = Image.open(img_path)
                ax.imshow(im)
                ax.axis("off")
            plt.tight_layout()
            plt.show()

        eval_tools = self.assembly.evaluation.tools.values()
        tools_str = ", ".join([tool.name for tool in eval_tools])
        described_tools = [t for t in eval_tools if getattr(t, "description", None)]
        if described_tools:
            tool_descriptions_block = "Tool descriptions:\n" + "\n".join(
                f"- {t.name}: {t.description}" for t in described_tools
            )
        else:
            tool_descriptions_block = ""

        # --- Build user message: per-call specifics + images ---
        allowed = f"{tools_str}, none"
        if allow_unclear:
            allowed += (
                ", unclear (only if the images lack sufficient detail — DO NOT GUESS)"
            )

        if static_mode:
            view_lines = "\n".join(
                f"- {name} view of '{obj.name}'" for name, _ in part_imgs
            )
            user_text = (
                f"Target part: {obj.name}\n"
                f"Available tools: {tools_str}\n"
                + (
                    tool_descriptions_block + "\n\n"
                    if tool_descriptions_block
                    else "\n"
                )
                + f"Allowed responses: {allowed}\n\n"
                f"Images provided ({len(part_imgs)} static views of the part):\n"
                f"{view_lines}"
            )
            user_content = [{"type": "text", "text": user_text}]
            for name, img in part_imgs:
                user_content.append({"type": "text", "text": f"{name} view"})
                user_content.append(image_part(img))
            system_prompt = _SYSTEM_TOOL_ANALYST_STATIC
        else:
            user_text = (
                f"Target part: {obj.name}\n"
                f"Available tools: {tools_str}\n"
                + (
                    tool_descriptions_block + "\n\n"
                    if tool_descriptions_block
                    else "\n"
                )
                + f"Allowed responses: {allowed}\n\n"
                "Images provided:\n"
            )
            if show_part:
                user_text += f"- Image 1: isolated isometric view of '{obj.name}' (inspect for tool-specific geometry features)\n"
            user_text += "- Remaining images: time-lapse of the part being removed from the assembly (examine trajectory and motion)"

            user_content = [{"type": "text", "text": user_text}]
            if show_part:
                img = obj.image_paths.get("iso1")
                user_content.append(
                    {"type": "text", "text": f"Isometric view of '{obj.name}'"}
                )
                user_content.append(image_part(img))
            for i, img in enumerate(img_paths):
                user_content.append({"type": "text", "text": f"State {i}"})
                user_content.append(image_part(img))
            system_prompt = _SYSTEM_TOOL_ANALYST

        messages = self._make_messages(system_prompt, user_content)

        if self.assembly.evaluation and self.assembly.evaluation.tokens_exhausted(
            f"check_tool_needed (obj {obj_idx})"
        ):
            return None, None

        try:
            if logprobs:
                response = llm_chat(
                    self.assembly.evaluation,
                    self.assembly.openai_api_key,
                    parse=True,
                    messages=messages,
                    **prompt_args,
                )
            else:
                response = llm_chat(
                    self.assembly.evaluation,
                    self.assembly.openai_api_key,
                    messages=messages,
                    **prompt_args,
                )
        except Exception as e:
            print(f"Error during OpenAI API call in check_tool_needed: {e}")
            return None, None

        confidence = None
        if logprobs:
            tool_suggested = response.choices[0].message.parsed.word.lower()
            try:
                confidence = np.exp(
                    response.choices[0].logprobs.content[3].top_logprobs[0].logprob
                )
            except (AttributeError, IndexError, TypeError):
                print("Logprobs not available for confidence estimation.")

            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Suggested Tool: {tool_suggested}")
                print(f"Token Cost: {response.usage.total_tokens}")
                if response.choices[0].logprobs:
                    print("--- Alternative Words Considered ---")
                    token_logprob = response.choices[0].logprobs.content[3]
                    for top_logprob in token_logprob.top_logprobs:
                        prob = np.exp(top_logprob.logprob) * 100
                        print(f"Word: '{top_logprob.token}' | Confidence: {prob:.2f}%")
        else:
            tool_suggested = (
                response.choices[0]
                .message.content.strip()
                .lower()
                .strip('"')
                .strip("'")
            )

        if "none" in tool_suggested:
            print(f"No tool required to assemble object {obj.name}.")
            return "none", confidence
        elif "unclear" in tool_suggested:
            print(
                f"Unclear whether a tool is required to assemble object {obj.name} based on the provided images. The analysis may have lacked sufficient detail or clarity to make a confident decision."
            )
            return "unclear", confidence

        # The LLM is shown tool *names* (e.g. "Phillips screwdriver") but
        # scaled_tools is keyed by *id* (e.g. "screwdriverphillipshead").
        # Resolve the suggestion via name first, fall back to id match.
        def _norm(s):
            return "".join(ch for ch in str(s).lower() if ch.isalnum())

        suggested_norm = _norm(tool_suggested)
        resolved_id = None
        if tool_suggested in self.assembly.scaled_tools:
            resolved_id = tool_suggested
        else:
            for tid, tool in self.assembly.scaled_tools.items():
                if (
                    _norm(getattr(tool, "name", "")) == suggested_norm
                    or _norm(tid) == suggested_norm
                ):
                    resolved_id = tid
                    break

        if resolved_id is None:
            tool_options = [
                f"{getattr(t, 'name', tid)} (id={tid})"
                for tid, t in self.assembly.scaled_tools.items()
            ]
            print(
                f"Tool '{tool_suggested}' not found in available tools. "
                f"Available tools: {tool_options}"
            )
            return "error", None

        return resolved_id, confidence

    def analyze_assembly_tools(
        self,
        overwrite=False,
        allow_unclear=False,
        logprobs=False,
        force_static=True,
    ):
        """Pre-compute, for every part of the assembly, which tool (if any) is needed,
        and ensure both that tool's and that part's orientation/contact axes are cached.

        Decisions are persisted to ``<storage_dir>/tool_decisions.json`` as
        ``{part_id: {"tool": <name|"none"|"unclear"|"error"|None>, "confidence": <float|None>}}``
        so subsequent runs (including sequence finding) can skip the VLM call entirely.
        After all parts have been classified, ``select_axes`` is invoked for the tools
        that ended up being needed and for the parts that need them — both routines
        write their own per-object JSON cache next to the source .obj file.

        Args:
            overwrite: if True, re-run the VLM check even when a cached entry exists.
            allow_unclear: forwarded to ``check_tool_needed``.
            logprobs: forwarded to ``check_tool_needed``.
            force_static: when True (default) the static-image prompt is used, which
                works before any disassembly sequence has been planned. Set to False
                to use the existing motion-mode prompt when sequence images exist.

        Returns:
            The decisions dict (the same content written to disk).
        """
        cache_path = self.assembly.storage_dir / "tool_decisions.json"
        cache = {}
        if cache_path.exists() and not overwrite:
            with open(cache_path) as f:
                cache = json.load(f)
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Loaded {len(cache)} cached tool decisions from {cache_path}")

        # Static prompt needs the multi-angle part renders; create them up-front for
        # parts that haven't been rendered yet.
        if force_static:
            self.assembly.images

        needed_tool_ids = set()
        terminal_states = {None, "none", "unclear", "error"}

        for obj_id in self.assembly.objects:
            entry = cache.get(obj_id)
            if entry is not None and not overwrite:
                tool_cached = entry.get("tool")
                if tool_cached not in terminal_states:
                    needed_tool_ids.add(tool_cached)
                continue

            tool, conf = self.check_tool_needed(
                obj_idx=obj_id,
                show=False,
                show_part=True,
                allow_unclear=allow_unclear,
                logprobs=logprobs,
                force_static=force_static,
            )
            cache[obj_id] = {"tool": tool, "confidence": conf}
            if tool not in terminal_states:
                needed_tool_ids.add(tool)

            # Persist progress after every part so an interrupted run isn't wasted.
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)

        # Orient the tools that we actually need.
        axes_dir = self.assembly.output_dir / "convex_decomp"
        tools_to_axis = {
            tid: self.assembly.evaluation.tools[tid]
            for tid in needed_tool_ids
            if tid in self.assembly.evaluation.tools
        }
        if tools_to_axis:
            self.select_axes(tools_to_axis, axes_dir)
            self.assembly.init_tools()

        # Orient the parts that need tools.
        parts_to_axis = {
            oid: self.assembly.objects[oid]
            for oid, entry in cache.items()
            if entry.get("tool") not in terminal_states and oid in self.assembly.objects
        }
        if parts_to_axis:
            self.select_axes(parts_to_axis, axes_dir)

        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(f"[analyze_assembly_tools] decisions saved to {cache_path}")

        return cache

    def get_cached_tool_decision(self, obj_id):
        """Return the cached tool decision for ``obj_id`` (loaded from
        ``<storage_dir>/tool_decisions.json``), or None if no cache entry exists.

        Result shape mirrors ``check_tool_needed``: ``(tool, confidence)`` where
        ``tool`` is a tool name, ``"none"``, ``"unclear"``, ``"error"``, or None.
        """
        cache_path = self.assembly.storage_dir / "tool_decisions.json"
        if not cache_path.exists():
            return None
        with open(cache_path) as f:
            cache = json.load(f)
        entry = cache.get(str(obj_id))
        if entry is None:
            return None
        return entry.get("tool"), entry.get("confidence")

    def apply_tool(self, tool_name, obj_id, show=False, invert=False):
        obj = self.assembly.objects[obj_id]
        tool = self.assembly.scaled_tools[tool_name]

        # Tool axes are computed on the unscaled tool so the JSON cache stays universal across
        # assemblies; re-running init_tools rescales contact_point for this assembly.
        if tool.direction is None or tool.contact_point is None:
            axis_save_dir = self.assembly.output_dir / "convex_decomp"
            original_tool = self.assembly.evaluation.tools[tool_name]
            self.select_axes({tool_name: original_tool}, axis_save_dir)
            self.assembly.init_tools()
            tool = self.assembly.scaled_tools[tool_name]

        print(
            f"Tool: {tool_name}, direction: {tool.direction}, contact point: {tool.contact_point}"
        )
        save_dir = self.assembly.storage_dir / "tool_data"
        return ToolAnalyzer._apply_tool_geometric(
            tool, obj, save_dir=save_dir, show=show, invert=invert
        )

    @staticmethod
    def _apply_tool_geometric(tool, obj, save_dir=None, show=False, invert=False):
        """Pure-geometry tool placement: position the tool's mesh against the part using
        cached `direction`/`contact_point`. Falls back to PCA / extreme vertex when an
        axis or contact is missing on the part (and to a Z-up default on the tool).

        Args:
            tool: a Tool instance with `tri_mesh` (and ideally `direction`, `contact_point`).
            obj: an Object instance with `tri_mesh`.
            save_dir: optional Path to write a debug screenshot to. None disables screenshots.
            show: open the pyvista plotter interactively after rendering.
            invert: flip the part's approach direction (used when the regular orientation
                collides with neighbours).

        Returns:
            (transformed_tool_mesh, screenshot_path_or_None) on success, or None when
            either mesh is not watertight.
        """
        # Approach direction on the part: cached obj.direction, fall back to minor principal axis
        if obj.direction is not None:
            obj_dir = obj.direction
        else:
            axes_transform = np.round(
                get_PC_transform(obj.tri_mesh, draw=False), decimals=8
            )
            v = axes_transform[2, :3]
            obj_dir = v / np.linalg.norm(v)
        if invert:
            obj_dir = -obj_dir

        # Contact point on the part. obj.contact_point is already projected onto the
        # convex-part axis by _pick_orientation, so use it directly. Fall back to projecting
        # the extreme full-mesh vertex when no cached value is available.
        if obj.contact_point is not None:
            obj_contact = obj.contact_point
        else:
            g_obj = obj.tri_mesh.center_mass
            dists = np.dot(obj.tri_mesh.vertices, obj_dir)
            extreme_v = obj.tri_mesh.vertices[np.argmax(dists)]
            obj_contact = g_obj + obj_dir * np.dot(extreme_v - g_obj, obj_dir)

        tool_dir = (
            tool.direction if tool.direction is not None else np.array([0.0, 0.0, 1.0])
        )
        # tool.contact_point is already on the convex-part axis; use it directly.
        # Only compute an extreme-vertex fallback when no cached value exists.
        if tool.contact_point is not None:
            tool_tip = tool.contact_point
        else:
            tool_tip = tool.tri_mesh.vertices[
                np.argmax(tool.tri_mesh.vertices @ tool_dir)
            ]

        # tool_dir (approach direction, toward part) is anti-parallel to obj_dir (outward
        # from contact surface, toward tool), so rotate tool_dir onto -obj_dir.
        rotation = _rotation_between(tool_dir, -obj_dir)
        tool_copy = tool.tri_mesh.copy()
        tool_copy.apply_transform(rotation)
        tool_tip_rotated = trimesh.transformations.transform_points(
            [tool_tip], rotation
        )[0]
        tool_copy.apply_translation(obj_contact - tool_tip_rotated)

        if (not tool_copy.is_watertight) or (not obj.tri_mesh.is_watertight):
            print(
                "Warning: Tool or object mesh is not watertight. Cannot perform boolean operation."
            )
            return None

        save_path = None
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(
                pv.Arrow(start=obj_contact, direction=-obj_dir, scale=0.5),
                color="cyan",
                label="Tool approach direction",
            )
            plotter.show_bounds(grid=True, location="outer")
            plotter.add_axes()
            plotter.add_mesh(
                obj.tri_mesh, color="blue", opacity=0.7, label="Original Part"
            )
            plotter.add_mesh(tool_copy, color="red", opacity=0.7, label="Applied Tool")
            plotter.add_points(
                np.array([obj_contact]),
                color="lime",
                point_size=14,
                render_points_as_spheres=True,
                label="Object Contact",
            )
            plotter.add_points(
                np.array([tool_tip_rotated]),
                color="orange",
                point_size=14,
                render_points_as_spheres=True,
                label="Tool Tip (placed)",
            )
            plotter.add_legend()

            idx = 0
            save_path = save_dir / f"{obj.id}_{tool.id}_{idx}.png"
            while save_path.exists():
                idx += 1
                save_path = save_dir / f"{obj.id}_{tool.id}_{idx}.png"

            plotter.screenshot(save_path)
            if show:
                plotter.show()
            plotter.close()

        return tool_copy, save_path

    # ------------------------------------------------------------------
    # Helpers shared by axis selection and orientation selection
    # ------------------------------------------------------------------

    def _render_arrows(self, tool_mesh, arrows, cam_pos, bg="white"):
        """Off-screen render of tool_mesh with a list of arrows.

        arrows: list of (origin, direction, color_hex, label) tuples.
        Returns a screenshot as an RGBA numpy array.
        """
        axis_length = np.linalg.norm(tool_mesh.bounding_box.extents) / 4.5
        label_color = "black" if bg == "white" else "white"
        tool_color = "lightsteelblue" if bg == "white" else "lightgray"

        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        plotter.background_color = bg
        plotter.add_mesh(
            pv.wrap(tool_mesh), color=tool_color, opacity=0.45, show_edges=False
        )

        label_pts, label_texts = [], []
        for origin, direction, color_hex, label in arrows:
            plotter.add_mesh(
                pv.Arrow(
                    start=origin,
                    direction=direction,
                    scale=axis_length,
                    tip_length=0.25,
                    tip_radius=0.08,
                    shaft_radius=0.035,
                ),
                color=color_hex,
            )
            label_pts.append(origin + np.array(direction) * axis_length * 1.15)
            label_texts.append(label)

        if label_pts:
            plotter.add_point_labels(
                label_pts,
                label_texts,
                font_size=22,
                bold=True,
                text_color=label_color,
                point_size=0,
                always_visible=True,
                shape_opacity=0.0,
            )
        plotter.camera_position = cam_pos
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img

    def _vlm_pick(self, system_prompt, img_path, user_text):
        """Send one image to the VLM and return the first integer found in the reply."""
        if self.assembly.evaluation and self.assembly.evaluation.tokens_exhausted(
            "_vlm_pick"
        ):
            return None

        user_content = [
            {"type": "text", "text": user_text},
            image_part(img_path),
        ]
        response = llm_chat(
            self.assembly.evaluation,
            self.assembly.openai_api_key,
            model=settings.LLM_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content.strip()
        print(f"  AI response: '{raw}'")
        match = re.search(r"\d+", raw)
        return int(match.group()) if match else None

    # ------------------------------------------------------------------
    # Step 1 — pick the best axis from convex-decomposition candidates
    # ------------------------------------------------------------------

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
                    axes.append((center, direction, part.tri_mesh))

        def _axis_color(i):
            rgb = cmap(i / max(len(axes) - 1, 1))[:3]
            return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"

        arrows = [
            (origin, direction, _axis_color(i), str(i))
            for i, (origin, direction, _) in enumerate(axes)
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
                f"Choose the application axis (0 - {len(axes) - 1})",
                fontsize=14,
            )
            ax.axis("off")
            ax.legend(handles=patches, loc="lower right", fontsize=11, framealpha=0.85)
            plt.tight_layout()
            plt.show()
            plt.close(fig)

            while True:
                raw = input(f"[{obj_id}] Axis (0 - {len(axes) - 1}): ").strip()
                if raw.isdigit() and 0 <= int(raw) < len(axes):
                    break
                print(f"  Invalid - enter a number between 0 and {len(axes) - 1}.")
            chosen = int(raw)

        else:  # ai — three views + numerical legend for reliable reasoning
            view_specs = [("iso", "Isometric"), ("xy", "Top (XY)"), ("yz", "Side (YZ)")]
            ai_patches = [
                Patch(
                    color=_axis_color(i),
                    label=f"Axis {i}  [{', '.join(f'{v:+.2f}' for v in d)}]",
                )
                for i, (_, d, _pm) in enumerate(axes)
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
                for i, (_, d, _pm) in enumerate(axes)
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
        _, chosen_direction, chosen_part_mesh = axes[chosen]
        return chosen_direction, chosen_part_mesh

    # ------------------------------------------------------------------
    # Step 2 — pick orientation (+/−) along the chosen axis
    # ------------------------------------------------------------------

    def _pick_orientation(self, obj, obj_id, axis, part_mesh, save_dir, mode):
        """Find the two extreme points along axis, show opposing arrows, and pick one.

        part_mesh is the specific convex part whose axis was selected in _pick_axis;
        contact points are derived from it rather than the full object mesh.
        Returns the final signed direction vector and the chosen contact point.
        """
        g = part_mesh.center_mass
        axis_distances = np.dot(part_mesh.vertices, axis)
        v_pos = part_mesh.vertices[np.argmax(axis_distances)]
        v_neg = part_mesh.vertices[np.argmin(axis_distances)]
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

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

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
            axis, part_mesh = self._pick_axis(obj, obj_id, obj_dir, minor_only, mode)

            print(f"\n[{obj_id}] Step 2/2 — orientation selection  (mode={mode})")
            direction, contact = self._pick_orientation(
                obj, obj_id, axis, part_mesh, obj_dir, mode
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
        result = self.select_axes(
            self.assembly.evaluation.tools,
            output_dir,
            minor_only,
            mode,
            allow_overwrite,
        )
        self.assembly.init_tools()
        return result

    def select_object_axes(
        self, output_dir, minor_only=True, mode="ai", allow_overwrite=False
    ):
        return self.select_axes(
            self.assembly.objects, output_dir, minor_only, mode, allow_overwrite
        )

    def check_tool_collision(
        self,
        tool_mesh,
        tool_name="tool",
        move_id=None,
        show=True,
        mode="depth",
        step_nr=None,
    ):
        if step_nr is not None:
            removed = {self.assembly.sequence[i].obj_id for i in range(step_nr)}
            active_objects = [
                o for o in self.assembly.objects.values() if o.id not in removed
            ]
        else:
            active_objects = list(self.assembly.objects.values())
        verbose = bool(self.assembly.evaluation and self.assembly.evaluation.verbose)
        return ToolAnalyzer._check_tool_collision_geometric(
            tool_mesh,
            active_objects,
            tool_name=tool_name,
            move_id=move_id,
            show=show,
            mode=mode,
            verbose=verbose,
        )

    @staticmethod
    def _check_tool_collision_geometric(
        tool_mesh,
        active_objects,
        tool_name="tool",
        move_id=None,
        show=False,
        mode="depth",
        verbose=False,
        save_path=None,
    ):
        """Collision check between a placed tool mesh and a list of active parts.

        Args:
            tool_mesh: trimesh.Trimesh of the positioned tool.
            active_objects: iterable of Object instances (each must expose `tri_mesh`).
            tool_name, move_id, show, mode: see check_tool_collision.
            verbose: print per-pair diagnostics.
            save_path: if given, render the scene off-screen to this PNG path
                (tool yellow, target part green, colliders red, others lightblue).
                Independent of `show`; both can be set.

        Returns:
            list of (obj_name, penetration_or_volume) for each part exceeding the
            depth / overlap threshold.
        """
        if mode not in ("depth", "boolean"):
            raise ValueError(
                "Invalid mode. Mode should be either 'depth' for depth-based collision checking or 'boolean' for boolean intersection-based collision checking."
            )

        overlap_threshold = 0.0005
        depth_threshold = 0.001
        if verbose:
            print(f"Checking for collisions between {tool_name} and assembly parts...")

        manager = trimesh.collision.CollisionManager()
        manager.add_object(name=tool_name, mesh=tool_mesh)

        collisions = []
        for obj in active_objects:
            if not obj.tri_mesh.is_watertight:
                print(
                    f"Warning: Part {obj.id} is not watertight. Skipping collision check."
                )
                continue

            is_colliding, _, contact_data = manager.in_collision_single(
                mesh=obj.tri_mesh, return_names=True, return_data=True
            )
            if is_colliding:
                if mode == "boolean":
                    overlap_mesh = tool_mesh.intersection(obj.tri_mesh)
                    v_mesh = tool_mesh.volume if tool_mesh.is_volume else 0
                    v_other = obj.tri_mesh.volume if obj.tri_mesh.is_volume else 0
                    v_tot = v_mesh + v_other
                    v_overlap = overlap_mesh.volume
                    overlap_ratio = v_overlap * 2 / v_tot if v_tot > 0 else 0

                    if overlap_ratio >= overlap_threshold:
                        collisions.append((obj.name, v_overlap))
                    if verbose:
                        print(
                            f"Checked collision between {tool_name} and {obj.name}: Overlap Ratio = {overlap_ratio:.4f}"
                        )

                if mode == "depth":
                    penetration_depths = [contact.depth for contact in contact_data]
                    if penetration_depths:
                        max_penetration = np.max(penetration_depths)
                        if max_penetration >= depth_threshold:
                            collisions.append((obj.name, max_penetration))
                        if verbose:
                            print(
                                f"Checked collision between {tool_name} and {obj.name}: Max Penetration Depth = {max_penetration:.6f}"
                            )
                    elif verbose:
                        print(
                            f"Checked collision between {tool_name} and {obj.name}: No contact points found."
                        )

        if collisions:
            print(
                f"Collisions detected between {tool_name} and: {[c[0] for c in collisions]}"
            )
        elif verbose:
            print(
                f"No collisions detected between {tool_name} and any assembly parts. Total collisions: {len(collisions)}"
            )

        def _build_scene(off_screen):
            p = pv.Plotter(off_screen=off_screen)
            p.add_mesh(tool_mesh, color="yellow", opacity=0.7, label="Tool")
            collision_names = {c[0] for c in collisions}
            for obj in active_objects:
                if obj.id == move_id:
                    color, opacity = "green", 0.8
                elif obj.id in collision_names:
                    color, opacity = "red", 0.8
                else:
                    color, opacity = "lightblue", 0.2
                p.add_mesh(obj.mesh, color=color, opacity=opacity, label=str(obj.id))
            p.add_legend()
            return p

        if show:
            _build_scene(off_screen=False).show()

        if save_path is not None:
            plotter = _build_scene(off_screen=True)
            plotter.camera_position = "iso"
            plotter.screenshot(str(save_path))
            plotter.close()

        return collisions

    def check_tool_assemblable(self, tool_mesh, ass_obj, show=False, step_nr=None):
        if step_nr is not None:
            removed = {self.assembly.sequence[i].obj_id for i in range(step_nr)}
            active_objects = [
                o for o in self.assembly.objects.values() if o.id not in removed
            ]
        else:
            active_objects = list(self.assembly.objects.values())
        asset_folder = os.path.join(project_base_dir, "ATA", "assets")
        verbose = bool(self.assembly.evaluation and self.assembly.evaluation.verbose)
        return ToolAnalyzer._check_tool_assemblable_geometric(
            tool_mesh,
            ass_obj,
            active_objects,
            asset_folder,
            self.assembly.assembly_dir,
            show=show,
            verbose=verbose,
        )

    @staticmethod
    def _check_tool_assemblable_geometric(
        tool_mesh,
        ass_obj,
        active_objects,
        asset_folder,
        assembly_src_dir,
        show=False,
        verbose=False,
        max_time=60,
        record_path=None,
    ):
        """Run BFS path planning to verify the placed tool (and tool+part combined)
        can be removed without colliding with the other active parts.

        Args:
            tool_mesh: trimesh.Trimesh of the positioned tool.
            ass_obj: Object instance of the part being removed.
            active_objects: iterable of Object instances (the active parts incl. ass_obj).
            asset_folder: redmax/BFSPlanner asset root (e.g. <base>/ATA/assets).
            assembly_src_dir: directory containing the active parts' .obj files; these
                will be copied into a scratch sub-directory used by the planner.
            show, verbose, max_time: forwarded / for logging.
            record_path: if given, forwarded to BFSPlanner so the BFS attempt is
                rendered to this GIF (single body: `<record_path>`; combined body:
                `<stem>_combined.gif`). BFSPlanner only renders on success, so the
                files may be absent on failure — callers must tolerate that.

        Returns:
            True iff BFSPlanner succeeds for both the tool-alone and tool+part-combined
            cases. The scratch directory is removed on exit.
        """
        if verbose:
            print("Checking if the tool can be applied without collisions...")

        active_ids = {o.id for o in active_objects}
        tool = Object(id="tool", path=None)
        tool.tri_mesh = tool_mesh
        # Parallel workers share the same on-disk assembly directory, so the
        # scratch sub-dir must be unique per call — otherwise one worker's
        # cleanup (shutil.rmtree in the outer finally) wipes files another
        # worker is still loading, causing FileNotFoundError races.
        import uuid as _uuid

        _scratch_tag = f"{os.getpid()}_{_uuid.uuid4().hex[:8]}"
        tool_planning_dir = Path(
            os.path.join(
                asset_folder,
                str(assembly_src_dir),
                f"tool_planning_{tool.id}_{_scratch_tag}",
            )
        )
        tool_planning_dir.mkdir(parents=True, exist_ok=True)

        try:
            for obj_file in Path(assembly_src_dir).glob("*.obj"):
                if obj_file.stem in active_ids:
                    shutil.copy(obj_file, tool_planning_dir / obj_file.name)

            tool.path = tool_planning_dir / f"{tool.id}.obj"
            tool_mesh.export(tool.path)

            try:
                path_planner = BFSPlanner(
                    asset_folder=asset_folder,
                    assembly_dir=str(tool_planning_dir),
                    move_ids=[tool.id],
                    still_ids=list(active_ids),
                    body_type="sdf",
                    force_mag=100,
                    save_sdf=True,
                )
                render_flag = bool(show or record_path is not None)
                status_single, _ = path_planner.plan(
                    max_time=max_time,
                    render=render_flag,
                    return_path=False,
                    record_path=str(record_path) if record_path is not None else None,
                )
            finally:
                for sdf_file in tool_planning_dir.glob("*.sdf"):
                    sdf_file.unlink()

            if verbose:
                print(
                    f"Path planner status for {tool.name}, tool only: {status_single}"
                )

            tool_combined = trimesh.boolean.union(
                [tool.tri_mesh, ass_obj.tri_mesh], engine="manifold"
            )
            tool_combined.export(tool.path)

            try:
                Path(tool_planning_dir / f"{ass_obj.id}.obj").unlink()
                path_planner = BFSPlanner(
                    asset_folder=asset_folder,
                    assembly_dir=str(tool_planning_dir),
                    move_ids=[tool.id],
                    still_ids=[oid for oid in active_ids if oid != ass_obj.id],
                    body_type="sdf",
                    force_mag=100,
                    save_sdf=True,
                )
                combined_record_path = None
                if record_path is not None:
                    p = Path(record_path)
                    combined_record_path = str(
                        p.with_name(f"{p.stem}_combined{p.suffix}")
                    )
                render_flag = bool(show or record_path is not None)
                status_combined, _ = path_planner.plan(
                    max_time=max_time,
                    render=render_flag,
                    return_path=False,
                    record_path=combined_record_path,
                )
            finally:
                for sdf_file in tool_planning_dir.glob("*.sdf"):
                    sdf_file.unlink()

            if verbose:
                print(
                    f"Path planner status for {tool.name}, tool and part combined: {status_combined}"
                )

        finally:
            if tool_planning_dir.exists():
                shutil.rmtree(tool_planning_dir)

        return status_single == "Success" and status_combined == "Success"

    @staticmethod
    def check_tool_pipeline(
        asset_folder,
        assembly_dir,
        parts,
        part_move,
        tools,
        output_dir=None,
        show=False,
        verbose=False,
        collision_mode="depth",
        check_assemblability=None,
        diagnostics=None,
        failure_record_dir=None,
    ):
        """Geometric tool feasibility for a generic (parts, tools) input.

        For each tool in `tools`, attempt to:
          1. Place the tool against `part_move` (regular orientation; if the placement
             collides with another active part, try the inverted orientation).
          2. Verify no collision with the other active parts (depth/boolean mode).
          3. (optional) Run BFS path planning on the tool alone and on the tool+part
             union, to confirm both can be removed from the assembly.

        Step 3 is the expensive one. It is controlled by ``settings.tool_assemblability``
        by default; pass ``check_assemblability=True``/``False`` to override per-call.

        The first tool that passes all enabled steps is returned. The pipeline avoids
        any VLM/LLM calls (no `check_tool_needed`, no orientation-validation calls),
        so it is safe to use during sequence planning. Tools must therefore have their
        `direction` and `contact_point` cached (otherwise the placement falls back
        to a Z-axis default).

        Args:
            asset_folder: redmax asset root used by BFSPlanner (typically
                `<base>/ATA/assets`).
            assembly_dir: directory containing the active parts' `.obj` files
                (matched by part-id stem).
            parts: list of active part IDs in the assembly; must include `part_move`.
            part_move: ID of the part the tool needs to remove.
            tools: list of `Tool` instances (already scaled to the assembly).
            output_dir: optional directory for intermediate placement screenshots.
            show, verbose: forwarded.
            collision_mode: "depth" or "boolean".
            check_assemblability: explicit override for the BFS-assemblability step;
                ``None`` (default) uses ``settings.tool_assemblability``.

        Returns:
            dict {'tool_id', 'tool_mesh', 'inverted'} on success, or None when no
            tool in `tools` is feasible.
        """
        if check_assemblability is None:
            check_assemblability = getattr(settings, "tool_assemblability", True)
        parts = [str(p) for p in parts]
        part_move = str(part_move)
        if part_move not in parts:
            raise ValueError(f"part_move {part_move!r} not in parts {parts!r}")

        assembly_dir = Path(assembly_dir)
        active_objects = []
        move_obj = None
        for part_id in parts:
            obj_path = assembly_dir / f"{part_id}.obj"
            if not obj_path.exists():
                raise FileNotFoundError(
                    f"Missing part file for id={part_id}: {obj_path}"
                )
            obj = Object(id=part_id, path=obj_path)
            active_objects.append(obj)
            if part_id == part_move:
                move_obj = obj
        assert move_obj is not None

        save_dir = Path(output_dir) if output_dir is not None else None
        fail_dir = Path(failure_record_dir) if failure_record_dir is not None else None
        if fail_dir is not None:
            fail_dir.mkdir(parents=True, exist_ok=True)

        if diagnostics is not None:
            diagnostics["tried_tools"] = []
            diagnostics["subkind"] = None
            diagnostics["colliding_parts"] = []
            diagnostics["evidence_paths"] = []

        for tool in tools:
            if verbose:
                print(
                    f"[check_tool_pipeline] Trying tool '{tool.id}' on part '{part_move}'"
                )
            if diagnostics is not None:
                diagnostics["tried_tools"].append(str(tool.id))

            # Step 1: place tool, regular orientation
            placement = ToolAnalyzer._apply_tool_geometric(
                tool,
                move_obj,
                save_dir=save_dir,
                show=False,
                invert=False,
            )
            if placement is None:
                if verbose:
                    print(
                        f"  Skipping '{tool.id}': placement failed (non-watertight mesh)."
                    )
                continue
            tool_mesh, _ = placement
            inverted = False

            # Step 2: collision check
            coll_save = (
                str(fail_dir / f"tool_{tool.id}_regular_collision.png")
                if fail_dir is not None
                else None
            )
            collisions = ToolAnalyzer._check_tool_collision_geometric(
                tool_mesh,
                active_objects,
                tool_name=str(tool.id),
                move_id=move_obj.id,
                show=False,
                mode=collision_mode,
                verbose=verbose,
                save_path=coll_save,
            )
            if collisions:
                if verbose:
                    print(
                        f"  '{tool.id}' collides in regular orientation; trying inverted."
                    )
                placement = ToolAnalyzer._apply_tool_geometric(
                    tool,
                    move_obj,
                    save_dir=save_dir,
                    show=False,
                    invert=True,
                )
                if placement is None:
                    if diagnostics is not None:
                        diagnostics["subkind"] = "collision"
                        for c in collisions:
                            if c not in diagnostics["colliding_parts"]:
                                diagnostics["colliding_parts"].append(c)
                        if coll_save is not None and os.path.exists(coll_save):
                            diagnostics["evidence_paths"].append(coll_save)
                    continue
                tool_mesh, _ = placement
                inverted = True
                coll_save_inv = (
                    str(fail_dir / f"tool_{tool.id}_inverted_collision.png")
                    if fail_dir is not None
                    else None
                )
                collisions = ToolAnalyzer._check_tool_collision_geometric(
                    tool_mesh,
                    active_objects,
                    tool_name=str(tool.id),
                    move_id=move_obj.id,
                    show=False,
                    mode=collision_mode,
                    verbose=verbose,
                    save_path=coll_save_inv,
                )
                if collisions:
                    if verbose:
                        print(f"  '{tool.id}' collides in both orientations; skipping.")
                    if diagnostics is not None:
                        diagnostics["subkind"] = "collision"
                        for c in collisions:
                            if c not in diagnostics["colliding_parts"]:
                                diagnostics["colliding_parts"].append(c)
                        if coll_save_inv is not None and os.path.exists(coll_save_inv):
                            diagnostics["evidence_paths"].append(coll_save_inv)
                    continue

            # Step 3: BFS path planning (optional — gated by settings.tool_assemblability).
            if not check_assemblability:
                if verbose:
                    print(
                        f"  '{tool.id}' placeable & collision-free; assemblability check skipped (inverted={inverted})."
                    )
                return {
                    "tool_id": tool.id,
                    "tool_mesh": tool_mesh,
                    "inverted": inverted,
                }

            access_record = (
                str(fail_dir / f"tool_{tool.id}_access.gif")
                if fail_dir is not None
                else None
            )
            if ToolAnalyzer._check_tool_assemblable_geometric(
                tool_mesh,
                move_obj,
                active_objects,
                asset_folder,
                assembly_dir,
                show=show,
                verbose=verbose,
                record_path=access_record,
            ):
                if verbose:
                    print(
                        f"  '{tool.id}' is geometrically feasible (inverted={inverted})."
                    )
                return {
                    "tool_id": tool.id,
                    "tool_mesh": tool_mesh,
                    "inverted": inverted,
                }
            else:
                if verbose:
                    print(f"  '{tool.id}' fails BFS path planning.")
                if diagnostics is not None:
                    # access failure overrides collision if the tool got past collision
                    diagnostics["subkind"] = "access"
                    if access_record is not None and os.path.exists(access_record):
                        diagnostics["evidence_paths"].append(access_record)

        return None


def load_scaled_tools(tools_dir, assembly_dir=None):
    """Load `Tool` instances from `tools_dir`, scaled to match an assembly's
    normalization. Each tool subdirectory must contain a `.obj` file and may
    contain a `normalization.json` with a `scale` key. Per-tool axes
    (`<id>_axes.json`) are lazy-loaded by the `Tool`/`Object` cached properties.

    Args:
        tools_dir: directory containing tool subdirectories (e.g. `<base>/assets/tools`).
        assembly_dir: optional assembly directory used to look up a sibling
            `normalization.json`. When omitted (or missing), tools are kept at
            their own normalized scale.

    Returns:
        list of `Tool` instances; `tri_mesh` and (if present) `contact_point` are
        rescaled so that the tool sits in the assembly's coordinate space.
    """
    tools_dir = Path(tools_dir)
    assembly_scale = 1.0
    if assembly_dir is not None:
        assembly_norm_file = Path(assembly_dir) / "normalization.json"
        if assembly_norm_file.exists():
            with open(assembly_norm_file) as f:
                assembly_scale = json.load(f).get("scale", 1.0)

    tools = []
    for tool_subdir in sorted(tools_dir.iterdir()):
        if not tool_subdir.is_dir():
            continue
        obj_files = sorted(f for f in tool_subdir.iterdir() if f.suffix == ".obj")
        if not obj_files:
            continue
        tool_file = obj_files[0]
        tool_norm_file = tool_subdir / "normalization.json"
        tool_scale = 1.0
        if tool_norm_file.exists():
            with open(tool_norm_file) as f:
                tool_scale = json.load(f).get("scale", 1.0)
        relative_scale = assembly_scale / tool_scale

        tool = Tool(
            id=tool_file.stem.lower(), path=tool_file, scaling_factor=relative_scale
        )
        scaled_mesh = tool.tri_mesh.copy()
        scale_transform = np.eye(4)
        scale_transform[:3, :3] *= relative_scale
        scaled_mesh.apply_transform(scale_transform)
        tool.__dict__["tri_mesh"] = scaled_mesh
        # JSON cache stores contact_point in the tool's own normalized space; scale it here.
        if tool.contact_point is not None:
            tool.__dict__["contact_point"] = tool.contact_point * relative_scale
        tools.append(tool)
    return tools
