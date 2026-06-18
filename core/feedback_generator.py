import json

import cv2
import matplotlib.pyplot as plt
import pyvista as pv
from openai import OpenAI

import settings
from core.llm import llm_chat
from core.models import (
    PartAnalysis,
    PartNameResponse,
    SelectiveStepFeedback,
    encode_image,
)

_SYSTEM_PART_NAMING = (
    "You are an expert mechanical engineer, machinist, and CAD specialist. "
    "Your task is to identify CAD parts from their rendered images using precise engineering terminology. "
    "Respond with exactly 1 to 3 words per part — the standard industry name "
    "(e.g. 'flange bearing', 'countersunk screw', 'L-bracket', 'spur gear', 'woodruff key'). "
    "Output ONLY the part name. No explanations, conversational filler, or punctuation."
)

_SYSTEM_PART_NAMING_COT = (
    "You are an expert mechanical engineer, machinist, and CAD specialist. "
    "Your task is to give one CAD part a precise standard engineering name "
    "that a technician could use to pick the correct part out of a pile.\n\n"
    "Work through the problem in this order before committing to a name:\n"
    "  1. ASSEMBLY PURPOSE — from the assembly-context render (with this part "
    "highlighted in red), infer what the overall assembled product or mechanism "
    "is. Be concrete; avoid generic answers like 'an assembly' or 'a structure'.\n"
    "  2. PART ROLE — what mechanical role does this specific part play in that "
    "product? (fastener, support, transmission element, cover, pivot, brace, seal, etc.)\n"
    "  3. LOCATION — where in the assembly does the part sit, and what does it "
    "interface with?\n"
    "  4. DISTINGUISHING FEATURES — what geometric features (holes, threads, "
    "slots, flanges, taper, profile) uniquely identify this part by name?\n"
    "  5. NAME — choose a 1-3 word description that reflects both "
    "the FUNCTION and the IDENTIFYING FEATURES. Use engineering terminology, "
    "if the part is a standard component. Prefer a name that lets the "
    "reader pick the correct part out of a pile just from the words.\n\n"
    "Output the structured JSON the schema requires; do not add commentary outside it."
)

_SYSTEM_FAILURE_ANALYST = (
    "You are an expert design engineer specializing in Design for Assembly. "
    "Your task is to analyze images of failed assembly attempts and identify the geometric root cause. "
    "You may ONLY recommend changes to part geometries — never to the assembly sequence or path, "
    "since it is guaranteed that no valid path exists for the given part geometries regardless of order.\n\n"
    "Be CONCISE. Reply with 2-4 short sentences total: one sentence stating the "
    "root cause grounded in the image, and 1-3 sentences proposing concrete "
    "geometric fixes. No headings, no bullet lists, no preamble."
)

_SYSTEM_DFA_SELECTIVE = (
    "You are an expert design engineer specialising in Design for Assembly (DFA) "
    "and human ergonomics. You review ONE assembly step and produce HIGH-SIGNAL, "
    "SELECTIVE feedback.\n\n"
    "Rules:\n"
    "1. Each feedback point must be NON-TRIVIAL and specific to this part's geometry "
    "or this step's situation. Skip generic DFA advice and self-evident remarks.\n"
    "2. If nothing actionable stands out, set needs_feedback=false and leave the "
    "points list empty. Do NOT pad with weak content.\n"
    "3. Categorise each point as 'observation' (something specific in the images), "
    "'dfa_issue' (a real DFA pitfall present here), or 'recommendation' (a concrete "
    "geometric change). Skip categories that don't apply.\n"
    "4. Reference visible part features or state numbers. Don't restate definitions.\n\n"
    "### DFA pitfalls to consider (cite only when actually present)\n"
    "1. Lack of clearance and access: blind insertions, no visual access to mating surfaces.\n"
    "2. Poor alignment features: flat-on-flat insertions, missing chamfers/tapers/pins.\n"
    "3. Ergonomic strain: awkward wrist angles, multi-axis simultaneous alignment.\n"
    "4. Ambiguous orientation: slight asymmetries that allow incorrect installation.\n"
    "5. Lack of self-retention: parts that fall before fastening.\n\n"
    "### Planning-failure evidence\n"
    "When the user message includes failure-mode evidence (stability collapse GIF, "
    "tool-collision PNG, tool-access blocked GIF, blocked-disassembly-path GIF, "
    "static collision overlay), tie your feedback to the SPECIFIC failure mode and "
    "the NAMED parts/IDs involved. Recommend geometric changes that would unblock the "
    "exact failure shown — do not speculate beyond what the evidence depicts."
)

_SYSTEM_INSTRUCTIONS = (
    "You are a technical writer specializing in assembly manuals for mechanical products. "
    "Your task is to write ONE short, high-level assembly instruction for a single step.\n\n"
    "The manual ALREADY shows direction, path and rotation visually (arrows, ghost "
    "positions, before/after frames). DO NOT REPEAT THE DIRECTION IN WORDS. The text "
    "must focus on the WHAT and the WHERE-IT-GOES, NOT the HOW-IT-MOVES.\n\n"
    "You receive:\n"
    "- The moving part's name\n"
    "- Optional: whether a rotation is required to seat the part\n"
    "- Optional: which other parts must be held fixed during this step\n"
    "- Optional: the tool required for this step\n"
    "- Key frames and (optional) trajectory matrices, for context only\n\n"
    "Rules for the output text:\n"
    "1. Write 1 to 3 short sentences a technician can read alongside the image.\n"
    "2. Name the moving part and name the INTERFACE on the rest of the assembly "
    "where it lands (e.g. 'the designated hole on the support bracket', 'the slot "
    "in the side panel', 'the seat on top of the base plate'). Make the final "
    "positioning clear in words.\n"
    "3. Stay HIGH-LEVEL — examples of the right register:\n"
    "   • 'Insert the screw into the designated hole of the support bracket.'\n"
    "   • 'Place the dowel into the matching socket on the bottom rail.'\n"
    "   • 'Fit the top rail onto the two side-panel pins.'\n"
    "4. DO NOT mention any global axis ('+X', '-Z'), any compass direction "
    "(downward, upward, from the left, from the side), any distance, any "
    "trajectory shape (diagonal, bent), or any percentage / numeric value.\n"
    "5. If a tool is required, mention it briefly (e.g. 'Fasten with a Phillips "
    "screwdriver.').\n"
    "6. If parts must be held fixed, name them briefly (e.g. 'Hold the base "
    "plate steady.').\n"
    "7. Output ONLY the instruction text — no preamble, no step number."
)


def convert_angle_pv_pos(angle):
    """Map angle keys to PyVista camera_position tuples.

    These mirror the camera_pos / camera_lookat passed to the sim renderer in
    sequence_planner._render_plan, so the manual page is rendered from the
    exact same viewpoint as the GIF that the SSIM ranking sees. The previous
    use of PyVista's built-in "iso" preset for iso1 produced a 90° rotation
    about Z because the preset places the camera at the (+x, +y, +z) corner
    while the sim places iso1 at (+x, -y, +z).
    """
    _positions = {
        # (camera_position, focal_point, view_up) — matched to sequence_planner.
        "iso1": [(1.25, -1.5, 1.5), (-1.0, 1.0, 0.0), (0, 0, 1)],
        "iso2": [(-1.25, 1.5, 1.5), (1.0, -1.0, 0.0), (0, 0, 1)],
        "iso3": [(-1.25, -1.5, 1.5), (1.0, 1.0, 0.0), (0, 0, 1)],
        "iso4": [(1.25, 1.5, 1.5), (-1.0, -1.0, 0.0), (0, 0, 1)],
    }
    return _positions.get(angle, "iso")


class FeedbackGenerator:
    """DFA feedback + part naming + instruction generation.

    Shared state (`assembly.instructions` dict, `_make_messages`, `_to_canonical`)
    lives on the parent Assembly. Manual generation lives in ManualGenerator.
    """

    def __init__(self, assembly):
        self.assembly = assembly

    @staticmethod
    def _format_failure_block(entry):
        """Render a `FailureEntry` as a short text block for the LLM prompt."""
        lines = [
            f"Planning-failure evidence for part '{entry.child_part}':",
            f"  Failure mode: {entry.fail_reason}",
        ]
        if entry.fail_reason == "stability" and entry.unstable_parts:
            lines.append(f"  Unstable / falling parts observed: {entry.unstable_parts}")
        if entry.fail_reason == "assembly" and entry.directions:
            n_dirs = len(entry.directions)
            n_failed = sum(1 for d in entry.directions if not d.get("success"))
            lines.append(
                f"  No valid disassembly motion: {n_failed}/{n_dirs} probe directions failed."
            )
            best = max(
                entry.directions, key=lambda d: d.get("path_len", 0), default=None
            )
            if best is not None:
                lines.append(
                    f"  Most-promising direction stalled after {best.get('path_len', 0)} steps along {best.get('action')}."
                )
        if entry.fail_reason == "tool":
            if entry.subkind:
                lines.append(
                    f"  Tool failure subkind: {entry.subkind} "
                    f"(collision = tool body overlaps surrounding parts; "
                    f"access = tool cannot be inserted/extracted without collision)."
                )
            if entry.colliding_parts:
                lines.append(f"  Tool-collision parts: {entry.colliding_parts}")
            if entry.tried_tools:
                lines.append(f"  Tools tried (all failed): {entry.tried_tools}")
        return "\n".join(lines)

    @staticmethod
    def _extract_gif_last_frame(gif_path):
        """Extract the last frame of a GIF as a PNG sibling file and return its
        path; returns None on failure. Used so the LLM sees a still image
        (image_url does not animate)."""
        from PIL import Image as _Image

        try:
            png_path = gif_path.with_suffix(gif_path.suffix + ".lastframe.png")
            if png_path.exists():
                return png_path
            with _Image.open(gif_path) as im:
                last_idx = 0
                try:
                    while True:
                        im.seek(im.tell() + 1)
                        last_idx += 1
                except EOFError:
                    pass
                im.seek(last_idx)
                im.convert("RGB").save(png_path)
            return png_path
        except Exception as e:
            print(f"_extract_gif_last_frame: failed for {gif_path}: {e}")
            return None

    def _render_part_in_context_to_file(self, part_id, save_path, size=(1024, 1024)):
        """Render the fully-assembled assembly with `part_id` highlighted in red
        and every other part in lightgray. Camera angle is taken from the
        corresponding step's `images` dict (already ranked by usefulness), with
        a fallback to 'iso' when no step / no ranked angle is available."""
        obj = self.assembly.objects[part_id]
        angle = "iso"
        step_nr = getattr(obj, "step_nr", None)
        if step_nr is not None and 0 <= step_nr < len(self.assembly.sequence):
            step_images = self.assembly.sequence[step_nr].images
            if step_images:
                angle = next(iter(step_images), "iso")

        plotter = pv.Plotter(off_screen=True, window_size=size)
        plotter.set_background("white")
        for o in self.assembly.objects.values():
            if o.id == part_id:
                plotter.add_mesh(o.tri_mesh, color="red", opacity=1.0)
            else:
                plotter.add_mesh(o.tri_mesh, color="lightgray", opacity=1.0)

        cam = convert_angle_pv_pos(angle)
        plotter.camera_position = cam
        if not isinstance(cam, str):
            plotter.reset_camera()
        plotter.screenshot(str(save_path))
        plotter.close()
        return angle

    def name_parts(self, iso_only=False, log_probs=False):
        # The non-iso_only branch uses the PartNameResponse chain-of-thought
        # schema (5 fields), which needs a much larger completion budget than
        # the single-name iso_only path.
        if iso_only:
            iso_args = {
                "model": "gpt-4o",
                "max_tokens": 50,
                "logprobs": True,
                "top_logprobs": 3,
            }
            cot_args = None
        else:
            iso_args = None
            cot_args = {"model": "gpt-4o", "max_tokens": 600}
            if log_probs:
                cot_args.update({"logprobs": True, "top_logprobs": 3})
        # Keep `prompt_args` for backward-compat with the iso_only block below.
        prompt_args = iso_args if iso_only else cot_args

        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(f"\nNaming {len(self.assembly.objects)} parts...")

        names_file_path = self.assembly.storage_dir / "part_names.json"
        if names_file_path.exists():
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Loading part names from {names_file_path}...")
            with open(names_file_path) as f:
                data = json.load(f)
                for obj_id, name in data.items():
                    if obj_id in self.assembly.objects:
                        self.assembly.objects[obj_id].name = name
            return True

        token_cost = 0

        if iso_only:
            if self.assembly.evaluation and self.assembly.evaluation.tokens_exhausted(
                "name_parts (iso_only)"
            ):
                return None

            user_content = [
                {
                    "type": "text",
                    "text": (
                        "Identify each of the following parts. "
                        "Provide one 1-3 word engineering name per part, in the order shown."
                    ),
                }
            ]
            for obj in self.assembly.objects.values():
                user_content.append({"type": "text", "text": f"Part {obj.id}"})
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(obj.image_paths['iso1'])}",
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
                    messages=self.assembly._make_messages(
                        _SYSTEM_PART_NAMING, user_content
                    ),
                    response_format=PartAnalysis,
                )
                if self.assembly.evaluation and self.assembly.evaluation.verbose:
                    print(response.choices[0].message.parsed)

                names_dict = {}
                for obj in self.assembly.objects.values():
                    obj.name = response.choices[0].message.parsed.obj_names[int(obj.id)]
                    names_dict[obj.id] = obj.name

                with open(names_file_path, "w") as f:
                    json.dump(names_dict, f)

                return response.choices[0].message.parsed
            except Exception as e:
                print(f"Error during OpenAI API call in name_parts: {e}")
                return None

        else:
            context_dir = self.assembly.storage_dir / "part_context_renders"
            context_dir.mkdir(parents=True, exist_ok=True)

            for obj in self.assembly.objects.values():
                if (
                    self.assembly.evaluation
                    and self.assembly.evaluation.tokens_exhausted(
                        f"name_parts (part {obj.id})"
                    )
                ):
                    obj.name = f"Part_{obj.id}"
                    continue

                user_content = [
                    {
                        "type": "text",
                        "text": (
                            "Identify this part by going through these four reasoning steps "
                            "BEFORE choosing a name, and fill out every field in the JSON schema:\n"
                            "  1. Look at the assembly-context render and infer what the OVERALL "
                            "PRODUCT or mechanism is.\n"
                            "  2. Decide what FUNCTIONAL ROLE this red-highlighted part plays in "
                            "that product.\n"
                            "  3. Note WHERE the part sits in the assembly and what it interfaces with.\n"
                            "  4. List the distinctive GEOMETRIC FEATURES of the part (from the "
                            "isolated views) that an engineer would use to recognise it.\n"
                            "Then pick a 1-3 word standard engineering name that reflects the "
                            "function and the identifying features."
                        ),
                    }
                ]
                user_content.append(
                    {"type": "text", "text": "Isolated views of the part:"}
                )
                for img in obj.image_paths.values():
                    if img is not None:
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(img)}"
                                },
                            }
                        )

                context_path = context_dir / f"{obj.id}_context.png"
                try:
                    self._render_part_in_context_to_file(obj.id, context_path)
                    user_content.append(
                        {
                            "type": "text",
                            "text": "Assembly context (this part highlighted in red):",
                        }
                    )
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_image(context_path)}"
                            },
                        }
                    )
                except Exception as e:
                    print(f"  context render failed for part {obj.id}: {e}")

                try:
                    response = llm_chat(
                        self.assembly.evaluation,
                        self.assembly.openai_api_key,
                        parse=True,
                        **prompt_args,
                        messages=self.assembly._make_messages(
                            _SYSTEM_PART_NAMING_COT, user_content
                        ),
                        response_format=PartNameResponse,
                    )
                    parsed = response.choices[0].message.parsed
                    response_string = parsed.name.strip().strip('"').strip("'")

                    if self.assembly.evaluation and self.assembly.evaluation.verbose:
                        print(f"[name_parts {obj.id}]")
                        print(f"  assembly_purpose:       {parsed.assembly_purpose}")
                        print(f"  part_role:              {parsed.part_role}")
                        print(f"  location:               {parsed.location}")
                        print(
                            f"  distinguishing_features:{parsed.distinguishing_features}"
                        )
                        print(f"  name:                   {response_string}")

                    obj.name = response_string
                    token_cost += response.usage.total_tokens

                except Exception as e:
                    print(
                        f"Error during OpenAI API call in name_parts for part {obj.id}: {e}"
                    )
                    obj.name = f"Part_{obj.id}"
                    print(f"Defaulting to name '{obj.name}' for part {obj.id}")

            names_dict = {obj.id: obj.name for obj in self.assembly.objects.values()}
            with open(names_file_path, "w") as f:
                json.dump(names_dict, f)

            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Total Token Cost for naming parts: {token_cost}")
            return True

    def generate_feedback_selective(self, step, show=False):
        """Selective DFA feedback: the LLM decides whether each potential point is
        worth raising and skips trivial / inapplicable categories instead of
        forcing a fixed three-section template per step. Returns the parsed
        ``SelectiveStepFeedback`` (or None if no images / API failure).

        Inputs to the model:
          - text prompt: DFA task framing augmented with
            the "only report non-trivial points" rule
          - the per-step state frames from ``step.images``, labelled in order
          - the isolated isometric render of the part (``obj.image_paths['iso1']``),
            labelled, for geometry reference
        """
        if step.images is None:
            self.assembly.planner.update_sequence(
                [s.obj_id for s in self.assembly.sequence]
            )
        assembly_images = next(iter(step.images.values()), []) if step.images else []

        planning_failures = self.assembly.planning_failures
        failure_entry = (
            planning_failures.for_part(step.obj_id) if planning_failures else None
        )

        if not assembly_images and failure_entry is None:
            print(
                f"generate_feedback_selective: no state images for step {step.obj_id}."
            )
            return None

        n_imgs = len(assembly_images)

        out_dir = self.assembly.output_dir / "selective_feedback"
        out_dir.mkdir(parents=True, exist_ok=True)

        if n_imgs > 0:
            fig, axes = plt.subplots(1, n_imgs, figsize=(4 * n_imgs, 4))
            if n_imgs == 1:
                axes = [axes]
            for idx, (ax, img_path) in enumerate(
                zip(axes, reversed(assembly_images), strict=False)
            ):
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(f"State {idx}")
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"step_{step.obj_id}_debug.png")
            if show:
                plt.show()
            plt.close(fig)

        obj = self.assembly.objects[step.obj_id]
        assemblable = step in self.assembly.sequence

        if assemblable:
            user_text = (
                f"Review the assembly step for the {obj.name}.\n"
                f"You see {n_imgs} state frames in increasing order; state {n_imgs - 1} is "
                f"the fully-assembled target. You also see an isolated isometric render of "
                f"the part itself for geometry reference.\n\n"
                "Report ONLY non-trivial, step-specific feedback. If nothing actionable "
                "stands out, set needs_feedback=false and return an empty points list."
            )
        elif n_imgs > 0:
            user_text = (
                f"Analyse this FAILED assembly attempt for the {obj.name}. "
                f"You see {n_imgs} state frames; state {n_imgs - 1} is "
                "the desired (but geometrically unreachable) final state. You also see an "
                "isolated isometric render of the part for geometry reference.\n\n"
                "Report ONLY non-trivial, step-specific feedback. Recommend GEOMETRY changes "
                "(never sequence/path changes — no valid path exists regardless of order). "
                "If the failure mode is captured by a single point, don't pad with extras."
            )
        else:
            user_text = (
                f"Analyse this FAILED assembly attempt for the {obj.name}. "
                "The sequence planner could not reach this part — "
                "no state frames are available, only the isolated geometry view and "
                "the planner's failure-mode evidence shown below.\n\n"
                "Report ONLY non-trivial, step-specific feedback. Recommend GEOMETRY changes "
                "(never sequence/path changes). If the failure mode is captured by a single "
                "point, don't pad with extras."
            )

        user_content = [{"type": "text", "text": user_text}]

        for i, img_path in enumerate(reversed(assembly_images)):
            user_content.append({"type": "text", "text": f"State {i}"})
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(img_path)}"
                    },
                }
            )

        iso_path = (obj.image_paths or {}).get("iso1") if obj.image_paths else None
        if iso_path:
            user_content.append(
                {
                    "type": "text",
                    "text": f"Isolated part view (iso) for the {obj.name}",
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(iso_path)}"
                    },
                }
            )

        if self.assembly.collisions:
            for (obj1, obj2), img_path in self.assembly.collisions.items():
                if step.obj_id not in (obj1, obj2):
                    continue
                other_id = obj2 if step.obj_id == obj1 else obj1
                other_name = self.assembly.objects[other_id].name
                user_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Static collision overlay: the {obj.name} and the {other_name} "
                            "overlap geometrically in the as-modeled assembly."
                        ),
                    }
                )
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(img_path)}"
                        },
                    }
                )

        if failure_entry is not None:
            text_block = self._format_failure_block(failure_entry)
            user_content.append({"type": "text", "text": text_block})
            ev_paths = list(failure_entry.all_evidence_images) or (
                [failure_entry.evidence_image] if failure_entry.evidence_image else []
            )
            for rel in ev_paths:
                resolved = planning_failures.resolve(rel) if planning_failures else None
                if resolved is None or not resolved.exists():
                    continue
                if resolved.suffix.lower() == ".gif":
                    frame_png = self._extract_gif_last_frame(resolved)
                    if frame_png is None:
                        continue
                    img_for_llm = frame_png
                else:
                    img_for_llm = resolved
                user_content.append(
                    {"type": "text", "text": f"Failure evidence ({rel}):"}
                )
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(img_for_llm)}"
                        },
                    }
                )

        if self.assembly.evaluation and self.assembly.evaluation.tokens_exhausted(
            f"generate_feedback_selective (step {step.obj_id})"
        ):
            return None

        try:
            response = llm_chat(
                self.assembly.evaluation,
                self.assembly.openai_api_key,
                parse=True,
                model=settings.LLM_model,
                messages=self.assembly._make_messages(
                    _SYSTEM_DFA_SELECTIVE, user_content
                ),
                response_format=SelectiveStepFeedback,
            )
            parsed = response.choices[0].message.parsed
        except Exception as e:
            print(f"Error during OpenAI API call in generate_feedback_selective: {e}")
            return None

        if not parsed.needs_feedback or not parsed.points:
            summary = f"Step {obj.name} ({step.obj_id}): no significant feedback."
            print("--- " + summary + " ---\n")
        else:
            lines = [f"Feedback for step {obj.name} ({step.obj_id}):"]
            for pt in parsed.points:
                lines.append(f"[{pt.category}] {pt.text}")
            summary = "\n".join(lines)
            print("--- " + summary + " ---\n")
            self.assembly.instructions["Feedback"].append(summary)

        text_payload = summary
        if failure_entry is not None:
            text_payload = (
                text_payload + "\n\n" + self._format_failure_block(failure_entry)
            )

        with open(out_dir / f"{step.obj_id}_feedback.txt", "w") as f:
            f.write(text_payload)

        graphic_path = out_dir / f"{step.obj_id}_feedback.png"
        try:
            self._compose_feedback_graphic(step, summary, graphic_path)
        except Exception as e:
            print(f"Failed to compose feedback graphic for {step.obj_id}: {e}")

        return parsed

    def _compose_feedback_graphic(
        self, step, feedback_text, save_path, canvas_width=1024
    ):
        """Compose a single PNG: the disassembled-state frame on top, the feedback
        text wrapped below it. Returns the save path on success."""
        from PIL import Image, ImageDraw, ImageFont

        assembly_images = next(iter(step.images.values()), []) if step.images else []
        if not assembly_images:
            return None

        disassembled_path = assembly_images[-1]

        img = Image.open(disassembled_path).convert("RGB")
        if img.width != canvas_width:
            scale = canvas_width / img.width
            img = img.resize(
                (canvas_width, max(1, int(img.height * scale))), Image.LANCZOS
            )
        img_h = img.height

        body = None
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]:
            try:
                body = ImageFont.truetype(p, 20)
                break
            except OSError:
                continue
        if body is None:
            body = ImageFont.load_default()

        margin = 30
        max_text_w = canvas_width - 2 * margin
        line_h = 28

        tmp_draw = ImageDraw.Draw(Image.new("RGB", (10, 10), "white"))

        wrapped = []
        for paragraph in (feedback_text or "").split("\n"):
            if not paragraph:
                wrapped.append("")
                continue
            words = paragraph.split()
            cur = ""
            for w in words:
                trial = (cur + " " + w).strip()
                if tmp_draw.textlength(trial, font=body) <= max_text_w:
                    cur = trial
                else:
                    if cur:
                        wrapped.append(cur)
                    cur = w
            if cur:
                wrapped.append(cur)
        if not wrapped:
            wrapped = [""]

        text_h = len(wrapped) * line_h + 2 * margin
        total_h = img_h + text_h

        canvas = Image.new("RGB", (canvas_width, total_h), "white")
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        y = img_h + margin
        for line in wrapped:
            draw.text((margin, y), line, fill="black", font=body)
            y += line_h

        canvas.save(save_path)
        return save_path

    # ------------------------------------------------------------------
    # Failure-mode feedback: one targeted item per detected failure,
    # covering all four DFA failure modes.  Used when the planner cannot
    # produce a complete disassembly sequence — replaces the per-step
    # iteration that the assemblable branch uses.
    # ------------------------------------------------------------------
    _FAILURE_MODES = ("collision", "assembly", "tool", "stability")

    def _name_of(self, pid):
        """obj_id → human name (defaults to id when no Object exists)."""
        if not pid:
            return str(pid)
        obj = self.assembly.objects.get(str(pid)) or self.assembly.objects.get(pid)
        return obj.name if obj is not None else str(pid)

    def _llm_failure_text(self, user_text, image_paths, cache_tag):
        """Wrap an LLM call with the shared FAILURE_ANALYST system prompt.
        image_paths: list of PNG / JPG paths the VLM sees.  Returns text or None."""
        if self.assembly.evaluation and self.assembly.evaluation.tokens_exhausted(
            f"failure_feedback ({cache_tag})"
        ):
            return None
        content = [{"type": "text", "text": user_text}]
        for p in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(p)}"},
                }
            )
        try:
            resp = llm_chat(
                self.assembly.evaluation,
                self.assembly.openai_api_key,
                model=settings.LLM_model,
                messages=self.assembly._make_messages(_SYSTEM_FAILURE_ANALYST, content),
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  LLM failure-feedback call failed ({cache_tag}): {e}")
            return None

    def _resolve_evidence_paths(self, entry, planning_failures):
        """Resolve each evidence reference on `entry` to an absolute Path.
        GIFs are converted to their last-frame PNG so the VLM gets a still."""
        from pathlib import Path as _Path

        out = []
        refs = list(entry.all_evidence_images) or (
            [entry.evidence_image] if entry.evidence_image else []
        )
        for rel in refs:
            if rel is None:
                continue
            resolved = (
                planning_failures.resolve(rel) if planning_failures else _Path(rel)
            )
            if resolved is None or not resolved.exists():
                continue
            if resolved.suffix.lower() == ".gif":
                frame = self._extract_gif_last_frame(resolved)
                if frame is not None:
                    out.append(frame)
            else:
                out.append(resolved)
        return out

    def _failure_collision(self, p1, p2, img_path, out_dir):
        n1 = self._name_of(p1)
        n2 = self._name_of(p2)
        user_text = (
            f"FAILURE: static overlap between two parts in their installed "
            f"positions.\n"
            f"The image contains FOUR sub-views of the same situation, "
            f"arranged in a 2x2 grid and labelled Iso (top-left), Top "
            f"(top-right), Side (bottom-left), Front (bottom-right). Cross-"
            f"reference all four to localize the overlap in 3D.\n"
            f"  - RED mesh   = {n1}\n"
            f"  - BLUE mesh  = {n2}\n"
            f"  - PURPLE volume (low opacity) = the intersection of the two "
            f"meshes, i.e. exactly the region where they overlap.\n\n"
            f"Think step by step (silently) before answering:\n"
            f"  1. Locate the purple overlap region across the Iso/Top/Side/"
            f"Front views — which feature of {n1} (RED) intersects which "
            f"feature of {n2} (BLUE)?\n"
            f"  2. Identify the geometric root cause (e.g. boss too tall, hole "
            f"too shallow, flange overshoots, missing clearance, wrong wall "
            f"thickness).\n"
            f"  3. Decide which part should yield, and propose a concrete, "
            f"minimal geometric change that eliminates the overlap without "
            f"breaking the part's function.\n\n"
            f"Then respond with the final answer only (no chain-of-thought, no "
            f"headings, no bullets): 1 sentence stating where they overlap and "
            f"the root cause, followed by 1-2 sentences with the concrete "
            f"geometric fix naming the part by name."
        )
        text = self._llm_failure_text(
            user_text, [img_path], cache_tag=f"collision_{p1}_{p2}"
        )
        if text is None:
            text = (
                f"Static collision: the {n1} and the {n2} overlap in their "
                f"installed positions. The geometry must be modified so the "
                f"parts no longer intersect."
            )
        graphic = out_dir / f"collision_{p1}_{p2}.png"
        try:
            self._compose_failure_graphic(img_path, text, graphic)
        except Exception as e:
            print(f"  failed to compose collision graphic: {e}")
            graphic = None
        return {
            "mode": "collision",
            "parts": [str(p1), str(p2)],
            "text": text,
            "image_path": str(img_path),
            "evidence_paths": [str(img_path)],
            "graphic_path": str(graphic) if graphic else None,
        }

    def _failure_assembly(self, entry, planning_failures, out_dir):
        name = self._name_of(entry.child_part)
        evidence = self._resolve_evidence_paths(entry, planning_failures)
        n_dirs = len(entry.directions)
        n_failed = sum(1 for d in entry.directions if not d.get("success"))
        best = max(entry.directions, key=lambda d: d.get("path_len", 0), default=None)
        best_action = best.get("action") if best else None
        best_len = best.get("path_len", 0) if best else 0
        user_text = (
            f"FAILURE: {name} cannot be extracted in any direction "
            f"({n_failed}/{n_dirs} probes failed; best was '{best_action}' "
            f"after {best_len} steps). Image shows it stuck. Briefly name the "
            f"blocking interference and propose a concrete geometric fix. "
            f"Geometry changes only — no sequence changes."
        )
        text = self._llm_failure_text(
            user_text, evidence, cache_tag=f"assembly_{entry.child_part}"
        )
        if text is None:
            text = (
                f"No valid disassembly path: the {name} is geometrically "
                f"blocked in every probed direction. Reshape interfering "
                f"features to open a clearance."
            )
        graphic = out_dir / f"assembly_{entry.child_part}.png"
        primary_img = evidence[0] if evidence else None
        if primary_img is not None:
            try:
                self._compose_failure_graphic(primary_img, text, graphic)
            except Exception as e:
                print(f"  failed to compose assembly graphic: {e}")
                graphic = None
        else:
            graphic = None
        return {
            "mode": "assembly",
            "parts": [str(entry.child_part)],
            "text": text,
            "image_path": str(primary_img) if primary_img else None,
            "evidence_paths": [str(p) for p in evidence],
            "graphic_path": str(graphic) if graphic else None,
        }

    def _failure_tool(self, entry, planning_failures, out_dir):
        name = self._name_of(entry.child_part)
        evidence = self._resolve_evidence_paths(entry, planning_failures)
        sub = entry.subkind or "unknown"
        colliding_names = [self._name_of(p) for p in entry.colliding_parts]
        tried = ", ".join(entry.tried_tools) if entry.tried_tools else "(none recorded)"
        sub_explain = {
            "collision": (
                "the tool BODY collides with surrounding parts when "
                "brought into position"
            ),
            "access": (
                "the tool cannot be inserted into or extracted from "
                "the work area without collision"
            ),
        }.get(sub, "the tool could not be applied")
        coll_str = (
            f"Tool collides with: {', '.join(colliding_names)}. "
            if colliding_names
            else ""
        )
        user_text = (
            f"FAILURE: tool can't be applied to {name} ({sub}). "
            f"All tried tools ({tried}) failed — {sub_explain}. {coll_str}"
            f"Image shows the blockage. Briefly identify what's in the way and "
            f"propose a concrete clearance fix (pocket, lowered wall, relocated "
            f"fastener, etc.)."
        )
        text = self._llm_failure_text(
            user_text, evidence, cache_tag=f"tool_{entry.child_part}_{sub}"
        )
        if text is None:
            text = (
                f"Tool cannot be applied to the {name} ({sub}). "
                f"Surrounding geometry must be modified to provide tool "
                f"clearance."
            )
        graphic = out_dir / f"tool_{entry.child_part}_{sub}.png"
        primary_img = evidence[0] if evidence else None
        if primary_img is not None:
            try:
                self._compose_failure_graphic(primary_img, text, graphic)
            except Exception as e:
                print(f"  failed to compose tool graphic: {e}")
                graphic = None
        else:
            graphic = None
        return {
            "mode": "tool",
            "parts": [str(entry.child_part)] + [str(p) for p in entry.colliding_parts],
            "text": text,
            "image_path": str(primary_img) if primary_img else None,
            "evidence_paths": [str(p) for p in evidence],
            "graphic_path": str(graphic) if graphic else None,
        }

    def _failure_stability(self, entry, planning_failures, out_dir):
        name = self._name_of(entry.child_part)
        unstable_names = [self._name_of(p) for p in entry.unstable_parts]
        unst_str = (
            ", ".join(unstable_names)
            if unstable_names
            else "one or more parts that previously rested in place"
        )

        # Use the existing PyVista renderer that highlights falling parts in
        # crimson over a lightgray context (rather than the last frame of the
        # stability GIF, which often shows parts already on the floor).
        unstable_png = out_dir / f"stability_{entry.child_part}_unstable.png"
        evidence = []
        try:
            import sys as _sys
            from pathlib import Path as _Path

            _asapx = _Path(__file__).resolve().parent / "ASAPx"
            # ATA and ASAPx both ship top-level packages named 'assets',
            # 'utils', 'plan_sequence', etc. Evict whichever copy is loaded
            # so the import below picks ASAPx's versions. Same pattern used
            # in sequence_planner._render_plan / get_assembly_plans_ASAP.
            _asapx_pkgs = {
                "assets",
                "utils",
                "simulation",
                "plan_path",
                "plan_robot",
                "plan_sequence",
                "settings",
            }
            for _mod in list(_sys.modules.keys()):
                if _mod in _asapx_pkgs or any(
                    _mod.startswith(p + ".") for p in _asapx_pkgs
                ):
                    del _sys.modules[_mod]
            if str(_asapx) not in _sys.path:
                _sys.path.insert(0, str(_asapx))
            else:
                # Re-prioritise so ASAPx wins over any ATA path that may
                # have been inserted earlier.
                _sys.path.remove(str(_asapx))
                _sys.path.insert(0, str(_asapx))
            from plan_sequence.planner._renders import render_unstable_parts

            parts_at_failure = [
                p for p in self.assembly.objects if p != entry.child_part
            ]
            png_path = render_unstable_parts(
                assembly_dir=str(self.assembly.assembly_dir),
                parts=parts_at_failure,
                observed_fallen=entry.unstable_parts or [],
                save_path=unstable_png,
            )
            if png_path is not None and _Path(png_path).exists():
                evidence = [_Path(png_path)]
        except Exception as e:
            print(f"  render_unstable_parts failed for {entry.child_part}: {e}")
        # Fallback to the original GIF-last-frame chain if the highlighted PNG
        # could not be produced.
        if not evidence:
            evidence = self._resolve_evidence_paths(entry, planning_failures)

        user_text = (
            f"FAILURE: at the {name} step, the sub-assembly is not self-"
            f"supporting — {unst_str} fall under gravity. Image shows the "
            f"remaining parts with the falling ones in CRIMSON. Briefly "
            f"explain why they fall and propose a concrete self-retaining "
            f"geometry change (interlock, detent, wider footprint, etc.)."
        )
        text = self._llm_failure_text(
            user_text, evidence, cache_tag=f"stability_{entry.child_part}"
        )
        if text is None:
            text = (
                f"Gravity instability at {name}: parts fall before the next "
                f"installation step. Add self-retaining geometry so the "
                f"sub-assembly stays together hands-free."
            )
        graphic = out_dir / f"stability_{entry.child_part}.png"
        primary_img = evidence[0] if evidence else None
        if primary_img is not None:
            try:
                self._compose_failure_graphic(primary_img, text, graphic)
            except Exception as e:
                print(f"  failed to compose stability graphic: {e}")
                graphic = None
        else:
            graphic = None
        return {
            "mode": "stability",
            "parts": [str(entry.child_part)] + [str(p) for p in entry.unstable_parts],
            "text": text,
            "image_path": str(primary_img) if primary_img else None,
            "evidence_paths": [str(p) for p in evidence],
            "graphic_path": str(graphic) if graphic else None,
        }

    def _compose_failure_graphic(self, image_path, text, save_path, canvas_width=1024):
        """Compose a single PNG: evidence image on top, wrapped feedback text below."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(image_path).convert("RGB")
        if img.width != canvas_width:
            scale = canvas_width / img.width
            img = img.resize(
                (canvas_width, max(1, int(img.height * scale))), Image.LANCZOS
            )
        body = None
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]:
            try:
                body = ImageFont.truetype(p, 20)
                break
            except OSError:
                continue
        if body is None:
            body = ImageFont.load_default()
        margin = 30
        line_h = 28
        max_text_w = canvas_width - 2 * margin
        tmp = ImageDraw.Draw(Image.new("RGB", (10, 10), "white"))
        wrapped = []
        for paragraph in (text or "").split("\n"):
            if not paragraph:
                wrapped.append("")
                continue
            cur = ""
            for w in paragraph.split():
                trial = (cur + " " + w).strip()
                if tmp.textlength(trial, font=body) <= max_text_w:
                    cur = trial
                else:
                    if cur:
                        wrapped.append(cur)
                    cur = w
            if cur:
                wrapped.append(cur)
        if not wrapped:
            wrapped = [""]
        text_h = len(wrapped) * line_h + 2 * margin
        canvas = Image.new("RGB", (canvas_width, img.height + text_h), "white")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        y = img.height + margin
        for line in wrapped:
            draw.text((margin, y), line, fill="black", font=body)
            y += line_h
        canvas.save(save_path)
        return save_path

    def generate_failure_feedback(self):
        """Failure-mode-centric feedback for a non-assemblable assembly.

        Walks every detected failure (static collisions + planning-failures)
        and produces one targeted feedback item per failure.  Each item:
          • is tied to a SPECIFIC failure mode (collision / assembly / tool /
            stability) and the SPECIFIC parts involved;
          • carries the LLM-generated text;
          • carries the evidence image path(s) the VLM was shown;
          • is also saved as a composed PNG (image-on-top + wrapped-text)
            under <output_dir>/failure_feedback/.

        Returns: list of dicts with keys
            mode, parts, text, image_path, evidence_paths, graphic_path.
        """
        out_dir = self.assembly.output_dir / "failure_feedback"
        out_dir.mkdir(parents=True, exist_ok=True)

        items = []

        # Mode 1: static geometric overlaps.
        collisions = getattr(self.assembly, "collisions", None) or {}
        for (p1, p2), img_path in collisions.items():
            try:
                item = self._failure_collision(p1, p2, img_path, out_dir)
                if item is not None:
                    items.append(item)
            except Exception as e:
                print(f"  collision feedback failed for ({p1}, {p2}): {e}")

        # Modes 2-4: planning-time failures.
        failures = getattr(self.assembly, "planning_failures", None)
        entries = list(getattr(failures, "failures", []) or []) if failures else []

        # Priority filter: tool / stability failures point at concrete physical
        # blockers (the tool can't reach, the sub-assembly collapses) and are
        # almost always the root cause of any 'assembly' (no-valid-direction)
        # entries that surface alongside them. If ANY tool or stability entry
        # is present in this deepest layer, drop the 'assembly' entries so we
        # only report the higher-priority root causes.
        priority_reasons = {"tool", "stability"}
        if any(e.fail_reason in priority_reasons for e in entries):
            dropped = [e for e in entries if e.fail_reason == "assembly"]
            entries = [e for e in entries if e.fail_reason != "assembly"]
            if (
                dropped
                and self.assembly.evaluation
                and self.assembly.evaluation.verbose
            ):
                print(
                    f"  failure_feedback: suppressing {len(dropped)} 'assembly' "
                    f"entries because higher-priority tool/stability "
                    f"failures are present in the same layer."
                )

        for entry in entries:
            try:
                if entry.fail_reason == "assembly":
                    item = self._failure_assembly(entry, failures, out_dir)
                elif entry.fail_reason == "tool":
                    item = self._failure_tool(entry, failures, out_dir)
                elif entry.fail_reason == "stability":
                    item = self._failure_stability(entry, failures, out_dir)
                else:
                    # 'grasp' or other reasons aren't currently visualised;
                    # surface them as text-only so the user at least sees them.
                    name = self._name_of(entry.child_part)
                    items.append(
                        {
                            "mode": entry.fail_reason or "unknown",
                            "parts": [str(entry.child_part)],
                            "text": (
                                f"Unhandled failure mode "
                                f"'{entry.fail_reason}' for the {name}."
                            ),
                            "image_path": None,
                            "evidence_paths": [],
                            "graphic_path": None,
                        }
                    )
                    continue
                if item is not None:
                    items.append(item)
            except Exception as e:
                print(
                    f"  planning-failure feedback failed for "
                    f"({entry.fail_reason}, {entry.child_part}): {e}"
                )

        # Persist a summary index alongside the individual graphics.
        try:
            import json as _json

            (out_dir / "failure_feedback.json").write_text(
                _json.dumps(items, indent=2, default=str)
            )
        except OSError as e:
            print(f"  failed to write failure_feedback.json: {e}")

        # Inject text into the assembly.instructions["Feedback"] list too so
        # downstream document compilers still see the content.
        for it in items:
            if it.get("text"):
                self.assembly.instructions.setdefault("Feedback", []).append(
                    f"[{it['mode'].upper()}] {it['text']}"
                )

        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(f"\n--- Failure-mode feedback: {len(items)} item(s) ---")
            for it in items:
                print(
                    f"  [{it['mode']}] parts={it['parts']}  "
                    f"image={it.get('image_path')}"
                )
        return items

    def generate_instructions(self):
        self.assembly.instructions["Steps"] = []
        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print("\nGenerating assembly instructions...")

        client = OpenAI(api_key=self.assembly.openai_api_key)
        full_text = self.assembly.instructions.get("Title", "")

        for i, step in enumerate(self.assembly.sequence):
            instruction = self.generate_instructions_from_paths(
                step, step_number=i + 1, client=client
            )
            if instruction:
                full_text += f"{i + 1}. {instruction}\n"
                self.assembly.instructions["Steps"].append(instruction)

        output_dir = self.assembly.output_dir / "assembly_instructions.txt"
        with open(output_dir, "w") as f:
            f.write(full_text)

        print(full_text)
        return full_text

    def generate_instructions_from_paths(
        self, step, step_number=None, client=None, camera_angle=None
    ):
        """Generate one assembly instruction for `step`.

        camera_angle: if given, the image frames sent to the LLM are taken
            from `step.images[camera_angle]` so they match the angle the
            manual renderer locked in. When None, the previous behaviour
            (first non-iso2 key in the per-step ranked dict) is used.
        """
        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(
                f"\nGenerating instruction for step {step_number} (part {step.obj_id})..."
            )

        obj = self.assembly.objects[step.obj_id]
        label = f"Step {step_number}: " if step_number else ""

        meta_lines = [f"{label}Part '{obj.name}'."]
        if step.parts_fix:
            names = [self.assembly.objects[p].name for p in step.parts_fix]
            meta_lines.append(
                f"Other parts present in the assembly: {', '.join(names)}."
            )
        if step.tool:
            meta_lines.append(f"Tool required: {step.tool}.")
        meta_lines.append(
            "Write the high-level assembly instruction for this step. "
            "Identify the INTERFACE on the rest of the assembly where the part lands "
            "(hole, slot, socket, seat, pin, etc.) by inspecting the image. "
            "Do not describe the motion direction in words — the image already shows it."
        )

        user_content = [{"type": "text", "text": "\n".join(meta_lines)}]

        if step.images:
            if camera_angle is not None and step.images.get(camera_angle):
                imgs = step.images[camera_angle]
            else:
                imgs = next(iter(step.images.values()), [])
        else:
            imgs = []
        key_frames = []
        if len(imgs) >= 2:
            key_frames = [
                (imgs[-1], "Before: part not yet in place"),
                (imgs[len(imgs) // 2], "During: mid-insertion"),
                (imgs[0], "After: part fully assembled"),
            ]
        elif imgs:
            key_frames = [(imgs[0], "Assembled state")]

        for img_path, frame_label in key_frames:
            user_content.append({"type": "text", "text": frame_label})
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(img_path)}",
                        "detail": "low",
                    },
                }
            )

        if self.assembly.evaluation and self.assembly.evaluation.tokens_exhausted(
            f"generate_instructions (step {step_number}, part {step.obj_id})"
        ):
            return None

        try:
            if client is None:
                client = OpenAI(api_key=self.assembly.openai_api_key)
            response = client.chat.completions.create(
                model=settings.LLM_model,
                messages=self.assembly._make_messages(
                    _SYSTEM_INSTRUCTIONS, user_content
                ),
            )

            if self.assembly.evaluation:
                self.assembly.evaluation.tokens_used += response.usage.total_tokens

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(
                f"Error during OpenAI API call in generate_instructions_from_paths: {e}"
            )
            return None
