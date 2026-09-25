import json

import numpy as np
import pyvista as pv
import trimesh
from openai import OpenAI

import settings
from core.feedback_generator import convert_angle_pv_pos
from core.models import AnnotationDecision

# Token estimate for one client.images.edit call (gpt-image-1 @ 1024x1024).
# images.edit doesn't report usage, so this is added manually after each call.
# Breakdown: ~150 prompt + ~1100/input image + ~1100-4160 output image.
_IMAGE_EDIT_TOKEN_ESTIMATE = 4000


_SYSTEM_ANNOTATION = (
    "You are an assembly-manual annotation designer. The page geometry (part outlines and "
    "assembly arrow) is already drawn correctly from projected mesh data — DO NOT try to alter "
    "geometry. You decide ONLY annotation metadata for one step:\n"
    "  - label_text: a short human-readable name for the moving part\n"
    "  - callout_position: where to place the label, avoiding the moving-part centroid by ~50 px "
    "and staying inside the canvas with at least 40 px margin\n"
    "  - show_tool_icon / tool_icon_position: include only if a tool is required\n"
    "  - add_zoom_inset: true only when the part is small or has fine-detail features that benefit "
    "from magnification\n"
    "  - step_note: one short instruction line if it adds value (null otherwise)\n"
    "Respect the provided canvas size. Coordinates are in pixels with origin at top-left."
)


class ManualGenerator:
    """Visual manual page generation (image_edit / wireframe / annotated / offline / iterative).

    Shared state (`assembly.instructions`, `_make_messages`, `_to_canonical`) lives on
    the parent Assembly. Step-instruction text reuse goes through a FeedbackGenerator
    reference passed at init time.
    """

    # Inset of panel content from the panel's rounded outline. The corner
    # radius is 24, so anything closer than this crowds the border arc.
    PANEL_PAD_X = 22
    PANEL_PAD_TOP = 20

    # Rotation-arrow geometry, as multiples of the parts' bounding-sphere
    # radius: the arc is drawn just outside the parts so it encircles them,
    # and the arrowhead extends past the arc's leading end.
    ARROW_ARC_RADIUS = 1.12
    ARROW_HEAD_LEN = 0.28
    # Radius the arrow panel has to keep in view. Derived from the two above so
    # resizing the arrow cannot silently leave it clipped or over-padded.
    ARROW_VIEW_RADIUS_FACTOR = ARROW_ARC_RADIUS + ARROW_HEAD_LEN + 0.10

    def __init__(self, assembly, feedback):
        self.assembly = assembly
        self.feedback = feedback

    def _present_ids(self, step_idx):
        """Return the set of part IDs present in the assembly for assembly step step_idx.

        Assembly order is the reverse of the disassembly sequence: step 0 of the
        assembly manual corresponds to the LAST disassembly step (fewest parts
        already installed).  The parts visible at this step are the current part
        plus all parts removed LATER in disassembly (already installed before
        this step in assembly), plus assembly.remaining (never disassembled).

        Under a subassembly plan the result is additionally clipped to the block
        the step belongs to.  A step inside S is performed while R is a separate
        body on the bench, so R must not appear in its context -- and because the
        plan orders the sequence prefix -> split -> S -> R, "removed at or after
        this step, and in this block" is exactly "installed so far in this
        block".  With no plan every step's block is the whole assembly and this
        reduces to the flat behaviour.
        """
        seq = self.assembly.sequence
        present = {seq[i].obj_id for i in range(step_idx, len(seq))}
        present |= self._base_ids()
        group = list(getattr(seq[step_idx], "subassembly", None) or [])
        return present & self._block_parts(group)

    # ------------------------------------------------------------------
    # Subassembly plan accessors
    # ------------------------------------------------------------------
    def _split_steps(self):
        """Flattened subassembly plan in disassembly order, or [] when the run
        produced none.  Entries are 'remove' (one per part, mapping onto
        assembly.sequence) and 'join' (a unified S/R mating, which has no Step
        because it is not a tree edge)."""
        return list(getattr(self.assembly, "split_steps", None) or [])

    def _block_parts(self, group):
        """Part IDs of the plan block at `group` (a path such as ["S", "R"]),
        including every nested sub-block.

        Returns every part in the assembly when there is no plan, when `group`
        is empty (the root block), or when the path does not resolve -- so
        callers can intersect with it unconditionally."""
        plan = getattr(self.assembly, "split_plan", None)
        all_ids = set(self.assembly.objects)
        if not plan or not group:
            return all_ids
        block = plan
        for side in group:
            nxt = block.get(side) if isinstance(block, dict) else None
            if not isinstance(nxt, dict):
                break
            block = nxt
        parts = block.get("parts") if isinstance(block, dict) else None
        return set(parts) if parts else all_ids

    @staticmethod
    def _shade(rgb, factor):
        """Blend `rgb` toward white (factor > 0) or black (factor < 0)."""
        if not factor:
            return tuple(int(c) for c in rgb)
        if factor > 0:
            shaded = (c + (255 - c) * factor for c in rgb)
        else:
            shaded = (c * (1.0 + factor) for c in rgb)
        return tuple(max(0, min(255, round(c))) for c in shaded)

    @staticmethod
    def _side_color(side):
        """The unshaded hue for a side. Used for TEXT, where the nesting tone
        is not worth the legibility: a level-1 light green on white is about
        2:1 contrast in thin glyphs, while the same tone reads fine as a 10px
        ring or a shaded 3D body. The label already spells the depth out
        ("Subassembly S-S"), so the tone would be redundant there anyway."""
        palette = getattr(settings, "subassembly_colors", None) or {}
        default = {"S": (26, 152, 80), "R": (123, 50, 148)}
        rgb = palette.get(side) or default.get(side)
        return tuple(int(c) for c in rgb) if rgb else None

    @classmethod
    def _group_colors(cls, group):
        """Outline colour per nesting level of `group`, outermost first.

        Two independent channels: settings.subassembly_colors maps the side
        ("S"/"R") to a hue, and settings.subassembly_shade_ladder maps the
        nesting level to a tone. So a step in S.R is framed base green outside
        (it is in S) and light purple inside (it is the R half of S) -- and a
        step in S.S gets base green outside and light green inside, which is
        what keeps two nested blocks of the same side apart."""
        palette = getattr(settings, "subassembly_colors", None) or {}
        default = {"S": (26, 152, 80), "R": (123, 50, 148)}
        ladder = tuple(
            getattr(settings, "subassembly_shade_ladder", None)
            or (0.0, 0.40, -0.40, 0.65, -0.65)
        )
        out = []
        for level, side in enumerate(group or []):
            rgb = palette.get(side) or default.get(side)
            if rgb is None:
                continue
            factor = ladder[min(level, len(ladder) - 1)] if ladder else 0.0
            out.append(cls._shade(rgb, factor))
        return out

    @staticmethod
    def _subassembly_label(group):
        """Human-readable block name, e.g. "Subassembly S" or "Subassembly S-R",
        or None for a step that belongs to no subassembly."""
        if not group:
            return None
        return "Subassembly " + "-".join(group)

    def _draw_subassembly_frame(self, canvas, group):
        """Draw one coloured ring per nesting level around the page.

        The rings are what mark a page as belonging to a subassembly: green for
        an S block, purple for an R block, outermost ring = outermost block.
        No-ops for a step outside any block."""
        colors = self._group_colors(group)
        if not colors:
            return
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        width = int(getattr(settings, "subassembly_border_width", 10))
        gap = int(getattr(settings, "subassembly_border_gap", 6))
        w, h = canvas.size
        for level, color in enumerate(colors):
            inset = level * (width + gap)
            # PIL strokes rectangles inward from the given box, so offsetting by
            # half the width would double-count; inset by the full ring pitch
            # and let each ring occupy its own band.
            draw.rectangle(
                [inset, inset, w - 1 - inset, h - 1 - inset],
                outline=tuple(color),
                width=width,
            )

    def _base_ids(self):
        """Part IDs that are never disassembled — the base the rest is built onto.

        SequencePlanner.update_sequence pops the last part of a complete
        disassembly into assembly.remaining (and puts every un-disassembled part
        there on a partial plan), so these parts are already on the bench before
        the first installation step and need an assembly page of their own.
        """
        seq_ids = {s.obj_id for s in self.assembly.sequence}
        return {s.obj_id for s in self.assembly.remaining} - seq_ids

    def _assembly_page_order(self):
        """Every manual page in assembly order, as descriptors.

        Entries are {"kind": "base"}, {"kind": "step", "step_idx": i} and
        {"kind": "join", "index": j} (an index into `_split_steps`).  Assembly
        order is the reverse of disassembly order, so under a subassembly plan
        the joins land exactly where they belong: after both halves have been
        built, before the block's prefix parts go on.

        Without a plan this is the base page followed by the steps in descending
        step_idx -- the ordering compile_manual_pdf has always used."""
        pages = []
        if self._base_ids():
            pages.append({"kind": "base"})

        entries = self._split_steps()
        if not entries:
            for step_idx in range(len(self.assembly.sequence) - 1, -1, -1):
                pages.append({"kind": "step", "step_idx": step_idx})
            return pages

        seq_idx = {step.obj_id: i for i, step in enumerate(self.assembly.sequence)}
        for j in range(len(entries) - 1, -1, -1):
            entry = entries[j]
            if entry.get("kind") == "join":
                pages.append({"kind": "join", "index": j})
                continue
            # Parts with no Step are the ones the disassembly left behind; the
            # base page already covers them.
            step_idx = seq_idx.get(entry.get("part"))
            if step_idx is not None:
                pages.append({"kind": "step", "step_idx": step_idx})
        return pages

    def _page_number(self, kind, key=None):
        """1-based position of a page in assembly order, or None if absent."""
        for number, page in enumerate(self._assembly_page_order(), start=1):
            if page["kind"] != kind:
                continue
            if kind == "base":
                return number
            if kind == "step" and page["step_idx"] == key:
                return number
            if kind == "join" and page["index"] == key:
                return number
        return None

    @staticmethod
    def _join_is_mating(entry):
        """True when the two halves of a join actually touch.

        The cut score rewards few contacts crossing the cut, so a split whose
        sides never touch scores well and is common -- 04489 splits into two
        legs joined only by a crossbar that is part of the prefix. There is no
        mating to perform on such a page: the halves are independent sub-builds
        that a later part bridges, and the page has to say so rather than tell
        the reader to seat one onto the other. Older plans without the count
        recorded fall back to assuming a real mating."""
        contacts = entry.get("contact_edges")
        return contacts is None or contacts > 0

    def _title_parts(self, number, group=None, join_group=None,
                     join_is_mating=True):
        """Coloured title segments for a page: the step number in black, then
        the subassembly it belongs to in that block's colour.

        `join_group` marks a join page, whose title names both halves in their
        own colours so the reader can match them to the bordered pages that
        built them."""
        parts = [(f"Step {number}" if number else "Step", (0, 0, 0))]
        if join_group is not None:
            parts.append(("  |  Join " if join_is_mating else "  |  Combine ",
                          (0, 0, 0)))
            parts.append(("S", self._side_color("S") or (0, 0, 0)))
            parts.append((" + ", (0, 0, 0)))
            parts.append(("R", self._side_color("R") or (0, 0, 0)))
            label = self._subassembly_label(join_group)
            if label and join_group:
                parts.append((f" of {label}", self._side_color(join_group[-1])
                              or (0, 0, 0)))
            return parts
        label = self._subassembly_label(group)
        if label and group:
            parts.append((f"  |  {label}", self._side_color(group[-1]) or (0, 0, 0)))
        return parts

    @staticmethod
    def _best_angle(step, skip=()):
        """Return the best-ranked angle from step.images that is not in `skip`.

        After Step.rank_angles() the dict is ordered best→worst by SSIM, so
        this returns the BEST non-skipped angle — i.e. the second-best when
        iso2 happens to rank first.

        Frame availability is INTENTIONALLY ignored: this picks a camera
        DIRECTION (a string label), and downstream consumers that need actual
        frames handle empty slots with their own fallback.
        """
        images = step.images or {}
        for angle in images:
            if angle not in skip:
                return angle
        for angle, frames in images.items():
            if frames:
                return angle
        return "iso1"

    # ------------------------------------------------------------------
    # image_edit backend (full color render -> VLM polish)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Wireframe backend (HLR render -> VLM polish)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Annotated backend (raw render + corner panel + VLM polish pass)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Initial-position necessity test
    # ------------------------------------------------------------------
    @staticmethod
    def _count_differing_pixels(img_x, img_y, tol=8):
        """P(x - y): how many pixels differ in colour between two renders.

        `tol` absorbs the anti-aliased boundary noise that survives even flat
        shading; a pixel genuinely covered by a different surface differs by far
        more than a few levels.
        """
        diff = np.abs(img_x.astype(np.int16) - img_y.astype(np.int16)).max(axis=-1)
        return int(np.count_nonzero(diff > tol))

    def _visible_fraction(
        self, step, present_ids=None, camera_angle=None, size=(1024, 768)
    ):
        """How much of the moving part's silhouette survives occlusion by the
        rest of the assembly at this step.

        Four flat-shaded probe renders are taken from ONE camera — actors are
        toggled and the camera is never reset between shots, so the framing is
        identical and the pixel counts are directly comparable:
            a  context only (every present part except the moving one)
            b  the moving part alone
            c  context + moving part
            d  empty scene (background only)
        With P(x - y) the number of differing pixels, P(c - a) is the part's
        visible area (pixels the part actually claims once everything else is
        drawn) and P(b - d) its unoccluded area, so P(c - a) / P(b - d) is the
        visible fraction.

        Lighting is off so each render is a flat silhouette: a pixel either
        belongs to the part or it does not, with no shading gradient for the
        difference test to trip over.

        The camera is fitted to the assembled state alone — no disassembled
        ghost — so the part is framed slightly larger here than on the manual
        page. Both counts scale with the framing, so the ratio is unaffected.

        Returns (visible_fraction, visible_px, total_px), with visible_fraction
        None when the part projects to too few pixels for the ratio to mean
        anything.
        """
        plotter = pv.Plotter(off_screen=True, window_size=size)
        plotter.set_background("white")
        pose = (
            np.asarray(step.pose, dtype=float) if step.pose is not None else np.eye(4)
        )

        context_actors = []
        for obj in self.assembly.objects.values():
            if obj.id == step.obj_id:
                continue
            if present_ids is not None and obj.id not in present_ids:
                continue
            mesh = obj.tri_mesh.copy().apply_transform(pose)
            context_actors.append(
                plotter.add_mesh(mesh, color="lightgray", opacity=1.0, lighting=False)
            )

        moving = self.assembly.objects[step.obj_id]
        moving_actor = plotter.add_mesh(
            moving.tri_mesh.copy().apply_transform(pose),
            color="blue",
            opacity=1.0,
            lighting=False,
        )

        plotter.camera_position = convert_angle_pv_pos(
            camera_angle if camera_angle is not None else "iso1"
        )
        plotter.reset_camera()

        def shot(show_context, show_moving):
            for actor in context_actors:
                actor.SetVisibility(show_context)
            moving_actor.SetVisibility(show_moving)
            # SetVisibility alone does not invalidate the rendered frame, so
            # without this every probe would screenshot the same image and both
            # pixel counts would come back 0.
            plotter.render()
            return np.asarray(plotter.screenshot(return_img=True))

        try:
            img_c = shot(True, True)
            img_a = shot(True, False)
            img_b = shot(False, True)
            img_d = shot(False, False)
        finally:
            plotter.close()

        visible_px = self._count_differing_pixels(img_c, img_a)
        total_px = self._count_differing_pixels(img_b, img_d)
        min_px = max(int(getattr(settings, "manual_visibility_min_pixels", 200)), 1)
        if total_px < min_px:
            return None, visible_px, total_px
        return visible_px / total_px, visible_px, total_px

    def _should_show_initial_position(self, step, present_ids=None, camera_angle=None):
        """Whether this step needs the disassembled-position ghost and the path
        trail, or reads clearly from the assembled position alone.

        A step is clear when enough of the moving part stays visible in the
        assembled state for a reader to see where it goes; only when the part
        largely disappears into the assembly does the page have to show where it
        comes from.  Any failure to measure falls back to showing them, so a
        broken render can never silently strip information off a page.

        Returns (show_initial, info); info records the measurement for the
        per-step log.
        """
        threshold = float(getattr(settings, "manual_visibility_threshold", 0.6))
        info = {"threshold": threshold}
        try:
            fraction, visible_px, total_px = self._visible_fraction(
                step, present_ids=present_ids, camera_angle=camera_angle
            )
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
            return True, info
        info["visible_px"] = visible_px
        info["total_px"] = total_px
        info["visible_fraction"] = fraction
        if fraction is None:
            info["reason"] = "part silhouette too small to measure"
            return True, info
        return fraction < threshold, info

    def _render_base_composite_to_file(
        self,
        step,
        save_path,
        size=(1024, 768),
        present_ids=None,
        camera_angle=None,
        include_motion=True,
    ):
        """Solid-surface composite rendered in the simulation's pose frame for
        this step (step.pose applied to every part).  Moving part in BLUE at
        its assembled position, the SAME moving part in RED at its disassembled
        position (transformed by step.matrices[-1], which is already in pose
        frame), and every other present part in lightgray.

        present_ids: if given, only parts whose id is in this set are rendered.
        camera_angle: explicit angle key (e.g. "iso1") to override per-step
            ranking; use this to keep a consistent viewing direction across steps.
        include_motion: if False, the red disassembled-position ghost AND the
            purple intermediate-trail dots are skipped — only the blue assembled
            position is drawn.  Set by the caller either from the per-step
            visibility test (see `_should_show_initial_position`, which turns
            them off for steps whose assembled position is plainly visible) or
            by the validator's ablation studies, to measure the contribution of
            the assembly-process visualisation."""
        plotter = pv.Plotter(off_screen=True, window_size=size)
        plotter.set_background("white")
        angle = (
            camera_angle
            if camera_angle is not None
            else (self._best_angle(step) if step.images else "iso1")
        )

        pose = (
            np.asarray(step.pose, dtype=float) if step.pose is not None else np.eye(4)
        )

        for obj in self.assembly.objects.values():
            if obj.id == step.obj_id:
                continue
            if present_ids is not None and obj.id not in present_ids:
                continue
            mesh = obj.tri_mesh.copy().apply_transform(pose)
            plotter.add_mesh(mesh, color="lightgray", opacity=1.0)

        moving = self.assembly.objects[step.obj_id]
        blue_mesh = moving.tri_mesh.copy().apply_transform(pose)
        plotter.add_mesh(blue_mesh, color="blue", opacity=1.0)

        matrices = step.matrices if step.matrices is not None else []
        if include_motion:
            # Optional path trail: small purple dots at the part's centre of mass
            # for each intermediate frame between assembled (blue) and disassembled
            # (red).  Frame indices 1..len-2 are sampled (endpoints excluded).
            if (
                getattr(settings, "manual_show_path_trail", False)
                and len(matrices) >= 3
            ):
                max_ghosts = int(getattr(settings, "manual_path_trail_max", 5))
                inner = list(range(1, len(matrices) - 1))
                if max_ghosts > 0 and len(inner) > max_ghosts:
                    idxs = (
                        np.linspace(0, len(inner) - 1, max_ghosts).round().astype(int)
                    )
                    inner = [inner[i] for i in idxs]
                com_local = np.asarray(moving.tri_mesh.center_mass, dtype=float)
                com_h = np.append(com_local, 1.0)
                ext = np.asarray(moving.tri_mesh.extents, dtype=float)
                dot_radius = 0.04 * float(np.linalg.norm(ext))
                for idx in inner:
                    T = np.asarray(matrices[idx], dtype=float)
                    world = (T @ com_h)[:3]
                    dot = pv.Sphere(radius=dot_radius, center=tuple(world.tolist()))
                    plotter.add_mesh(dot, color="purple", opacity=1.0)

            # step.matrices is already in pose frame, so apply it directly.
            # With no recorded motion there is no disassembled position to show:
            # drawing the ghost at `pose` would just stack a red copy on the
            # blue one. That happens for a part the flat render skipped, which a
            # subassembly plan can move out of the sequence's last slot.
            if len(matrices):
                red_mesh = moving.tri_mesh.copy().apply_transform(matrices[-1])
                plotter.add_mesh(red_mesh, color="red", opacity=1.0)

        cam = convert_angle_pv_pos(angle)
        plotter.camera_position = cam
        plotter.reset_camera()  # keeps direction, fits to present parts
        plotter.screenshot(str(save_path))
        plotter.close()

    @staticmethod
    def _load_fonts():
        """Best-effort font lookup; falls back to PIL default."""
        from PIL import ImageFont

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        body_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        title = body = big = None
        for p in candidates:
            try:
                title = ImageFont.truetype(p, 16)
                big = ImageFont.truetype(p, 22)
                break
            except OSError:
                continue
        for p in body_candidates:
            try:
                body = ImageFont.truetype(p, 14)
                break
            except OSError:
                continue
        if title is None:
            title = ImageFont.load_default()
        if body is None:
            body = ImageFont.load_default()
        if big is None:
            big = title
        return title, body, big

    @staticmethod
    def _load_italic_title_font():
        """Italic counterpart of the title font from _load_fonts, same weight and
        size. Returns None when no italic face is installed, so callers fall
        back to the upright title font rather than a mismatched default."""
        from PIL import ImageFont

        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
        ):
            try:
                return ImageFont.truetype(path, 16)
            except OSError:
                continue
        return None

    def _fetch_step_instruction(self, step, step_idx, step_dir, camera_angle=None):
        """Reuse FeedbackGenerator.generate_instructions_from_paths to produce a
        one-sentence instruction. Falls back to a simple template if the LLM
        call fails or is skipped by the token budget.

        camera_angle: forwarded so the LLM sees frames from the same camera
            angle the manual page is being rendered with."""
        instruction = None
        try:
            client = OpenAI(api_key=self.assembly.openai_api_key)
            instruction = self.feedback.generate_instructions_from_paths(
                step,
                step_number=step_idx + 1,
                client=client,
                camera_angle=camera_angle,
            )
        except Exception as e:
            print(f"  instruction generation failed: {e}")
        if not instruction:
            moving_name = self.assembly.objects[step.obj_id].name
            instruction = f"Install {moving_name}."
        (step_dir / "instruction.txt").write_text(instruction)
        return instruction

    def _draw_bottom_instruction_text(self, canvas, instruction, region, title=None,
                                      title_parts=None):
        """Draw word-wrapped instruction text in the bottom region (x1,y1,x2,y2).

        If title is given, it is drawn as a bold centered header above the body
        text.  `title_parts` -- a list of (text, rgb) segments -- overrides it
        and is drawn as one centered run, which is how a page states the
        subassembly it belongs to in that block's colour."""
        from PIL import ImageDraw

        x1, y1, x2, _y2 = region
        draw = ImageDraw.Draw(canvas)
        draw.line([(x1 + 20, y1), (x2 - 20, y1)], fill="black", width=2)

        title_font, _, big_font = self._load_fonts()
        cx = (x1 + x2) // 2
        y = y1 + 10

        if title_parts:
            total = sum(draw.textlength(t, font=title_font) for t, _ in title_parts)
            x = cx - total // 2
            for text, color in title_parts:
                draw.text((x, y), text, fill=tuple(color), font=title_font)
                x += draw.textlength(text, font=title_font)
            y += 30
        elif title:
            tw = draw.textlength(title, font=title_font)
            draw.text((cx - tw // 2, y), title, fill="black", font=title_font)
            y += 30

        max_width = (x2 - x1) - 40
        words = (instruction or "").split()
        lines = []
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            if draw.textlength(trial, font=big_font) <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        line_h = 28
        lines = lines[:5]
        for line in lines:
            w = draw.textlength(line, font=big_font)
            draw.text((cx - w // 2, y), line, fill="black", font=big_font)
            y += line_h

    # ------------------------------------------------------------------
    # Offline backend (same layout as annotated, no LLM/VLM calls)
    # ------------------------------------------------------------------

    def _render_isolated_mesh_solid_to_pil(
        self, mesh, size=(240, 140), color="lightgray"
    ):
        """Solid-shaded isometric render (matches the main composite's style:
        opaque colored surface, no wireframe overlay). Returns PIL RGBA."""
        import os
        import tempfile

        from PIL import Image

        plotter = pv.Plotter(off_screen=True, window_size=size)
        plotter.set_background("white")
        plotter.add_mesh(mesh, color=color, opacity=1.0)
        plotter.camera_position = "iso"
        plotter.reset_camera()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        plotter.screenshot(tmp_path)
        plotter.close()
        img = Image.open(tmp_path).convert("RGBA")
        os.unlink(tmp_path)
        return img

    @staticmethod
    def _relative_rotation_axis_angle(prev_pose, cur_pose):
        """Axis and angle of the rotation taking the previous step's orientation
        to the current step's, expressed in the world frame the rotation panel
        renders in.

        The panel draws parts at prev_pose, so the world-frame relative rotation
        R_rel = R_cur @ R_prev^T has exactly the axis that should be drawn in the
        render (equivalently: the axis in the previous step's own frame,
        R_prev^T @ R_cur, carried through prev_pose). Returns (unit_axis (3,),
        angle_rad) or None if the relative rotation is negligible/degenerate."""
        R_prev = np.asarray(prev_pose, dtype=float)[:3, :3]
        R_cur = np.asarray(cur_pose, dtype=float)[:3, :3]
        R = R_cur @ R_prev.T
        cos_angle = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        angle = float(np.arccos(cos_angle))
        if angle < np.radians(2.0):
            return None
        if angle > np.radians(178.0):
            # Near 180 deg the (Rz - Ry, ...) form vanishes; take the axis from
            # the dominant column of R + I (sign is inherently ambiguous here).
            A = R + np.eye(3)
            axis = A[:, int(np.argmax(np.linalg.norm(A, axis=0)))]
        else:
            axis = np.array(
                [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
                dtype=float,
            )
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            return None
        return axis / n, angle

    def _add_rotation_arrow(self, plotter, center, radius, axis, angle, color="red"):
        """Add a curved rotation arrow (arc + arrowhead) about `axis` through
        `center`, sized to `radius`. The arc sweeps in the positive
        (right-hand-rule) direction about `axis`, so it conveys the sense of the
        reorientation; the magnitude is only clamped for legibility and no
        numeric value is drawn."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        center = np.asarray(center, dtype=float)

        # Orthonormal basis (u, v) of the plane perpendicular to axis, with
        # (u, v, axis) right-handed so increasing t rotates u -> v about +axis.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, axis))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, ref)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(axis, u)

        # Encircle the parts (arc just outside the bounding sphere) so the arrow
        # reads clearly; the far side is naturally occluded, conveying depth.
        arc_r = self.ARROW_ARC_RADIUS * radius
        # Honour the true direction; clamp the drawn sweep so it stays legible.
        sweep = float(np.clip(angle, np.radians(80.0), np.radians(300.0)))
        ts = np.linspace(0.0, sweep, 48)
        pts = center + arc_r * (np.outer(np.cos(ts), u) + np.outer(np.sin(ts), v))
        plotter.add_mesh(pv.Spline(pts, 48).tube(radius=0.05 * radius), color=color)

        # Arrowhead (cone) at the leading end, tangent to the arc.
        tangent = -np.sin(sweep) * u + np.cos(sweep) * v
        tangent /= np.linalg.norm(tangent) + 1e-12
        tip_base = center + arc_r * (np.cos(sweep) * u + np.sin(sweep) * v)
        head_len = self.ARROW_HEAD_LEN * radius
        plotter.add_mesh(
            pv.Cone(
                center=tip_base + tangent * (head_len / 2.0),
                direction=tangent,
                height=head_len,
                radius=0.13 * radius,
                resolution=24,
            ),
            color=color,
        )

    def _render_rest_of_assembly_to_pil(
        self,
        step,
        size=(240, 200),
        color="lightgray",
        present_ids=None,
        pose_override=None,
        camera_angle=None,
        rotation_axis=None,
        rotation_angle=None,
    ):
        """Render every part of the assembly EXCEPT the currently-moving part,
        using the same solid-surface styling as the main composite (no wireframe).
        Returns a PIL RGBA image (transparent if no parts remain).

        present_ids: if given, only parts whose id is in this set are rendered.
        pose_override: if given, use this 4x4 matrix as the pose for every part
            instead of step.pose (used by the rotation panel to show the previous
            assembly step's pose — i.e., the orientation BEFORE the rotation).
        camera_angle: if given, an angle key (e.g. "iso1") mapped via
            convert_angle_pv_pos to the same viewpoint as the main composite;
            otherwise the neutral "iso" preset is used.
        rotation_axis / rotation_angle: if given, a unit axis (in this render's
            world frame) and angle (rad); a curved rotation arrow about that axis
            through the assembly centroid is added, showing the reorientation
            sense. Only used by the rotation panel; other callers leave it off so
            their renders are unchanged."""
        import os
        import tempfile

        from PIL import Image

        plotter = pv.Plotter(off_screen=True, window_size=size)
        plotter.set_background("white")
        if pose_override is not None:
            pose = np.asarray(pose_override, dtype=float)
        elif step.pose is not None:
            pose = np.asarray(step.pose, dtype=float)
        else:
            pose = np.eye(4)
        added = 0
        all_min = all_max = None
        for obj in self.assembly.objects.values():
            if obj.id == step.obj_id:
                continue
            if present_ids is not None and obj.id not in present_ids:
                continue
            mesh = obj.tri_mesh.copy().apply_transform(pose)
            plotter.add_mesh(mesh, color=color, opacity=1.0)
            b = np.asarray(mesh.bounds, dtype=float)  # (2, 3): [min; max]
            if all_min is None:
                all_min, all_max = b[0].copy(), b[1].copy()
            else:
                all_min = np.minimum(all_min, b[0])
                all_max = np.maximum(all_max, b[1])
            added += 1
        if added == 0:
            plotter.close()
            return Image.new("RGBA", size, (255, 255, 255, 0))
        plotter.camera_position = (
            convert_angle_pv_pos(camera_angle) if camera_angle is not None else "iso"
        )
        if rotation_axis is not None and all_min is not None:
            center = (all_min + all_max) / 2.0
            radius = 0.5 * float(np.linalg.norm(all_max - all_min))
            if radius > 1e-9:
                try:
                    self._add_rotation_arrow(
                        plotter,
                        center,
                        radius,
                        rotation_axis,
                        rotation_angle if rotation_angle is not None else np.pi / 2,
                    )
                except Exception as e:
                    print(f"  rotation arrow render failed: {e}")
                # Frame to a fixed multiple of the PARTS' extent (a cube around
                # their centroid), independent of the arrow's bounds. This keeps
                # part size consistent between steps regardless of the arrow's
                # axis or sweep, while leaving room for the encircling arc.
                #
                # reset_camera(bounds) frames the box's bounding SPHERE, i.e.
                # half its diagonal -- a cube of half-side h shows a radius of
                # h*sqrt(3), not h. Size the cube from the radius we actually
                # want in view so the parts land at the same scale as the plain
                # reset_camera() the arrow-less panels use, rather than a
                # further 1.7x zoom-out.
                view_radius = self.ARROW_VIEW_RADIUS_FACTOR * radius
                pad = view_radius / np.sqrt(3.0)
                plotter.reset_camera(
                    bounds=[
                        center[0] - pad,
                        center[0] + pad,
                        center[1] - pad,
                        center[1] + pad,
                        center[2] - pad,
                        center[2] + pad,
                    ]
                )
            else:
                plotter.reset_camera()
        else:
            plotter.reset_camera()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        plotter.screenshot(tmp_path)
        plotter.close()
        img = Image.open(tmp_path).convert("RGBA")
        os.unlink(tmp_path)
        return img

    def _draw_offline_rotation_panel(
        self,
        step,
        panel_size=(280, 380),
        present_ids=None,
        prev_pose=None,
        camera_angle=None,
        rotation_axis=None,
        rotation_angle=None,
    ):
        """Left corner panel: rotation widget over a wireframe render of the
        already-installed parts (everything except the moving part), plus the
        hold list. Caller guarantees step.rotated and step.pose.

        prev_pose: if given, render the present parts in this pose (the previous
            assembly step's pose) to show the orientation BEFORE the rotation.
        camera_angle: the step's chosen angle key, forwarded to the render so the
            panel shares the main composite's viewpoint instead of a fixed one.
        rotation_axis / rotation_angle: previous->current rotation (axis in the
            previous step's rendered frame, angle in rad); forwarded to draw the
            rotation arrow over the parts."""
        from PIL import Image, ImageDraw

        panel = Image.new("RGBA", panel_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            [0, 0, panel_size[0] - 1, panel_size[1] - 1],
            radius=24,
            fill=(255, 255, 255, 240),
            outline="black",
            width=2,
        )
        title_font, body_font, _ = self._load_fonts()
        from PIL import ImageFont

        hold_font = body_font
        for p in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ):
            try:
                hold_font = ImageFont.truetype(p, 18)
                break
            except OSError:
                continue

        pad = self.PANEL_PAD_X
        y = self.PANEL_PAD_TOP
        icon_size = (panel_size[0] - 2 * pad, 200)

        draw.text((pad, y), "Reorientation from", fill="black", font=title_font)
        y += 20
        draw.text((pad, y), "previous step:", fill="black", font=title_font)
        y += 22
        try:
            rest_img = self._render_rest_of_assembly_to_pil(
                step,
                size=icon_size,
                present_ids=present_ids,
                pose_override=prev_pose,
                camera_angle=camera_angle,
                rotation_axis=rotation_axis,
                rotation_angle=rotation_angle,
            )
            panel.paste(rest_img, (pad, y), rest_img)
        except Exception as e:
            print(f"  rest-of-assembly icon render failed: {e}")
        y += icon_size[1] + 10

        if step.parts_fix:
            draw.text((pad, y), "Hold:", fill="black", font=title_font)
            y += 24
            names = []
            for pid in step.parts_fix:
                obj = self.assembly.objects.get(str(pid)) or self.assembly.objects.get(
                    pid
                )
                names.append(obj.name if obj else str(pid))
            for name in names[:4]:
                draw.text((pad + 10, y), f"• {name}", fill="black", font=hold_font)
                y += 22
            if len(names) > 4:
                draw.text(
                    (pad + 10, y),
                    f"... +{len(names) - 4} more",
                    fill="gray",
                    font=hold_font,
                )
        return panel

    def _paste_mesh_icon(self, panel, draw, mesh, pos, size, body_font):
        """Render `mesh` isometrically (same solid-surface styling as the main
        composite) and paste it into `panel` at `pos`; on a missing mesh or a
        render failure, draw a gray placeholder instead. Returns the y just
        below the icon area."""
        x, y = pos
        if mesh is not None:
            try:
                icon = self._render_isolated_mesh_solid_to_pil(mesh, size=size)
                panel.paste(icon, (x, y), icon)
            except Exception as e:
                print(f"  icon render failed: {e}")
                draw.text((x, y), "(icon render failed)", fill="gray", font=body_font)
        else:
            draw.text((x, y), "(no mesh available)", fill="gray", font=body_font)
        return y + size[1]

    def _draw_offline_part_panel(self, step, panel_size=(280, 380)):
        """Right corner panel: the part being installed this step (name + solid
        isometric render), and — when the step requires a tool — the tool (name
        + render) stacked below it. Same solid-surface styling as the main
        composite (no wireframe)."""
        from PIL import Image, ImageDraw

        panel = Image.new("RGBA", panel_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            [0, 0, panel_size[0] - 1, panel_size[1] - 1],
            radius=24,
            fill=(255, 255, 255, 240),
            outline="black",
            width=2,
        )
        title_font, body_font, _ = self._load_fonts()

        pad = self.PANEL_PAD_X
        icon_w = panel_size[0] - 2 * pad
        has_tool = bool(step.tool)
        # Two stacked sections (part + tool) use shorter icons so both fit the
        # fixed panel height; a part-only panel keeps the full-height icon.
        icon_h = 130 if has_tool else 200

        y = self.PANEL_PAD_TOP

        # Part section (always shown). The name stands on its own -- the panel
        # is the part panel, so a "Part:" label only repeats that -- and is set
        # in italics to read as a name rather than a heading.
        part = self.assembly.objects.get(step.obj_id)
        part_name = part.name if part is not None else str(step.obj_id)
        draw.text(
            (pad, y),
            part_name,
            fill="black",
            font=self._load_italic_title_font() or title_font,
        )
        y += 24
        y = self._paste_mesh_icon(
            panel,
            draw,
            getattr(part, "tri_mesh", None),
            (pad, y),
            (icon_w, icon_h),
            body_font,
        )

        # Tool section (only when the step requires a tool).
        if has_tool:
            y += 8
            draw.text((pad, y), f"Tool: {step.tool}", fill="black", font=title_font)
            y += 24
            scaled = getattr(self.assembly, "scaled_tools", None) or {}
            tool_obj = scaled.get(step.tool)
            tool_mesh = (
                tool_obj.tri_mesh
                if tool_obj is not None and hasattr(tool_obj, "tri_mesh")
                else None
            )
            y = self._paste_mesh_icon(
                panel, draw, tool_mesh, (pad, y), (icon_w, icon_h), body_font
            )

        return panel

    def _render_parts_neutral_to_file(
        self, obj_ids, pose, save_path, size=(1024, 768), camera_angle="iso1"
    ):
        """Render the given parts in lightgray with no blue/red position markers,
        in the given pose frame.  Used for the initial-state assembly page, where
        nothing is being installed yet."""
        plotter = pv.Plotter(off_screen=True, window_size=size)
        plotter.set_background("white")
        pose = np.asarray(pose, dtype=float) if pose is not None else np.eye(4)
        for obj_id in obj_ids:
            mesh = self.assembly.objects[obj_id].tri_mesh.copy().apply_transform(pose)
            plotter.add_mesh(mesh, color="lightgray", opacity=1.0)
        plotter.camera_position = convert_angle_pv_pos(camera_angle)
        plotter.reset_camera()
        plotter.screenshot(str(save_path))
        plotter.close()

    @staticmethod
    def _rgb_hex(rgb):
        r, g, b = (int(c) for c in rgb)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _render_join_composite_to_file(
        self,
        s_ids,
        r_ids,
        group,
        pose,
        save_path,
        size=(1024, 768),
        camera_angle="iso1",
    ):
        """Render the mating of two subassemblies: both at their joined
        position, each in its own side colour and nesting tone.

        Deliberately NO red pre-assembly ghost, unlike the per-part step pages.
        On a step page the red copy is the one part being installed shown at its
        starting position, which reads clearly.  Here it would be a whole second
        subassembly floating off to one side, and readers took it for a third
        body rather than "R, before it goes on".  The two coloured bodies plus
        the instruction line carry the step on their own.

        `group` is the block path of the join, so each half is coloured one
        level deeper -- matching the inner ring of the pages that built it."""
        plotter = pv.Plotter(off_screen=True, window_size=size)
        plotter.set_background("white")

        pose = np.asarray(pose, dtype=float) if pose is not None else np.eye(4)
        group = list(group or [])
        s_color = self._rgb_hex(
            (self._group_colors([*group, "S"]) or [(26, 152, 80)])[-1]
        )
        r_color = self._rgb_hex(
            (self._group_colors([*group, "R"]) or [(123, 50, 148)])[-1]
        )

        def posed(obj_id):
            return self.assembly.objects[obj_id].tri_mesh.copy().apply_transform(pose)

        s_meshes = [posed(i) for i in s_ids if i in self.assembly.objects]
        r_meshes = [posed(i) for i in r_ids if i in self.assembly.objects]
        if not s_meshes or not r_meshes:
            return False

        for mesh in s_meshes:
            plotter.add_mesh(mesh, color=s_color, opacity=1.0)
        for mesh in r_meshes:
            plotter.add_mesh(mesh, color=r_color, opacity=1.0)

        plotter.camera_position = convert_angle_pv_pos(camera_angle)
        plotter.reset_camera()
        plotter.screenshot(str(save_path))
        plotter.close()
        return True

    def generate_manual_join_offline(self, join_index, ablation="full"):
        """Manual page for one unified S/R mating.

        A join is not a tree edge -- no part moves on its own -- so it has no
        Step and no rendered GIF.  The page is built straight from the meshes,
        in the frame of the page that follows it in assembly order (the first
        entry of the block's own sequence in disassembly order), so the reader's
        viewpoint carries over.

        Returns the page path, or None when the join cannot be rendered."""
        from PIL import Image

        entries = self._split_steps()
        if join_index >= len(entries):
            return None
        entry = entries[join_index]
        if entry.get("kind") != "join":
            return None

        s_ids = [i for i in (entry.get("S") or []) if i in self.assembly.objects]
        r_ids = [i for i in (entry.get("R") or []) if i in self.assembly.objects]
        if not s_ids or not r_ids:
            return None

        group = list(entry.get("group") or [])
        save_dir = self.assembly.output_dir / "manual"
        save_dir.mkdir(parents=True, exist_ok=True)
        step_dir = save_dir / f"step_join_{join_index}_offline"
        step_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if ablation == "full" else f"_{ablation}"

        # Share the frame of the neighbouring page: in assembly order the join
        # is immediately preceded by the last step of the S block, which in
        # disassembly order is the first 'remove' entry after this join.
        by_id = {step.obj_id: step for step in self.assembly.sequence}
        anchor = None
        for follower in entries[join_index + 1 :]:
            if follower.get("kind") != "remove":
                continue
            anchor = by_id.get(follower.get("part"))
            if anchor is not None:
                break
        pose = anchor.pose if anchor is not None else None
        if ablation == "no_angle_ranking" or anchor is None:
            camera_angle = "iso1"
        else:
            camera_angle = self._best_angle(anchor) if anchor.images else "iso1"

        base_path = step_dir / f"01_base_render{suffix}.png"
        ok = self._render_join_composite_to_file(
            s_ids,
            r_ids,
            group,
            pose,
            base_path,
            size=(1024, 768),
            camera_angle=camera_angle,
        )
        if not ok:
            return None

        s_label = self._subassembly_label([*group, "S"]) or "subassembly S"
        r_label = self._subassembly_label([*group, "R"]) or "subassembly R"
        is_mating = self._join_is_mating(entry)
        if is_mating:
            instruction = (
                f"Fit {r_label} ({len(r_ids)} parts) onto {s_label} "
                f"({len(s_ids)} parts) and seat the two together."
            )
        else:
            instruction = (
                f"Set {r_label} ({len(r_ids)} parts) and {s_label} "
                f"({len(s_ids)} parts) in the relative position shown. They do "
                f"not touch yet; the parts added next join them."
            )
        (step_dir / "instruction.txt").write_text(instruction)

        canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        canvas.paste(Image.open(base_path).convert("RGBA"), (0, 0))
        self._draw_bottom_instruction_text(
            canvas,
            "" if ablation == "no_text" else instruction,
            region=(0, 768, 1024, 1024),
            title_parts=self._title_parts(
                self._page_number("join", join_index), join_group=group,
                join_is_mating=is_mating,
            ),
        )
        self._draw_subassembly_frame(canvas, group)
        canvas_rgb = canvas.convert("RGB")

        output_top = save_dir / f"join_{join_index}_manual_offline{suffix}.png"
        canvas_rgb.save(output_top)
        if ablation == "full":
            canvas_rgb.save(step_dir / "final.png")
            self.assembly.instructions["Manual"].append(str(output_top))
        return str(output_top)

    # Supported ablation flags for generate_manual_offline (used by the
    # manual_validator's per-component evaluation).
    ABLATIONS = ("full", "no_angle_ranking", "no_text", "no_motion")

    def generate_manual_offline(self, step_idx, ablation="full"):
        """Fully offline manual page: composite render + programmatic corner
        panel + LLM instruction sentence.

        Every disassembly step is a real installation page.  The initial state —
        the parts left over by the disassembly (assembly._base_ids) — gets its own
        page from generate_manual_base_offline, which this method triggers once,
        on the last disassembly step.  Only when the disassembly left nothing
        behind does the last step itself become the simplified initial-state page
        (a neutral single-part view with a short fixed caption and no corner
        panels).

        ablation: one of self.ABLATIONS.  Controls which features are disabled
            in this rendering so the validator can measure their contribution:
              "full"             — baseline; all features on.
              "no_angle_ranking" — force camera_angle="iso1" instead of using
                                   the SSIM-best angle for this step.
              "no_text"          — bottom-region instruction text is blanked
                                   (title is still shown so the page reads as
                                   "Step N", just with no sentences).
              "no_motion"        — base composite omits the red disassembled-
                                   position ghost and the purple intermediate
                                   path trail, on every step, overriding the
                                   per-step visibility test.
            The output filename gets a per-ablation suffix so all four pages
            coexist on disk under the same step folder.
        """
        if ablation not in self.ABLATIONS:
            raise ValueError(
                f"Unknown ablation '{ablation}'; expected one of {self.ABLATIONS}"
            )
        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(f"\nCreating offline manual (ablation={ablation})...")

        from PIL import Image

        step = self.assembly.sequence[step_idx]
        save_dir = self.assembly.output_dir / "manual"
        save_dir.mkdir(parents=True, exist_ok=True)
        step_dir = save_dir / f"step_{step_idx}_{step.obj_id}_offline"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Ablation flags.
        use_angle_ranking = ablation != "no_angle_ranking"
        include_text = ablation != "no_text"
        include_motion = ablation != "no_motion"

        if use_angle_ranking:
            camera_angle = self._best_angle(step) if step.images else "iso1"
        else:
            camera_angle = "iso1"

        n_steps = len(self.assembly.sequence)
        base_ids = self._base_ids()
        group = list(getattr(step, "subassembly", None) or [])
        # Numbering comes from the assembly-order page list rather than from
        # step_idx arithmetic, because a subassembly plan inserts join pages
        # between the installation steps.
        assembly_step_nr = self._page_number("step", step_idx)
        step_title = f"Step {assembly_step_nr}" if assembly_step_nr else "Step"
        # Only the last disassembly step can be the initial-state page, and only
        # when the disassembly left nothing behind for the base page to show.
        is_initial = step_idx == n_steps - 1 and not base_ids

        present_ids = self._present_ids(step_idx)
        # Use ablation-specific base render so we don't clobber the baseline.
        base_suffix = "" if ablation == "full" else f"_{ablation}"
        base_path = step_dir / f"01_base_render{base_suffix}.png"

        if is_initial:
            self._render_parts_neutral_to_file(
                [step.obj_id],
                step.pose,
                base_path,
                size=(1024, 768),
                camera_angle=camera_angle,
            )
            part_name = self.assembly.objects[step.obj_id].name
            instruction = (
                f"Place {part_name} on the work surface as the starting component."
            )
            (step_dir / "instruction.txt").write_text(instruction)
        else:
            # Steps whose assembled position is plainly visible don't need the
            # disassembled ghost and the path trail — drawing both positions
            # there only adds clutter.  The no_motion ablation has already
            # turned them off, so leave that case alone.
            if include_motion and getattr(
                settings, "manual_auto_initial_position", True
            ):
                include_motion, visibility = self._should_show_initial_position(
                    step, present_ids=present_ids, camera_angle=camera_angle
                )
                visibility["show_initial_position"] = include_motion
                (step_dir / f"visibility{base_suffix}.json").write_text(
                    json.dumps(visibility, indent=2)
                )
                if self.assembly.evaluation and self.assembly.evaluation.verbose:
                    frac = visibility.get("visible_fraction")
                    print(
                        f"  step {step_idx} ({step.obj_id}): visible fraction "
                        f"{'n/a' if frac is None else f'{frac:.2f}'} "
                        f"(threshold {visibility['threshold']:.2f}) -> "
                        f"{'showing' if include_motion else 'hiding'} initial position"
                    )
            self._render_base_composite_to_file(
                step,
                base_path,
                size=(1024, 768),
                present_ids=present_ids,
                camera_angle=camera_angle,
                include_motion=include_motion,
            )
            if include_text:
                instruction = self._fetch_step_instruction(
                    step, step_idx, step_dir, camera_angle=camera_angle
                )
            else:
                instruction = ""

        base = Image.open(base_path).convert("RGBA")

        panel_size = (280, 380)
        left_panel = None
        right_panel = None
        if not is_initial:
            # The last disassembly step is the FIRST installation step, so there
            # is no previous step to have been reoriented from: the base page it
            # follows is rendered in this step's own frame.
            has_prev_step = step_idx + 1 < n_steps
            if step.rotated and step.pose is not None and has_prev_step:
                # The panel depicts the orientation BEFORE this step's
                # reorientation, so it must use the previous assembly step's
                # viewpoint — both its pose AND its chosen camera angle — not
                # the current step's. (In assembly order the previous step is
                # sequence[step_idx + 1], since the sequence is disassembly
                # order.)
                prev_pose = None
                prev_step = self.assembly.sequence[step_idx + 1]
                if prev_step.pose is not None:
                    prev_pose = np.asarray(prev_step.pose, dtype=float)
                if use_angle_ranking:
                    prev_camera_angle = (
                        self._best_angle(prev_step) if prev_step.images else "iso1"
                    )
                else:
                    prev_camera_angle = "iso1"
                # Axis+angle of the previous->current reorientation, in the
                # previous step's rendered frame, for the rotation arrow.
                rotation_axis = rotation_angle = None
                if prev_pose is not None:
                    aa = self._relative_rotation_axis_angle(prev_pose, step.pose)
                    if aa is not None:
                        rotation_axis, rotation_angle = aa
                left_panel = self._draw_offline_rotation_panel(
                    step,
                    panel_size=panel_size,
                    present_ids=present_ids,
                    prev_pose=prev_pose,
                    camera_angle=prev_camera_angle,
                    rotation_axis=rotation_axis,
                    rotation_angle=rotation_angle,
                )
                left_panel.save(step_dir / "02_rotation_panel.png")
            right_panel = self._draw_offline_part_panel(step, panel_size=panel_size)
            right_panel.save(step_dir / "03_part_panel.png")

        canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        canvas.paste(base, (0, 0))
        if left_panel is not None:
            canvas.alpha_composite(left_panel, (0, 0))
        if right_panel is not None:
            canvas.alpha_composite(right_panel, (1024 - right_panel.width, 0))
        self._draw_bottom_instruction_text(
            canvas,
            instruction,
            region=(0, 768, 1024, 1024),
            title=step_title,
            title_parts=self._title_parts(assembly_step_nr, group=group),
        )
        # Drawn last so the coloured rings sit on top of the composite and the
        # corner panels, never underneath them.
        self._draw_subassembly_frame(canvas, group)
        canvas_rgb = canvas.convert("RGB")

        suffix = "" if ablation == "full" else f"_{ablation}"
        output_top = save_dir / f"{step_idx}_{step.obj_id}_manual_offline{suffix}.png"
        canvas_rgb.save(output_top)
        if ablation == "full":
            canvas_rgb.save(step_dir / "final.png")
            self.assembly.instructions["Manual"].append(str(output_top))

        # The base page has no step of its own to be driven from, so emit it
        # alongside the step it precedes — the last disassembly step.
        if base_ids and step_idx == n_steps - 1:
            self.generate_manual_base_offline(ablation=ablation)
        return str(output_top)

    def generate_manual_base_offline(self, ablation="full"):
        """Assembly Step 1 page: the parts the disassembly left behind.

        Those parts (see `_base_ids`) are already on the bench when the first
        installation step starts, so without this page they would silently
        appear in the grey context of Step 2.  The page is rendered in the frame
        of the step that follows it, so no reorientation separates the two.

        Returns the page path, or None when the disassembly left nothing behind.
        """
        from PIL import Image

        base_ids = sorted(self._base_ids())
        if not base_ids:
            return None

        save_dir = self.assembly.output_dir / "manual"
        save_dir.mkdir(parents=True, exist_ok=True)
        step_dir = save_dir / "step_base_offline"
        step_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if ablation == "full" else f"_{ablation}"

        # Share the frame of the first installation step (the last disassembly
        # step) so the reader's viewpoint carries over unchanged.
        next_step = self.assembly.sequence[-1] if self.assembly.sequence else None
        pose = next_step.pose if next_step is not None else None
        if ablation == "no_angle_ranking" or next_step is None:
            camera_angle = "iso1"
        else:
            camera_angle = self._best_angle(next_step) if next_step.images else "iso1"

        base_path = step_dir / f"01_base_render{suffix}.png"
        self._render_parts_neutral_to_file(
            base_ids, pose, base_path, size=(1024, 768), camera_angle=camera_angle
        )

        names = [self.assembly.objects[i].name or f"part {i}" for i in base_ids]
        if len(names) == 1:
            instruction = (
                f"Place {names[0]} on the work surface as the starting component."
            )
        else:
            instruction = (
                f"Place {', '.join(names[:-1])} and {names[-1]} on the work surface "
                "as the starting components."
            )
        (step_dir / "instruction.txt").write_text(instruction)

        # The leftover parts sit in the deepest R block under a subassembly
        # plan, so the base page is framed like the pages that follow it.
        base_steps = [
            step for step in self.assembly.remaining if step.obj_id in set(base_ids)
        ]
        group = list(getattr(base_steps[0], "subassembly", None) or []) if base_steps else []

        canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        canvas.paste(Image.open(base_path).convert("RGBA"), (0, 0))
        self._draw_bottom_instruction_text(
            canvas,
            "" if ablation == "no_text" else instruction,
            region=(0, 768, 1024, 1024),
            title_parts=self._title_parts(self._page_number("base"), group=group),
        )
        self._draw_subassembly_frame(canvas, group)
        canvas_rgb = canvas.convert("RGB")

        output_top = save_dir / f"base_manual_offline{suffix}.png"
        canvas_rgb.save(output_top)
        if ablation == "full":
            canvas_rgb.save(step_dir / "final.png")
            self.assembly.instructions["Manual"].append(str(output_top))
        return str(output_top)

    def compile_manual_pdf(
        self, ablation="full", rows=3, cols=2, dpi=150, output_name=None
    ):
        """Stitch the per-step offline manual pages into one multi-page PDF.

        Pages are laid out as a rows x cols grid (default 3 rows x 2 cols = 6
        steps) on DIN A4 portrait sheets, in assembly order (Step 1 first) as
        given by `_assembly_page_order`: the base page (the parts the
        disassembly left behind) when there is one, then the installation steps
        in reverse disassembly order, with a join page wherever a subassembly
        plan mates two halves.

        Reads the per-step PNGs written by generate_manual_offline for the given
        ablation; missing base and join pages are generated here, since neither
        has a step of its own to be driven from. Returns the PDF path, or None
        if no pages were found."""
        from PIL import Image

        save_dir = self.assembly.output_dir / "manual"
        suffix = "" if ablation == "full" else f"_{ablation}"

        page_pngs = []
        for page in self._assembly_page_order():
            if page["kind"] == "base":
                png = save_dir / f"base_manual_offline{suffix}.png"
                if not png.exists():
                    self.generate_manual_base_offline(ablation=ablation)
            elif page["kind"] == "join":
                png = save_dir / f"join_{page['index']}_manual_offline{suffix}.png"
                if not png.exists():
                    self.generate_manual_join_offline(page["index"], ablation=ablation)
            else:
                step = self.assembly.sequence[page["step_idx"]]
                png = (
                    save_dir
                    / f"{page['step_idx']}_{step.obj_id}_manual_offline{suffix}.png"
                )
            if png.exists():
                page_pngs.append(png)
        if not page_pngs:
            print("  compile_manual_pdf: no manual pages found; skipping PDF.")
            return None

        # DIN A4 portrait in pixels at the requested DPI.
        a4_w = round(210.0 / 25.4 * dpi)
        a4_h = round(297.0 / 25.4 * dpi)
        per_page = rows * cols
        margin = round(dpi * 0.2)  # ~5mm outer margin
        gutter = round(dpi * 0.1)  # ~2.5mm between cells
        cell_w = (a4_w - 2 * margin - (cols - 1) * gutter) // cols
        cell_h = (a4_h - 2 * margin - (rows - 1) * gutter) // rows

        pages = []
        for start in range(0, len(page_pngs), per_page):
            sheet = Image.new("RGB", (a4_w, a4_h), (255, 255, 255))
            for cell_idx, png in enumerate(page_pngs[start : start + per_page]):
                r, c = divmod(cell_idx, cols)
                img = Image.open(png).convert("RGB")
                scale = min(cell_w / img.width, cell_h / img.height)
                img = img.resize(
                    (
                        max(1, round(img.width * scale)),
                        max(1, round(img.height * scale)),
                    ),
                    Image.LANCZOS,
                )
                x0 = margin + c * (cell_w + gutter) + (cell_w - img.width) // 2
                y0 = margin + r * (cell_h + gutter) + (cell_h - img.height) // 2
                sheet.paste(img, (x0, y0))
            pages.append(sheet)

        out_path = save_dir / (output_name or f"manual{suffix}.pdf")
        pages[0].save(
            out_path,
            "PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=float(dpi),
        )
        if ablation == "full":
            self.assembly.instructions["Manual"].append(str(out_path))
        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(
                f"  compiled manual PDF ({len(pages)} page(s), "
                f"{len(page_pngs)} step(s)) -> {out_path}"
            )
        return str(out_path)

    # ------------------------------------------------------------------
    # Iterative / geometric backend (mesh -> 2D SVG, LLM only for annotation)
    # ------------------------------------------------------------------
    @staticmethod
    def _view_matrix_from_angle(angle, scene_center, scene_radius):
        """Build a 4x4 view matrix matching the convention used by convert_angle_pv_pos.

        Returns a matrix M whose rotation rows are the camera basis (right, up, -forward)
        expressed in world space. M @ [x,y,z,1]^T gives the point in camera space, where
        +x is screen-right, +y is screen-up, -z is the view direction.
        """
        # Eye directions mirror the sim camera_pos values in
        # sequence_planner._render_plan so the geometric-SVG pipeline lines up
        # with the sim-rendered GIFs.
        if angle == "iso2":
            eye_dir = np.array([-1.0, 1.0, 1.0])
            up = np.array([0.0, 0.0, 1.0])
        elif angle == "iso3":
            eye_dir = np.array([-1.0, -1.0, 1.0])
            up = np.array([0.0, 0.0, 1.0])
        elif angle == "iso4":
            eye_dir = np.array([1.0, 1.0, 1.0])
            up = np.array([0.0, 0.0, 1.0])
        else:  # iso1 (default)
            eye_dir = np.array([1.0, -1.0, 1.0])
            up = np.array([0.0, 0.0, 1.0])

        eye_dir /= np.linalg.norm(eye_dir)
        eye = scene_center + eye_dir * scene_radius * 3.0
        forward = scene_center - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)

        R = np.stack([right, true_up, -forward], axis=0)
        t = -R @ eye
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M, forward

    @staticmethod
    def _project_points(points_world, view_matrix):
        """Apply 4x4 view matrix; return Nx3 camera-space points (x=right, y=up, z=depth)."""
        h = np.hstack([points_world, np.ones((len(points_world), 1))])
        return (view_matrix @ h.T).T[:, :3]

    @staticmethod
    def _classify_edges(mesh, view_dir_world, crease_deg=30.0):
        """Return (silhouette_edges, crease_edges) as Nx2 vertex-index arrays."""
        face_normals = np.asarray(mesh.face_normals)
        face_facing = (face_normals @ view_dir_world) < 0

        sil = []
        crease = []
        for ei, (f1, f2) in enumerate(mesh.face_adjacency):
            if face_facing[f1] != face_facing[f2]:
                sil.append(mesh.face_adjacency_edges[ei])
            elif mesh.face_adjacency_angles[ei] > np.radians(crease_deg):
                if face_facing[f1] and face_facing[f2]:
                    crease.append(mesh.face_adjacency_edges[ei])
        return sil, crease

    @staticmethod
    def _filter_visible_edges(
        source_vertices,
        edges,
        occluder_mesh,
        view_dir_world,
        view_matrix,
        samples_per_edge=3,
    ):
        """Fast vectorized hidden-line removal for an orthographic camera.

        For each edge, sample `samples_per_edge` points along it. Each sample is
        projected to camera space and tested against every front-facing triangle
        of the occluder mesh via a 2D point-in-triangle barycentric check + linear
        depth comparison. A sample is hidden iff some triangle's projected outline
        contains it AND that triangle's interpolated depth at the sample point is
        closer to the camera. An edge survives if ANY of its samples is visible.

        Vectorized in numpy; runs in milliseconds where the trimesh ray engine
        would take minutes.
        """
        if not len(edges):
            return edges
        edges_arr = np.asarray(edges, dtype=int)
        n_edges = len(edges_arr)
        v0w = source_vertices[edges_arr[:, 0]]
        v1w = source_vertices[edges_arr[:, 1]]

        ts = np.linspace(0.15, 0.85, samples_per_edge)
        samples_world = np.concatenate([v0w * (1 - t) + v1w * t for t in ts], axis=0)

        s_h = np.hstack([samples_world, np.ones((len(samples_world), 1))])
        samples_cam = (view_matrix @ s_h.T).T[:, :3]
        sx = samples_cam[:, 0]
        sy = samples_cam[:, 1]
        sz = samples_cam[:, 2]

        face_normals = np.asarray(occluder_mesh.face_normals)
        front_mask = (face_normals @ view_dir_world) < 0
        if not front_mask.any():
            return edges

        tri_v_world = occluder_mesh.vertices[occluder_mesh.faces[front_mask]]
        m = len(tri_v_world)
        tri_h = np.concatenate([tri_v_world, np.ones((m, 3, 1))], axis=2)
        tri_cam = np.einsum("ij,mvj->mvi", view_matrix, tri_h)[..., :3]

        ax, ay, az = tri_cam[:, 0, 0], tri_cam[:, 0, 1], tri_cam[:, 0, 2]
        bx, by, bz = tri_cam[:, 1, 0], tri_cam[:, 1, 1], tri_cam[:, 1, 2]
        cx, cy, cz = tri_cam[:, 2, 0], tri_cam[:, 2, 1], tri_cam[:, 2, 2]

        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        valid = np.abs(denom) > 1e-12
        denom_safe = np.where(valid, denom, 1.0)

        sx_b = sx[:, None]
        sy_b = sy[:, None]
        a_bary = ((by - cy) * (sx_b - cx) + (cx - bx) * (sy_b - cy)) / denom_safe
        b_bary = ((cy - ay) * (sx_b - cx) + (ax - cx) * (sy_b - cy)) / denom_safe
        c_bary = 1.0 - a_bary - b_bary

        eps_bary = 1e-7
        inside = (a_bary >= -eps_bary) & (b_bary >= -eps_bary) & (c_bary >= -eps_bary)
        inside &= valid[None, :]

        tri_depth = a_bary * az + b_bary * bz + c_bary * cz

        bbox = occluder_mesh.bounds[1] - occluder_mesh.bounds[0]
        eps_depth = float(np.linalg.norm(bbox)) * 1e-3

        occluded = inside & (tri_depth > sz[:, None] + eps_depth)
        sample_hidden = occluded.any(axis=1)

        sample_hidden = sample_hidden.reshape(samples_per_edge, n_edges)
        any_visible = (~sample_hidden).any(axis=0)
        return edges_arr[any_visible].tolist()

    @staticmethod
    def _normalize_to_canvas(cam_xy_groups, canvas_size, margin=0.08):
        """Given a dict of group_id -> Nx2 camera-space (x, y) arrays, compute a shared
        affine map that fits the union into canvas with `margin` fraction of padding.
        Returns a function `to_px(arr)` that maps Nx2 cam-space points to image pixels."""
        all_pts = np.vstack(list(cam_xy_groups.values()))
        mn = all_pts.min(0)
        mx = all_pts.max(0)
        span = np.maximum(mx - mn, 1e-9)
        usable_w = canvas_size[0] * (1.0 - 2.0 * margin)
        usable_h = canvas_size[1] * (1.0 - 2.0 * margin)
        scale = min(usable_w / span[0], usable_h / span[1])
        offset_x = (canvas_size[0] - span[0] * scale) / 2.0 - mn[0] * scale
        offset_y = (canvas_size[1] - span[1] * scale) / 2.0 - mn[1] * scale

        def to_px(arr):
            x_px = arr[:, 0] * scale + offset_x
            y_px = canvas_size[1] - (arr[:, 1] * scale + offset_y)
            return np.stack([x_px, y_px], axis=1)

        return to_px

    @staticmethod
    def _svg_open(width, height):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n'
            f'  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>\n'
            "  <defs>\n"
            '    <marker id="arrowhead" markerWidth="8" markerHeight="8" '
            'refX="7" refY="4" orient="auto" markerUnits="strokeWidth">\n'
            '      <polygon points="0,0 8,4 0,8" fill="black"/>\n'
            "    </marker>\n"
            "  </defs>\n"
        )

    @staticmethod
    def _svg_close():
        return "</svg>\n"

    @staticmethod
    def _svg_edges_layer(
        group_id, edges_px, stroke="#888", stroke_width=1.0, dasharray=None
    ):
        if not len(edges_px):
            return ""
        dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
        lines = [
            f'  <g id="{group_id}" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'fill="none" stroke-linecap="round"{dash_attr}>'
        ]
        for (x1, y1), (x2, y2) in edges_px:
            lines.append(
                f'    <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>'
            )
        lines.append("  </g>")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _svg_arrow(start_px, end_px, stroke="black", stroke_width=2.0, dasharray="8,5"):
        x1, y1 = start_px
        x2, y2 = end_px
        return (
            f'  <g id="assembly_arrow" stroke="{stroke}" stroke-width="{stroke_width}" fill="none">\n'
            f'    <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke-dasharray="{dasharray}" marker-end="url(#arrowhead)"/>\n'
            "  </g>\n"
        )

    @staticmethod
    def _svg_text(x, y, text, font_size=20, anchor="start", fill="black"):
        safe = (
            (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return (
            f'  <text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
            f'font-size="{font_size}" fill="{fill}" text-anchor="{anchor}">{safe}</text>\n'
        )

    @staticmethod
    def _svg_circle(cx, cy, r, fill="none", stroke="black", stroke_width=2.0):
        return (
            f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>\n'
        )

    def _svg_to_png(self, svg_text, png_path, width=1024, height=1024):
        """Rasterize an SVG string to PNG. Falls back through cairosvg → svglib → no-op."""
        try:
            import cairosvg

            cairosvg.svg2png(
                bytestring=svg_text.encode("utf-8"),
                write_to=str(png_path),
                output_width=width,
                output_height=height,
            )
            return True
        except ImportError:
            pass

        try:
            from io import BytesIO

            from reportlab.graphics import renderPM
            from svglib.svglib import svg2rlg

            drawing = svg2rlg(BytesIO(svg_text.encode("utf-8")))
            renderPM.drawToFile(drawing, str(png_path), fmt="PNG")
            return True
        except ImportError:
            print(
                "Warning: neither cairosvg nor svglib installed. "
                "SVG saved but PNG render skipped — install cairosvg for full pipeline."
            )
            return False
        except Exception as e:
            print(f"Warning: SVG-to-PNG rasterization failed: {e}")
            return False

    def _build_geometric_svg(self, step, canvas_size=(1024, 1024), margin_frac=0.05):
        """Project all assembly parts to 2D and compose a base SVG with:
          - already-assembled parts in light gray
          - moving part in black, positioned so its bbox is JUST OUTSIDE the
            rest-of-assembly bbox along the assembly direction (orientation from
            the full assembly path is preserved, only the along-axis offset is
            recomputed). A small margin gap is added between the two bboxes.
          - per-silhouette-vertex motion trails from drawn -> assembled position
          - one small dashed central arrow with arrowhead

        `margin_frac` is the gap between the two bboxes as a fraction of the
        rest-of-assembly bbox diagonal.
        """
        angle = self._best_angle(step) if step.images else "iso"

        moving_obj = self.assembly.objects[step.obj_id]
        moving_mesh_assembled = moving_obj.tri_mesh

        transform = (
            self.assembly._to_canonical(step, step.matrices[-1])
            if step.matrices
            else np.eye(4)
        )
        moving_mesh_disassembled_full = moving_mesh_assembled.copy().apply_transform(
            transform
        )

        rest_meshes = [
            obj.tri_mesh.vertices
            for obj in self.assembly.objects.values()
            if obj.id != step.obj_id
        ]
        if rest_meshes:
            rest_verts = np.vstack(rest_meshes)
            rest_diag = float(np.linalg.norm(rest_verts.max(0) - rest_verts.min(0)))
        else:
            rest_verts = moving_mesh_assembled.vertices
            rest_diag = float(np.linalg.norm(rest_verts.max(0) - rest_verts.min(0)))

        disp = moving_mesh_disassembled_full.centroid - moving_mesh_assembled.centroid
        disp_len = float(np.linalg.norm(disp))

        v = np.array([0.0, 0.0, 1.0]) if disp_len < 1e-09 else disp / disp_len

        margin = rest_diag * margin_frac
        rest_proj = rest_verts @ v
        moving_proj_assembled = moving_mesh_assembled.vertices @ v
        required_along_axis = (rest_proj.max() + margin) - moving_proj_assembled.min()

        correction = v * required_along_axis - disp
        extra = np.eye(4)
        extra[:3, 3] = correction
        moving_mesh_drawn = moving_mesh_disassembled_full.copy().apply_transform(extra)

        all_verts = [obj.tri_mesh.vertices for obj in self.assembly.objects.values()]
        all_verts.append(moving_mesh_drawn.vertices)
        stacked = np.vstack(all_verts)
        scene_center = (stacked.min(0) + stacked.max(0)) / 2.0
        scene_radius = float(np.linalg.norm(stacked.max(0) - stacked.min(0)) / 2.0)

        view_matrix, view_dir_world = self._view_matrix_from_angle(
            angle, scene_center, scene_radius
        )

        non_moving = [
            obj for obj in self.assembly.objects.values() if obj.id != step.obj_id
        ]
        combined_rest = None
        if non_moving:
            try:
                combined_rest = trimesh.boolean.union(
                    [obj.tri_mesh for obj in non_moving]
                )
            except Exception as e:
                print(f"  boolean.union failed ({e}); falling back to concatenate.")
                combined_rest = trimesh.util.concatenate(
                    [obj.tri_mesh.copy() for obj in non_moving]
                )
                combined_rest.merge_vertices()

        cam_xy_groups = {}
        per_part_cam = {}
        for obj in non_moving:
            cam_pts = self._project_points(obj.tri_mesh.vertices, view_matrix)
            per_part_cam[obj.id] = cam_pts
            cam_xy_groups[obj.id] = cam_pts[:, :2]

        if combined_rest is not None:
            combined_cam = self._project_points(combined_rest.vertices, view_matrix)
            cam_xy_groups["__rest__"] = combined_cam[:, :2]
            rest_sil, rest_crease = self._classify_edges(combined_rest, view_dir_world)
            rest_sil = self._filter_visible_edges(
                combined_rest.vertices,
                rest_sil,
                combined_rest,
                view_dir_world,
                view_matrix,
            )
            rest_crease = self._filter_visible_edges(
                combined_rest.vertices,
                rest_crease,
                combined_rest,
                view_dir_world,
                view_matrix,
            )

        cam_pts_moving = self._project_points(moving_mesh_drawn.vertices, view_matrix)
        cam_xy_groups["__moving__"] = cam_pts_moving[:, :2]
        sil_m, crease_m = self._classify_edges(moving_mesh_drawn, view_dir_world)
        if combined_rest is not None:
            world_mesh = trimesh.util.concatenate(
                [moving_mesh_drawn.copy(), combined_rest.copy()]
            )
        else:
            world_mesh = moving_mesh_drawn
        sil_m = self._filter_visible_edges(
            moving_mesh_drawn.vertices, sil_m, world_mesh, view_dir_world, view_matrix
        )
        crease_m = self._filter_visible_edges(
            moving_mesh_drawn.vertices,
            crease_m,
            world_mesh,
            view_dir_world,
            view_matrix,
        )

        to_px = self._normalize_to_canvas(cam_xy_groups, canvas_size)

        svg_parts = [self._svg_open(*canvas_size)]
        assembled_parts_info = []

        if combined_rest is not None:
            combined_px = to_px(combined_cam[:, :2])
            sil_lines = [(combined_px[a], combined_px[b]) for a, b in rest_sil]
            crease_lines = [(combined_px[a], combined_px[b]) for a, b in rest_crease]
            svg_parts.append(
                self._svg_edges_layer(
                    "rest_assembly_sil", sil_lines, stroke="#666", stroke_width=1.2
                )
            )
            svg_parts.append(
                self._svg_edges_layer(
                    "rest_assembly_crease",
                    crease_lines,
                    stroke="#aaa",
                    stroke_width=0.6,
                )
            )

            for obj in non_moving:
                pts_px = to_px(per_part_cam[obj.id][:, :2])
                assembled_parts_info.append(
                    {
                        "name": obj.name,
                        "centroid_px": pts_px.mean(0).tolist(),
                    }
                )

        mv_px = to_px(cam_pts_moving[:, :2])
        mv_sil_lines = [(mv_px[a], mv_px[b]) for a, b in sil_m]
        mv_crease_lines = [(mv_px[a], mv_px[b]) for a, b in crease_m]
        svg_parts.append(
            self._svg_edges_layer(
                f"part_{step.obj_id}_sil",
                mv_sil_lines,
                stroke="black",
                stroke_width=2.2,
            )
        )
        svg_parts.append(
            self._svg_edges_layer(
                f"part_{step.obj_id}_crease",
                mv_crease_lines,
                stroke="#333",
                stroke_width=0.9,
            )
        )
        moving_centroid_px = mv_px.mean(0).tolist()
        mv = {"sil": sil_m}

        if len(mv["sil"]):
            sil_vidx = np.unique(np.asarray(mv["sil"]).reshape(-1))
            drawn_world = moving_mesh_drawn.vertices[sil_vidx]
            assembled_world = moving_mesh_assembled.vertices[sil_vidx]
            drawn_sub_px = to_px(self._project_points(drawn_world, view_matrix)[:, :2])
            assembled_sub_px = to_px(
                self._project_points(assembled_world, view_matrix)[:, :2]
            )
            motion_lines = list(zip(drawn_sub_px, assembled_sub_px, strict=False))
            svg_parts.append(
                self._svg_edges_layer(
                    "motion_trails",
                    motion_lines,
                    stroke="#999",
                    stroke_width=0.5,
                    dasharray="4,3",
                )
            )

        cam_assembled = self._project_points(
            np.array([moving_mesh_assembled.centroid]), view_matrix
        )
        cam_drawn = self._project_points(
            np.array([moving_mesh_drawn.centroid]), view_matrix
        )
        end_px_arr = to_px(cam_assembled[:, :2])[0]
        start_px_arr = to_px(cam_drawn[:, :2])[0]
        svg_parts.append(self._svg_arrow(start_px_arr.tolist(), end_px_arr.tolist()))

        svg_parts.append(self._svg_close())
        svg_text = "".join(svg_parts)

        info = {
            "angle": angle,
            "canvas_size": list(canvas_size),
            "arrow_start_px": [float(start_px_arr[0]), float(start_px_arr[1])],
            "arrow_end_px": [float(end_px_arr[0]), float(end_px_arr[1])],
            "moving_part_centroid_px": [
                float(moving_centroid_px[0]),
                float(moving_centroid_px[1]),
            ],
            "moving_part_name": moving_obj.name,
            "assembled_parts": assembled_parts_info,
            "assembly_direction_world": (
                (
                    moving_mesh_assembled.centroid
                    - moving_mesh_disassembled_full.centroid
                ).tolist()
            ),
            "rest_assembly_diagonal": rest_diag,
            "required_offset_along_axis": float(required_along_axis),
            "original_disassembly_distance": disp_len,
        }
        return svg_text, info

    def _llm_decide_annotations(self, client, step, info):
        """Ask the LLM to decide labels / callout positions / tool icon / zoom inset.
        Receives structured data only — no rendered image."""
        moving_obj = self.assembly.objects[step.obj_id]
        parts_fix_names = [
            self.assembly.objects[pid].name
            if pid in self.assembly.objects
            else str(pid)
            for pid in (step.parts_fix or [])
        ]
        step_description = {
            "moving_part_name": moving_obj.name,
            "assembly_direction_world": info["assembly_direction_world"],
            "tool_required": step.tool,
            "rotation_required": bool(step.rotated),
            "parts_held_fixed": parts_fix_names,
            "moving_part_centroid_2d": info["moving_part_centroid_px"],
            "arrow_start_2d": info["arrow_start_px"],
            "arrow_end_2d": info["arrow_end_px"],
            "canvas_size": info["canvas_size"],
            "assembled_parts": info["assembled_parts"],
        }
        user_content = [
            {
                "type": "text",
                "text": (
                    "Decide annotation placement for one IKEA-style manual page. "
                    "Geometry is already drawn correctly; you only choose labels and metadata.\n\n"
                    f"Step data:\n{json.dumps(step_description, indent=2)}"
                ),
            }
        ]
        response = client.chat.completions.parse(
            model=settings.LLM_model,
            messages=self.assembly._make_messages(_SYSTEM_ANNOTATION, user_content),
            response_format=AnnotationDecision,
        )
        if self.assembly.evaluation:
            self.assembly.evaluation.tokens_used += response.usage.total_tokens
        return response.choices[0].message.parsed

    def _compose_final_svg(self, base_svg, annotation, info, canvas_size=(1024, 1024)):
        """Re-emit the SVG with annotation overlays (label, optional tool icon, optional zoom inset)."""
        body = base_svg[: base_svg.rfind("</svg>")]
        overlay = []

        if annotation.label_text:
            lx, ly = annotation.callout_position
            mx, my = info["moving_part_centroid_px"]
            overlay.append(
                f'  <line x1="{lx:.2f}" y1="{ly:.2f}" x2="{mx:.2f}" y2="{my:.2f}" '
                f'stroke="black" stroke-width="1" stroke-dasharray="3,3"/>\n'
            )
            overlay.append(
                self._svg_text(
                    lx, ly, annotation.label_text, font_size=22, anchor="middle"
                )
            )

        step_tool = getattr(self, "_last_step_tool", None)
        if annotation.show_tool_icon and step_tool:
            tx, ty = annotation.tool_icon_position
            overlay.append(
                self._svg_circle(
                    tx, ty, 24, fill="white", stroke="black", stroke_width=2
                )
            )
            overlay.append(
                self._svg_text(tx, ty + 6, "T", font_size=20, anchor="middle")
            )
            overlay.append(
                self._svg_text(
                    tx, ty + 44, str(step_tool), font_size=14, anchor="middle"
                )
            )

        if annotation.add_zoom_inset:
            iw = int(canvas_size[0] * 0.28)
            ih = int(canvas_size[1] * 0.28)
            x0 = canvas_size[0] - iw - 30
            y0 = 30
            mx, my = info["moving_part_centroid_px"]
            overlay.append(
                f'  <rect x="{x0}" y="{y0}" width="{iw}" height="{ih}" '
                f'fill="white" stroke="black" stroke-width="2"/>\n'
            )
            overlay.append(
                f'  <line x1="{mx:.2f}" y1="{my:.2f}" x2="{x0}" y2="{y0 + ih}" '
                f'stroke="black" stroke-width="1" stroke-dasharray="3,3"/>\n'
            )
            overlay.append(
                self._svg_text(
                    x0 + iw / 2,
                    y0 + ih / 2,
                    "zoom",
                    font_size=14,
                    anchor="middle",
                    fill="#888",
                )
            )

        if annotation.step_note:
            overlay.append(
                self._svg_text(
                    canvas_size[0] / 2,
                    canvas_size[1] - 30,
                    annotation.step_note,
                    font_size=18,
                    anchor="middle",
                )
            )

        return body + "".join(overlay) + "</svg>\n"

    def generate_manual_iterative(self, step_idx):
        """Geometric manual generation: mesh -> 2D projection -> SVG composition,
        with an LLM call only for annotation placement.

        Outputs:
          manual/step_{idx}_{obj_id}/geometric.svg     — base, no annotations
          manual/step_{idx}_{obj_id}/info.json         — projection metadata
          manual/step_{idx}_{obj_id}/annotation.json   — LLM decision
          manual/step_{idx}_{obj_id}/final.svg + .png
          manual/{idx}_{obj_id}_manual.svg + .png      — copies for backward compat
        """
        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print(f"\nCreating geometric manual for step {step_idx}...")

        step = self.assembly.sequence[step_idx]
        save_dir = (
            self.assembly.output_dir / "manual" / f"step_{step_idx}_{step.obj_id}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            base_svg, info = self._build_geometric_svg(step)
        except Exception as e:
            print(f"Geometric SVG build failed for step {step_idx}: {e}")
            return None

        (save_dir / "geometric.svg").write_text(base_svg)
        (save_dir / "info.json").write_text(json.dumps(info, indent=2))

        annotation = AnnotationDecision(
            label_text=self.assembly.objects[step.obj_id].name,
            callout_position=[
                min(
                    max(info["moving_part_centroid_px"][0] + 80, 60),
                    info["canvas_size"][0] - 60,
                ),
                max(info["moving_part_centroid_px"][1] - 80, 40),
            ],
            show_tool_icon=bool(step.tool),
            tool_icon_position=[
                info["canvas_size"][0] - 80,
                info["canvas_size"][1] - 80,
            ],
            add_zoom_inset=False,
            step_note=None,
        )

        if not (
            self.assembly.evaluation
            and self.assembly.evaluation.tokens_exhausted(
                f"generate_manual_iterative annotate (step {step_idx})"
            )
        ):
            try:
                client = OpenAI(api_key=self.assembly.openai_api_key)
                annotation = self._llm_decide_annotations(client, step, info)
            except Exception as e:
                print(
                    f"Annotation LLM call failed for step {step_idx}: {e} — using defaults."
                )

        (save_dir / "annotation.json").write_text(annotation.model_dump_json(indent=2))

        self._last_step_tool = step.tool
        final_svg = self._compose_final_svg(
            base_svg, annotation, info, canvas_size=tuple(info["canvas_size"])
        )
        final_svg_path = save_dir / "final.svg"
        final_png_path = save_dir / "final.png"
        final_svg_path.write_text(final_svg)
        png_ok = self._svg_to_png(
            final_svg,
            final_png_path,
            width=info["canvas_size"][0],
            height=info["canvas_size"][1],
        )

        top_svg = (
            self.assembly.output_dir / "manual" / f"{step_idx}_{step.obj_id}_manual.svg"
        )
        top_png = (
            self.assembly.output_dir / "manual" / f"{step_idx}_{step.obj_id}_manual.png"
        )
        top_svg.write_text(final_svg)
        if png_ok:
            top_png.write_bytes(final_png_path.read_bytes())
            self.assembly.instructions["Manual"].append(str(top_png))
            return str(top_png)
        self.assembly.instructions["Manual"].append(str(top_svg))
        return str(top_svg)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def make_manual(self, step_idx, ablation="full"):
        """Dispatch to a manual-generation backend based on settings.manual_method.

        Two backends are supported:
          "offline"   — generate_manual_offline: no VLM/LLM calls.
          "geometric" — generate_manual_iterative: SVG silhouette/crease +
                        iterative LLM annotations (the generative backend).

        ablation: forwarded to the offline backend (the only one supporting
            component-ablation runs); ignored by the geometric backend."""
        method = getattr(settings, "manual_method", "offline")
        if method == "geometric":
            return self.generate_manual_iterative(step_idx)
        if method != "offline":
            print(
                f"Unknown settings.manual_method='{method}'; falling back to offline."
            )
        return self.generate_manual_offline(step_idx, ablation=ablation)
