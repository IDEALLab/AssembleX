import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from ATA.assets.save import clear_saved_sdfs
from ATA.examples.run_multi_plan import ProgressiveQueueSequencePlanner

import settings
from core.models import Step, extract_gif_frames

# This file lives at <repo_root>/core/sequence_planner.py, so the repo root
# (which holds the assets/, ATA/ and ASAPx/ trees) is one level up.
project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)


# Eval object active for the duration of a get_assembly_plans_ASAP call.
# Set/cleared around the seq_plan(...) entry point so module-level helpers
# (most importantly choose_nodes_via_llm) can charge tokens against the same
# budget the rest of the pipeline uses.
_ACTIVE_EVAL = None


def choose_nodes_via_llm(
    history_items, candidate_items, k=1, model=None, cache_dir=None, evaluation=None
):
    """Ask a vision LLM to pick `k` of `candidate_items` to expand next.

    Used by ASAPx's `LLMDFASequencePlanner`. The call is synchronous, retried
    up to 3x with exponential backoff, and cached on disk by hash of all input
    PNGs + labels + k + model so re-runs with the same seed skip the API.

    Token tracking: `evaluation` (an `assembly.Eval`) is used to (a) skip the
    call entirely when `evaluation.tokens_exhausted(...)` is True, and (b)
    accumulate `response.usage.total_tokens` after a successful live API call
    (cache hits don't charge tokens). When `evaluation` is None the function
    falls back to module-level `_ACTIVE_EVAL`, which `get_assembly_plans_ASAP`
    sets for the duration of a seq_plan run.

    Args:
        history_items:   list of (png_path, label) — previous removal steps,
                         most-recent last. May be empty (e.g. first iteration).
        candidate_items: list of (png_path, label) — the candidates to choose
                         from, indexed 0..len-1.
        k:               number of indices to return.
        model:           OpenAI model id; defaults to settings.LLM_model.
        cache_dir:       optional directory for response cache JSON files.
        evaluation:      optional Eval; overrides _ACTIVE_EVAL.

    Returns:
        (indices, raw_response_text). `indices` is `None` when the call was
        skipped (token-budget exhausted) or failed after retries — the caller
        should handle this by falling back to its own default. Otherwise it's
        a list[int] of length k.
    """
    import base64
    import hashlib
    import json
    import time

    chosen_model = model or getattr(settings, "LLM_model", "gpt-4o")

    if evaluation is None:
        evaluation = _ACTIVE_EVAL

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Content-hash all inputs so identical (history, candidates, k, model)
    # tuples hit the cache, regardless of timestamps or file paths.
    h = hashlib.sha256()
    for path, label in list(history_items) + list(candidate_items):
        h.update(b"\x00" + label.encode("utf-8") + b"\x00")
        try:
            with open(path, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(str(path).encode("utf-8"))
    h.update(f"k={k}|model={chosen_model}".encode())
    cache_key = h.hexdigest()[:24]

    if cache_dir is not None:
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                # Cache hits are free — return them regardless of budget state.
                return cached["indices"], cached["response"]
            except (OSError, json.JSONDecodeError, KeyError):
                pass  # treat corrupt cache as miss

    # Budget gate: only checked when we'd actually hit the API.
    if evaluation is not None and evaluation.tokens_exhausted("llm_planner_selection"):
        return None, "SKIPPED: token budget exhausted"

    def _to_data_url(path):
        with open(path, "rb") as f:
            b = f.read()
        return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

    intro = (
        "You are advising on the manual disassembly of a multi-part object. "
        "Each image shows the current state of the partial assembly with one "
        "part highlighted in red — that part is the one being removed in the "
        "step the image represents.\n\n"
        "First you will see PREVIOUS STEPS already taken (most recent last). "
        "Then you will see CANDIDATE NEXT MOVES, numbered 0 through N-1. "
        f"Pick the {k} candidate(s) that a human technician would find easiest "
        "to remove next without disturbing the rest of the assembly. Reason "
        "about accessibility, leverage, and which removal best opens up the "
        "remaining structure for subsequent steps.\n\n"
        "Reply with ONLY a JSON object of the form "
        f'{{"choices": [<{k} distinct integer index/indices into the candidates>]}}.'
    )

    content = [{"type": "text", "text": intro}]
    if history_items:
        content.append({"type": "text", "text": "PREVIOUS STEPS (most recent last):"})
        for path, label in history_items:
            content.append({"type": "text", "text": label})
            content.append(
                {"type": "image_url", "image_url": {"url": _to_data_url(path)}}
            )
    else:
        content.append(
            {"type": "text", "text": "PREVIOUS STEPS: (none yet — this is the start)."}
        )
    content.append({"type": "text", "text": "CANDIDATE NEXT MOVES:"})
    for path, label in candidate_items:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": _to_data_url(path)}})

    indices = None
    response_text = None
    last_err = None
    for attempt in range(3):
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=chosen_model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            # Charge tokens against the budget on every live API call,
            # even before parsing — failed-parse responses still consumed tokens.
            usage = getattr(resp, "usage", None)
            if evaluation is not None and usage is not None:
                evaluation.tokens_used += int(getattr(usage, "total_tokens", 0) or 0)
            response_text = resp.choices[0].message.content or ""
            parsed = json.loads(response_text)
            raw_choices = parsed.get("choices") or []
            picked = []
            seen = set()
            for v in raw_choices:
                try:
                    i = int(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < len(candidate_items) and i not in seen:
                    picked.append(i)
                    seen.add(i)
            if len(picked) < k:
                raise ValueError(f"got {len(picked)} valid index/indices, need {k}")
            indices = picked[:k]
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2**attempt)
            continue

    if indices is None:
        # Persistent failure: hand control back to the caller. Returning None
        # signals "use your own fallback" (planner uses its heuristic best).
        response_text = f"ERROR: {last_err}"

    if cache_path is not None and indices is not None:
        try:
            with open(cache_path, "w") as f:
                json.dump({"indices": indices, "response": response_text}, f)
        except OSError:
            pass

    return indices, response_text


def _extract_step_details(tree):
    """Walk the solution path in the tree and return per-step pose and parts_fix.

    Targets the deepest feasible node: a size-1 leaf on full success, or the
    deepest feasible prefix when the planner got stuck. Tie-break matches
    SequencePlanner.find_partial_sequence so step_details and the rendered
    sequence follow the same spine.
    """
    # Prefer a full solution (first size-1 feasible leaf), matching
    # SequencePlanner.find_sequence so step_details and stats['sequence'] agree.
    leaf_node = None
    for node in tree.nodes:
        if len(node) == 1 and tree.nodes[node]["n_gripper"] is not None:
            leaf_node = node
            break
    # Stuck run: fall back to the deepest feasible prefix, tie-broken the same
    # way as SequencePlanner.find_partial_sequence.
    if leaf_node is None:
        for node in tree.nodes:
            info = tree.nodes[node]
            if info["n_gripper"] is None:
                continue
            leaf_info = tree.nodes[leaf_node] if leaf_node is not None else None
            if leaf_node is None or (len(node), info["n_gripper"], info["n_eval"]) < (
                len(leaf_node),
                leaf_info["n_gripper"],
                leaf_info["n_eval"],
            ):
                leaf_node = node
    if leaf_node is None or tree.in_degree(leaf_node) == 0:
        return []

    steps = []
    node = leaf_node
    while tree.in_degree(node) > 0:
        node_info = tree.nodes[node]
        for parent_node in tree.predecessors(node):
            parent_gripper = tree.nodes[parent_node]["n_gripper"]
            if parent_gripper is not None and parent_gripper <= node_info["n_gripper"]:
                sim_info = tree.edges[parent_node, node]["sim_info"]
                pose = sim_info["pose"]
                steps.insert(
                    0,
                    {
                        "part": sim_info["part_move"],
                        "parts_fix": sim_info["parts_fix"] or [],
                        "pose": pose.tolist() if pose is not None else None,
                        "rotated": pose is not None,
                    },
                )
                node = parent_node
                break
        else:
            break  # no feasible parent — stop walking
    return steps


class SequencePlanner:
    def __init__(self, assembly):
        self.assembly = assembly

    def fetch_sequence_matrices(self):
        paths_dir = self.assembly.storage_dir / "paths"
        if not paths_dir.exists():
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(
                    f"No paths directory at {paths_dir}. Assembly may be completely rigid."
                )
            return self.assembly.sequence

        # ASAPx writes paths/{i}_{obj_id}/{frame}/part{obj_id}.npy (nested frame dirs);
        # ATA writes paths/{i}_{obj_id}/{frame}.npy. Dispatch by inspecting the layout.
        sample = next((p for p in paths_dir.iterdir() if p.is_dir()), None)
        if sample is not None and any(child.is_dir() for child in sample.iterdir()):
            return self.fetch_sequence_matrices_asap()

        # Build lookup once so directory → step matching is O(1) instead of O(n·m).
        step_by_id = {step.obj_id: step for step in self.assembly.sequence}
        loaded = 0

        for part_dir in sorted(paths_dir.iterdir()):
            if not part_dir.is_dir():
                continue
            obj_id = "_".join(part_dir.name.split("_")[1:])
            step = step_by_id.get(obj_id)
            if step is None:
                continue

            npy_files = sorted(part_dir.glob("*.npy"))
            if not npy_files:
                continue

            step.matrices = [np.load(f) for f in npy_files]
            loaded += 1

        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            if loaded:
                print(
                    f"Loaded path matrices for {loaded}/{len(self.assembly.sequence)} steps."
                )
            else:
                print("No path matrices found. Assembly may be completely rigid.")

        return self.assembly.sequence

    def fetch_sequence_matrices_asap(self):
        """Load per-frame transforms of the moving part from the ASAPx path layout.

        ASAPx's play_logged_plan stores motion via save_path_all_objects, producing
        paths/{i}_{obj_id}/{frame_idx}/part{body}.npy — one absolute 4x4 world
        transform per body per frame. We keep only the moving part's transforms
        (part{obj_id}.npy) so step.matrices remains a flat list of 4x4 matrices,
        compatible with the ATA path consumers.
        """
        paths_dir = self.assembly.storage_dir / "paths"
        if not paths_dir.exists():
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(
                    f"No paths directory at {paths_dir}. Assembly may be completely rigid."
                )
            return self.assembly.sequence

        step_by_id = {step.obj_id: step for step in self.assembly.sequence}
        loaded = 0

        for part_dir in sorted(paths_dir.iterdir()):
            if not part_dir.is_dir():
                continue
            obj_id = "_".join(part_dir.name.split("_")[1:])
            step = step_by_id.get(obj_id)
            if step is None:
                continue

            # Frame directories are named "0", "1", ... — sort numerically, not lexically.
            frame_dirs = sorted(
                (d for d in part_dir.iterdir() if d.is_dir() and d.name.isdigit()),
                key=lambda d: int(d.name),
            )
            if not frame_dirs:
                continue

            part_file = f"part{obj_id}.npy"
            matrices = [
                np.load(d / part_file) for d in frame_dirs if (d / part_file).exists()
            ]
            if not matrices:
                continue

            step.matrices = matrices
            loaded += 1

        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            if loaded:
                print(
                    f"Loaded ASAP path matrices for {loaded}/{len(self.assembly.sequence)} steps."
                )
            else:
                print("No path matrices found. Assembly may be completely rigid.")

        return self.assembly.sequence

    def update_sequence(self, sequence, assign_step_nr=True):
        self.assembly.sequence = [Step(obj_id=obj_id) for obj_id in sequence]

        if len(self.assembly.sequence) == len(self.assembly.objects):
            self.assembly.remaining.add(self.assembly.sequence.pop())
            self.fetch_sequence_matrices()
            self.fetch_imgs_and_gifs(self.assembly.sequence, assign_step_nr)

        elif len(sequence) < len(self.assembly.objects):
            self.fetch_sequence_matrices()
            self.fetch_imgs_and_gifs(self.assembly.sequence, assign_step_nr)
            for obj_id in self.assembly.objects:
                if obj_id not in sequence:
                    self.assembly.remaining.add(Step(obj_id=obj_id))
            self.fetch_imgs_and_gifs(self.assembly.remaining, assign_step_nr=False)
        else:
            raise ValueError(
                f"Length of sequence ({len(sequence)}) is greater than the number of objects ({len(self.assembly.objects)}). Please check the sequence and object initialization."
            )

        return self.assembly.sequence, self.assembly.remaining

    def _apply_step_details(self, step_details):
        for step, detail in zip(self.assembly.sequence, step_details, strict=False):
            step.pose = detail.get("pose")
            step.rotated = detail.get("rotated", False)
            step.parts_fix = detail.get("parts_fix", [])
            tool = detail.get("tool")
            if tool is not None:
                step.tool = tool

    def _pre_plan_tool_check(self, args):
        """Run per-part tool-needed VLM analysis and orientation selection before the
        sequence planner kicks in, gated by ``--tool-check``. Cache-aware: re-runs are
        cheap when ``storage_dir/tool_decisions.json`` already covers every part.
        """
        if not getattr(args, "tool_check", False):
            return
        if self.assembly.evaluation and self.assembly.evaluation.verbose:
            print("[get_assembly_plans] Pre-computing tool decisions for assembly...")
        self.assembly.analyze_assembly_tools()

    def _load_tool_decisions(self):
        """Load the per-part tool-decision dict written by ``analyze_assembly_tools``.

        Returns ``{}`` when no decisions file exists. Each entry has the shape
        ``{"tool": <name|"none"|"unclear"|"error"|None>, "confidence": <float|None>}``.
        """
        cache_path = self.assembly.storage_dir / "tool_decisions.json"
        if not cache_path.exists():
            return {}
        with open(cache_path) as f:
            return json.load(f)

    @staticmethod
    def _annotate_steps_with_tool(step_details, tool_decisions):
        """Attach the cached tool name (if any) to each step-detail in place."""
        for detail in step_details:
            part_id = detail.get("part")
            if part_id is None:
                continue
            entry = tool_decisions.get(str(part_id))
            if entry is None:
                continue
            tool = entry.get("tool")
            if tool is not None:
                detail["tool"] = tool
        return step_details

    def _apply_tool_decisions(self, tool_decisions):
        """Write cached tool names onto every ``Step`` in the assembly sequence."""
        if not tool_decisions:
            return
        for step in self.assembly.sequence:
            entry = tool_decisions.get(str(step.obj_id))
            if entry is None:
                continue
            tool = entry.get("tool")
            if tool is not None:
                step.tool = tool

    def _build_tool_meshes_for_plan(self, plan_sequence):
        """For each part in ``plan_sequence`` that has a cached tool decision, return
        a positioned tool mesh ready to be passed to ``play_logged_plan``.

        The mesh is produced by ``ToolAnalyzer._apply_tool_geometric`` using the
        part's and tool's cached direction/contact_point — so the tool's geometry
        is already in the moving part's OBJ frame and can be attached as a fixed
        child link inside redmax (see ``MultiPartPathPlanner.render_with_tool``).

        Returns an empty dict when no decisions are cached or no tools are valid.
        """
        tool_decisions = self._load_tool_decisions()
        if not tool_decisions:
            return {}
        terminal_states = {None, "none", "unclear", "error"}
        meshes = {}
        for part_id in plan_sequence:
            entry = tool_decisions.get(str(part_id))
            if entry is None:
                continue
            tool_name = entry.get("tool")
            if tool_name in terminal_states:
                continue
            tool = (self.assembly.scaled_tools or {}).get(tool_name)
            if tool is None:
                continue
            obj = self.assembly.objects.get(str(part_id))
            if obj is None:
                continue
            try:
                from core.tool_analyzer import ToolAnalyzer

                placement = ToolAnalyzer._apply_tool_geometric(tool, obj, save_dir=None)
            except Exception as e:
                if self.assembly.evaluation and self.assembly.evaluation.verbose:
                    print(f"[_build_tool_meshes_for_plan] {part_id}: skipped ({e})")
                continue
            if placement is None:
                continue
            tool_mesh, _ = placement
            meshes[str(part_id)] = tool_mesh
        if self.assembly.evaluation and self.assembly.evaluation.verbose and meshes:
            print(
                f"[_build_tool_meshes_for_plan] attaching tools to {len(meshes)} step(s): {sorted(meshes.keys())}"
            )
        return meshes

    def fetch_imgs_and_gifs(self, steps, assign_step_nr):
        for i, step in enumerate(steps):
            for angle in step.images:
                imgs_path = (
                    self.assembly.storage_dir
                    / "./frames"
                    / f"object_{step.obj_id}"
                    / angle
                )

                if angle == "iso1":
                    gif_name = f"*_{step.obj_id}.gif"
                elif angle == "iso2":
                    gif_name = f"*_{step.obj_id}_opposite.gif"
                else:
                    gif_name = f"*_{step.obj_id}_{angle}.gif"

                gif_path = next(self.assembly.storage_dir.glob(gif_name), None)

                if i == len(steps) - 1 and gif_path is None:
                    if self.assembly.evaluation and self.assembly.evaluation.verbose:
                        print(
                            f"Skipping last step {step.obj_id}, since assembly is static."
                        )
                    break

                if gif_path is None:
                    if self.assembly.evaluation and self.assembly.evaluation.verbose:
                        print(
                            f"Skipping {angle} for step {step.obj_id}: no GIF matching {gif_name}"
                        )
                    continue

                if self.assembly.evaluation and self.assembly.evaluation.verbose:
                    print(f"Found GIF: {gif_path}")

                img_paths = extract_gif_frames(gif_path, imgs_path)
                step.images[angle] = img_paths
                step.gifs[angle] = gif_path

            if assign_step_nr:
                self.assembly.objects[step.obj_id].step_nr = i

    def get_assembly_plans(self, args):
        if args.seq_planner == "ASAP":
            return self.get_assembly_plans_ASAP(args)
        elif args.seq_planner == "ASAP-archive":
            return self.get_assembly_plans_ASAP_archive(args)
        elif args.seq_planner == "ATA":
            return self.get_assembly_plans_ATA(args)
        else:
            raise NotImplementedError(
                f"Sequence planner {args.seq_planner} not implemented. Currently only 'progressive_queue', 'asap', and 'multi-plan' are supported."
            )

    def get_assembly_plans_ATA(self, args):
        asset_folder = os.path.join(project_base_dir, "./assets")
        assembly_dir = os.path.join(asset_folder, args.dir, args.id)

        seq_file_path = self.assembly.storage_dir / "sequence.json"
        if seq_file_path.exists():
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Loading sequence from {seq_file_path}...")
            with open(seq_file_path) as f:
                data = json.load(f)
                self.update_sequence(data["sequence"])
                self._apply_tool_decisions(data.get("tool_decisions", {}))
                return data["assemblable"]

        if not args.use_previous_sdf:
            clear_saved_sdfs(assembly_dir)

        if (
            args.verbose
            and self.assembly.evaluation
            and self.assembly.evaluation.verbose
        ):
            print(
                f"Running sequence planner... Storage path: {self.assembly.storage_dir}"
            )

        self._pre_plan_tool_check(args)

        seq_planner = ProgressiveQueueSequencePlanner(asset_folder, assembly_dir)
        seq_status, sequence, _seq_count, _t_plan = seq_planner.plan_sequence(
            "bfs",
            args.rotation,
            "sdf",
            args.sdf_dx,
            args.collision_th,
            args.force_mag,
            args.frame_skip,
            args.seq_max_time,
            args.path_max_time,
            args.seed,
            render=True,
            record_dir=self.assembly.storage_dir.parent,
            save_dir=self.assembly.storage_dir.parent,
            n_save_state=5,
            verbose=args.verbose,
            max_iterations=args.max_iterations,
            two_angles=True,
        )
        print(
            f"Final result for assembly {args.id}: {seq_status} | Sequence: {sequence}"
        )

        assemblable = "Success" in seq_status
        tool_decisions = self._load_tool_decisions()
        with open(seq_file_path, "w") as f:
            json.dump(
                {
                    "sequence": sequence,
                    "assemblable": assemblable,
                    "tool_decisions": tool_decisions,
                },
                f,
            )
        self.update_sequence(sequence)
        self._apply_tool_decisions(tool_decisions)
        return assemblable

    def get_assembly_plans_ASAP(self, args):
        asset_folder = os.path.join(project_base_dir, "./assets")
        assembly_dir = self.assembly.assembly_dir

        seq_file_path = self.assembly.storage_dir / "sequence.json"
        if seq_file_path.exists():
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Loading sequence from {seq_file_path}...")
            with open(seq_file_path) as f:
                data = json.load(f)
                self.update_sequence(data["sequence"])
                self._apply_step_details(data.get("steps", []))
                self._apply_tool_decisions(data.get("tool_decisions", {}))
                # Restore planning-failure evidence from the cached log dir so
                # failure-mode feedback can be regenerated on re-runs without
                # re-planning. Mirrors the same load in the non-cached branch.
                from core.models import PlanningFailures

                log_dir = self.assembly.storage_dir / "log"
                if log_dir.exists():
                    self.assembly.planning_failures = PlanningFailures.from_json(
                        log_dir
                    )
                return data["assemblable"]

        if not getattr(args, "use_previous_sdf", False):
            clear_saved_sdfs(assembly_dir)

        if (
            getattr(args, "verbose", False)
            and self.assembly.evaluation
            and self.assembly.evaluation.verbose
        ):
            print(
                f"Running sequence planner (ASAP)... Storage path: {self.assembly.storage_dir}"
            )

        self._pre_plan_tool_check(args)

        asap_dir = os.path.join(project_base_dir, "ASAPx")
        if asap_dir not in sys.path:
            sys.path.insert(0, asap_dir)
        # Evict all top-level packages that both ATA and ASAPx define, so ASAPx
        # re-imports its own versions now that its dir is first in sys.path.
        # NOTE: "settings" is deliberately NOT in this set. There is exactly
        # one settings.py in the repo (the root one) -- neither ATA nor ASAPx
        # ships its own -- so evicting it cannot resolve a name clash; it only
        # forces a fresh re-read from disk, silently discarding every runtime
        # override the caller set (heuristic_weights_source="optuna",
        # render_sequence=False, debug_stability=False, ...) before planning.
        _asapx_pkgs = {
            "assets",
            "utils",
            "simulation",
            "plan_path",
            "plan_robot",
            "plan_sequence",
        }
        for _mod in list(sys.modules.keys()):
            if _mod in _asapx_pkgs or any(
                _mod.startswith(p + ".") for p in _asapx_pkgs
            ):
                del sys.modules[_mod]
        from ASAPx.plan_sequence.run_seq_plan import seq_plan

        log_dir = self.assembly.storage_dir / "log"
        log_dir.mkdir(exist_ok=True)

        seq_tools = None
        # The geometric tool check during sequence planning is the canonical
        # tool-feasibility pass. It's enabled by settings.tool_assemblability
        # (which also gates the BFS-assemblability step inside the pipeline);
        # --seq-tool-check is kept as an explicit CLI override.
        _tool_check_enabled = getattr(
            settings, "tool_assemblability", False
        ) or getattr(args, "seq_tool_check", False)
        if _tool_check_enabled:
            scaled = self.assembly.scaled_tools or {}
            if not scaled:
                print(
                    "[get_assembly_plans_ASAP] tool check enabled but no scaled tools available; skipping in-sequence tool check."
                )
            else:
                # Prefer per-part decisions cached by analyze_assembly_tools()
                # so the planner only tries the VLM-decided tool per step
                # rather than iterating every tool. Parts decided as 'none'
                # get an empty list (no tool required); parts with no cache
                # entry fall back to the full scaled list.
                decisions_path = self.assembly.storage_dir / "tool_decisions.json"
                terminal_states = {None, "none", "unclear", "error"}
                seq_tools = {}
                if decisions_path.exists():
                    try:
                        with open(decisions_path) as _f:
                            decisions = json.load(_f)
                    except (OSError, json.JSONDecodeError) as _e:
                        print(
                            f"[get_assembly_plans_ASAP] could not load {decisions_path}: {_e}; falling back to full tool list."
                        )
                        decisions = {}
                    for part_id in self.assembly.objects:
                        entry = decisions.get(part_id) or {}
                        tool_id = entry.get("tool")
                        if tool_id in terminal_states:
                            seq_tools[str(part_id)] = []
                        elif tool_id in scaled:
                            seq_tools[str(part_id)] = [scaled[tool_id]]
                        else:
                            # Unknown tool id or no decision — fall back to all.
                            seq_tools[str(part_id)] = list(scaled.values())
                    n_decided = sum(1 for v in seq_tools.values() if len(v) == 1)
                    n_none = sum(1 for v in seq_tools.values() if len(v) == 0)
                    print(
                        f"[get_assembly_plans_ASAP] using per-part tool decisions: {n_decided} decided, {n_none} no-tool, {len(seq_tools) - n_decided - n_none} fallback (full list)."
                    )
                else:
                    # No pre-planning tool decision — same flat-list behavior
                    # as before, just expressed as a per-part dict so the
                    # planner has a uniform interface.
                    flat = list(scaled.values())
                    for part_id in self.assembly.objects:
                        seq_tools[str(part_id)] = flat
                    print(
                        "[get_assembly_plans_ASAP] no tool_decisions.json; passing full tool list per part (run with --tool-check to enable VLM pre-decision)."
                    )

        # Step 1: plan the disassembly sequence
        # Expose the active Eval to module-level helpers (e.g. choose_nodes_via_llm)
        # for the duration of the seq_plan call so they share the same token budget.
        global _ACTIVE_EVAL
        _ACTIVE_EVAL = self.assembly.evaluation
        try:
            seq_plan(
                asset_folder=asset_folder,
                assembly_dir=assembly_dir,
                generator_name=getattr(args, "generator", "rand"),
                planner_name=getattr(args, "planner", "heuristic"),
                num_proc=getattr(args, "num_proc", 80),
                seed=getattr(args, "seed", 0),
                budget=getattr(args, "budget", 6000),
                max_gripper=getattr(args, "max_gripper", 2),
                max_pose=getattr(args, "max_pose", 3),
                # pose_reuse=0 (the upstream ASAP default) forces the planner to
                # call get_stable_poses(node_mesh, max_num=max_pose) for every
                # subassembly during iteration. With pose_reuse=max_pose, max_num
                # drops to 0 and get_stable_poses early-returns [] — every node
                # silently inherits its parent's single working pose, so the
                # rendered disassembly never re-orients between steps. Override
                # only with the parallel arg if you have a perf reason and accept
                # the loss of per-step pose recomputation.
                pose_reuse=getattr(args, "pose_reuse", 0),
                # early_term gates the outer planner loop's "stop on first full
                # sequence" exit (see base.py / dfa.py). With it off, the
                # planner keeps exploring until budget / timeout /
                # tree-fully-explored — yielding a richer tree (more candidate
                # sub-sequences, per-step pose variation across siblings) at
                # the cost of using more of the budget. This does NOT disable
                # the DFA per-parent n_success_term quota (which gates the
                # inner parallel batch separately) nor the always-on inner
                # pose-loop break on first feasibility.
                early_term=getattr(args, "early_term", True),
                timeout=getattr(args, "timeout", None),
                base_part=getattr(args, "base_part", None),
                save_sdf=not getattr(args, "disable_save_sdf", False),
                clear_sdf=False,
                plan_grasp=getattr(args, "plan_grasp", False),
                plan_arm=getattr(args, "plan_arm", False),
                gripper_type=getattr(args, "gripper_type", "rod"),
                gripper_scale=getattr(args, "gripper_scale", 0.4),
                optimizer=getattr(args, "optimizer", "L-BFGS-B"),
                debug=getattr(args, "verbose", 0),
                render=False,
                record_dir=None,
                log_dir=str(log_dir),
                allow_gap=getattr(args, "allow_gap", False),
                n_success_term=settings.n_success_term,
                connect_path=getattr(args, "connect_path", False),
                get_dof=settings.get_dof,
                tools=seq_tools,
                skip_stability=settings.skip_stability,
                max_frontier=settings.max_frontier,
                seq_optimizer=getattr(args, "seq_optimizer", None),
            )
        finally:
            _ACTIVE_EVAL = None

        with open(log_dir / "stats.json") as f:
            stats = json.load(f)
        with open(log_dir / "tree.pkl", "rb") as f:
            tree = pickle.load(f)

        from core.models import PlanningFailures

        self.assembly.planning_failures = PlanningFailures.from_json(log_dir)

        plan_sequence = stats.get("sequence") or []
        assemblable = bool(stats.get("success", False))

        # Extend the saved sequence with the trivially-remaining base part so
        # update_sequence receives a complete list. Keep plan_sequence (the
        # original ASAP output) for play_logged_plan, which looks up tree edges
        # and has no entry for the last part.
        sequence = plan_sequence
        if assemblable and plan_sequence:
            missing = set(self.assembly.objects.keys()) - set(plan_sequence)
            if missing:
                sequence = plan_sequence + list(missing)

        print(
            f"Final result for assembly {args.id}: {'Success' if assemblable else 'Failure'} | Sequence: {sequence}"
        )

        step_details = _extract_step_details(tree) if plan_sequence else []
        tool_decisions = self._load_tool_decisions()
        self._annotate_steps_with_tool(step_details, tool_decisions)
        with open(seq_file_path, "w") as f:
            json.dump(
                {
                    "sequence": sequence,
                    "assemblable": assemblable,
                    "steps": step_details,
                    "tool_decisions": tool_decisions,
                },
                f,
            )

        if plan_sequence and getattr(settings, "render_sequence", True):
            self._render_plan(asset_folder, assembly_dir, plan_sequence, tree, args)

        self.update_sequence(sequence)
        self._apply_step_details(step_details)
        return assemblable

    def get_assembly_plans_ASAP_archive(self, args):
        """Run the upstream archive/ASAP sequence planner (smaller seq_plan
        signature, only 'dfs'/'beam'/'randseq' planners + 'rand'/'heur-*'/'learn'
        generators). Reuses the same sequence.json cache and post-processing as
        get_assembly_plans_ASAP, but skips the ASAPx-specific render step (the
        archive planner has its own renderer and the play_logged_plan call here
        targets ASAPx).
        """
        asset_folder = os.path.join(project_base_dir, "./assets")
        assembly_dir = self.assembly.assembly_dir

        seq_file_path = self.assembly.storage_dir / "sequence.json"
        if seq_file_path.exists():
            if self.assembly.evaluation and self.assembly.evaluation.verbose:
                print(f"Loading sequence from {seq_file_path}...")
            with open(seq_file_path) as f:
                data = json.load(f)
                self.update_sequence(data["sequence"])
                self._apply_step_details(data.get("steps", []))
                self._apply_tool_decisions(data.get("tool_decisions", {}))
                # Restore planning-failure evidence from the cached log dir so
                # failure-mode feedback can be regenerated on re-runs without
                # re-planning. Mirrors the same load in the non-cached branch.
                from core.models import PlanningFailures

                log_dir = self.assembly.storage_dir / "log"
                if log_dir.exists():
                    self.assembly.planning_failures = PlanningFailures.from_json(
                        log_dir
                    )
                return data["assemblable"]

        if not getattr(args, "use_previous_sdf", False):
            clear_saved_sdfs(assembly_dir)

        self._pre_plan_tool_check(args)

        # archive.ASAP uses fully-qualified imports throughout, so no sys.path
        # gymnastics needed (unlike get_assembly_plans_ASAP, which has to
        # swap ASAPx in for ATA's same-named packages).
        from archive.ASAP.plan_sequence.run_seq_plan import (
            seq_plan as _seq_plan_archive,
        )

        log_dir = self.assembly.storage_dir / "log"
        log_dir.mkdir(exist_ok=True)

        # Archive planners: dfs/beam/randseq; generators: rand/heur-vol/heur-out/learn.
        # If --planner is one of the ASAPx-only ones (heuristic/dfa/llm/...),
        # fall back to dfs so the call doesn't KeyError on the planners dict.
        _archive_planners = {"dfs", "beam", "randseq"}
        planner_name = getattr(args, "planner", "dfs")
        if planner_name not in _archive_planners:
            print(
                f"[get_assembly_plans_ASAP_archive] planner {planner_name!r} not in "
                f"archive set {sorted(_archive_planners)}; using 'dfs'"
            )
            planner_name = "dfs"

        _seq_plan_archive(
            asset_folder=asset_folder,
            assembly_dir=assembly_dir,
            generator_name=getattr(args, "generator", "rand"),
            planner_name=planner_name,
            num_proc=getattr(args, "num_proc", 80),
            seed=getattr(args, "seed", 0),
            budget=getattr(args, "budget", 6000),
            max_gripper=getattr(args, "max_gripper", 1),
            max_pose=getattr(args, "max_pose", 3),
            pose_reuse=getattr(args, "pose_reuse", None),
            early_term=getattr(args, "early_term", True),
            timeout=getattr(args, "timeout", None),
            base_part=getattr(args, "base_part", None),
            save_sdf=not getattr(args, "disable_save_sdf", False),
            clear_sdf=False,
            plan_grasp=getattr(args, "plan_grasp", False),
            plan_arm=getattr(args, "plan_arm", False),
            gripper_type=getattr(args, "gripper_type", "robotiq-140"),
            gripper_scale=getattr(args, "gripper_scale", 0.4),
            optimizer=getattr(args, "optimizer", "L-BFGS-B"),
            debug=getattr(args, "verbose", 0),
            render=False,
            record_dir=None,
            log_dir=str(log_dir),
        )

        with open(log_dir / "stats.json") as f:
            stats = json.load(f)
        with open(log_dir / "tree.pkl", "rb") as f:
            tree = pickle.load(f)

        plan_sequence = stats.get("sequence") or []
        assemblable = bool(stats.get("success", False))

        sequence = plan_sequence
        if assemblable and plan_sequence:
            missing = set(self.assembly.objects.keys()) - set(plan_sequence)
            if missing:
                sequence = plan_sequence + list(missing)

        print(
            f"Final result for assembly {args.id}: {'Success' if assemblable else 'Failure'} | Sequence: {sequence}"
        )

        step_details = _extract_step_details(tree) if plan_sequence else []
        tool_decisions = self._load_tool_decisions()
        self._annotate_steps_with_tool(step_details, tool_decisions)
        with open(seq_file_path, "w") as f:
            json.dump(
                {
                    "sequence": sequence,
                    "assemblable": assemblable,
                    "steps": step_details,
                    "tool_decisions": tool_decisions,
                },
                f,
            )

        self.update_sequence(sequence)
        self._apply_step_details(step_details)
        return assemblable

    def _render_plan(self, asset_folder, assembly_dir, plan_sequence, tree, args):
        asap_dir = os.path.join(project_base_dir, "ASAPx")
        if asap_dir not in sys.path:
            sys.path.insert(0, asap_dir)
        # NOTE: "settings" is deliberately NOT in this set. There is exactly
        # one settings.py in the repo (the root one) -- neither ATA nor ASAPx
        # ships its own -- so evicting it cannot resolve a name clash; it only
        # forces a fresh re-read from disk, silently discarding every runtime
        # override the caller set (heuristic_weights_source="optuna",
        # render_sequence=False, debug_stability=False, ...) before planning.
        _asapx_pkgs = {
            "assets",
            "utils",
            "simulation",
            "plan_path",
            "plan_robot",
            "plan_sequence",
        }
        for _mod in list(sys.modules.keys()):
            if _mod in _asapx_pkgs or any(
                _mod.startswith(p + ".") for p in _asapx_pkgs
            ):
                del sys.modules[_mod]
        from ASAPx.plan_sequence.play_logged_plan import (
            play_logged_plan,
            play_subassembly_split,
        )

        render_iso1 = self.assembly.storage_dir / "_render_iso1"
        render_iso2 = self.assembly.storage_dir / "_render_iso2"
        render_iso3 = self.assembly.storage_dir / "_render_iso3"
        render_iso4 = self.assembly.storage_dir / "_render_iso4"

        tool_meshes_per_step = self._build_tool_meshes_for_plan(plan_sequence)

        _connect_path = getattr(args, "connect_path", False)
        _plan_arm = getattr(args, "plan_arm", False)
        _gripper_type = getattr(args, "gripper_type", "rod")
        _gripper_scale = getattr(args, "gripper_scale", 0.4)
        # settings.contact_model = 'rod' overrides the CLI to use the rod
        # contact model regardless of --gripper-type. 'gripper' keeps the
        # CLI choice (panda / robotiq-85 / robotiq-140 / rod).
        _contact_model = getattr(settings, "contact_model", "rod")
        if _contact_model == "rod":
            _gripper_type = "rod"

        # Two-stage arm planning over the whole sequence. Runs before
        # play_logged_plan so the render workers can replay the precomputed
        # arm trajectories instead of doing their own per-step IK + RRT.
        # Gated by both --plan-arm (CLI intent) and settings.arm_continuous
        # (the structural switch). When the latter is False, render workers
        # fall back to the lazy per-step planning they already had.
        _log_dir = self.assembly.storage_dir / "log"
        if _plan_arm and getattr(settings, "arm_continuous", True) and plan_sequence:
            try:
                from ASAPx.plan_robot.arm_pipeline import plan_arm_sequence

                arm_asset_folder = os.path.join(asap_dir, "assets")
                plan_arm_sequence(
                    arm_asset_folder,
                    assembly_dir,
                    plan_sequence,
                    tree,
                    gripper_type=_gripper_type,
                    gripper_scale=_gripper_scale,
                    log_dir=str(_log_dir),
                    num_proc=10,
                )
            except Exception as _arm_e:
                # Non-fatal: render workers will fall back to lazy per-step
                # arm planning (the previous behavior).
                print(
                    f"[sequence_planner] arm_pipeline failed: {_arm_e}; "
                    "renderer will fall back to lazy per-step arm planning."
                )
                import traceback as _tb

                print(_tb.format_exc())

        # When the arm pipeline is in simplified mode, the per-step
        # arm_path_full is None — but `show_arm=True` would still trigger
        # play_logged_plan's lazy fallback (GraspArmPlanner.plan + RRT) per
        # step, which is exactly what simplified mode was meant to avoid.
        # Force show_arm/show_grasp off and drop arm_plans_path so the
        # renderer never enters any arm code path. Disassembly GIFs still
        # render via the part-only MultiPartPathPlanner replay.
        _simplified_arm = bool(getattr(settings, "arm_simplified_mode", False))
        _render_show_arm = _plan_arm and not _simplified_arm
        _render_show_grasp = _plan_arm and not _simplified_arm
        if _simplified_arm and _plan_arm:
            print(
                "[sequence_planner] arm_simplified_mode=True — "
                "rendering with show_arm/show_grasp=False to avoid the "
                "play_logged_plan lazy GraspArmPlanner fallback."
            )

        play_logged_plan(
            asset_folder,
            assembly_dir,
            plan_sequence,
            tree,
            result_dir=str(render_iso1),
            save_mesh=False,
            save_pose=False,
            save_part=False,
            save_path=True,
            save_record=True,
            save_all=False,
            camera_pos=[1.25, -1.5, 1.5],
            camera_lookat=[-1.0, 1.0, 0.0],
            connect_path=_connect_path,
            # When --plan-arm is on AND simplified mode is OFF, show_arm
            # switches the per-step renderer to render_path_with_grasp_and_arm,
            # which is what triggers the RRT-Connect reach/retreat wiring
            # inside render_grasp_arm.py. In simplified mode both flags are
            # forced False here so the renderer stays purely part-based.
            show_grasp=_render_show_grasp,
            show_arm=_render_show_arm,
            gripper_type=_gripper_type,
            gripper_scale=_gripper_scale,
            extra_views=[
                (str(render_iso2 / "record"), [-1.25, 1.5, 1.5], [1.0, -1.0, 0.0]),
                (str(render_iso3 / "record"), [-1.25, -1.5, 1.5], [1.0, 1.0, 0.0]),
                (str(render_iso4 / "record"), [1.25, 1.5, 1.5], [-1.0, -1.0, 0.0]),
            ],
            tool_meshes_per_step=tool_meshes_per_step,
            # Dropped from 72 to 10: at higher parallelism the OpenGL render
            # workers saturate the X server's GLX resources and the whole
            # session crashes mid-batch (see the "X connection broken / fatal
            # IO error" symptom). 10 has comfortable headroom.
            num_proc=10,
            n_frame=settings.n_save_state,
            arm_plans_path=(
                str(_log_dir / "arm_plans.json")
                if _render_show_arm and getattr(settings, "arm_continuous", True)
                else None
            ),
        )

        # Reorganize files to match fetch_sequence_matrices / fetch_imgs_and_gifs expectations:
        # iso1 path/ → storage_dir/paths/
        src_path = render_iso1 / "path"
        if src_path.exists():
            shutil.move(str(src_path), str(self.assembly.storage_dir / "paths"))

        # iso1 GIFs → storage_dir/{i}_{obj_id}.gif  (matches glob *_{obj_id}.gif)
        src_record = render_iso1 / "record"
        if src_record.exists():
            for gif in src_record.glob("*.gif"):
                shutil.move(str(gif), str(self.assembly.storage_dir / gif.name))

        # Arm-overlay GIFs (only produced when --plan-arm is on; written to
        # record_grasp by the per-step worker via record_dir_grasp). Without
        # this move the rmtree below silently deletes them.
        src_record_grasp = render_iso1 / "record_grasp"
        if src_record_grasp.exists():
            for gif in src_record_grasp.glob("*.gif"):
                shutil.move(
                    str(gif), str(self.assembly.storage_dir / f"{gif.stem}_arm.gif")
                )

        # iso2 GIFs → storage_dir/{i}_{obj_id}_opposite.gif  (matches glob *_{obj_id}_opposite.gif)
        src_record_iso2 = render_iso2 / "record"
        if src_record_iso2.exists():
            for gif in src_record_iso2.glob("*.gif"):
                shutil.move(
                    str(gif),
                    str(self.assembly.storage_dir / f"{gif.stem}_opposite.gif"),
                )

        # iso3 GIFs → storage_dir/{i}_{obj_id}_iso3.gif  (matches glob *_{obj_id}_iso3.gif)
        src_record_iso3 = render_iso3 / "record"
        if src_record_iso3.exists():
            for gif in src_record_iso3.glob("*.gif"):
                shutil.move(
                    str(gif), str(self.assembly.storage_dir / f"{gif.stem}_iso3.gif")
                )

        # iso4 GIFs → storage_dir/{i}_{obj_id}_iso4.gif  (matches glob *_{obj_id}_iso4.gif)
        src_record_iso4 = render_iso4 / "record"
        if src_record_iso4.exists():
            for gif in src_record_iso4.glob("*.gif"):
                shutil.move(
                    str(gif), str(self.assembly.storage_dir / f"{gif.stem}_iso4.gif")
                )

        shutil.rmtree(str(render_iso1), ignore_errors=True)
        shutil.rmtree(str(render_iso2), ignore_errors=True)
        shutil.rmtree(str(render_iso3), ignore_errors=True)
        shutil.rmtree(str(render_iso4), ignore_errors=True)

        # Subassembly-split render: visualise the divide-optimizer split (if one
        # was persisted). Renders R separating from S plus each subassembly's
        # internal disassembly into storage_dir/subassembly/.
        stats_file = self.assembly.storage_dir / "log" / "stats.json"
        divide_split = None
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    divide_split = json.load(f).get("divide_split")
            except (json.JSONDecodeError, OSError):
                divide_split = None
        if divide_split:
            play_subassembly_split(
                asset_folder,
                assembly_dir,
                divide_split,
                plan_sequence,
                tree,
                result_dir=str(self.assembly.storage_dir / "subassembly"),
                connect_path=_connect_path,
                camera_pos=[1.25, -1.5, 1.5],
                camera_lookat=[-1.0, 1.0, 0.0],
            )
