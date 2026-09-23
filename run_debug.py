"""Debugging / diagnostic subcommands: focused visual checks and
experiments (collision, PCA, tools, gravity, divide-optimizer, param
sweeps). Not part of the product pipeline or the data outputs."""

import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pyvista as pv

import settings
from run_common import resolve_ids
from core.assembly import Eval
from core.perturbation import resolve_collision
from core.simulation import ContactTree


def _setup_param_sweep_imports():
    """Import ASAPx modules once and install a seq_plan wrapper that applies
    the current sweep params (KN, DAMPING, COL_TH_STABLE) before every call.

    The wrapper is needed because get_assembly_plans_ASAP evicts plan_sequence.*
    from sys.modules on every call and then re-imports run_seq_plan via the
    ASAPx-prefixed package. ASAPx.plan_sequence.run_seq_plan is NOT evicted
    (the eviction filter only matches non-prefixed module paths), so attaching
    a wrapper to its seq_plan attribute persists across iterations.

    Returns (params_dict, sim_string_module, physics_planner_module, DivideOptimizer).
    Mutate params_dict before each iteration; the wrapper reads from it lazily.
    """
    project_base_dir = os.path.dirname(os.path.abspath(__file__))
    asap_dir = os.path.join(project_base_dir, "ASAPx")
    if asap_dir not in sys.path:
        sys.path.insert(0, asap_dir)

    _asapx_pkgs = {
        "assets",
        "utils",
        "simulation",
        "plan_path",
        "plan_robot",
        "plan_sequence",
        "settings",
    }
    for _mod in list(sys.modules.keys()):
        if _mod in _asapx_pkgs or any(_mod.startswith(p + ".") for p in _asapx_pkgs):
            del sys.modules[_mod]

    # Eagerly import optimizer too, so its module-level bindings point at the
    # same sim_string/physics_planner module objects that run_seq_plan does.
    # If we re-imported optimizer later (after the planner evicts plan_sequence.*),
    # optimizer would rebind to fresh sim_string/physics_planner modules whose
    # KN/DAMPING/COL_TH_STABLE are at compile-time defaults, defeating the sweep.
    from plan_sequence.optimizer import DivideOptimizer

    import ASAPx.plan_sequence.run_seq_plan as _rsp

    # Capture the same module objects that run_seq_plan's and optimizer's
    # functions reference via __globals__. Patching attributes on these objects
    # affects the running planner and optimizer even after sys.modules is later
    # cleared by get_assembly_plans_ASAP's eviction loop.
    _sim = sys.modules["plan_sequence.sim_string"]
    _phys = sys.modules["plan_sequence.physics_planner"]

    _params = {
        "KN": _sim.KN,
        "DAMPING": _sim.DAMPING,
        "COL_TH_STABLE": _phys.COL_TH_STABLE,
    }

    if not hasattr(_rsp, "_param_sweep_orig_seq_plan"):
        _rsp._param_sweep_orig_seq_plan = _rsp.seq_plan
    _orig_seq_plan = _rsp._param_sweep_orig_seq_plan

    def _patched_seq_plan(*a, **kw):
        _sim.KN = _params["KN"]
        _sim.DAMPING = _params["DAMPING"]
        _phys.COL_TH_STABLE = _params["COL_TH_STABLE"]
        _phys.MultiPartStabilityPlanner.col_th = _params["COL_TH_STABLE"]
        print(
            f"[sweep] patched: KN={_sim.KN:g}  DAMPING={_sim.DAMPING:g}  "
            f"COL_TH_STABLE={_phys.MultiPartStabilityPlanner.col_th:g}"
        )
        return _orig_seq_plan(*a, **kw)

    _rsp.seq_plan = _patched_seq_plan
    return _params, _sim, _phys, DivideOptimizer


def _compute_dfa_failure_stats(tree):
    """Per-edge fail-reason counts for a sequence-planning DFA tree.

    Each edge in the tree carries a sim_info dict whose `fail_reason` is one of
    {None, 'assembly', 'stability', 'tool', 'grasp'}. Returns counts plus the
    total. None means feasible.
    """
    counts = {"feasible": 0, "assembly": 0, "stability": 0, "tool": 0, "grasp": 0}
    total = 0
    for _, _, data in tree.edges(data=True):
        sim_info = data.get("sim_info") or {}
        reason = sim_info.get("fail_reason")
        key = "feasible" if reason is None else reason
        counts[key] = counts.get(key, 0) + 1
        total += 1
    return total, counts


def _run_divide_optimizer_on_tree(ass, args, tree, output_dir, DivideOptimizer):
    """Run the same divide-optimizer pipeline as test_divide_optimizer, but
    on the in-memory `tree` and with all artifacts going to `output_dir`.

    DivideOptimizer must be the class captured at sweep-setup time, so its
    module bindings still point at the patched sim_string / physics_planner.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = DivideOptimizer(
        tree,
        asset_folder=str(Path("assets").resolve()),
        assembly_dir=str(ass.assembly_dir),
    )
    if opt.build_obstruction_graph() is None:
        return None

    opt.visualize_obstruction_graph(
        save_path=str(output_dir / "obstruction_graph.png"), show=False
    )
    opt.visualize_symmetric_additions(
        save_path=str(output_dir / "obstruction_graph_symmetric.png"), show=False
    )
    opt.find_locally_free_subassemblies(timeout=100)
    opt.verify_locally_free(top_k=10, num_proc=getattr(args, "num_proc", 8))

    meshes = {obj_id: obj.mesh for obj_id, obj in ass.objects.items()}
    opt.visualize_subassemblies(
        meshes=meshes,
        output_dir=output_dir / "subassemblies",
        top_n=10,
        bottom_n=10,
    )
    return opt


def run_test_collision(args, test_eval, output_folder, assembly_dir):
    for ass in test_eval.assemblies:
        # ass.get_assembly_plans(args)
        names_colliding_boolean, _, _ = ass.get_collisions(mode="boolean", draw=True)
        names_colliding_depth, _, _ = ass.get_collisions(mode="depth", draw=True)
        print(
            f"Assembly {ass.id} — Boolean method: {'Collisions detected' if names_colliding_boolean else 'No collisions detected'}"
        )
        if names_colliding_boolean:
            print(f"Overlapping parts: {names_colliding_boolean}")
            ass.view_collisions(names_colliding_boolean)

        print(
            f"Assembly {ass.id} — Depth method: {'Collisions detected' if names_colliding_depth else 'No collisions detected'}"
        )
        if names_colliding_depth:
            print(f"Overlapping parts: {names_colliding_depth}")
            ass.view_collisions(names_colliding_depth)


def run_test_PCA(args, test_eval, output_folder, assembly_dir):
    test_eval = Eval(output_dir=output_folder)
    for _id in resolve_ids(
        args.id, assembly_dir, max_parts=args.max_parts, min_parts=args.min_parts
    ):
        test_eval.add_assembly(dir=assembly_dir, id=_id)
    tool = "screwdriver"
    for ass in test_eval.assemblies:
        for obj in ass.objects.values():
            tool_position, _ = ass.apply_tool(tool, obj.id, show=True)
            if ass.check_tool_collision(tool_position, part_name=obj.name, show=True):
                print(
                    f"Tool collision detected for part {obj.name}. Trying inverse position."
                )
                tool_position, _ = ass.apply_tool(tool, obj.id, show=True, invert=True)
                if ass.check_tool_collision(
                    tool_position, part_name=obj.name, show=True
                ):
                    print(
                        f"Inverse position also causes collision for part {obj.name}. Tool cannot be applied to position"
                    )
                    continue

            if not ass.check_tool_assemblable(tool_position, show=True, ass_obj=obj):
                print(f"Part {obj.name} can not be reached by the given tool.")


def run_test_tools(args, test_eval, output_folder, assembly_dir):
    for ass in test_eval.assemblies:
        ass.images
        ass.name_parts(iso_only=False, log_probs=False)
        print(f"generated names: {[obj.name for obj in ass.objects.values()]}")
        ass.get_assembly_plans(args)

        if settings.angle_ranking:
            ass.planner.rank_sequence_angles(show=False)

        summary = []
        for step_idx, step in enumerate(ass.sequence):
            obj = ass.objects[step.obj_id]
            tool, confidence = ass.check_tool_needed(
                obj_idx=obj.id, show=False, show_part=False, logprobs=True
            )
            print(f"Decision: {tool}, Confidence: {confidence}.")

            if tool is None or confidence is None:
                print(f"Unable to determine tool for part {obj.name}.")
                summary.append(
                    {
                        "part": obj.name,
                        "part_nr": obj.id,
                        "tool": "Unknown",
                        "confidence": "N/A",
                        "applied": False,
                        "reason": "Model unable to determine tool",
                    }
                )
                continue

            if tool == "unclear" or confidence < 0.85:
                print("Chattie not confident, trying different angle.")
                orig_tool, orig_confidence = tool, confidence
                tool, confidence = ass.check_tool_needed(
                    obj_idx=obj.id,
                    show=False,
                    show_part=False,
                    opposite=True,
                    logprobs=True,
                )
                print(f"Opposite Angle. Decision: {tool}, Confidence: {confidence}.")

                if tool == "unclear" or confidence < 0.85:
                    if orig_confidence is not None and (
                        confidence is None or orig_confidence > confidence
                    ):
                        tool, confidence = orig_tool, orig_confidence
                        print(
                            f"Reverting to original angle. Decision: {tool}, Confidence: {confidence}."
                        )

            if tool in {"error", "unclear"}:
                print("Chattie feedback ambigous, moving on to next part")
                summary.append(
                    {
                        "part": obj.name,
                        "part_nr": obj.id,
                        "tool": tool,
                        "confidence": confidence,
                        "applied": False,
                        "reason": "Ambiguous model feedback",
                    }
                )
                continue
            elif tool == "none":
                summary.append(
                    {
                        "part": obj.name,
                        "part_nr": obj.id,
                        "tool": "none",
                        "confidence": confidence,
                        "applied": True,
                        "reason": "No tool needed",
                    }
                )
                continue

            if tool:
                tool_position, _ = ass.apply_tool(tool, obj.id, show=True)

                if ass.check_tool_collision(
                    tool_position, move_id=obj.id, show=False, step_nr=step_idx
                ):
                    print(
                        f"Tool collision detected for part {obj.name}. Trying inverse position."
                    )
                    tool_position, _ = ass.apply_tool(
                        tool, obj.id, show=True, invert=True
                    )
                    if ass.check_tool_collision(
                        tool_position, move_id=obj.id, show=False, step_nr=step_idx
                    ):
                        print(
                            f"Inverse position also causes collision for part {obj.name}. Tool cannot be applied to position"
                        )
                        summary.append(
                            {
                                "part": obj.name,
                                "part_nr": obj.id,
                                "tool": tool,
                                "confidence": confidence,
                                "applied": False,
                                "reason": "Collision detected in normal and inverse tool position",
                            }
                        )
                        continue

                if not ass.check_tool_assemblable(
                    tool_position, show=False, ass_obj=obj, step_nr=step_idx
                ):
                    print(f"Part {obj.name} can not be reached by the given tool.")
                    summary.append(
                        {
                            "part": obj.name,
                            "part_nr": obj.id,
                            "tool": tool,
                            "confidence": confidence,
                            "applied": False,
                            "reason": "Part cannot be reached by the required tool",
                        }
                    )
                else:
                    print(f"Part {obj.name} has to to be assembled using {tool}")
                    summary.append(
                        {
                            "part": obj.name,
                            "part_nr": obj.id,
                            "tool": tool,
                            "confidence": confidence,
                            "applied": True,
                            "reason": "",
                        }
                    )

        print("\n========== TOOL ANALYSIS SUMMARY ==========")
        print(
            f"Sequence: {[ass.sequence[step].obj_id for step in range(len(ass.sequence))]}"
        )

        print("---Settings---")  # WARNING: MOSTLY HARDCODED!
        print(f"Part naming: {settings.part_naming}")
        print(f"Angle Ranking: {settings.angle_ranking}")
        print(f"Color Scheme: {settings.color_scheme}")
        print("Confidence Adjustment: False")
        print(f"LLM Model: {settings.LLM_model}")
        for entry in summary:
            print(f"Part: {entry['part']}, Nr: {entry['part_nr']}")
            print(f"  - Tool needed: {entry['tool']}")
            print(f"  - Confidence:  {entry['confidence']}")
            print(f"  - Applicable?: {'Yes' if entry['applied'] else 'No'}")
            if not entry["applied"]:
                print(f"  - Reason:      {entry['reason']}")
        print("===========================================\n")


def run_test_collision_resolver(args, test_eval, output_folder, assembly_dir):
    for test_ass in test_eval.assemblies:
        names_colliding, _, _ = test_ass.get_collisions(draw=True, show=True)
        if names_colliding:
            for obj1_name, obj2_name in names_colliding:
                obj1 = test_ass.objects[obj1_name]
                test_ass.objects[obj2_name]
                # test_ass.view_collisions(names_overlapping = [(obj1_name, obj2_name)], show=True)
                mesh_list = [
                    obj.tri_mesh
                    for obj in test_ass.objects.values()
                    if obj.id != obj1.id
                ]
                _converged, _dist, mesh = resolve_collision(
                    meshes=mesh_list, moving_part=obj1.tri_mesh, draw=True
                )
                obj1.tri_mesh = mesh
                obj1.mesh = pv.wrap(mesh)
                test_ass.view_collisions(
                    names_overlapping=[(obj1_name, obj2_name)], show=True
                )
        else:
            print(f"\nNo collisions found for assembly {test_ass.id}")


def run_test_collision_graph_batch(args, test_eval, output_folder, assembly_dir):
    print(
        f"Building collision graphs for {len(test_eval.assemblies)} assembly ID(s)..."
    )
    for ass in test_eval.assemblies:
        print(f"Processing assembly {ass.id}...")

        # Render full assembly from iso view (off-screen)
        plotter = pv.Plotter(off_screen=True)
        for obj in ass.objects.values():
            plotter.add_mesh(obj.mesh, color="lightgray", show_edges=False)
        plotter.camera_position = "iso"
        assembly_img = plotter.screenshot(return_img=True)
        plotter.close()

        # Build collision contact graph
        meshes = {obj.id: obj.tri_mesh for obj in ass.objects.values()}
        tree = ContactTree(meshes)
        print(f"  Edges: {list(tree.G.edges())}")

        # Compose side-by-side: assembly render | collision graph
        fig, (ax_render, ax_graph) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Assembly {ass.id}", fontsize=14)

        ax_render.imshow(assembly_img)
        ax_render.axis("off")
        ax_render.set_title("Isometric view")

        nx.draw(tree.G, with_labels=True, ax=ax_graph)
        ax_graph.set_title(f"Collision graph  ({len(tree.G.edges())} edge(s))")

        fig.tight_layout()
        fig.savefig(output_folder / f"batch_{ass.id}.png", dpi=100)
        plt.close(fig)


def run_test_divide_optimizer(args, test_eval, output_folder, assembly_dir):
    project_base_dir = os.path.dirname(os.path.abspath(__file__))
    asap_dir = os.path.join(project_base_dir, "ASAPx")
    if asap_dir not in sys.path:
        sys.path.insert(0, asap_dir)
    _asapx_pkgs = {
        "assets",
        "utils",
        "simulation",
        "plan_path",
        "plan_robot",
        "plan_sequence",
        "settings",
    }
    for _mod in list(sys.modules.keys()):
        if _mod in _asapx_pkgs or any(_mod.startswith(p + ".") for p in _asapx_pkgs):
            del sys.modules[_mod]
    from plan_sequence.optimizer import DivideOptimizer

    for ass in test_eval.assemblies:
        tree_file = ass.storage_dir / "log" / "tree.pkl"
        if not tree_file.exists():
            print(f"[{ass.id}] Missing {tree_file}. Run sequence planning first.")
            continue
        with open(tree_file, "rb") as f:
            tree = pickle.load(f)

        opt = DivideOptimizer(
            tree,
            asset_folder=str(Path("assets").resolve()),
            assembly_dir=str(ass.assembly_dir),
        )
        if opt.build_obstruction_graph() is None:
            continue

        save_path = ass.storage_dir / "obstruction_graph.png"
        opt.visualize_obstruction_graph(save_path=str(save_path), show=False)
        print(f"[{ass.id}] Wrote obstruction graph to {save_path}")

        sym_save = ass.storage_dir / "obstruction_graph_symmetric.png"
        opt.visualize_symmetric_additions(save_path=str(sym_save), show=False)
        print(f"[{ass.id}] Wrote symmetric-additions graph to {sym_save}")

        opt.find_locally_free_subassemblies(timeout=100)

        opt.verify_locally_free(
            top_k=10,
            num_proc=getattr(args, "num_proc", 80),
        )

        meshes = {obj_id: obj.mesh for obj_id, obj in ass.objects.items()}
        sub_dir = ass.storage_dir / "subassemblies"
        opt.visualize_subassemblies(
            meshes=meshes, output_dir=sub_dir, top_n=10, bottom_n=10
        )

        # Cost comparison: original flat sequence vs split-based sequence
        # (prefix + unified-split step + re-planned S + re-planned R).
        verified = getattr(opt, "verified_locally_free", None) or []
        if verified:
            from plan_sequence.optimizer.compare import (
                compare_with_split,
                print_comparison,
            )

            chosen = verified[0]
            split_dict = {"S": list(chosen[0]), "R": list(chosen[1])}
            result = compare_with_split(
                tree=tree,
                asset_folder=str(Path("assets").resolve()),
                assembly_dir=str(ass.assembly_dir),
                split=split_dict,
                num_proc=getattr(args, "num_proc", 80),
                debug=1,
            )
            print_comparison(result)
            out = ass.storage_dir / "split_comparison.json"

            def _serialize(o):
                if hasattr(o, "tolist"):  # numpy arrays / scalars
                    return o.tolist()
                if isinstance(o, (frozenset, set)):
                    return sorted(o)
                if isinstance(o, tuple):
                    return list(o)
                return str(o)

            with open(out, "w") as f:
                json.dump(result, f, default=_serialize, indent=2)
            print(f"[{ass.id}] split comparison written to {out}")
        else:
            print(f"[{ass.id}] no verified split; skipping comparison")


def run_test_tool_naming(args, test_eval, output_folder, assembly_dir):
    for ass in test_eval.assemblies:
        # Render an iso1 view for every canonical tool (none exist on disk by default)
        tools_dir = ass.storage_dir / "tool_images"
        tools_dir.mkdir(parents=True, exist_ok=True)
        for tool in test_eval.tools.values():
            if tool.image_paths is None:
                tool.image_paths = {}
            img_path = tools_dir / f"{tool.id}_iso1.png"
            if not img_path.exists():
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(tool.mesh, color="lightgray", show_edges=False)
                plotter.camera_position = "iso"
                plotter.screenshot(img_path)
                plotter.close()
            tool.image_paths["iso1"] = img_path

        names_cache = ass.storage_dir / "tool_names.json"
        if names_cache.exists():
            print(
                f"[{ass.id}] Removing stale cache {names_cache} to force fresh API call"
            )
            names_cache.unlink()

        print(f"\n[{ass.id}] --- First call: batched API request (iso_only=True) ---")
        ass.tool_analyzer.name_tools(iso_only=True)
        for tool in test_eval.tools.values():
            print(f"  {tool.id}: name={tool.name!r}  desc={tool.description!r}")
        assert names_cache.exists(), "Cache file was not written"

        print(f"\n[{ass.id}] --- Second call: should load from cache, no API ---")
        tokens_before = test_eval.tokens_used
        for tool in test_eval.tools.values():
            tool.name = None
            tool.description = None
        ass.tool_analyzer.name_tools(iso_only=True)
        assert test_eval.tokens_used == tokens_before, (
            "Cache hit unexpectedly consumed tokens"
        )
        for tool in test_eval.tools.values():
            print(f"  {tool.id}: name={tool.name!r}  desc={tool.description!r}")
            assert tool.name and tool.description, (
                f"Tool {tool.id} missing name/description after cache load"
            )

        # Confirm names and descriptions also reach the per-assembly scaled tools
        for tool_id, scaled in ass.scaled_tools.items():
            src = test_eval.tools[tool_id]
            assert scaled.name == src.name and scaled.description == src.description, (
                f"scaled_tools[{tool_id}] not in sync with evaluation.tools"
            )
        print(f"[{ass.id}] OK — names & descriptions propagated to scaled_tools")


def run_test_gravity(args, test_eval, output_folder, assembly_dir):
    for test_ass in test_eval.assemblies:
        # test_ass.simulation.test_gravity(show=True)
        test_ass.simulation.test_stable(show=True)


def run_test_param_sweep(args, test_eval, output_folder, assembly_dir):
    import numpy as _np

    # Force fresh sequence runs so cached sequence.json files aren't reused.
    test_eval.cache = "new"

    # Parameter grid
    KN_VALUES = [1e3, 1e6]
    COL_TH_VALUES = [0.00, 0.01]
    DAMPING_VALUES = [1e3, 5e1]

    sweep_root = Path(output_folder) / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    _params, _sim_mod, _phys_mod, _DivideOptimizer = _setup_param_sweep_imports()

    # results[config_label] = {
    #   'params': {...},
    #   'assemblies': {ass_id: {'counts': {...}, 'total': N, 'assemblable': bool}},
    #   'aggregate': {'total': N, 'assembly_pct': %, 'stability_pct': %, 'tool_pct': %, 'feasible_pct': %},
    # }
    results = {}

    combos = [
        (kn, col, damp)
        for kn in KN_VALUES
        for col in COL_TH_VALUES
        for damp in DAMPING_VALUES
    ]
    print(
        f"\n[test_param_sweep] {len(combos)} configurations × {len(test_eval.assemblies)} assemblies "
        f"= {len(combos) * len(test_eval.assemblies)} runs total.\n"
    )

    for kn, col_th, damp in combos:
        config_label = f"KN{kn:g}_COL{col_th:g}_DAMP{damp:g}"
        config_dir = sweep_root / config_label
        config_dir.mkdir(parents=True, exist_ok=True)

        # Update closure params; the seq_plan wrapper reads these lazily.
        _params["KN"] = kn
        _params["DAMPING"] = damp
        _params["COL_TH_STABLE"] = col_th

        # Also patch immediately so the modules reflect the new values
        # even outside of the wrapped seq_plan call (e.g. divide optimizer).
        _sim_mod.KN = kn
        _sim_mod.DAMPING = damp
        _phys_mod.COL_TH_STABLE = col_th
        _phys_mod.MultiPartStabilityPlanner.col_th = col_th

        print(f"\n{'=' * 70}\n[sweep] config {config_label}\n{'=' * 70}")

        config_record = {
            "params": {"KN": kn, "COL_TH_STABLE": col_th, "DAMPING": damp},
            "assemblies": {},
        }

        for ass in test_eval.assemblies:
            ass_out = config_dir / ass.id
            ass_out.mkdir(parents=True, exist_ok=True)
            print(f"\n[sweep:{config_label}] assembly {ass.id} ...")

            # Wipe stale sequence cache so the planner re-runs.
            seq_json = ass.storage_dir / "sequence.json"
            log_dir = ass.storage_dir / "log"
            if seq_json.exists():
                seq_json.unlink()
            if log_dir.exists():
                shutil.rmtree(str(log_dir))

            record = {
                "counts": None,
                "total": 0,
                "assemblable": None,
                "seq_planner_error": None,
                "divide_optimizer_error": None,
            }

            # Sequence planner
            try:
                assemblable = ass.planner.get_assembly_plans(args)
                record["assemblable"] = bool(assemblable)
            except Exception as _exc:
                import traceback as _tb

                print(
                    f"[sweep:{config_label}:{ass.id}] sequence planner FAILED: {_exc}"
                )
                _tb.print_exc()
                record["seq_planner_error"] = str(_exc)

            # Load tree, compute stats, copy artifacts to per-config dir
            tree_path = ass.storage_dir / "log" / "tree.pkl"
            tree = None
            if tree_path.exists():
                try:
                    with open(tree_path, "rb") as f:
                        tree = pickle.load(f)
                    total, counts = _compute_dfa_failure_stats(tree)
                    record["counts"] = counts
                    record["total"] = total
                    shutil.copy(str(tree_path), str(ass_out / "tree.pkl"))
                    stats_path = ass.storage_dir / "log" / "stats.json"
                    if stats_path.exists():
                        shutil.copy(str(stats_path), str(ass_out / "stats.json"))
                    if seq_json.exists():
                        shutil.copy(str(seq_json), str(ass_out / "sequence.json"))
                    print(f"[sweep:{config_label}:{ass.id}] edges={total}  {counts}")
                except Exception as _exc:
                    print(f"[sweep:{config_label}:{ass.id}] tree-load error: {_exc}")
            else:
                print(f"[sweep:{config_label}:{ass.id}] no tree.pkl produced")

            # Divide optimizer (same steps as test_divide_optimizer)
            if tree is not None:
                try:
                    _run_divide_optimizer_on_tree(
                        ass,
                        args,
                        tree,
                        ass_out / "divide_optimizer",
                        _DivideOptimizer,
                    )
                except Exception as _exc:
                    import traceback as _tb

                    print(
                        f"[sweep:{config_label}:{ass.id}] divide optimizer FAILED: {_exc}"
                    )
                    _tb.print_exc()
                    record["divide_optimizer_error"] = str(_exc)

            config_record["assemblies"][ass.id] = record

        # Aggregate across assemblies for this config
        total_edges = sum(r["total"] for r in config_record["assemblies"].values())
        summed = {"feasible": 0, "assembly": 0, "stability": 0, "tool": 0, "grasp": 0}
        for r in config_record["assemblies"].values():
            if r["counts"]:
                for k, v in r["counts"].items():
                    summed[k] = summed.get(k, 0) + v
        if total_edges > 0:
            aggregate = {
                "total": total_edges,
                "assembly_pct": 100.0 * summed["assembly"] / total_edges,
                "stability_pct": 100.0 * summed["stability"] / total_edges,
                "tool_pct": 100.0 * summed["tool"] / total_edges,
                "grasp_pct": 100.0 * summed["grasp"] / total_edges,
                "feasible_pct": 100.0 * summed["feasible"] / total_edges,
                "counts": summed,
            }
        else:
            aggregate = {
                "total": 0,
                "assembly_pct": 0.0,
                "stability_pct": 0.0,
                "tool_pct": 0.0,
                "grasp_pct": 0.0,
                "feasible_pct": 0.0,
                "counts": summed,
            }
        config_record["aggregate"] = aggregate

        with open(config_dir / "summary.json", "w") as _f:
            json.dump(config_record, _f, indent=2)
        results[config_label] = config_record

    # ----- Save combined JSON -----
    with open(sweep_root / "results.json", "w") as _f:
        json.dump(
            {
                "params_grid": {
                    "KN": KN_VALUES,
                    "COL_TH_STABLE": COL_TH_VALUES,
                    "DAMPING": DAMPING_VALUES,
                },
                "configs": results,
            },
            _f,
            indent=2,
        )

    # ----- Plot (b): grouped bars per config, both metrics overlaid -----
    labels = list(results.keys())
    asm_pcts = [results[c]["aggregate"]["assembly_pct"] for c in labels]
    stb_pcts = [results[c]["aggregate"]["stability_pct"] for c in labels]
    x = _np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(labels)), 6))
    ax.bar(x - width / 2, asm_pcts, width, label="Assembly fail %", color="#d95f5f")
    ax.bar(x + width / 2, stb_pcts, width, label="Stability fail %", color="#5f8fd9")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Per-DFA-node failure %")
    ax.set_title("Sequence-planner failure rates across (KN, COL_TH_STABLE, DAMPING)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(sweep_root / "combined_bars.png", dpi=150)
    plt.close(fig)

    # ----- Plot (c): two heatmaps, x=KN, y=(COL_TH × DAMPING) -----
    ycombos = [(c, d) for c in COL_TH_VALUES for d in DAMPING_VALUES]
    ylabels = [f"col={c:g}\ndamp={d:g}" for c, d in ycombos]
    xlabels = [f"KN={k:g}" for k in KN_VALUES]
    asm_grid = _np.zeros((len(ycombos), len(KN_VALUES)))
    stb_grid = _np.zeros((len(ycombos), len(KN_VALUES)))
    for i, (col_th, damp) in enumerate(ycombos):
        for j, kn in enumerate(KN_VALUES):
            label = f"KN{kn:g}_COL{col_th:g}_DAMP{damp:g}"
            agg = results[label]["aggregate"]
            asm_grid[i, j] = agg["assembly_pct"]
            stb_grid[i, j] = agg["stability_pct"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4 + 0.3 * len(ycombos)))
    for ax, grid, title in [
        (axes[0], asm_grid, "Assembly fail %"),
        (axes[1], stb_grid, "Stability fail %"),
    ]:
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_title(title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if grid[i, j] < grid.max() * 0.6 else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, shrink=0.8, label="%")
    fig.suptitle("Per-DFA-node fail % (rows = COL_TH_STABLE × DAMPING, cols = KN)")
    fig.tight_layout()
    fig.savefig(sweep_root / "heatmap.png", dpi=150)
    plt.close(fig)

    # ----- Plain-text overview -----
    with open(sweep_root / "overview.txt", "w") as _f:
        _f.write("Parameter sweep overview\n")
        _f.write("=" * 70 + "\n\n")
        _f.write(f"Assemblies: {[a.id for a in test_eval.assemblies]}\n")
        _f.write(
            f"Grid: KN={KN_VALUES}  COL_TH_STABLE={COL_TH_VALUES}  DAMPING={DAMPING_VALUES}\n\n"
        )
        _f.write(
            f"{'config':<40}  {'edges':>7}  {'asm%':>7}  {'stb%':>7}  {'tool%':>7}  {'ok%':>7}\n"
        )
        _f.write("-" * 80 + "\n")
        for label, rec in results.items():
            agg = rec["aggregate"]
            _f.write(
                f"{label:<40}  {agg['total']:>7d}  "
                f"{agg['assembly_pct']:>6.2f}%  {agg['stability_pct']:>6.2f}%  "
                f"{agg['tool_pct']:>6.2f}%  {agg['feasible_pct']:>6.2f}%\n"
            )
    print(f"\n[test_param_sweep] outputs written to {sweep_root}")


def run_test_archive_ASAP(args, test_eval, output_folder, assembly_dir):
    # ------------------------------------------------------------------
    # Run the upstream archive/ASAP sequence planner on each assembly
    # via SequencePlanner.get_assembly_plans_ASAP_archive (dispatched by
    # setting args.seq_planner = "ASAP-archive"). Mirrors data_validate_cost:
    # wipe stale cache per assembly so the planner actually re-runs.
    # ------------------------------------------------------------------
    _orig_seq_planner = getattr(args, "seq_planner", None)
    _orig_cache = test_eval.cache
    try:
        args.seq_planner = "ASAP-archive"
        test_eval.cache = "new"

        for ass in test_eval.assemblies:
            print(f"\n[archive-ASAP] ===== assembly {ass.id} =====")
            try:
                seq_json = ass.storage_dir / "sequence.json"
                log_dir = ass.storage_dir / "log"
                if seq_json.exists():
                    seq_json.unlink()
                if log_dir.exists():
                    shutil.rmtree(str(log_dir))

                _t0 = time.perf_counter()
                ass.planner.get_assembly_plans(args)
                print(f"[{ass.id}] OK  ({time.perf_counter() - _t0:.1f}s)")
            except Exception as _exc:
                import traceback as _tb

                print(f"[archive-ASAP] {ass.id} ERROR: {_exc}")
                _tb.print_exc()
    finally:
        if _orig_seq_planner is not None:
            args.seq_planner = _orig_seq_planner
        test_eval.cache = _orig_cache
