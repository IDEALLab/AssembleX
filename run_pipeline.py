"""Pipeline subcommands: the end-to-end disassembly+manual pipeline
(test_pipeline), its parallel batch driver (test_pipeline_batch), and
re-rendering from saved plans (test_render)."""

import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import pyvista as pv

import settings
from run_common import _write_comparison_batch_summary, _write_llm_batch_summary
from core.perturbation import resolve_collision


def _balance_concurrency(max_frontier, budget=80, max_poses=3):
    """Pick (n_concurrent, per_assembly_num_proc) from a fixed worker budget.

    Each assembly's per-iter parallel batch is roughly `max_frontier * max_poses`
    sim tasks (one parent at a time × candidate parts × poses). When that's
    small, a single assembly leaves many workers idle, so we want many
    assemblies in flight; when it's large, one or two assemblies will saturate
    the cores on their own and parallelizing further just oversubscribes.

    Examples (budget=80, max_poses=3):
        max_frontier=1  → 26 concurrent × 3 workers each
        max_frontier=3  →  8 concurrent × 10 workers each
        max_frontier=5  →  5 concurrent × 16 workers each
        max_frontier=10 →  2 concurrent × 40 workers each
        max_frontier=20+→  1 concurrent × 80 workers
    """
    per_assembly_estimate = max(1, int(max_frontier) * int(max_poses or 1))
    n_concurrent = max(1, budget // per_assembly_estimate)
    per_assembly_num_proc = max(1, budget // n_concurrent)
    return n_concurrent, per_assembly_num_proc


def _build_child_pipeline_cmd(parent_argv, ass_id, storage_dir, num_proc):
    """Rebuild a single-id `test_pipeline` invocation by forwarding the parent's
    argv with the test_type swapped to `test_pipeline` and the conflicting flags
    (`--id`, `--num-proc`, `--storage-dir`) replaced.

    Any other flag the user passed to the parent (--planner, --cache,
    --token-limit, -x, --allow-gap, etc.) is preserved verbatim. `-u` is added
    so the subprocess writes stdout/stderr unbuffered — critical for being able
    to see the actual error if it crashes early.
    """
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "test_pipeline"]
    overrides = {"--id", "--num-proc", "--storage-dir"}
    skipped_first_positional = False
    i = 0
    while i < len(parent_argv):
        tok = parent_argv[i]
        if not skipped_first_positional and not tok.startswith("-"):
            skipped_first_positional = True  # consume the original test_type
            i += 1
            continue
        if tok in overrides:
            i += 2  # skip flag + its value
            continue
        cmd.append(tok)
        i += 1
    cmd.extend(
        [
            "--id",
            str(ass_id),
            "--num-proc",
            str(num_proc),
            "--storage-dir",
            str(storage_dir),
        ]
    )
    return cmd


def _run_one_assembly_subprocess(ass_id, storage_dir, num_proc, parent_argv, log_path):
    """Run one assembly's pipeline as a subprocess; redirect stdout+stderr to
    `log_path`. Returns (ass_id, returncode)."""
    import subprocess as _sp

    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_child_pipeline_cmd(parent_argv, ass_id, storage_dir, num_proc)
    with open(log_path, "w") as log_f:
        log_f.write(f"$ {' '.join(cmd)}\n\n")
        log_f.flush()
        proc = _sp.Popen(cmd, stdout=log_f, stderr=_sp.STDOUT)
        rc = proc.wait()
    return ass_id, rc


def _run_assemblies_parallel(
    assemblies, parent_argv, output_folder, n_concurrent, per_assembly_num_proc
):
    """Run `assemblies` via parallel subprocesses with `n_concurrent` in flight.

    Returns dict[ass_id] = {"status": "ok"/"error (rc=N)"/"exception: ...",
                            "timings": {}}.

    Each subprocess's full stdout/stderr lands in
    `<output_folder>/subprocess_logs/<assembly_id>.log` to avoid interleaving.
    """
    import concurrent.futures as _futures
    import threading as _threading

    results = {}
    logs_dir = Path(output_folder) / "subprocess_logs"
    print_lock = _threading.Lock()

    def _safe_print(msg):
        with print_lock:
            print(msg, flush=True)

    # Build the cmd for the very first assembly and print it once so a quick
    # rc=2 / argparse failure is diagnosable without grepping log files.
    if assemblies:
        _preview_cmd = _build_child_pipeline_cmd(
            parent_argv,
            assemblies[0].id,
            assemblies[0].storage_dir,
            per_assembly_num_proc,
        )
        _safe_print("[batch] subprocess template cmd (first assembly):")
        _safe_print("        " + " ".join(_preview_cmd))

    with _futures.ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = {}
        for ass in assemblies:
            log_path = logs_dir / f"{ass.id}.log"
            fut = pool.submit(
                _run_one_assembly_subprocess,
                ass.id,
                ass.storage_dir,
                per_assembly_num_proc,
                parent_argv,
                log_path,
            )
            futures[fut] = (ass, log_path)
            _safe_print(f"[batch] [{ass.id}] queued  (log: {log_path})")
        for fut in _futures.as_completed(futures):
            ass, log_path = futures[fut]
            try:
                ass_id, rc = fut.result()
                status = "ok" if rc == 0 else f"error (rc={rc})"
                results[ass_id] = {"status": status, "timings": {}}
                _safe_print(f"[batch] [{ass_id}] done   {status}  (log: {log_path})")
            except Exception as e:
                results[ass.id] = {"status": f"exception: {e}", "timings": {}}
                _safe_print(f"[batch] [{ass.id}] EXCEPTION: {e}")
    return results


def _ensure_preprocessed(ass):
    """Preprocess an assembly's raw meshes in place when it has not been
    preprocessed yet, detected by the absence of a normalization.json in its
    directory. When present, the assembly is assumed already preprocessed and
    this is a no-op.

    Preprocessing normalizes the part meshes to a unit bounding box and writes
    normalization.json plus the normalized per-part .obj files. Whether the
    source first needs splitting into per-part .obj files (GLB scenes) is
    detected automatically inside preprocess_direct, so no CLI flag is needed.
    The in-memory Assembly's objects and scaled tools are reloaded afterwards so
    downstream stages see the normalized meshes and the new scale.

    Returns True when the assembly is ready (already preprocessed or freshly
    preprocessed), False when preprocessing failed (e.g. non-watertight meshes),
    in which case the caller should skip the assembly.
    """
    from types import SimpleNamespace

    from core.preprocess import preprocess_direct

    asm_dir = Path(ass.assembly_dir)
    if (asm_dir / "normalization.json").exists():
        return True

    print(
        f"[{ass.id}] no normalization.json found; preprocessing meshes in {asm_dir} ..."
    )
    try:
        ok = preprocess_direct(
            SimpleNamespace(
                source_dir=str(asm_dir),
                target_dir=str(asm_dir),
                subdivide=False,
                scale=1.0,
            )
        )
    except Exception as exc:
        print(f"[{ass.id}] preprocessing errored: {exc}; skipping assembly.")
        return False
    if not ok:
        print(
            f"[{ass.id}] preprocessing failed (e.g. non-watertight meshes); skipping assembly."
        )
        return False

    # Refresh the in-memory assembly so the normalized meshes and scale are used
    # by every downstream stage. Renderer/planner/simulation hold only a
    # reference to the assembly and read objects lazily, so updating in place is
    # enough.
    ass.objects = ass.init_objects_()
    ass.init_tools()
    print(f"[{ass.id}] preprocessing complete ({len(ass.objects)} parts).")
    return True


def run_test_pipeline(args, test_eval, output_folder, assembly_dir):
    # Determine active components.
    # No -x: all run. -x gc: only g and c. -x xgc: all except g and c.
    _ALL = set("gcitmfv")
    if args.x is None:
        _active = _ALL
    elif "x" in args.x:
        _active = _ALL - (set(args.x) - {"x"})
    else:
        _active = set(args.x) & _ALL

    _timings = {}

    # Run the pipeline for each assembly
    for ass in test_eval.assemblies:
        # End-to-end entry point: preprocess raw meshes on first use. No-op when
        # the assembly directory already has a normalization.json.
        if not _ensure_preprocessed(ass):
            continue

        if args.user:
            axes_dir = ass.output_dir / "convex_decomp"
            ass.tool_analyzer.select_tool_axes(
                output_dir=axes_dir, mode="user", allow_overwrite=True
            )
            # ass.tool_analyzer.select_object_axes(output_dir=axes_dir, mode="user", allow_overwrite=True)

        # Check if the assembly is stable under gravity
        if "g" in _active:
            _t = time.perf_counter()
            unstable_parts = ass.simulation.test_stable(show=True)
            _timings["gravity"] = (
                _timings.get("gravity", 0.0) + time.perf_counter() - _t
            )
            if unstable_parts:
                print(f"Unstable parts found in assembly {ass.id}: {unstable_parts}")
                input("Attempt to resolve unstable parts? Else continue. [y/n]")
                if input().lower() == "y":
                    raise NotImplementedError()

        # Check for collisions that might break the sequence finding
        if "c" in _active:
            _t = time.perf_counter()
            names_colliding, _, _ = ass.get_collisions(draw=True, show=True)

            if names_colliding:
                print(
                    f"Collisions detected between the following parts: {names_colliding}."
                )

                # Optional: attempt to nudge each colliding part with
                # the geometric collision resolver. Gated by
                # settings.resolve_collisions (default False). Resolved
                # pairs are popped from ass.collisions so only the
                # unresolved remainder is later surfaced as feedback.
                if getattr(settings, "resolve_collisions", False):
                    for obj1_name, obj2_name in names_colliding:
                        obj1 = ass.objects[obj1_name]
                        obj2 = ass.objects[obj2_name]
                        ass.view_collisions(
                            names_overlapping=[(obj1_name, obj2_name)], show=True
                        )
                        mesh_list = [
                            obj.tri_mesh
                            for obj in ass.objects.values()
                            if obj.id != obj1.id
                        ]
                        converged, dist, mesh = resolve_collision(
                            meshes=mesh_list, moving_part=obj1.tri_mesh, draw=True
                        )
                        if converged:
                            obj1.tri_mesh = mesh  # TODO: this won't update the actual mesh files, so the sequence finding will rely on the original mesh
                            obj1.mesh = pv.wrap(mesh)
                            ass.collisions.pop((obj1.id, obj2.id))
                            print(
                                f"Collision resolved for part {obj1.name} and {obj2.name}"
                            )
                            ass.view_collisions(
                                names_overlapping=[(obj1_name, obj2_name)], show=True
                            )
                        else:
                            mesh_list = [
                                obj.tri_mesh
                                for obj in ass.objects.values()
                                if obj.id != obj2.id
                            ]
                            converged, _dist, mesh = resolve_collision(
                                meshes=mesh_list, moving_part=obj2.tri_mesh, draw=True
                            )
                            if converged:
                                obj2.tri_mesh = mesh
                                obj2.mesh = pv.wrap(mesh)
                                ass.collisions.pop((obj1.id, obj2.id))
                                print(
                                    f"Collision resolved for part {obj1.name} and {obj2.name}"
                                )
                                ass.view_collisions(
                                    names_overlapping=[(obj1_name, obj2_name)],
                                    show=True,
                                )
                    print(
                        f"The following collisions could not be resolved: {list(ass.collisions.keys())}"
                    )
                else:
                    print(
                        f"[collisions] resolver disabled "
                        f"(settings.resolve_collisions=False); leaving "
                        f"{len(ass.collisions)} collision(s) for the "
                        f"failure-feedback generator."
                    )

                # Ask the user whether to continue with sequence
                # planning. "n" still runs failure feedback on the
                # remaining collisions so they can see what's wrong
                # before this assembly is skipped.
                _choice = (
                    input("Continue with sequence planning? [Y/n] ").strip().lower()
                )
                if _choice == "n":
                    if ass.collisions:
                        print(
                            f"[collisions] aborting assembly {ass.id}; "
                            f"running failure feedback on "
                            f"{len(ass.collisions)} remaining "
                            f"collision(s) first..."
                        )
                        # Failure feedback references parts by name, so
                        # the same naming pass that the normal flow does
                        # below must happen here too — otherwise the
                        # LLM only sees raw part ids.
                        try:
                            ass.images
                        except Exception as _ie:
                            print(f"[collisions] part image generation errored: {_ie}")
                        if settings.part_naming:
                            try:
                                ass.name_parts()
                            except Exception as _ne:
                                print(f"[collisions] part naming errored: {_ne}")
                        try:
                            ass.feedback.generate_failure_feedback()
                        except Exception as _fe:
                            print(f"[collisions] failure feedback errored: {_fe}")
                    else:
                        print(
                            f"[collisions] aborting assembly {ass.id} "
                            f"(no collisions left to report)."
                        )
                    _timings["collisions"] = (
                        _timings.get("collisions", 0.0) + time.perf_counter() - _t
                    )
                    continue
            _timings["collisions"] = (
                _timings.get("collisions", 0.0) + time.perf_counter() - _t
            )

        # Create images of the individual parts
        _t = time.perf_counter()
        ass.images
        _timings["create_images"] = time.perf_counter() - _t

        # Generate names for the parts based on images
        if settings.part_naming:
            _t = time.perf_counter()
            ass.name_parts()
            _timings["part_naming"] = time.perf_counter() - _t

        # Generate names and descriptions for the available tools
        if settings.tool_naming:
            _t = time.perf_counter()
            # name_tools(iso_only=True) reads tool.image_paths['iso1'],
            # which is None for tools by default. Render an iso1 view
            # once per tool so the API call has something to send.
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

            ass.tool_analyzer.name_tools(iso_only=True)
            _timings["tool_naming"] = time.perf_counter() - _t

        # Pre-planning per-part tool decision: VLM picks one tool (or
        # 'none') per part and caches into storage_dir/tool_decisions.json.
        # The sequence planner then consumes those decisions, trying
        # only the chosen tool per step instead of every available tool.
        if getattr(args, "tool_check", False):
            _t = time.perf_counter()
            ass.analyze_assembly_tools()
            _timings["tool_decisions"] = time.perf_counter() - _t

        # Get assembly sequence
        _t = time.perf_counter()
        assemblable = ass.planner.get_assembly_plans(args)
        _timings["sequence_planning"] = time.perf_counter() - _t

        if "v" in _active:
            _t = time.perf_counter()
            out_gif = ass.renderer.stitch_reversed(angle="iso1")
            _timings["stitch_reversed"] = time.perf_counter() - _t
            print(f"[{ass.id}] Reversed assembly GIF: {out_gif}")

        """
        # Divide optimizer (same steps as test_divide_optimizer)
        _t = time.perf_counter()
        tree_path = ass.storage_dir / "log" / "tree.pkl"
        if tree_path.exists():
            _asap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ASAPx")
            if _asap_dir not in sys.path:
                sys.path.insert(0, _asap_dir)
            from plan_sequence.optimizer import DivideOptimizer
            with open(tree_path, "rb") as f:
                _tree = pickle.load(f)
            _run_divide_optimizer_on_tree(
                ass, args, _tree,
                ass.storage_dir / "divide_optimizer",
                DivideOptimizer,
            )
        else:
            print(f"[{ass.id}] No tree.pkl produced; skipping divide optimizer.")
        _timings["divide_optimizer"] = time.perf_counter() - _t
        """

        # Rank the angles of the assembly sequence images
        if settings.angle_ranking:
            _t = time.perf_counter()
            for step in ass.sequence:
                step.rank_angles(show=False)
            _timings["angle_ranking"] = time.perf_counter() - _t

        # Find out if the assembly requires tools
        if assemblable:
            if "i" in _active:
                _t = time.perf_counter()
                ass.generate_instructions()
                _timings["instructions"] = time.perf_counter() - _t
            for step_idx, step in enumerate(ass.sequence):
                # Per-step LLM tool check is disabled: the canonical
                # tool feasibility pass is now the geometric one run
                # inside sequence planning, gated by
                # settings.tool_assemblability. The "t" component is
                # therefore a no-op here.
                if "f" in _active and getattr(settings, "feedback_on_success", False):
                    _t = time.perf_counter()
                    ass.feedback.generate_feedback_selective(step)
                    _timings["feedback"] = (
                        _timings.get("feedback", 0.0) + time.perf_counter() - _t
                    )
                if "m" in _active:
                    _t = time.perf_counter()
                    ass.manual.make_manual(step_idx)
                    _timings["manual"] = (
                        _timings.get("manual", 0.0) + time.perf_counter() - _t
                    )
            if "m" in _active:
                _t = time.perf_counter()
                ass.manual.compile_manual_pdf()
                _timings["manual"] = (
                    _timings.get("manual", 0.0) + time.perf_counter() - _t
                )

        # Non-assemblable: instead of iterating every remaining part
        # with generic per-step feedback, run ONE failure-mode pass
        # that targets each detected failure (collision / assembly /
        # tool / stability) with its own evidence image + text.
        elif "f" in _active:
            _t = time.perf_counter()
            ass.feedback.generate_failure_feedback()
            _timings["feedback"] = (
                _timings.get("feedback", 0.0) + time.perf_counter() - _t
            )

        # Even when the assembly is fully planable, surface any static
        # collisions (overlapping parts) that were detected pre-planning.
        # generate_failure_feedback iterates ass.collisions for the
        # 'collision' mode and is a no-op when there is nothing to
        # report, so it is safe to call unconditionally — but only
        # when there are collisions left and the failure branch hasn't
        # already covered them.
        if (
            assemblable
            and "f" in _active
            and getattr(settings, "feedback_on_success", False)
            and getattr(ass, "collisions", None)
        ):
            _t = time.perf_counter()
            ass.feedback.generate_failure_feedback()
            _timings["feedback"] = (
                _timings.get("feedback", 0.0) + time.perf_counter() - _t
            )
    return _timings


def run_test_pipeline_batch(args, test_eval, output_folder, assembly_dir):
    _ALL = set("gcitmf")
    if args.x is None:
        _active = _ALL
    elif "x" in args.x:
        _active = _ALL - (set(args.x) - {"x"})
    else:
        _active = set(args.x) & _ALL

    _batch_results = {}  # id -> {"status": "ok"/"error", "error": str, "timings": dict}
    _is_llm_run = getattr(args, "planner", None) == "llm"
    _is_comparison_run = getattr(args, "planner", None) == "comparison"
    # Force a fresh planner invocation per assembly when the planner
    # writes per-step decision logs that we want for every assembly.
    _force_fresh = _is_llm_run or _is_comparison_run

    # Choose how to schedule assemblies against the fixed worker budget.
    # Low max_frontier → many in flight with small per-assembly num_proc;
    # high max_frontier → fewer in flight (saturate cores per assembly).
    _BUDGET = 80
    _n_concurrent, _per_assembly = _balance_concurrency(
        max_frontier=getattr(settings, "max_frontier", 1) or 1,
        budget=_BUDGET,
        max_poses=getattr(args, "max_pose", 3) or 3,
    )
    _n_concurrent = min(_n_concurrent, max(1, len(test_eval.assemblies)))
    _per_assembly = max(1, _BUDGET // _n_concurrent)
    args.num_proc = _per_assembly  # used by serial path; passed via CLI for parallel
    print(
        f"\n[batch] {len(test_eval.assemblies)} assemblies | "
        f"max_frontier={getattr(settings, 'max_frontier', 1)} | "
        f"budget={_BUDGET} | "
        f"running {_n_concurrent} in parallel × {_per_assembly} workers each"
    )

    # Pre-clear stale per-assembly cache once so both code paths see
    # the same starting state (cached sequence.json short-circuits the
    # planner before any decision-log file ever gets written).
    if _force_fresh:
        for ass in test_eval.assemblies:
            _seq_json = ass.storage_dir / "sequence.json"
            _log_dir = ass.storage_dir / "log"
            if _seq_json.exists():
                _seq_json.unlink()
            if _log_dir.exists():
                shutil.rmtree(str(_log_dir))

    try:
        if _n_concurrent > 1:
            # Parallel: spawn `test_pipeline --id <single>` subprocesses with
            # n_concurrent in flight. Each subprocess is otherwise the same
            # invocation as the parent (forwarded CLI flags), and writes its
            # per-assembly outputs into ass.storage_dir, which the batch
            # aggregator picks up in the finally below.
            _batch_results = _run_assemblies_parallel(
                test_eval.assemblies,
                sys.argv[1:],
                output_folder,
                n_concurrent=_n_concurrent,
                per_assembly_num_proc=_per_assembly,
            )
        else:
            for ass in test_eval.assemblies:
                print(f"\n{'=' * 60}")
                print(f"[{ass.id}] Starting pipeline...")
                _ass_timings = {}

                try:
                    # Gravity check
                    if "g" in _active:
                        _t = time.perf_counter()
                        unstable_parts = ass.simulation.test_stable(show=False)
                        _ass_timings["gravity"] = time.perf_counter() - _t
                        if unstable_parts:
                            print(f"[{ass.id}] Unstable parts: {unstable_parts}")

                    # Collision check
                    if "c" in _active:
                        _t = time.perf_counter()
                        names_colliding, _, _ = ass.get_collisions(
                            draw=False, show=False
                        )
                        _ass_timings["collisions"] = time.perf_counter() - _t
                        if names_colliding:
                            print(f"[{ass.id}] Collisions detected: {names_colliding}")

                    # Part images
                    _t = time.perf_counter()
                    ass.images
                    _ass_timings["create_images"] = time.perf_counter() - _t

                    # Part naming
                    if settings.part_naming:
                        _t = time.perf_counter()
                        ass.name_parts()
                        _ass_timings["part_naming"] = time.perf_counter() - _t

                    # Tool naming
                    if settings.tool_naming:
                        _t = time.perf_counter()
                        tools_dir = ass.storage_dir / "tool_images"
                        tools_dir.mkdir(parents=True, exist_ok=True)
                        for tool in test_eval.tools.values():
                            if tool.image_paths is None:
                                tool.image_paths = {}
                            img_path = tools_dir / f"{tool.id}_iso1.png"
                            if not img_path.exists():
                                plotter = pv.Plotter(off_screen=True)
                                plotter.add_mesh(
                                    tool.mesh, color="lightgray", show_edges=False
                                )
                                plotter.camera_position = "iso"
                                plotter.screenshot(img_path)
                                plotter.close()
                            tool.image_paths["iso1"] = img_path
                        ass.tool_analyzer.name_tools(iso_only=True)
                        _ass_timings["tool_naming"] = time.perf_counter() - _t

                    # Pre-planning per-part tool decision (see test_pipeline).
                    if getattr(args, "tool_check", False):
                        _t = time.perf_counter()
                        ass.analyze_assembly_tools()
                        _ass_timings["tool_decisions"] = time.perf_counter() - _t

                    # Sequence planning
                    _t = time.perf_counter()
                    assemblable = ass.planner.get_assembly_plans(args)
                    _ass_timings["sequence_planning"] = time.perf_counter() - _t

                    # Angle ranking
                    if settings.angle_ranking:
                        _t = time.perf_counter()
                        for step in ass.sequence:
                            step.rank_angles(show=False)
                        _ass_timings["angle_ranking"] = time.perf_counter() - _t

                    if assemblable:
                        if "i" in _active:
                            _t = time.perf_counter()
                            ass.generate_instructions()
                            _ass_timings["instructions"] = time.perf_counter() - _t
                        for step_idx, step in enumerate(ass.sequence):
                            # Per-step LLM tool check disabled — the geometric
                            # tool check inside sequence planning is the
                            # canonical pass (see settings.tool_assemblability).
                            if "f" in _active and getattr(
                                settings, "feedback_on_success", False
                            ):
                                _t = time.perf_counter()
                                ass.feedback.generate_feedback_selective(step)
                                _ass_timings["feedback"] = (
                                    _ass_timings.get("feedback", 0.0)
                                    + time.perf_counter()
                                    - _t
                                )
                            if "m" in _active:
                                _t = time.perf_counter()
                                ass.manual.make_manual(step_idx)
                                _ass_timings["manual"] = (
                                    _ass_timings.get("manual", 0.0)
                                    + time.perf_counter()
                                    - _t
                                )
                        if "m" in _active:
                            _t = time.perf_counter()
                            ass.manual.compile_manual_pdf()
                            _ass_timings["manual"] = (
                                _ass_timings.get("manual", 0.0)
                                + time.perf_counter()
                                - _t
                            )
                    elif "f" in _active:
                        _t = time.perf_counter()
                        ass.feedback.generate_failure_feedback()
                        _ass_timings["feedback"] = (
                            _ass_timings.get("feedback", 0.0) + time.perf_counter() - _t
                        )

                    _batch_results[ass.id] = {"status": "ok", "timings": _ass_timings}
                    print(f"[{ass.id}] OK  ({sum(_ass_timings.values()):.1f}s)")

                except Exception as _exc:
                    import traceback as _tb

                    print(f"[{ass.id}] ERROR: {_exc}")
                    _tb.print_exc()
                    _batch_results[ass.id] = {
                        "status": "error",
                        "error": str(_exc),
                        "timings": _ass_timings,
                    }
    finally:
        # Always write the cross-assembly summary, even if the loop
        # was interrupted partway through or a fatal error escaped the
        # per-iteration try/except. Aggregators read each assembly's
        # per-run summary file (or report it missing).
        if _is_llm_run:
            try:
                _write_llm_batch_summary(test_eval.assemblies, output_folder)
            except Exception as _e:
                print(f"[llm-batch] failed to write batch summary: {_e}")
        if _is_comparison_run:
            try:
                _write_comparison_batch_summary(test_eval.assemblies, output_folder)
            except Exception as _e:
                print(f"[comparison-batch] failed to write batch summary: {_e}")

    # Batch summary
    _ok = [i for i, r in _batch_results.items() if r["status"] == "ok"]
    _err = [i for i, r in _batch_results.items() if r["status"] == "error"]
    print(f"\n{'=' * 60}")
    print(f"BATCH PIPELINE SUMMARY  ({len(_batch_results)} assemblies)")
    print(f"  OK:    {len(_ok)}")
    print(f"  ERROR: {len(_err)}")
    if _err:
        print("\nFailed assemblies:")
        for _id in _err:
            print(f"  [{_id}] {_batch_results[_id]['error']}")
    _timings = {
        step: sum(r["timings"].get(step, 0.0) for r in _batch_results.values())
        for step in {s for r in _batch_results.values() for s in r["timings"]}
    }


def run_test_render(args, test_eval, output_folder, assembly_dir):
    for ass in test_eval.assemblies:
        log_dir = ass.storage_dir / "log"
        stats_file = log_dir / "stats.json"
        tree_file = log_dir / "tree.pkl"

        if not stats_file.exists() or not tree_file.exists():
            print(
                f"[{ass.id}] Missing log files in {log_dir}. Run sequence planning first."
            )
            continue

        with open(stats_file) as f:
            stats = json.load(f)
        with open(tree_file, "rb") as f:
            tree = pickle.load(f)

        plan_sequence = stats.get("sequence") or []
        if not plan_sequence:
            print(f"[{ass.id}] No sequence found in stats.json, nothing to render.")
            continue

        # Remove stale render artifacts so _render_plan can place fresh ones
        stale_paths = ass.storage_dir / "paths"
        if stale_paths.exists():
            shutil.rmtree(str(stale_paths))
        for stale_gif in ass.storage_dir.glob("*.gif"):
            stale_gif.unlink()

        ass.planner._render_plan(
            asset_folder=str(Path("assets").resolve()),
            assembly_dir=str(ass.assembly_dir),
            plan_sequence=plan_sequence,
            tree=tree,
            args=args,
        )
        print(f"[{ass.id}] Re-render complete. GIFs written to {ass.storage_dir}")
