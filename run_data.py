"""Data-generation subcommands: benchmarks, weight training, and the
offline validation/eval batches that produce the project's research
outputs (plots, summaries, trained weights)."""

import contextlib
import csv
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv
from PIL import Image as PILImage

import settings
from run_common import _write_comparison_batch_summary
from core.simulation import ContactTree
from core.tool_eval import ToolEvaluator


def _enumerate_sequences(tree, root, scorer, weights, max_seqs=100_000):
    """Brute-force DFS over all root → terminal-leaf feasible paths in `tree`.

    A terminal leaf is any node with `len(node) <= 2` (matches the success
    criterion in DFASequencePlanner.plan). Each edge's contribution to the
    sequence cost is `scorer._cost_child(weights, child, sim_info, parent)`
    — the same per-edge cost the heuristic beam search minimises.

    Returns (sequences, truncated). `sequences` is a list of:
        {"path": [tuple, ...], "moves": [part_id, ...], "cost": float}
    sorted in DFS-discovery order. `truncated` is True if max_seqs was hit.
    """
    sequences = []
    truncated = [False]

    def dfs(node, path, moves, total_cost):
        if truncated[0]:
            return
        if len(sequences) >= max_seqs:
            truncated[0] = True
            return
        if len(node) <= 2:
            sequences.append(
                {
                    "path": list(path),
                    "moves": list(moves),
                    "cost": total_cost,
                }
            )
            return
        for _, child, edata in tree.out_edges(node, data=True):
            sim_info = edata.get("sim_info") or {}
            if not sim_info.get("feasible"):
                continue
            edge_cost = scorer._cost_child(weights, list(child), sim_info, list(node))
            removed = sim_info.get("part_move")
            dfs(child, [*path, child], [*moves, removed], total_cost + edge_cost)

    dfs(root, [root], [], 0.0)
    return sequences, truncated[0]


def _simulate_beam(tree, root, k, scorer, weights):
    """Simulate top-k beam search over a pre-computed `tree`, matching
    HeuristicDFASequencePlanner._select_next_frontier: each layer expands the
    current beam's feasible children, ranks them by per-edge _cost_child, and
    keeps the cheapest `k`. Terminal children (len <= 2) graduate to the
    completed-sequence list and don't consume a beam slot.

    Returns a list of finished sequences with the same shape as
    `_enumerate_sequences`.
    """
    frontier = [{"node": root, "path": [root], "moves": [], "cost": 0.0}]
    sequences = []

    while frontier:
        candidates = []
        for state in frontier:
            node = state["node"]
            if len(node) <= 2:
                sequences.append(state)
                continue
            for _, child, edata in tree.out_edges(node, data=True):
                sim_info = edata.get("sim_info") or {}
                if not sim_info.get("feasible"):
                    continue
                edge_cost = scorer._cost_child(
                    weights, list(child), sim_info, list(node)
                )
                removed = sim_info.get("part_move")
                candidates.append(
                    {
                        "node": child,
                        "path": state["path"] + [child],
                        "moves": state["moves"] + [removed],
                        "cost": state["cost"] + edge_cost,
                        "_edge_cost": edge_cost,
                    }
                )
        if not candidates:
            break
        # Match the planner: rank by per-edge cost (ascending), take cheapest k.
        candidates.sort(key=lambda c: c["_edge_cost"])
        frontier = candidates[:k]
        for c in frontier:
            c.pop("_edge_cost", None)

    return sequences


def run_data_filter_assemblies(args, test_eval, output_folder, assembly_dir):
    # ------------------------------------------------------------------
    # Interactive triage: for each assembly, show the iso render +
    # collision graph for 3 seconds, then prompt y/n. Keepers (y) get
    # copied to <output_folder>/filtered_assemblies/<id>/. The source
    # folder is never modified.
    #
    # --allow-gap: ASAPx's allow_gap flag has no direct meaning here
    # (it's a stability-sim knob, not a contact-graph one). The closest
    # semantic match for this triage is "consider parts in contact when
    # they're within CONTACT_EPS of each other," which is exactly what
    # ASAPx's get_contact_graph(..., contact_eps=...) does using
    # redmax's body_in_contact tolerance. When the flag is set we
    # swap the strict trimesh CollisionManager check for that
    # tolerance-aware version.
    # ------------------------------------------------------------------
    dest_root = Path(output_folder) / "filtered_assemblies"
    dest_root.mkdir(parents=True, exist_ok=True)
    kept, dropped = [], []
    print(f"Filtering {len(test_eval.assemblies)} assembly ID(s) → {dest_root}")

    _allow_gap = getattr(args, "allow_gap", False)
    _asap_contact_graph = None
    _asap_contact_eps = None
    _asap_asset_folder = None
    if _allow_gap:
        # Same ATA-eviction prologue used by other ASAPx-backed test_types
        # so the ASAPx packages shadow ATA's identically-named ones.
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
            if _mod in _asapx_pkgs or any(
                _mod.startswith(p + ".") for p in _asapx_pkgs
            ):
                del sys.modules[_mod]
        from plan_sequence.physics_planner import CONTACT_EPS as _asap_contact_eps
        from plan_sequence.physics_planner import (
            get_contact_graph as _asap_contact_graph,
        )

        _asap_asset_folder = str(Path("assets").resolve())
        print(
            f"[data_filter_assemblies] --allow-gap: using ASAPx "
            f"get_contact_graph with contact_eps={_asap_contact_eps}"
        )

    for ass in test_eval.assemblies:
        print(f"\nProcessing assembly {ass.id}...")

        plotter = pv.Plotter(off_screen=True)
        for obj in ass.objects.values():
            plotter.add_mesh(obj.mesh, color="lightgray", show_edges=False)
        # PyVista's "iso" shoots roughly from (+x, -y, +z) — that's
        # iso1 in sequence_planner's convention. We want iso3 here
        # (partially-opposite, not the perfectly mirrored iso2) but
        # at the same auto-fit distance "iso" would give. Compute the
        # camera position from the live scene bounds so the framing
        # matches what the original "iso" preset produced.
        import numpy as _np

        _bounds = plotter.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        _center = _np.array(
            [
                (_bounds[0] + _bounds[1]) * 0.5,
                (_bounds[2] + _bounds[3]) * 0.5,
                (_bounds[4] + _bounds[5]) * 0.5,
            ]
        )
        _half_diag = _np.linalg.norm(
            [
                (_bounds[1] - _bounds[0]) * 0.5,
                (_bounds[3] - _bounds[2]) * 0.5,
                (_bounds[5] - _bounds[4]) * 0.5,
            ]
        )
        # Same camera-to-center distance PyVista's "iso" uses
        # (≈ 2.5×half-diagonal for the default 30° vertical FOV).
        _cam_dist = max(_half_diag * 2.5, 1e-6)
        _iso3_dir = _np.array([-1.0, -1.0, 1.0]) / _np.sqrt(3.0)
        _cam_pos = _center + _iso3_dir * _cam_dist
        plotter.camera_position = [
            tuple(_cam_pos),
            tuple(_center),
            (0.0, 0.0, 1.0),
        ]
        assembly_img = plotter.screenshot(return_img=True)
        plotter.close()

        if _allow_gap:
            parts = sorted(ass.objects.keys())
            try:
                contact_g = _asap_contact_graph(
                    _asap_asset_folder,
                    str(ass.assembly_dir),
                    parts,
                    contact_eps=_asap_contact_eps,
                    save_sdf=False,
                )
            except Exception as _e:
                print(
                    f"  WARN ASAPx contact graph failed ({_e}); "
                    f"falling back to strict trimesh check"
                )
                meshes = {obj.id: obj.tri_mesh for obj in ass.objects.values()}
                contact_g = ContactTree(meshes).G
        else:
            meshes = {obj.id: obj.tri_mesh for obj in ass.objects.values()}
            contact_g = ContactTree(meshes).G
        print(f"  Edges: {list(contact_g.edges())}")

        fig, (ax_render, ax_graph) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Assembly {ass.id}", fontsize=14)
        ax_render.imshow(assembly_img)
        ax_render.axis("off")
        ax_render.set_title("Isometric view")
        nx.draw(contact_g, with_labels=True, ax=ax_graph)
        _eps_note = f"  (eps={_asap_contact_eps})" if _allow_gap else ""
        ax_graph.set_title(
            f"Collision graph  ({len(contact_g.edges())} edge(s)){_eps_note}"
        )
        fig.tight_layout()
        # Non-blocking show for 3 s; pause() drives the GUI event loop
        # so the window actually appears (a bare plt.show(block=False)
        # would return immediately without rendering).
        plt.show(block=False)
        plt.pause(3)
        plt.close(fig)

        ans = input(f"[{ass.id}] keep? [y/N]: ").strip().lower()
        if ans == "y":
            src = Path(ass.assembly_dir)
            dst = dest_root / ass.id
            if dst.exists():
                shutil.rmtree(str(dst))
            shutil.copytree(str(src), str(dst))
            kept.append(ass.id)
            print(f"  → copied to {dst}")
        else:
            dropped.append(ass.id)
            print("  → skipped")

    print(f"\nFiltering complete: kept {len(kept)}, dropped {len(dropped)}")
    print(f"  kept   : {kept}")
    print(f"  dropped: {dropped}")
    print(f"  output : {dest_root}")


def run_test_convex_decomp(args, test_eval, output_folder, assembly_dir):
    n_samples = max(1, args.ai_samples)
    view_specs = [("iso", "Isometric"), ("xy", "Top (XY)"), ("yz", "Side (YZ)")]

    for ass in test_eval.assemblies:
        evaluator = ToolEvaluator(ass)
        decomp_dir = output_folder / "convex_decomp" / ass.id

        # ── AI picks N times per tool ──────────────────────────────────
        print(
            f"\n--- AI selection for assembly {ass.id} ({n_samples} sample(s)/tool) ---"
        )
        if n_samples > 1:
            ai_samples = evaluator.sample_tool_axes(
                output_dir=decomp_dir,
                n_samples=n_samples,
                mode="ai",
            )
        else:
            ai_selected = evaluator.select_tool_axes(
                output_dir=decomp_dir,
                mode="ai",
            )
            ai_samples = {tid: [pick] for tid, pick in ai_selected.items()}

        # ── User confirms each AI pick (yes/no) ────────────────────────
        print(f"\n--- User confirmation for assembly {ass.id} ---")
        confirmations = {tid: [] for tid in ai_samples}

        for tool_id, samples in ai_samples.items():
            tool = ass.scaled_tools[tool_id]
            for s_idx, (direction, contact_point) in enumerate(samples):
                arrows = [(contact_point, direction, "#e63946", "AI pick")]

                fig, axs = plt.subplots(
                    1, len(view_specs), figsize=(5 * len(view_specs), 5)
                )
                if len(view_specs) == 1:
                    axs = [axs]
                for ax, (cam, label) in zip(axs, view_specs, strict=False):
                    img = evaluator._render_arrows(
                        tool.tri_mesh, arrows, cam, bg="white"
                    )
                    ax.imshow(img)
                    ax.set_title(label, fontsize=11)
                    ax.axis("off")
                fig.suptitle(
                    f"AI selection — tool '{tool.name}' ({tool_id}) "
                    f"sample {s_idx + 1}/{n_samples}\n"
                    f"direction=[{', '.join(f'{v:+.2f}' for v in direction)}]",
                    fontsize=13,
                )
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.5)

                prompt = (
                    f"  Accept AI sample {s_idx + 1}/{n_samples} for "
                    f"'{tool_id}'? [y/n]: "
                )
                while True:
                    raw = input(prompt).strip().lower()
                    if raw in ("y", "yes"):
                        confirmations[tool_id].append(True)
                        break
                    if raw in ("n", "no"):
                        confirmations[tool_id].append(False)
                        break
                    print("  Enter 'y' or 'n'.")
                plt.close(fig)

        # ── Per-tool stats: mean acceptance and standard error ─────────
        tool_ids = list(ai_samples.keys())
        stats = {}  # tool_id -> (mean, se, n, n_yes)
        for tid in tool_ids:
            confs = confirmations[tid]
            n = len(confs)
            n_yes = sum(confs)
            p = n_yes / n if n else 0.0
            se = (p * (1 - p) / n) ** 0.5 if n > 0 else 0.0
            stats[tid] = (p, se, n, n_yes)

        total_samples = sum(s[2] for s in stats.values())
        total_yes = sum(s[3] for s in stats.values())

        # ── Summary table ──────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print(
            f"  AI axis confirmations — assembly {ass.id}  "
            f"({total_yes}/{total_samples} accepted, "
            f"n={n_samples}/tool)"
        )
        print(f"{'=' * 70}")
        print(f"  {'Tool':<26} {'mean':>6}  {'SE':>6}  {'YES/n':>8}")
        print(f"  {'-' * 26} {'-' * 6}  {'-' * 6}  {'-' * 8}")
        for tid in tool_ids:
            p, se, n, n_yes = stats[tid]
            print(f"  {tid:<26} {p:>6.2f}  {se:>6.3f}  {n_yes:>3}/{n:<4}")
        print(f"{'=' * 70}")

        # ── Save results to CSV (one row per sample) ───────────────────
        csv_path = output_folder / f"tool_axes_{ass.id}.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "tool_id",
                    "sample_idx",
                    "dir_x",
                    "dir_y",
                    "dir_z",
                    "contact_x",
                    "contact_y",
                    "contact_z",
                    "confirmed",
                ]
            )
            for tid, samples in ai_samples.items():
                for s_idx, (direction, contact) in enumerate(samples):
                    confirmed = confirmations[tid][s_idx]
                    writer.writerow(
                        [
                            tid,
                            s_idx,
                            *direction.tolist(),
                            *contact.tolist(),
                            "YES" if confirmed else "NO",
                        ]
                    )
        print(f"  Saved results: {csv_path}")

        # ── Per-assembly plot: mean acceptance ± SE per tool ───────────
        means = [stats[t][0] for t in tool_ids]
        ses = [stats[t][1] for t in tool_ids]
        colors = ["#2ecc71" if m >= 0.5 else "#e74c3c" for m in means]

        fig, ax = plt.subplots(figsize=(max(6, len(tool_ids) * 0.9 + 2), 5))
        xs = np.arange(len(tool_ids))
        ax.bar(
            xs,
            means,
            yerr=ses,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"ecolor": "black", "lw": 1},
        )
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel(f"Mean acceptance  (n={n_samples} samples/tool)")
        ax.set_xlabel("Tool")
        title = (
            f"AI axis confirmations — assembly {ass.id}"
            f"  ({total_yes}/{total_samples} accepted)"
        )
        if n_samples > 1:
            title += "   mean ± SE"
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(tool_ids, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        plot_path = output_folder / f"tool_axes_{ass.id}.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  Saved plot:    {plot_path}")


def run_train_heuristic_weights(args, test_eval, output_folder, assembly_dir):
    # ------------------------------------------------------------------
    # Optuna-based black-box optimisation of the HeuristicDFASequencePlanner
    # weights, using the arm pipeline's total_s as the objective. Each
    # trial writes its candidate weights to assets/heuristic_weights_optuna.json
    # (read by the planner because the trainer flips
    # settings.heuristic_weights_source to "optuna" for the duration).
    # At study end, the best trial's weights are written back to the same
    # file as the final, frozen state — flip settings.heuristic_weights_source
    # to "optuna" in settings.py for inference, leave at "default" to
    # compare against the manually-tuned `settings.heuristic_weights`.
    # ------------------------------------------------------------------
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
    from plan_sequence.optimizer.weight_trainer import train_heuristic_weights

    n_trials = getattr(args, "optuna_trials", 50)
    persist = getattr(args, "optuna_resume", False)
    print(
        f"[train_heuristic_weights] training on {len(test_eval.assemblies)} "
        f"assemblies for {n_trials} trials  (persist_study={persist})"
    )
    study = train_heuristic_weights(
        test_eval=test_eval,
        args=args,
        n_trials=n_trials,
        persist_study=persist,
    )
    print(f"[train_heuristic_weights] study finished: {len(study.trials)} trials total")


def run_data_heuristic_validation(args, test_eval, output_folder, assembly_dir):
    # ------------------------------------------------------------------
    # Validate the heuristic beam-search planner: run a BFS-style sweep
    # (very wide frontier) per assembly, then post-hoc simulate beam
    # search at k=1..MAX_K over the resulting tree and check at which k
    # each assembly's globally-optimal sequence (by heuristic score) is
    # first recovered.
    # ------------------------------------------------------------------
    BFS_MAX_FRONTIER = 1000  # effectively unbounded for small assemblies
    MAX_K = 10  # beam widths to simulate
    BUDGET = 50000  # per-assembly sim-eval budget for the BFS run
    MAX_ENUM_SEQS = 100_000  # cap on enumerated complete sequences

    hv_dir = Path(output_folder) / "heuristic_validation"
    hv_dir.mkdir(parents=True, exist_ok=True)

    per_assembly = []  # accumulated for summary; written in `finally` below

    def _write_validation_summary():
        if not per_assembly:
            return
        ok_records = [a for a in per_assembly if a.get("status") == "ok"]
        n_ok = len(ok_records)
        n_err = sum(1 for a in per_assembly if a.get("status") == "error")
        n_no_seq = sum(
            1
            for a in per_assembly
            if a.get("status") in ("no_tree", "no_root", "no_complete_seq")
        )

        # Aggregate: how many assemblies hit the optimum at k <= K?
        found_at_k = [0] * (MAX_K + 1)
        for a in ok_records:
            lk = a.get("lowest_k_for_global_optimum")
            if lk is not None and lk <= MAX_K:
                for k in range(lk, MAX_K + 1):
                    found_at_k[k] += 1
        never_found = sum(
            1 for a in ok_records if a.get("lowest_k_for_global_optimum") is None
        )

        json_path = hv_dir / "validation_summary.json"
        with open(json_path, "w") as _f:
            json.dump(
                {
                    "n_assemblies": len(per_assembly),
                    "ok": n_ok,
                    "errors": n_err,
                    "no_sequence": n_no_seq,
                    "max_k_tested": MAX_K,
                    "bfs_max_frontier": BFS_MAX_FRONTIER,
                    "budget_per_run": BUDGET,
                    "aggregate_found_at_k": {
                        str(k): found_at_k[k] for k in range(1, MAX_K + 1)
                    },
                    "n_never_found_at_max_k": never_found,
                    "per_assembly": per_assembly,
                },
                _f,
                indent=2,
                default=str,
            )

        lines = []
        lines.append("Heuristic Beam Search Validation Summary")
        lines.append("=" * 72)
        lines.append(
            f"Assemblies tested:   {len(per_assembly)}  "
            f"(ok={n_ok}, errors={n_err}, no_seq={n_no_seq})"
        )
        lines.append(f"BFS max_frontier:    {BFS_MAX_FRONTIER}")
        lines.append(f"Budget per run:      {BUDGET}")
        lines.append(f"Max k tested:        {MAX_K}")
        lines.append("")
        lines.append(
            "Aggregate: # assemblies that recover the global optimum at k <= K"
        )
        lines.append(f"  {'k':>3}  {'count':>6}  {'%':>6}")
        for k in range(1, MAX_K + 1):
            pct = (100.0 * found_at_k[k] / n_ok) if n_ok > 0 else 0.0
            lines.append(f"  {k:>3}  {found_at_k[k]:>6d}  {pct:>5.1f}%")
        if n_ok > 0:
            lines.append(f"  not found at k<={MAX_K}: {never_found}/{n_ok}")
        lines.append("")

        for a in per_assembly:
            lines.append("-" * 72)
            lines.append(f"Assembly {a.get('id')}  [{a.get('status')}]")
            if a.get("status") == "error":
                lines.append(f"  ERROR: {a.get('error')}")
                continue
            if a.get("status") != "ok":
                if a.get("bfs_time_s") is not None:
                    lines.append(f"  bfs_time: {a['bfs_time_s']:.1f}s")
                continue
            trunc = " [truncated]" if a.get("sequence_enum_truncated") else ""
            lines.append(
                f"  n_parts={a['n_parts']}  tree=({a['n_tree_nodes']} nodes, "
                f"{a['n_tree_edges']} edges)  "
                f"complete_sequences={a['n_complete_sequences']}{trunc}"
            )
            lines.append(
                f"  bfs_time={a['bfs_time_s']:.1f}s  "
                f"global_best_cost={a['global_best_cost']:.4f}  "
                f"lowest_k_for_global_optimum={a['lowest_k_for_global_optimum']}"
            )
            lines.append(f"  global_best_moves: {a['global_best_moves']}")
            lines.append("")
            for k in range(1, MAX_K + 1):
                kres = a["k_results"].get(k) or a["k_results"].get(str(k))
                if kres is None:
                    continue
                lines.append(
                    f"  k={k:>2}  n_seqs={kres['n_sequences_found']:>4}  "
                    f"best={kres['best_cost']:.4f}  "
                    f"found_global={'Y' if kres['found_global_optimum'] else 'N'}"
                )
                for i, s in enumerate(kres["top5"]):
                    lines.append(
                        f"      #{i + 1}: cost={s['cost']:.4f}  moves={s['moves']}"
                    )
            lines.append("")

        txt_path = hv_dir / "validation_summary.txt"
        with open(txt_path, "w") as _f:
            _f.write("\n".join(lines) + "\n")
        print(
            f"\n[heuristic-val] summary written to {hv_dir}/validation_summary.{{txt,json}}"
        )

    # Stash originals so we can restore after the test.
    _orig_planner = getattr(args, "planner", None)
    _orig_budget = getattr(args, "budget", None)
    _orig_early = getattr(args, "early_term", None)
    _orig_mf = settings.max_frontier
    _orig_cache = test_eval.cache

    try:
        test_eval.cache = "new"  # force fresh BFS per run
        args.planner = "heuristic"
        args.budget = BUDGET
        args.early_term = False
        settings.max_frontier = BFS_MAX_FRONTIER

        for ass in test_eval.assemblies:
            print(f"\n[heuristic-val] ===== assembly {ass.id} =====")
            entry = {"id": ass.id, "status": "ok", "bfs_time_s": None}
            try:
                # Wipe stale cache so the planner actually re-runs.
                seq_json = ass.storage_dir / "sequence.json"
                log_dir = ass.storage_dir / "log"
                if seq_json.exists():
                    seq_json.unlink()
                if log_dir.exists():
                    shutil.rmtree(str(log_dir))

                _t0 = time.perf_counter()
                ass.planner.get_assembly_plans(args)
                entry["bfs_time_s"] = time.perf_counter() - _t0

                tree_path = ass.storage_dir / "log" / "tree.pkl"
                if not tree_path.exists():
                    entry["status"] = "no_tree"
                    per_assembly.append(entry)
                    continue
                with open(tree_path, "rb") as _f:
                    tree = pickle.load(_f)
                entry["n_tree_nodes"] = tree.number_of_nodes()
                entry["n_tree_edges"] = tree.number_of_edges()

                roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
                if not roots:
                    entry["status"] = "no_root"
                    per_assembly.append(entry)
                    continue
                root = roots[0]
                entry["n_parts"] = len(root)

                # Build a scorer planner (no re-plan, just for _cost_child).
                # ASAPx modules are already in sys.modules from the seq_plan call
                # above, so plan_sequence.* imports resolve to the same instances.
                from plan_sequence.generator import generators as _generators
                from plan_sequence.planner.heuristic import HeuristicDFASequencePlanner

                asset_folder = str(Path("assets").resolve())
                assembly_dir = str(ass.assembly_dir)
                gen = _generators["rand"](
                    asset_folder,
                    assembly_dir,
                    base_part=getattr(args, "base_part", None),
                    save_sdf=not getattr(args, "disable_save_sdf", False),
                )
                scorer = HeuristicDFASequencePlanner(
                    gen,
                    num_proc=1,
                    save_sdf=not getattr(args, "disable_save_sdf", False),
                    get_dof=settings.get_dof,
                    skip_stability=settings.skip_stability,
                )
                weights = scorer._load_weights()

                sequences, truncated = _enumerate_sequences(
                    tree,
                    root,
                    scorer,
                    weights,
                    max_seqs=MAX_ENUM_SEQS,
                )
                sequences.sort(key=lambda s: s["cost"])
                entry["n_complete_sequences"] = len(sequences)
                entry["sequence_enum_truncated"] = truncated
                if not sequences:
                    entry["status"] = "no_complete_seq"
                    per_assembly.append(entry)
                    continue

                global_best_cost = sequences[0]["cost"]
                entry["global_best_cost"] = global_best_cost
                entry["global_best_moves"] = sequences[0]["moves"]

                k_results = {}
                lowest_k = None
                for k in range(1, MAX_K + 1):
                    beam_seqs = _simulate_beam(tree, root, k, scorer, weights)
                    beam_seqs.sort(key=lambda s: s["cost"])
                    if beam_seqs:
                        best = beam_seqs[0]["cost"]
                        found = abs(best - global_best_cost) < 1e-9
                    else:
                        best = float("inf")
                        found = False
                    k_results[k] = {
                        "n_sequences_found": len(beam_seqs),
                        "best_cost": best,
                        "top5": [
                            {"moves": s["moves"], "cost": s["cost"]}
                            for s in beam_seqs[:5]
                        ],
                        "found_global_optimum": bool(found),
                    }
                    if found and lowest_k is None:
                        lowest_k = k
                entry["k_results"] = k_results
                entry["lowest_k_for_global_optimum"] = lowest_k

                per_assembly.append(entry)
                print(
                    f"[{ass.id}] best_cost={global_best_cost:.3f}  "
                    f"lowest_k={lowest_k}  n_complete_seqs={len(sequences)}"
                    f"{' (truncated)' if truncated else ''}"
                )

            except Exception as _exc:
                import traceback as _tb

                _tb.print_exc()
                entry["status"] = "error"
                entry["error"] = str(_exc)
                per_assembly.append(entry)

    finally:
        # Restore originals before propagating up to the outer finally.
        if _orig_planner is not None:
            args.planner = _orig_planner
        if _orig_budget is not None:
            args.budget = _orig_budget
        if _orig_early is not None:
            args.early_term = _orig_early
        settings.max_frontier = _orig_mf
        test_eval.cache = _orig_cache
        # Always write whatever we've collected, even on early termination.
        try:
            _write_validation_summary()
        except Exception as _e:
            print(f"[heuristic-val] failed to write summary: {_e}")


def _load_assembly_time_results(data_dir, runs, components):
    """Load a previous assembly-time run's timing into the in-memory
    `results` shape ({assembly_id: {run_label: timing_overview-or-error}}).

    Looks first for an `assembly_time_summary.json` (the single-file digest
    the benchmark writes), then falls back to scanning per-run
    `timing_overview.json` files laid out as
    `<…>/<assembly_id>/<run_label>/log/timing_overview.json`. Only run
    labels present in `runs` are kept, so sub-assembly timing (sub_S/sub_R)
    is ignored. Returns {} when nothing usable is found, so the caller can
    fall through to the full benchmark.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[assembly-time] --data-dir does not exist: {data_dir}")
        return {}
    valid_labels = {label for label, _, _ in runs}

    # 1) Prefer a previously-written summary JSON (single source of truth).
    # Glob a trailing suffix too (e.g. assembly_time_summary_good11.json) so
    # manually-renamed/archived summaries are still picked up; prefer the
    # canonical unsuffixed name when several are present.
    summary_candidates = sorted(
        set(data_dir.glob("assembly_time_summary*.json"))
        | set((data_dir / "assembly_time").glob("assembly_time_summary*.json")),
        key=lambda p: (p.name != "assembly_time_summary.json", str(p)),
    )
    if len(summary_candidates) > 1:
        print(
            f"[assembly-time] multiple summary files under {data_dir}: "
            f"{[p.name for p in summary_candidates]} — trying in order"
        )
    for cand in summary_candidates:
        try:
            with open(cand) as _f:
                summary = json.load(_f)
        except Exception as _e:
            print(f"[assembly-time] could not read {cand}: {_e}")
            continue
        per_assembly = (summary or {}).get("per_assembly") or {}
        results = {}
        for aid, runs_data in per_assembly.items():
            results[aid] = {}
            for label, entry in (runs_data or {}).items():
                if label not in valid_labels:
                    continue
                entry = entry or {}
                if entry.get("status") == "ok" and entry.get("totals"):
                    results[aid][label] = {
                        "status": "ok",
                        "totals": entry["totals"],
                        # per_step is only used for the n_steps_planned field
                        # in the re-written summary; length is all that matters.
                        "per_step": [None] * int(entry.get("n_steps_planned") or 0),
                    }
                else:
                    results[aid][label] = {
                        "status": entry.get("status", "missing"),
                        "error": entry.get("error"),
                    }
            if not results[aid]:
                del results[aid]
        if results:
            print(
                f"[assembly-time] loaded existing summary {cand} "
                f"({len(results)} assemblies)"
            )
            return results

    # 2) Fall back to scanning per-run timing_overview.json files.
    results = {}
    for tp in sorted(data_dir.rglob("timing_overview.json")):
        # Expect <assembly_id>/<run_label>/log/timing_overview.json.
        if tp.parent.name != "log" or len(tp.parents) < 3:
            continue
        run_label = tp.parents[1].name
        aid = tp.parents[2].name
        if run_label not in valid_labels:
            continue
        try:
            with open(tp) as _f:
                overview = json.load(_f)
        except Exception as _e:
            print(f"[assembly-time] skip {tp}: {_e}")
            continue
        overview.setdefault("status", "ok")
        results.setdefault(aid, {})[run_label] = overview
    if results:
        n_files = sum(len(v) for v in results.values())
        print(
            f"[assembly-time] loaded {n_files} timing_overview.json "
            f"file(s) under {data_dir} ({len(results)} assemblies)"
        )
    return results


def run_data_assembly_time(args, test_eval, output_folder, assembly_dir):
    # ------------------------------------------------------------------
    # End-to-end assembly-time comparison across generators.
    #
    # For each assembly in --id, plan + render (with --plan-arm) under
    # each (planner, generator) combination listed in RUNS, read the
    # resulting timing_overview.json, and emit:
    #   - per-assembly stacked-bar PNG (4 timing components per run)
    #   - cross-assembly mean-total bar PNG
    #   - aggregate JSON + TXT summary
    # Each run gets its own storage subdir so runs don't clobber each
    # other's tree.pkl / arm_plans.json / timing_overview.json.
    # ------------------------------------------------------------------
    RUNS = [
        # (label, planner, generator)
        ("heuristic", "heuristic", "rand"),
        (
            "heuristic_trained",
            "heuristic",
            "rand",
        ),  # heuristic w/ Optuna-trained weights
        (
            "heuristic+optimizer",
            "heuristic",
            "rand",
        ),  # reuses heuristic tree + divide optimizer
        ("gen:heur-out", "gen-adapter", "heur-out"),
        # ("gen:heur-vol",       "gen-adapter", "heur-vol"),
        # ("gen:learn",          "gen-adapter", "learn"),
        ("gen:rand", "gen-adapter", "rand"),
        # ("gen:dfa",           "gen-adapter", "dfa"),
    ]
    COMPONENTS = (
        "step_disassembly_s",
        "transitions_s",
        "base_travel_s",
        "reorientation_s",
        "hold_s",
    )

    at_dir = Path(output_folder) / "assembly_time"
    at_dir.mkdir(parents=True, exist_ok=True)

    # results[assembly_id][run_label] = timing_overview dict or
    # {"status": "error", "error": "..."} when the run failed.
    results = {}

    _orig_planner = getattr(args, "planner", None)
    _orig_generator = getattr(args, "generator", None)
    _orig_plan_arm = getattr(args, "plan_arm", False)
    _orig_storage = getattr(args, "storage_dir", None)
    _orig_cache = test_eval.cache

    def _write_assembly_time_summary():
        """Aggregate `results` into JSON + TXT + matplotlib charts.
        Safe to call after partial runs."""
        if not results:
            return
        # Compact per-assembly + per-run flat table for downstream tools.
        summary = {
            "runs": [
                {"label": label, "planner": p, "generator": g} for label, p, g in RUNS
            ],
            "components": list(COMPONENTS),
            "per_assembly": {},
        }
        for aid, runs_data in results.items():
            summary["per_assembly"][aid] = {}
            for label, _, _ in RUNS:
                entry = runs_data.get(label)
                if entry is None or entry.get("status") == "error":
                    summary["per_assembly"][aid][label] = {
                        "status": (entry or {}).get("status", "missing"),
                        "error": (entry or {}).get("error"),
                        "totals": None,
                    }
                    continue
                totals = entry.get("totals", {})
                summary["per_assembly"][aid][label] = {
                    "status": "ok",
                    "totals": {
                        k: float(totals.get(k, 0.0))
                        for k in [*list(COMPONENTS), "total_s"]
                    },
                    "n_steps_planned": len(entry.get("per_step", []) or []),
                }

        json_path = at_dir / "assembly_time_summary.json"
        with open(json_path, "w") as _f:
            json.dump(summary, _f, indent=2, default=str)

        # TXT digest
        lines = []
        lines.append("Assembly-Time Generator Comparison")
        lines.append("=" * 72)
        lines.append(f"Assemblies: {len(results)}")
        lines.append(f"Runs/assembly: {[label for label, _, _ in RUNS]}")
        lines.append("")
        # Per-run mean total across assemblies (ok-only).
        means = {label: [] for label, _, _ in RUNS}
        for aid, runs_data in results.items():
            for label, _, _ in RUNS:
                entry = runs_data.get(label) or {}
                if entry.get("status") != "ok":
                    continue
                means[label].append(float(entry["totals"].get("total_s", 0.0)))
        lines.append("Mean total time per generator (across OK runs):")
        lines.append(f"  {'run':<16}  {'n':>3}  {'mean_total_s':>13}")
        for label, _, _ in RUNS:
            vals = means[label]
            mean_s = (sum(vals) / len(vals)) if vals else 0.0
            lines.append(f"  {label:<16}  {len(vals):>3}  {mean_s:>13.2f}")
        lines.append("")
        # Per-assembly table.
        for aid, runs_data in results.items():
            lines.append("-" * 72)
            lines.append(f"Assembly {aid}")
            lines.append(
                f"  {'run':<16}  {'status':>8}  "
                + "  ".join(f"{c.replace('_s', ''):>10}" for c in COMPONENTS)
                + f"  {'total':>10}"
            )
            for label, _, _ in RUNS:
                entry = runs_data.get(label) or {}
                status = entry.get("status", "missing")
                if status != "ok":
                    lines.append(
                        f"  {label:<16}  {status:>8}  "
                        + "  ".join(f"{'-':>10}" for _ in COMPONENTS)
                        + f"  {'-':>10}"
                    )
                    continue
                t = entry["totals"]
                comp_strs = [f"{float(t.get(c, 0.0)):>10.2f}" for c in COMPONENTS]
                total = float(t.get("total_s", 0.0))
                lines.append(
                    f"  {label:<16}  {status:>8}  "
                    + "  ".join(comp_strs)
                    + f"  {total:>10.2f}"
                )
            lines.append("")

        txt_path = at_dir / "assembly_time_summary.txt"
        with open(txt_path, "w") as _f:
            _f.write("\n".join(lines) + "\n")
        print(f"[assembly-time] summary → {json_path}")
        print(f"[assembly-time] summary → {txt_path}")

        # ---- matplotlib charts ----
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            component_colors = {
                "step_disassembly_s": "#4C72B0",  # blue
                "transitions_s": "#55A868",  # green
                "base_travel_s": "#C44E52",  # red
                "reorientation_s": "#8172B2",  # purple
                "hold_s": "#CCB974",  # tan
            }
            component_labels = {
                "step_disassembly_s": "disassembly",
                "transitions_s": "transitions",
                "base_travel_s": "base travel",
                "reorientation_s": "reorientation",
                "hold_s": "hold penalty",
            }
            run_labels = [label for label, _, _ in RUNS]

            # 1) One stacked-bar chart per assembly.
            for aid, runs_data in results.items():
                fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(RUNS)), 4.5))
                x = np.arange(len(RUNS))
                bottom = np.zeros(len(RUNS))
                for comp in COMPONENTS:
                    vals = []
                    for label, _, _ in RUNS:
                        entry = runs_data.get(label) or {}
                        if entry.get("status") == "ok":
                            vals.append(float(entry["totals"].get(comp, 0.0)))
                        else:
                            vals.append(0.0)
                    ax.bar(
                        x,
                        vals,
                        bottom=bottom,
                        label=component_labels[comp],
                        color=component_colors[comp],
                    )
                    bottom += np.array(vals)
                # Mark failed runs with a hatched empty bar so they read as missing.
                for i, (label, _, _) in enumerate(RUNS):
                    entry = runs_data.get(label) or {}
                    if entry.get("status") != "ok":
                        ax.bar([x[i]], [0.0], color="lightgray")
                        ax.text(
                            x[i],
                            0.0,
                            "fail",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="dimgray",
                        )
                ax.set_xticks(x)
                ax.set_xticklabels(run_labels, rotation=30, ha="right")
                ax.set_ylabel("estimated time (s)")
                ax.set_title(f"Assembly {aid}: time breakdown by generator")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                out = at_dir / f"{aid}_breakdown.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
                print(f"[assembly-time] {aid}: {out.name}")

            # 2) Cross-assembly mean-total bar chart (mean ± std of total_s).
            fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(RUNS)), 4.5))
            x = np.arange(len(RUNS))
            mean_vals = []
            err_vals = []
            n_vals = []
            for label, _, _ in RUNS:
                v = means[label]
                n_vals.append(len(v))
                if v:
                    mean_vals.append(float(np.mean(v)))
                    err_vals.append(float(np.std(v)) if len(v) > 1 else 0.0)
                else:
                    mean_vals.append(0.0)
                    err_vals.append(0.0)
            ax.bar(x, mean_vals, yerr=err_vals, color="#4C72B0", capsize=4, alpha=0.85)
            for xi, (m, n) in enumerate(zip(mean_vals, n_vals, strict=False)):
                ax.text(xi, m, f"n={n}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(run_labels, rotation=30, ha="right")
            ax.set_ylabel("mean estimated assembly time (s)")
            ax.set_title(
                f"Mean total time per generator  "
                f"({len(results)} assemblies, OK runs only)"
            )
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            out = at_dir / "comparison_mean_total.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f"[assembly-time] {out.name}")

            # 3) Grouped stacked-bar chart: one group per assembly,
            # stacked components within each run-bar in the group.
            # Useful as the single "headline" figure.
            aids = list(results.keys())
            if aids:
                fig, ax = plt.subplots(
                    figsize=(max(8, 1.0 * len(aids) * len(RUNS)), 4.5)
                )
                n_runs = len(RUNS)
                bar_w = 0.8 / n_runs
                x = np.arange(len(aids))
                for r_idx, (label, _, _) in enumerate(RUNS):
                    bottom = np.zeros(len(aids))
                    for comp in COMPONENTS:
                        vals = []
                        for aid in aids:
                            entry = (results[aid] or {}).get(label) or {}
                            if entry.get("status") == "ok":
                                vals.append(float(entry["totals"].get(comp, 0.0)))
                            else:
                                vals.append(0.0)
                        offset = (r_idx - (n_runs - 1) / 2) * bar_w
                        ax.bar(
                            x + offset,
                            vals,
                            bar_w,
                            bottom=bottom,
                            label=(
                                f"{component_labels[comp]} ({label})"
                                if r_idx == 0
                                else None
                            ),
                            color=component_colors[comp],
                            edgecolor="white",
                            linewidth=0.3,
                        )
                        bottom += np.array(vals)
                    # Run-label tick above the group.
                    for ai in range(len(aids)):
                        ax.text(
                            x[ai] + offset,
                            0.0,
                            label[:8],
                            rotation=90,
                            fontsize=6,
                            ha="center",
                            va="top",
                            color="dimgray",
                        )
                ax.set_xticks(x)
                ax.set_xticklabels(aids)
                ax.set_ylabel("estimated time (s)")
                ax.set_title("Per-assembly time breakdown by generator")
                # Component legend only (run-label written on each bar).
                handles = [
                    plt.Rectangle((0, 0), 1, 1, color=component_colors[c])
                    for c in COMPONENTS
                ]
                ax.legend(
                    handles,
                    [component_labels[c] for c in COMPONENTS],
                    loc="upper right",
                    fontsize=8,
                )
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                out = at_dir / "comparison_per_assembly.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
                print(f"[assembly-time] {out.name}")
        except (Exception, KeyboardInterrupt) as _plot_e:
            # Catch KeyboardInterrupt too so an abort during plot
            # rendering doesn't lose the JSON/TXT we already wrote at
            # the top of this function. Re-raise after logging so the
            # outer try/finally still sees the interrupt.
            print(f"[assembly-time] plotting failed: {_plot_e}")
            if not isinstance(_plot_e, KeyboardInterrupt):
                import traceback as _tb

                _tb.print_exc()
            else:
                raise

    # Data-dir-first: if --data-dir points at a previous assembly-time run
    # with usable timing, just re-plot from it and skip planning entirely.
    # Falls through to the full benchmark when nothing usable is found.
    data_dir = getattr(args, "data_dir", None)
    if data_dir:
        existing = _load_assembly_time_results(data_dir, RUNS, COMPONENTS)
        if existing:
            results.update(existing)
            print(
                f"[assembly-time] re-plotting from existing data in "
                f"{data_dir} ({len(results)} assemblies); skipping planning"
            )
            _write_assembly_time_summary()
            return
        print(
            f"[assembly-time] --data-dir given ({data_dir}) but no usable "
            f"timing found there; running the full benchmark"
        )

    try:
        if not getattr(args, "plan_arm", False):
            print(
                "[assembly-time] --plan-arm was off; forcing it ON "
                "so arm_pipeline writes timing_overview.json"
            )
        args.plan_arm = True

        # Cache semantics
        # ---------------
        # args.cache == "read": if a previous run for this (assembly,
        #   run_label) left tree.pkl + stats.json in the cache_dir,
        #   reuse the sequence. Otherwise plan from scratch and
        #   populate the cache for next time.
        # args.cache == "new" / "update": always re-plan and overwrite
        #   the cache. Existing assets/output/.../assembly_time/...
        #   render output still gets wiped per run.
        #
        # Cache lives under each assembly's storage_dir; render output
        # goes under <output_folder>/assembly_time/<assembly_id>/
        # <run_label>/ so multiple test runs don't clobber each other
        # and the cache stays decoupled from the artifact directory.
        cache_mode = getattr(args, "cache", "read")
        reuse_allowed = cache_mode == "read"

        # ASAPx eviction helper — _render_plan does this internally on
        # entry, but when we call it twice in a row for different
        # storage dirs we want to be sure the module state is fresh.
        os.path.dirname(os.path.abspath(__file__))
        asset_folder = str(Path("assets").resolve())

        for ass in test_eval.assemblies:
            print(f"\n[assembly-time] ===== assembly {ass.id} =====")
            results[ass.id] = {}
            base_storage = Path(ass.storage_dir)
            _orig_ass_storage = ass.storage_dir
            # Set to True if any RUN reports "no self-stable initial
            # pose" for this assembly. Subsequent RUNS without a
            # usable cached sequence are skipped — the precheck is a
            # property of assembly geometry, not the planner, so they
            # would all fail the same way. RUNS that already have a
            # valid cached sequence still proceed (respects the
            # cache=read per-method semantics).
            assembly_precheck_failed = False

            for run_label, planner_name, generator_name in RUNS:
                # Incremental save BEFORE each run starts — captures all
                # completed runs from this and prior assemblies, so an
                # abort mid-run only loses the currently-executing one
                # (whose results entry hasn't been written yet anyway).
                try:
                    _write_assembly_time_summary()
                except Exception as _e:
                    print(f"[assembly-time] pre-run incremental save failed: {_e}")

                cache_dir = base_storage / "assembly_time" / run_label
                cache_dir.mkdir(parents=True, exist_ok=True)
                output_run_dir = at_dir / str(ass.id) / run_label
                output_run_dir.mkdir(parents=True, exist_ok=True)

                cache_log = cache_dir / "log"
                cache_tree = cache_log / "tree.pkl"
                cache_stats = cache_log / "stats.json"
                cache_hit = (
                    reuse_allowed and cache_tree.exists() and cache_stats.exists()
                )

                args.planner = planner_name
                args.generator = generator_name
                print(
                    f"[assembly-time] -- run {run_label}  "
                    f"(planner={planner_name}, generator={generator_name})  "
                    f"cache={'HIT' if cache_hit else 'MISS'}  "
                    f"cache_dir={cache_dir}  "
                    f"output={output_run_dir}"
                )

                # Early skip when an earlier RUN already established
                # this assembly has no self-stable initial pose. We
                # only skip if the cache here is either missing or
                # holds an empty sequence — a previously-cached
                # successful run is still valid and gets reused.
                if assembly_precheck_failed:
                    cached_seq_ok = False
                    if cache_hit:
                        try:
                            with open(cache_stats) as _cf:
                                cached_seq_ok = bool(json.load(_cf).get("sequence"))
                        except Exception:
                            cached_seq_ok = False
                    if not cached_seq_ok:
                        results[ass.id][run_label] = {
                            "status": "skipped_no_stable_pose",
                            "error": "skipped because an earlier method "
                            "found no self-stable initial pose "
                            "for this assembly",
                        }
                        print(
                            f"[assembly-time]    {run_label}: skipped "
                            f"(no self-stable initial pose; no usable cached sequence)"
                        )
                        continue

                # ------------------------------------------------------
                # Special-case: "heuristic+optimizer" reuses the heuristic
                # run's tree + timing (no re-planning), then runs the
                # divide optimizer to find a split; if one is found, the
                # two subassemblies are re-planned + re-rendered + re-
                # armed standalone, and the run's total time is built as
                #   prefix (heuristic per-step entries for parts not in
                #          S∪R) + separation step + S total + R total
                # mirroring the cost decomposition in compare.py. If no
                # split is verified, falls back to the heuristic timing.
                # ------------------------------------------------------
                if run_label == "heuristic+optimizer":
                    heur_cache_log = (
                        base_storage / "assembly_time" / "heuristic" / "log"
                    )
                    heur_output_log = at_dir / str(ass.id) / "heuristic" / "log"
                    req_paths = [
                        heur_cache_log / "tree.pkl",
                        heur_cache_log / "stats.json",
                        heur_output_log / "timing_overview.json",
                    ]
                    if not all(p.exists() for p in req_paths):
                        missing = [str(p) for p in req_paths if not p.exists()]
                        results[ass.id][run_label] = {
                            "status": "error",
                            "error": f"heuristic run artifacts missing: {missing}",
                        }
                        print(
                            f"[assembly-time]    {run_label}: heuristic artifacts missing"
                        )
                        continue
                    try:
                        with open(heur_cache_log / "tree.pkl", "rb") as _f:
                            heur_tree = pickle.load(_f)
                        with open(heur_cache_log / "stats.json") as _f:
                            heur_stats = json.load(_f)
                        with open(heur_output_log / "timing_overview.json") as _f:
                            heur_timing = json.load(_f)
                    except Exception as _e:
                        results[ass.id][run_label] = {
                            "status": "error",
                            "error": f"failed to load heuristic artifacts: {_e}",
                        }
                        continue
                    heur_seq = heur_stats.get("sequence") or []

                    # Run divide optimizer on the heuristic tree.
                    try:
                        from plan_sequence.optimizer import DivideOptimizer
                        from plan_sequence.optimizer.compare import (
                            _prepare_subassembly_dir,
                        )

                        opt = DivideOptimizer(
                            heur_tree,
                            asset_folder=asset_folder,
                            assembly_dir=str(ass.assembly_dir),
                        )
                        if opt.build_obstruction_graph() is None:
                            raise RuntimeError(
                                "no dof_info in heuristic tree (need get_dof=True)"
                            )
                        opt.find_locally_free_subassemblies(timeout=100)
                        opt.verify_locally_free(
                            top_k=10, num_proc=getattr(args, "num_proc", 8)
                        )
                        verified = getattr(opt, "verified_locally_free", None) or []
                    except Exception as _e:
                        import traceback as _tb

                        _tb.print_exc()
                        results[ass.id][run_label] = {
                            "status": "error",
                            "error": f"divide optimizer failed: {_e}",
                        }
                        continue

                    heur_per_step = heur_timing.get("per_step") or []

                    # Fallback: no verified split → copy heuristic timing.
                    if not verified:
                        overview = dict(heur_timing)
                        overview["status"] = "ok"
                        overview["note"] = "no_verified_split_fallback"
                        (output_run_dir / "log").mkdir(parents=True, exist_ok=True)
                        with open(
                            output_run_dir / "log" / "timing_overview.json", "w"
                        ) as _f:
                            json.dump(overview, _f, indent=2)
                        results[ass.id][run_label] = overview
                        tot = overview.get("totals", {})
                        print(
                            f"[assembly-time]    {run_label}: no split; "
                            f"reusing heuristic total={tot.get('total_s', 0.0):.2f}s"
                        )
                        continue

                    chosen = verified[0]
                    parts_S = sorted(chosen[0])
                    parts_R = sorted(chosen[1])
                    print(
                        f"[assembly-time]    {run_label}: split S={parts_S}  R={parts_R}"
                    )

                    # Re-run the full pipeline (plan + render + arm) for S
                    # and R standalone. Swap ass.assembly_dir / ass.storage_dir
                    # temporarily; the planner reads parts from assembly_dir
                    # directly, so the override is enough. Each subassembly
                    # is treated like a normal assembly: NO base_part
                    # override, so the planner runs its standard initial
                    # self-stable-pose precheck and computes a fresh
                    # stable pose per step. Stability behaviour follows
                    # settings.skip_stability — same as a non-split run.
                    sub_timings = {}
                    sub_failure = None
                    _saved_dir = ass.assembly_dir
                    _saved_storage = ass.storage_dir
                    _saved_cache = args.cache
                    try:
                        for sub_label, sub_parts in (("S", parts_S), ("R", parts_R)):
                            # Always source from the ORIGINAL assembly dir
                            # (_saved_dir), not from a possibly-overridden
                            # ass.assembly_dir left over from the previous
                            # sub-iteration.
                            sub_assembly_tmp = _prepare_subassembly_dir(
                                str(_saved_dir),
                                sub_parts,
                            )
                            sub_output_dir = output_run_dir / f"sub_{sub_label}"
                            sub_output_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                ass.assembly_dir = Path(sub_assembly_tmp)
                                ass.storage_dir = sub_output_dir
                                args.cache = "new"  # always re-plan subassembly
                                print(
                                    f"[assembly-time]    {run_label}: re-running pipeline "
                                    f"for sub-{sub_label} ({len(sub_parts)} parts)"
                                )
                                ass.planner.get_assembly_plans(args)
                                sub_timing_path = (
                                    sub_output_dir / "log" / "timing_overview.json"
                                )
                                if not sub_timing_path.exists():
                                    raise RuntimeError(
                                        f"sub-{sub_label} produced no timing_overview.json"
                                    )
                                with open(sub_timing_path) as _f:
                                    sub_timings[sub_label] = json.load(_f)
                            finally:
                                # Restore ass.assembly_dir / storage_dir
                                # between sub-iterations so the next call
                                # to _prepare_subassembly_dir resolves
                                # against the original assembly even if a
                                # downstream callee mutates the attribute.
                                ass.assembly_dir = _saved_dir
                                ass.storage_dir = _saved_storage
                                shutil.rmtree(sub_assembly_tmp, ignore_errors=True)
                    except Exception as _e:
                        import traceback as _tb

                        _tb.print_exc()
                        sub_failure = f"{_e}"
                    finally:
                        ass.assembly_dir = _saved_dir
                        ass.storage_dir = _saved_storage
                        args.cache = _saved_cache

                    if sub_failure or len(sub_timings) != 2:
                        results[ass.id][run_label] = {
                            "status": "error",
                            "error": f"subassembly pipeline failed: {sub_failure}",
                        }
                        continue

                    # Aggregate: prefix + separation + S + R.
                    in_split = set(parts_S) | set(parts_R)
                    prefix_parts = [p for p in heur_seq if p not in in_split]

                    prefix_components = dict.fromkeys(COMPONENTS, 0.0)
                    for step in heur_per_step:
                        if step.get("part") in prefix_parts:
                            for c in COMPONENTS:
                                prefix_components[c] += float(step.get(c, 0.0) or 0.0)

                    # Separation-step time estimate: one disassembly-step
                    # worth of motion, sized at the median per-step total
                    # of the heuristic run (matches the cost-side z_alignment
                    # placeholder in compare.py — one rough step's-worth).
                    sep_components = dict.fromkeys(COMPONENTS, 0.0)
                    if heur_per_step:
                        sorted_totals = sorted(
                            float(s.get("total_s", 0.0) or 0.0) for s in heur_per_step
                        )
                        median_total = sorted_totals[len(sorted_totals) // 2]
                        sep_components["step_disassembly_s"] = median_total

                    s_totals = sub_timings["S"].get("totals") or {}
                    r_totals = sub_timings["R"].get("totals") or {}

                    agg = dict.fromkeys(COMPONENTS, 0.0)
                    for c in COMPONENTS:
                        agg[c] = (
                            prefix_components[c]
                            + sep_components[c]
                            + float(s_totals.get(c, 0.0) or 0.0)
                            + float(r_totals.get(c, 0.0) or 0.0)
                        )
                    agg["total_s"] = sum(agg[c] for c in COMPONENTS)

                    overview = {
                        "status": "ok",
                        "totals": agg,
                        "components": {
                            "prefix": prefix_components,
                            "separation": sep_components,
                            "S_totals": {
                                c: float(s_totals.get(c, 0.0) or 0.0)
                                for c in COMPONENTS
                            },
                            "R_totals": {
                                c: float(r_totals.get(c, 0.0) or 0.0)
                                for c in COMPONENTS
                            },
                        },
                        "split": {
                            "S": parts_S,
                            "R": parts_R,
                            "divide_score": float(chosen[2]),
                        },
                        "prefix_sequence": prefix_parts,
                        "S_sequence": sub_timings["S"].get("sequence"),
                        "R_sequence": sub_timings["R"].get("sequence"),
                        "per_step": [],
                    }
                    (output_run_dir / "log").mkdir(parents=True, exist_ok=True)
                    with open(
                        output_run_dir / "log" / "timing_overview.json", "w"
                    ) as _f:
                        json.dump(overview, _f, indent=2)
                    results[ass.id][run_label] = overview
                    print(
                        f"[assembly-time]    {run_label}: total={agg['total_s']:.2f}s "
                        f"(prefix={sum(prefix_components.values()):.2f} + "
                        f"sep={sum(sep_components.values()):.2f} + "
                        f"S={float(s_totals.get('total_s', 0.0)):.2f} + "
                        f"R={float(r_totals.get('total_s', 0.0)):.2f})"
                    )
                    continue
                # ------------------------------------------------------

                # 1) Sequence: load from cache, or plan into cache_dir.
                tree = None
                plan_sequence = None
                if cache_hit:
                    try:
                        with open(cache_stats) as _f:
                            stats = json.load(_f)
                        with open(cache_tree, "rb") as _f:
                            tree = pickle.load(_f)
                        plan_sequence = stats.get("sequence") or []
                        if not plan_sequence:
                            print(
                                "[assembly-time]    cache had no sequence — re-planning"
                            )
                            cache_hit = False
                    except Exception as _ce:
                        print(
                            f"[assembly-time]    cache load failed: {_ce}; re-planning"
                        )
                        cache_hit = False

                if not cache_hit:
                    # Plan into cache_dir, with rendering temporarily
                    # disabled so we don't render twice (full render
                    # happens below into output_run_dir).
                    ass.storage_dir = cache_dir
                    stale_log = cache_dir / "log"
                    if stale_log.exists():
                        shutil.rmtree(str(stale_log))
                    for stale_gif in cache_dir.glob("*.gif"):
                        stale_gif.unlink()
                    stale_paths = cache_dir / "paths"
                    if stale_paths.exists():
                        shutil.rmtree(str(stale_paths))

                    _orig_render = settings.render_sequence
                    settings.render_sequence = False
                    # For the "heuristic_trained" run, point the
                    # heuristic planner at the Optuna-trained weights
                    # file for the duration of this planning call only.
                    # All other runs see the user's original setting.
                    _orig_hw_source = getattr(
                        settings, "heuristic_weights_source", "default"
                    )
                    if run_label == "heuristic_trained":
                        settings.heuristic_weights_source = "optuna"
                    try:
                        ass.planner.get_assembly_plans(args)
                    except Exception as _exc:
                        import traceback as _tb

                        _tb.print_exc()
                        settings.render_sequence = _orig_render
                        settings.heuristic_weights_source = _orig_hw_source
                        results[ass.id][run_label] = {
                            "status": "error",
                            "error": f"planning failed: {_exc}",
                        }
                        continue
                    settings.render_sequence = _orig_render
                    settings.heuristic_weights_source = _orig_hw_source

                    # Read back what the planner wrote.
                    if not cache_tree.exists() or not cache_stats.exists():
                        results[ass.id][run_label] = {
                            "status": "error",
                            "error": "planner produced no tree.pkl / stats.json",
                        }
                        print(
                            f"[assembly-time]    {run_label}: missing planner artifacts"
                        )
                        continue
                    try:
                        with open(cache_stats) as _f:
                            stats = json.load(_f)
                        with open(cache_tree, "rb") as _f:
                            tree = pickle.load(_f)
                        plan_sequence = stats.get("sequence") or []
                    except Exception as _ce:
                        results[ass.id][run_label] = {
                            "status": "error",
                            "error": f"failed to read planner artifacts: {_ce}",
                        }
                        continue

                # Precheck-failure: the planner exited because the
                # full assembly has no self-stable initial pose under
                # `settings.no_stable_pose_action='exit'`. Tag this
                # RUN and flip `assembly_precheck_failed` so the
                # remaining methods get the early-skip treatment
                # above (instead of re-running the planner and
                # hitting the same geometric dead end).
                if (
                    getattr(settings, "no_stable_pose_action", "exit") == "exit"
                    and stats.get("stop_msg") == "no self-stable initial pose"
                ):
                    results[ass.id][run_label] = {
                        "status": "no_stable_pose",
                        "error": "no self-stable initial pose for full assembly",
                    }
                    assembly_precheck_failed = True
                    print(
                        f"[assembly-time]    {run_label}: no self-stable "
                        f"initial pose; remaining methods without a "
                        f"cached sequence will be skipped"
                    )
                    continue

                if not plan_sequence:
                    results[ass.id][run_label] = {
                        "status": "error",
                        "error": "empty sequence (planner stopped before any feasible step)",
                    }
                    print(
                        f"[assembly-time]    {run_label}: empty sequence; skipping render"
                    )
                    continue

                # 2) Render: ALWAYS re-run, into the output dir.
                ass.storage_dir = output_run_dir
                # Wipe any stale render output from a prior run.
                out_log = output_run_dir / "log"
                if out_log.exists():
                    shutil.rmtree(str(out_log))
                for stale_gif in output_run_dir.glob("*.gif"):
                    stale_gif.unlink()
                stale_paths = output_run_dir / "paths"
                if stale_paths.exists():
                    shutil.rmtree(str(stale_paths))
                out_log.mkdir(parents=True, exist_ok=True)
                # Mirror tree.pkl + stats.json into the output log so
                # downstream tools (test_render, etc.) can target the
                # output directly without crawling back to the cache.
                shutil.copy2(str(cache_tree), str(out_log / "tree.pkl"))
                shutil.copy2(str(cache_stats), str(out_log / "stats.json"))

                try:
                    ass.planner._render_plan(
                        asset_folder=asset_folder,
                        assembly_dir=str(ass.assembly_dir),
                        plan_sequence=plan_sequence,
                        tree=tree,
                        args=args,
                    )
                except Exception as _exc:
                    import traceback as _tb

                    _tb.print_exc()
                    results[ass.id][run_label] = {
                        "status": "error",
                        "error": f"render/arm pipeline failed: {_exc}",
                    }
                    continue

                # 3) Read the timing overview from the output dir.
                timing_path = output_run_dir / "log" / "timing_overview.json"
                if not timing_path.exists():
                    results[ass.id][run_label] = {
                        "status": "error",
                        "error": "no timing_overview.json produced",
                    }
                    print(
                        f"[assembly-time]    {run_label}: missing timing_overview.json"
                    )
                    continue
                try:
                    with open(timing_path) as _f:
                        overview = json.load(_f)
                except Exception as _e:
                    results[ass.id][run_label] = {
                        "status": "error",
                        "error": f"failed to load timing_overview.json: {_e}",
                    }
                    continue
                overview["status"] = "ok"
                results[ass.id][run_label] = overview
                totals = overview.get("totals", {})
                print(
                    f"[assembly-time]    {run_label}: total={totals.get('total_s', 0.0):.2f}s "
                    f"(disasm={totals.get('step_disassembly_s', 0.0):.2f}, "
                    f"trans={totals.get('transitions_s', 0.0):.2f}, "
                    f"base={totals.get('base_travel_s', 0.0):.2f}, "
                    f"reorient={totals.get('reorientation_s', 0.0):.2f}, "
                    f"hold={totals.get('hold_s', 0.0):.2f})"
                )

            ass.storage_dir = _orig_ass_storage
            # Write incremental summary so partial batches are still useful.
            try:
                _write_assembly_time_summary()
            except Exception as _e:
                print(f"[assembly-time] incremental summary failed: {_e}")

    finally:
        if _orig_planner is not None:
            args.planner = _orig_planner
        if _orig_generator is not None:
            args.generator = _orig_generator
        args.plan_arm = _orig_plan_arm
        if _orig_storage is not None:
            args.storage_dir = _orig_storage
        test_eval.cache = _orig_cache
        try:
            _write_assembly_time_summary()
        except Exception as _e:
            print(f"[assembly-time] failed to write summary: {_e}")


def run_data_manual_validation(args, test_eval, output_folder, assembly_dir):
    # ------------------------------------------------------------------
    # Validate every step's generated manual by:
    #   A = canonical instruction (built from step metadata)
    #   B = vision-LLM description of the generated manual page
    #   verdict = independent LLM verifying every fact in A appears in B
    # Per-assembly outputs go to <storage_dir>/manual_validation/;
    # a cross-assembly digest lands under <output_folder>/.
    # try/finally so a partial batch still produces the digest.
    # ------------------------------------------------------------------
    from core.manual_validator import (
        validate_assembly_manual as _validate_manual,
    )
    from core.manual_validator import (
        write_batch_summary as _write_manual_batch_summary,
    )

    try:
        for ass in test_eval.assemblies:
            print(f"\n[manual-val] ===== assembly {ass.id} =====")
            try:
                # Per-object images first so name_parts / name_tools
                # have something to feed the VLM.
                ass.images
                if settings.part_naming:
                    ass.name_parts()
                if settings.tool_naming:
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

                # Make sure ass.sequence is populated.
                ass.planner.get_assembly_plans(args)
                # Make sure step.matrices are loaded for motion-axis
                # inference (no-op if paths/ are missing).
                with contextlib.suppress(Exception):
                    ass.planner.fetch_sequence_matrices()

                # Rank the per-step camera angles by usefulness so the
                # manual generator picks the best-ranked view (matches
                # the test_pipeline ordering).
                if settings.angle_ranking:
                    for step in ass.sequence:
                        step.rank_angles(show=False)

                _validate_manual(
                    ass,
                    test_eval,
                    run_manual_generation=True,
                )
            except Exception as _exc:
                import traceback as _tb

                print(f"[manual-val] {ass.id} ERROR: {_exc}")
                _tb.print_exc()
    finally:
        try:
            _write_manual_batch_summary(test_eval.assemblies, output_folder)
        except Exception as _e:
            print(f"[manual-val] failed to write batch summary: {_e}")


def run_data_validate_cost(args, test_eval, output_folder, assembly_dir):
    # ------------------------------------------------------------------
    # Run the comparison planner (ComparisonDFASequencePlanner) per
    # assembly: every frontier-truncation step asks all configured
    # selectors which candidate to take, logs their picks, and advances
    # by a uniform-random pick from the sample. The planner writes a
    # `comparison_summary.{json,txt}` under each assembly's log dir on
    # exit; we aggregate them into a batch digest under output_folder.
    # Selector list comes from `settings.comparison_planner['planners']`.
    # ------------------------------------------------------------------
    _orig_planner = getattr(args, "planner", None)
    _orig_cache = test_eval.cache

    try:
        args.planner = "comparison"
        test_eval.cache = "new"  # force fresh planner run

        for ass in test_eval.assemblies:
            print(f"\n[validate-cost] ===== assembly {ass.id} =====")
            try:
                # Wipe stale per-assembly cache; cached sequence.json
                # short-circuits the planner and would prevent any
                # comparison decisions from ever being logged.
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

                print(f"[validate-cost] {ass.id} ERROR: {_exc}")
                _tb.print_exc()
    finally:
        if _orig_planner is not None:
            args.planner = _orig_planner
        test_eval.cache = _orig_cache
        try:
            _write_comparison_batch_summary(test_eval.assemblies, output_folder)
        except Exception as _e:
            print(f"[validate-cost] failed to write batch summary: {_e}")


def run_test_tool_needed(args, test_eval, output_folder, assembly_dir):
    for ass in test_eval.assemblies:
        evaluator = ToolEvaluator(ass)
        out_dir = output_folder / "tool_needed" / ass.id
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Render 4-angle images for all parts ──────────────────────
        print(f"\nRendering images for assembly {ass.id}...")
        evaluator._create_tetra_images()

        tool_list = list(ass.evaluation.tools.keys())
        options = [*tool_list, "none"]

        # ── Phase 1: Human labeling ───────────────────────────────────
        print(f"\n{'=' * 62}")
        print(f"  HUMAN LABELING — assembly {ass.id}")
        print(f"{'=' * 62}")

        storage_human_json = ass.storage_dir / "human_labels.json"
        out_human_json = out_dir / "human_labels.json"

        if storage_human_json.exists():
            with open(storage_human_json) as f:
                cached = json.load(f)
            human_labels = cached.get("labels", {})
            print(
                f"  Loaded cached human labels from {storage_human_json}  "
                f"({len(human_labels)} parts)"
            )
        else:
            human_labels = {}
            for obj_id, obj in ass.objects.items():
                img_paths = [
                    p
                    for p in (obj.image_paths or {}).values()
                    if p is not None and Path(p).exists()
                ]

                if img_paths:
                    n = len(img_paths)
                    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
                    if n == 1:
                        axes = [axes]
                    for ax, ip in zip(axes, img_paths, strict=False):
                        ax.imshow(PILImage.open(ip))
                        ax.axis("off")
                    fig.suptitle(f"Part: {obj.name}  ({obj_id})", fontsize=13)
                    plt.tight_layout()
                    plt.show(block=False)
                    plt.pause(1)
                    plt.close(fig)

                print(f"\n  Part: {obj_id}  ({obj.name})")
                for i, t in enumerate(options, start=1):
                    print(f"    {i}: {t}")
                while True:
                    raw = input(f"  → Choice (1–{len(options)}): ").strip()
                    if raw.isdigit() and 1 <= int(raw) <= len(options):
                        break
                    print(f"    Invalid — enter a number between 1 and {len(options)}.")
                human_labels[obj_id] = options[int(raw) - 1]

        # Save to both the run output dir and the persistent storage cache.
        payload = {"assembly_id": ass.id, "labels": human_labels}
        for dest in (out_human_json, storage_human_json):
            with open(dest, "w") as f:
                json.dump(payload, f, indent=2)
        print("\n  Human labels saved:")
        print(f"    {out_human_json}")
        print(f"    {storage_human_json}")

        # ── Phase 2: AI labeling ──────────────────────────────────────
        print(f"\n{'=' * 62}")
        print(f"  AI LABELING — assembly {ass.id}")
        print(f"{'=' * 62}")

        # Ensure all parts have engineering names before running AI inference.
        if not (ass.storage_dir / "part_names.json").exists():
            print("  Generating part names...")
            ass.name_parts(iso_only=False, log_probs=False)
            print(f"  Tokens used: {test_eval.tokens_used:,} / {args.token_limit:,}")

        n_samples = max(1, args.ai_samples)
        ai_labels = {}  # obj_id -> list[str] of length n_samples
        ai_reasoning = {}  # obj_id -> list[dict|None] of length n_samples
        aborted = False
        for obj_id, obj in ass.objects.items():
            sample_labels = []
            sample_reasonings = []
            for s_idx in range(n_samples):
                if test_eval.tokens_used >= args.token_limit:
                    print(
                        f"\n  ⚠ Token limit reached "
                        f"({test_eval.tokens_used:,} / {args.token_limit:,}). "
                        "Aborting remaining AI calls."
                    )
                    sample_labels.append("aborted")
                    sample_reasonings.append(None)
                    aborted = True
                    continue

                tokens_before = test_eval.tokens_used
                print(
                    f"\n  Part {obj_id} ({obj.name}) — "
                    f"sample {s_idx + 1}/{n_samples}..."
                )
                try:
                    tool_suggested, reasoning = evaluator.check_tool_needed_cot(
                        obj_idx=obj_id
                    )
                    sample_labels.append(tool_suggested or "error")
                    sample_reasonings.append(reasoning)
                except Exception as e:
                    print(f"  Error processing {obj_id}: {e}")
                    sample_labels.append("error")
                    sample_reasonings.append(None)

                delta = test_eval.tokens_used - tokens_before
                pct = 100.0 * test_eval.tokens_used / args.token_limit
                print(
                    f"    Tokens: +{delta:,}  →  {test_eval.tokens_used:,} / "
                    f"{args.token_limit:,} ({pct:.1f}%)"
                )

            ai_labels[obj_id] = sample_labels
            ai_reasoning[obj_id] = sample_reasonings

        if aborted:
            print(
                f"\n  Note: assembly {ass.id} was partially processed due to the token limit."
            )

        ai_json = out_dir / "ai_labels.json"
        with open(ai_json, "w") as f:
            json.dump(
                {
                    "assembly_id": ass.id,
                    "n_samples": n_samples,
                    "labels": ai_labels,
                    "reasoning": ai_reasoning,
                },
                f,
                indent=2,
            )
        print(f"\n  AI labels saved: {ai_json}")

        # ── Phase 3: Comparison plot (mean ± SE across samples) ──────
        part_ids = list(human_labels.keys())

        # Flatten: one row per (part, sample).
        flat = []  # (part_id, human, ai, correct)
        for p in part_ids:
            h = human_labels[p]
            for ai in ai_labels.get(p, []):
                flat.append((p, h, ai, h == ai))

        total_samples = len(flat)
        correct_total = sum(1 for *_, c in flat if c)
        accuracy = correct_total / total_samples if total_samples else 0.0
        ((accuracy * (1 - accuracy) / total_samples) ** 0.5 if total_samples else 0.0)

        # Per-part mean accuracy ± SE.
        per_part_stats = []  # (part_id, mean, se, majority_ai)
        for p in part_ids:
            samples = ai_labels.get(p, [])
            n = len(samples)
            n_c = sum(1 for ai in samples if ai == human_labels[p])
            pr = n_c / n if n else 0.0
            se = (pr * (1 - pr) / n) ** 0.5 if n else 0.0
            maj = Counter(samples).most_common(1)[0][0] if samples else "—"
            per_part_stats.append((p, pr, se, maj))

        # Per-class stats (denominator = total samples where human = c).
        all_classes = sorted(
            (set(human_labels.values()) | {s[2] for s in flat})
            - {"error", "aborted", "missing"}
        )
        per_class_stats = []  # (class, n_total, n_correct, mean, se)
        for c in all_classes:
            cls_samples = [s for s in flat if s[1] == c]
            n = len(cls_samples)
            n_c = sum(1 for *_, ok in cls_samples if ok)
            pr = n_c / n if n else 0.0
            se = (pr * (1 - pr) / n) ** 0.5 if n else 0.0
            per_class_stats.append((c, n, n_c, pr, se))

        fig = plt.figure(figsize=(max(10, len(part_ids) * 1.0 + 2), 9))
        fig.suptitle(
            f"Tool detection accuracy — assembly {ass.id}  "
            f"({accuracy:.1%} overall, n={n_samples}/part)",
            fontsize=13,
            fontweight="bold",
        )

        # Top panel: per-part mean accuracy (with SE bars when n_samples > 1)
        ax_top = fig.add_subplot(2, 1, 1)
        xs = np.arange(len(part_ids))
        p_means = [s[1] for s in per_part_stats]
        p_ses = [s[2] for s in per_part_stats]
        bar_colors = ["#2ecc71" if m >= 0.5 else "#e74c3c" for m in p_means]
        ax_top.bar(
            xs,
            p_means,
            yerr=p_ses,
            capsize=3,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"ecolor": "black", "lw": 0.8},
        )
        ax_top.set_xticks(xs)
        ax_top.set_xticklabels(part_ids, rotation=45, ha="right", fontsize=8)
        ax_top.set_ylim(0, 1.25)
        ax_top.set_yticks([0.0, 0.5, 1.0])
        ax_top.set_ylabel("Mean accuracy")
        ax_top.set_title(
            f"Per-part mean accuracy "
            f"({'mean ± SE' if n_samples > 1 else 'green = correct'})"
        )
        ax_top.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax_top.set_axisbelow(True)
        for i, (p, _mean, se, maj) in enumerate(per_part_stats):
            h_lbl = human_labels[p]
            ax_top.text(
                i,
                0.05,
                f"AI: {maj}\nHuman: {h_lbl}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="black",
            )

        # Bottom panel: per-class mean accuracy ± SE
        ax_bot = fig.add_subplot(2, 1, 2)
        xs = np.arange(len(all_classes))
        c_means = [s[3] for s in per_class_stats]
        c_ses = [s[4] for s in per_class_stats]
        ax_bot.bar(
            xs,
            [m * 100 for m in c_means],
            yerr=[s * 100 for s in c_ses],
            capsize=4,
            color="#1f3a93",
            edgecolor="black",
            linewidth=0.5,
            error_kw={"ecolor": "black", "lw": 1},
        )
        ax_bot.set_xticks(xs)
        ax_bot.set_xticklabels(all_classes, rotation=30, ha="right", fontsize=9)
        ax_bot.set_ylabel("Accuracy (%)")
        ax_bot.set_ylim(0, 110)
        ax_bot.set_title("Accuracy per tool class  (mean ± SE)")
        ax_bot.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax_bot.set_axisbelow(True)
        for x, (_cls, n, n_c, pr, _) in zip(xs, per_class_stats, strict=False):
            ax_bot.text(
                x,
                pr * 100 + 2.0,
                f"{pr:.0%}\n({n_c}/{n})",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        fig.tight_layout()
        plot_path = out_dir / "accuracy.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"\n  Overall accuracy: {accuracy:.1%}")
        print(f"  Plot saved:       {plot_path}")
        print(
            f"  Total tokens after assembly {ass.id}: "
            f"{test_eval.tokens_used:,} / {args.token_limit:,}"
        )

        if test_eval.tokens_used >= args.token_limit:
            print(
                f"\n⚠ Global token limit reached "
                f"({test_eval.tokens_used:,} / {args.token_limit:,}). "
                "Skipping remaining assemblies."
            )
            break


def run_collect_tool_data(args, test_eval, output_folder, assembly_dir):
    if args.data_dir is None:
        raise ValueError(
            "collect_tool_data requires --data-dir pointing at a directory of human_labels.json / ai_labels.json files."
        )
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")

    # Walk the data dir for every AI-labels file (pattern allows
    # ai_labels.json, ai_labels_run1.json, ai_labels_v2.json, …) and pair
    # each one with the sibling human_labels.json in the same directory.
    ai_jsons = sorted(data_dir.rglob("ai_labels*.json"))
    if not ai_jsons:
        raise FileNotFoundError(f"No ai_labels*.json files found under {data_dir}")

    print(f"Found {len(ai_jsons)} ai_labels*.json file(s) under {data_dir}")

    # Each row: (assembly_id, part_id, sample_idx, human_label, ai_label).
    # Multiple AI samples for the same (assembly, part) are concatenated;
    # sample indices keep incrementing so each datapoint stays distinct.
    samples = []
    assemblies_seen = set()
    files_per_assembly = Counter()
    human_first = {}  # (ass_id, part_id) -> first human label seen
    sample_counters = {}  # (ass_id, part_id) -> next free sample idx
    skipped = 0
    for apath in ai_jsons:
        hpath = apath.parent / "human_labels.json"
        if not hpath.exists():
            print(f"  [skip] {apath}  (no sibling human_labels.json)")
            skipped += 1
            continue
        with open(hpath) as f:
            hdata = json.load(f)
        with open(apath) as f:
            adata = json.load(f)

        ass_id = (
            adata.get("assembly_id") or hdata.get("assembly_id") or apath.parent.name
        )
        hlabels = hdata.get("labels", {})
        alabels = adata.get("labels", {})

        file_rows = 0
        file_matches = 0
        ai_missing = 0
        new_parts = 0
        for part_id, hlabel in hlabels.items():
            key = (ass_id, part_id)

            if key not in human_first:
                human_first[key] = hlabel
                new_parts += 1
            elif human_first[key] != hlabel:
                print(
                    f"  [warn] Human label conflict for "
                    f"{ass_id}/{part_id}: kept {human_first[key]!r}, "
                    f"ignored {hlabel!r} from {apath.name}"
                )
            hlabel_used = human_first[key]

            # Backward compat: old format had string labels; new format has lists.
            if part_id in alabels:
                raw = alabels[part_id]
                ai_sample_list = raw if isinstance(raw, list) else [raw]
            else:
                ai_sample_list = ["missing"]
                ai_missing += 1

            for ai_lbl in ai_sample_list:
                s_idx = sample_counters.get(key, 0)
                sample_counters[key] = s_idx + 1
                samples.append((ass_id, part_id, s_idx, hlabel_used, ai_lbl))
                file_rows += 1
                if ai_lbl == hlabel_used:
                    file_matches += 1

        assemblies_seen.add(ass_id)
        files_per_assembly[ass_id] += 1
        file_acc = file_matches / file_rows if file_rows else 0.0
        merged = (
            f"  (file #{files_per_assembly[ass_id]} for this assembly, "
            f"+{new_parts} new parts)"
            if files_per_assembly[ass_id] > 1
            else ""
        )
        print(
            f"  [load] {apath.relative_to(data_dir)}: assembly={ass_id}  "
            f"parts={len(hlabels)}  rows={file_rows}  "
            f"matched={file_matches}/{file_rows} ({file_acc:.1%})  "
            f"ai_missing={ai_missing}{merged}"
        )

    # Summarise multi-file assemblies.
    merged_assemblies = {a: c for a, c in files_per_assembly.items() if c > 1}
    if merged_assemblies:
        print("\nMerged multiple files into a single run for:")
        for ass_id, n_files in sorted(merged_assemblies.items()):
            n_samples_ass = sum(1 for a, *_ in samples if a == ass_id)
            n_parts_ass = len({p for a, p, *_ in samples if a == ass_id})
            print(
                f"  {ass_id}: {n_files} files → {n_parts_ass} parts, "
                f"{n_samples_ass} samples"
            )

    if not samples:
        raise RuntimeError("No labeled parts loaded — nothing to plot.")

    total = len(samples)
    HUMAN_NON_TOOLS = {"none", "error", "aborted", "missing"}

    # ── Step 1: collapse runs to part level ───────────────────────────
    # part_data[(ass_id, part_id)] = (human_label, [ai_run_1, ai_run_2, ...])
    part_data = {}
    for ass_id, part_id, _s_idx, h, a in samples:
        key = (ass_id, part_id)
        if key not in part_data:
            part_data[key] = (h, [])
        part_data[key][1].append(a)
    n_parts = len(part_data)

    # Group parts by assembly for the clustered bootstrap.
    parts_by_assembly = {}
    for (ass_id, _), entry in part_data.items():
        parts_by_assembly.setdefault(ass_id, []).append(entry)

    # Distribution of number of runs per part (sanity check).
    runs_per_part = Counter(len(al) for _, al in part_data.values())
    print("\nRuns per part:", dict(runs_per_part))

    # ── Step 2: per-part rate + metric definitions ────────────────────
    def _rate(h, ai_list, predicate):
        """fraction of ai_list entries that satisfy predicate(h, a)."""
        if not ai_list:
            return None
        return sum(1 for a in ai_list if predicate(h, a)) / len(ai_list)

    def overall_per_part(h, al):
        return _rate(h, al, lambda h, a: h == a)

    def tool_need_per_part(h, al):
        # Recall on tool-needed parts: of parts where human picked a real tool,
        # fraction of AI runs that ALSO chose a tool (anything except "none").
        if h in HUMAN_NON_TOOLS:
            return None
        return _rate(h, al, lambda h, a: a != "none")

    def make_tool_metric(tool):
        # Per-tool accuracy: among parts where human = tool, fraction
        # of AI runs that picked the exact same tool.
        def _fn(h, al):
            if h != tool:
                return None
            return _rate(h, al, lambda h, a: a == tool)

        return _fn

    tool_classes = sorted({h for h, _ in part_data.values()} - HUMAN_NON_TOOLS)

    metrics = {"overall": overall_per_part, "tool_need": tool_need_per_part}
    for t in tool_classes:
        metrics[t] = make_tool_metric(t)

    def metric_mean(parts, metric_fn):
        vals = []
        for h, al in parts:
            v = metric_fn(h, al)
            if v is not None:
                vals.append(v)
        return (sum(vals) / len(vals)) if vals else 0.0, len(vals)

    # ── Point estimates on the full data ─────────────────────────────
    all_parts = list(part_data.values())
    point = {}  # name -> mean rate
    denom = {}  # name -> # of contributing parts
    for name, fn in metrics.items():
        p, n = metric_mean(all_parts, fn)
        point[name] = p
        denom[name] = n

    # ── Step 3: clustered bootstrap, resampling assemblies ────────────
    BOOT_ITER = 10_000
    BOOT_SEED = 42
    rng = random.Random(BOOT_SEED)
    ass_ids = list(parts_by_assembly.keys())
    n_ass = len(ass_ids)

    boot_vals = {name: [] for name in metrics}
    print(
        f"\nRunning clustered bootstrap "
        f"({BOOT_ITER:,} iters, resampling {n_ass} assemblies)..."
    )
    for _ in range(BOOT_ITER):
        # Draw assemblies with replacement, keep ALL their parts.
        resampled = []
        for _ in range(n_ass):
            resampled.extend(parts_by_assembly[ass_ids[rng.randrange(n_ass)]])
        for name, fn in metrics.items():
            p, _ = metric_mean(resampled, fn)
            boot_vals[name].append(p)

    # Use the 15th / 85th percentile of the bootstrap distribution as
    # the lower / upper whiskers. Stored as (lower_value, upper_value).
    LOWER_Q, UPPER_Q = 15.0, 85.0
    bands = {
        name: (
            float(np.percentile(vs, LOWER_Q)),
            float(np.percentile(vs, UPPER_Q)),
        )
        for name, vs in boot_vals.items()
    }

    overall_acc = point["overall"]
    overall_lo, overall_hi = bands["overall"]
    tool_need_recognized = point["tool_need"]
    tool_need_lo, tool_need_hi = bands["tool_need"]

    # ── Step 4: VLM consistency ──────────────────────────────────────
    multi_run_parts = [(h, al) for h, al in all_parts if len(al) > 1]
    if multi_run_parts:
        n_agreed = sum(1 for _, al in multi_run_parts if len(set(al)) == 1)
        vlm_consistency = n_agreed / len(multi_run_parts)
        vlm_consistency_note = (
            f"{vlm_consistency:.1%} "
            f"({n_agreed}/{len(multi_run_parts)} parts unanimous "
            f"across runs)"
        )
    else:
        vlm_consistency = float("nan")
        vlm_consistency_note = "n/a (only one run per part)"

    # Per-tool table (point + bootstrap p15/p85 band).
    per_tool = []  # (tool, n_parts_with_label, p, lo, hi)
    for t in tool_classes:
        lo, hi = bands[t]
        per_tool.append((t, denom[t], point[t], lo, hi))

    # ── Reports ──────────────────────────────────────────────────────
    ai_counter = Counter(a for *_, a in samples)
    print("\nAI label breakdown (over all runs):")
    for label, n in sorted(ai_counter.items(), key=lambda x: -x[1]):
        print(f"  {label:<20}  {n}")

    print(
        f"\nMetric                     mean    [p{LOWER_Q:.0f}, p{UPPER_Q:.0f}]"
        "     parts contributing"
    )
    print("------------------------   ------  --------------   ------------------")
    print(
        f"  Overall accuracy         {overall_acc:>5.1%}  "
        f"[{overall_lo:>5.1%}, {overall_hi:>5.1%}]   {denom['overall']}"
    )
    print(
        f"  Tool-need recognition    {tool_need_recognized:>5.1%}  "
        f"[{tool_need_lo:>5.1%}, {tool_need_hi:>5.1%}]   {denom['tool_need']}"
    )
    for t, n, p, lo, hi in per_tool:
        print(f"  {t:<24} {p:>5.1%}  [{lo:>5.1%}, {hi:>5.1%}]   {n}")

    print(f"\nVLM consistency: {vlm_consistency_note}")

    # Assemble the bar chart — bar height is the point-estimate mean, with a
    # boxplot of the clustered-bootstrap distribution overlaid per column
    # (box = p25–p75 IQR, centre line = median, whiskers = p15/p85 band).
    bar_labels = ["Overall", "Tool need\nrecognized"] + [t for t, *_ in per_tool]
    bar_values = [overall_acc * 100, tool_need_recognized * 100] + [
        p * 100 for *_, p, _, _ in per_tool
    ]
    lower_band = [overall_lo * 100, tool_need_lo * 100] + [
        lo * 100 for *_, lo, _ in per_tool
    ]
    upper_band = [overall_hi * 100, tool_need_hi * 100] + [
        hi * 100 for *_, hi in per_tool
    ]
    # yerr wants distance from bar height (non-negative), per side — these
    # draw the p15/p85 error band on top of the bars.
    yerr_lo = [max(0.0, v - lo) for v, lo in zip(bar_values, lower_band, strict=False)]
    yerr_hi = [max(0.0, hi - v) for v, hi in zip(bar_values, upper_band, strict=False)]
    # Clustered-bootstrap distributions (scaled to %) in the same column
    # order as the bars — these drive the overlaid boxplots.
    box_order = ["overall", "tool_need"] + [t for t, *_ in per_tool]
    box_data = [np.asarray(boot_vals[name]) * 100.0 for name in box_order]
    colors = ["#c0392b", "#c0392b"] + ["#1f3a93"] * len(per_tool)

    fig, ax = plt.subplots(figsize=(max(8, len(bar_labels) * 1.1 + 2), 6))
    xs = np.arange(len(bar_labels))
    ax.bar(
        xs,
        bar_values,
        yerr=[yerr_lo, yerr_hi],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.30,
        zorder=1,
        error_kw={"ecolor": "black", "lw": 1, "zorder": 4},
    )
    # Overlay one boxplot per column on the bootstrap distribution. Whiskers
    # are pinned to the reported p15/p85 band; bootstrap outliers are hidden.
    bp = ax.boxplot(
        box_data,
        positions=xs,
        widths=0.5,
        whis=(LOWER_Q, UPPER_Q),
        showfliers=False,
        patch_artist=True,
        zorder=3,
        medianprops={"color": "black", "lw": 1.5},
        whiskerprops={"color": "black", "lw": 1},
        capprops={"color": "black", "lw": 1},
        boxprops={"edgecolor": "black", "lw": 1},
    )
    for patch, c in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_yticks(range(0, 101, 10))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    min_runs_per_part = min((len(al) for _, al in part_data.values()), default=0)
    ax.set_title(
        f"Tool detection accuracy "
        f"(mean bar · clustered-bootstrap boxplot, "
        f"whiskers p{LOWER_Q:.0f}–p{UPPER_Q:.0f})\n"
        f"{n_ass} assemblies  ·  {n_parts} parts  ·  "
        f"{min_runs_per_part} runs/part",
        fontsize=12,
        fontweight="bold",
    )

    # Label each column with its mean, placed just above the upper whisker.
    for x, v, hi in zip(xs, bar_values, upper_band, strict=False):
        ax.text(
            x,
            hi + 2.0,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    plot_path = output_folder / "tool_data_combined.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    csv_path = output_folder / "tool_data_combined.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["assembly_id", "part_id", "sample_idx", "human", "ai", "match"]
        )
        for ass_id, part_id, s_idx, h, a in sorted(samples):
            writer.writerow(
                [
                    ass_id,
                    part_id,
                    s_idx,
                    h,
                    a,
                    "YES" if h == a else "NO",
                ]
            )

    # Also write a part-level summary (collapsed per-part rates).
    csv_parts_path = output_folder / "tool_data_combined_parts.csv"
    with open(csv_parts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "assembly_id",
                "part_id",
                "human",
                "n_runs",
                "n_correct",
                "p_correct",
                "all_agreed",
            ]
        )
        for (ass_id, part_id), (h, al) in sorted(part_data.items()):
            n_c = sum(1 for a in al if a == h)
            writer.writerow(
                [
                    ass_id,
                    part_id,
                    h,
                    len(al),
                    n_c,
                    f"{n_c / len(al):.4f}" if al else "",
                    "YES" if al and len(set(al)) == 1 else "NO",
                ]
            )

    print(
        f"\nAggregated {total} runs across {n_parts} parts from "
        f"{n_ass} assemblies ({skipped} files skipped)."
    )
    print(
        f"Overall accuracy: {overall_acc:.1%}  "
        f"[p{LOWER_Q:.0f} {overall_lo:.1%}, p{UPPER_Q:.0f} {overall_hi:.1%}] "
        f"(clustered bootstrap)"
    )
    print(f"Plot:             {plot_path}")
    print(f"CSV (runs):       {csv_path}")
    print(f"CSV (per part):   {csv_parts_path}")


def run_collect_tool_axes(args, test_eval, output_folder, assembly_dir):
    if args.data_dir is None:
        raise ValueError(
            "collect_tool_axes requires --data-dir pointing at a directory of tool_axes_*.csv files."
        )
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")

    csv_files = sorted(data_dir.rglob("tool_axes_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No tool_axes_*.csv files found under {data_dir}")

    print(f"Found {len(csv_files)} tool_axes_*.csv file(s) under {data_dir}")

    # List of (assembly_id, tool_id, sample_idx, confirmed bool) — keeping every sample.
    samples = []
    assemblies_seen = set()
    for cpath in csv_files:
        ass_id = cpath.stem.replace("tool_axes_", "", 1)
        file_records = 0
        file_yes = 0
        with open(cpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tool_id = row["tool_id"]
                conf_str = row.get("confirmed", "").strip().upper()
                if conf_str not in ("YES", "NO"):
                    continue
                # sample_idx is optional (older CSVs don't have it)
                try:
                    s_idx = int(row.get("sample_idx", 0))
                except (ValueError, TypeError):
                    s_idx = 0
                confirmed = conf_str == "YES"
                samples.append((ass_id, tool_id, s_idx, confirmed))
                file_records += 1
                if confirmed:
                    file_yes += 1
        assemblies_seen.add(ass_id)
        print(
            f"  [load] {cpath.parent.name}/{cpath.name}: assembly={ass_id}  "
            f"samples={file_records}  confirmed={file_yes}/{file_records}"
        )

    if not samples:
        raise RuntimeError("No usable rows loaded — nothing to plot.")

    total = len(samples)
    confirmed_total = sum(1 for *_, v in samples if v)
    overall_p = confirmed_total / total
    overall_se = (overall_p * (1 - overall_p) / total) ** 0.5

    # Per-tool stats across assemblies — every sample counts independently.
    tool_ids_all = sorted({tid for _, tid, _, _ in samples})
    per_tool = []  # (tool_id, n, n_yes, p, se)
    for tid in tool_ids_all:
        vs = [v for _, t, _, v in samples if t == tid]
        n = len(vs)
        n_yes = sum(vs)
        p = n_yes / n if n else 0.0
        se = (p * (1 - p) / n) ** 0.5 if n > 0 else 0.0
        per_tool.append((tid, n, n_yes, p, se))

    # Bar chart, same colour convention as collect_tool_data.
    bar_labels = ["Overall"] + [t for t, *_ in per_tool]
    bar_values = [overall_p * 100] + [p * 100 for *_, p, _ in per_tool]
    bar_errs = [overall_se * 100] + [se * 100 for *_, se in per_tool]
    colors = ["#c0392b"] + ["#1f3a93"] * len(per_tool)

    fig, ax = plt.subplots(figsize=(max(8, len(bar_labels) * 1.1 + 2), 6))
    xs = np.arange(len(bar_labels))
    bars = ax.bar(
        xs,
        bar_values,
        yerr=bar_errs,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        error_kw={"ecolor": "black", "lw": 1},
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("User confirmation rate (%)")
    ax.set_yticks(range(0, 101, 10))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        f"AI axis pick — user confirmation (mean ± SE)  ·  "
        f"{len(assemblies_seen)} assemblies  ·  {total} samples",
        fontsize=12,
        fontweight="bold",
    )
    for bar, v in zip(bars, bar_values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    plot_path = output_folder / "tool_axes_combined.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    csv_path = output_folder / "tool_axes_combined.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["assembly_id", "tool_id", "sample_idx", "confirmed"])
        for ass_id, tool_id, s_idx, v in sorted(samples):
            writer.writerow([ass_id, tool_id, s_idx, "YES" if v else "NO"])

    print(f"\nAggregated {total} samples from {len(assemblies_seen)} assemblies.")
    print(f"Overall confirmation rate: {overall_p:.1%}  ± {overall_se:.1%} SE")
    print(f"Plot:                      {plot_path}")
    print(f"CSV:                       {csv_path}")
