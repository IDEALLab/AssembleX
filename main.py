"""AssembleX entry point.

Parses CLI args, builds the shared Eval, and dispatches the chosen
`test_type` to its handler. Handlers live in run_pipeline / run_data /
run_debug; shared helpers in run_common.
"""

import json
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

from run_common import create_output_directory, resolve_ids
from run_data import (
    run_collect_tool_axes,
    run_collect_tool_data,
    run_data_assembly_time,
    run_data_filter_assemblies,
    run_data_heuristic_validation,
    run_data_manual_validation,
    run_data_validate_cost,
    run_test_convex_decomp,
    run_test_tool_needed,
    run_train_heuristic_weights,
)
from run_debug import (
    run_test_archive_ASAP,
    run_test_collision,
    run_test_collision_graph_batch,
    run_test_collision_resolver,
    run_test_divide_optimizer,
    run_test_gravity,
    run_test_param_sweep,
    run_test_PCA,
    run_test_tool_naming,
    run_test_tools,
)
from run_pipeline import (
    run_test_pipeline,
    run_test_pipeline_batch,
    run_test_render,
)
from core.assembly import Eval

# Maps each test_type to its handler. Every handler takes
# (args, test_eval, output_folder, assembly_dir); only run_test_pipeline
# returns a timings dict (consumed by the summary below), the rest None.
DISPATCH = {
    # Pipeline
    "test_pipeline": run_test_pipeline,
    "test_pipeline_batch": run_test_pipeline_batch,
    "test_render": run_test_render,
    # Data generation
    "data_assembly_time": run_data_assembly_time,
    "train_heuristic_weights": run_train_heuristic_weights,
    "data_heuristic_validation": run_data_heuristic_validation,
    "data_manual_validation": run_data_manual_validation,
    "data_validate_cost": run_data_validate_cost,
    "data_filter_assemblies": run_data_filter_assemblies,
    "test_convex_decomp": run_test_convex_decomp,
    "test_tool_needed": run_test_tool_needed,
    "collect_tool_data": run_collect_tool_data,
    "collect_tool_axes": run_collect_tool_axes,
    # Debugging / diagnostics
    "test_collision": run_test_collision,
    "test_PCA": run_test_PCA,
    "test_tools": run_test_tools,
    "test_collision_resolver": run_test_collision_resolver,
    "test_collision_graph_batch": run_test_collision_graph_batch,
    "test_divide_optimizer": run_test_divide_optimizer,
    "test_tool_naming": run_test_tool_naming,
    "test_gravity": run_test_gravity,
    "test_param_sweep": run_test_param_sweep,
    "test_archive_ASAP": run_test_archive_ASAP,
}


if __name__ == "__main__":
    output_folder = create_output_directory()

    parser = ArgumentParser()
    parser.add_argument(
        "test_type", type=str, choices=list(DISPATCH), help="type of test to run"
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="assembly id (e.g. 00000) or inclusive range (e.g. 00000-00010). Optional: omit for subcommands that do not operate on assemblies (e.g. collect_tool_data / collect_tool_axes, which only read --data-dir).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/multi_assembly",
        help="directory storing all assemblies",
    )
    parser.add_argument(
        "--cache",
        type=str,
        choices=["read", "update", "new"],
        default="read",
        help='cache mode for assembly processing. "read" will use existing cache if available, "update" will re-run sequence finding and update cache, "new" will re-run sequence finding and ignore cache.',
    )
    parser.add_argument(
        "-x",
        type=str,
        default=None,
        metavar="COMPONENTS",
        help=(
            "pipeline components to run (test_pipeline only): "
            "g=gravity, c=collisions, i=instructions, t=tools, f=feedback, m=manuals, v=video(stitch). "
            "Add 'x' to invert: all components on by default, skip the ones listed. "
            "Example: -x gc  runs only gravity+collisions; -x xgc  runs everything except those."
        ),
    )
    # region Situational flags for specific tests
    parser.add_argument("--rotation", default=False, action="store_true")
    parser.add_argument(
        "--seq-planner", type=str, default="ASAP", choices=["ASAP", "ATA"]
    )
    parser.add_argument(
        "--generator", type=str, default="rand", help="ASAP generator name (e.g. rand)"
    )
    parser.add_argument(
        "--planner", type=str, default="heuristic", help="ASAP planner name (e.g. dfs)"
    )
    parser.add_argument(
        "--sdf-dx", type=float, default=0.05, help="grid resolution of SDF"
    )
    parser.add_argument("--collision-th", type=float, default=1e-2)
    parser.add_argument(
        "--force-mag", type=float, default=100, help="magnitude of force"
    )
    parser.add_argument("--frame-skip", type=int, default=100, help="control frequency")
    parser.add_argument(
        "--seq-max-time", type=float, default=3600, help="sequence planning timeout"
    )
    parser.add_argument(
        "--path-max-time", type=float, default=120, help="path planning timeout"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--n-save-state", type=int, default=5)
    parser.add_argument(
        "--token-limit",
        type=int,
        default=200_000,
        help="cumulative LLM token budget across all assemblies. Once exceeded, all subsequent LLM calls are skipped.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="maximum iterations for sequence and path planning (overrides timeouts if set)",
    )
    parser.add_argument(
        "--render-num-proc",
        type=int,
        default=30,
        help=(
            "number of parallel worker processes for ASAP step rendering "
            "(play_logged_plan). Each step is rendered in its own process; "
            "set to 1 to keep the original serial behavior."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="verbosity level (0=off, 1=info, 2=debug, 3=trace); passed as debug= to the sequence finder",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="train_heuristic_weights: number of Optuna trials to run.",
    )
    parser.add_argument(
        "--optuna-resume",
        action="store_true",
        help="train_heuristic_weights: open a persistent SQLite-backed study so subsequent calls extend it.",
    )
    parser.add_argument(
        "--ai-samples",
        type=int,
        default=5,
        help="test_convex_decomp data-collection mode: number of independent AI runs per tool. Default 5. Validator-side, duplicate picks (|dot|>0.99) share a verdict so you only judge each unique answer once per tool.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="directory of per-assembly outputs to aggregate. Used by collect_tool_data (human_labels/ai_labels JSON) and collect_tool_axes (tool_axes_*.csv). For data_assembly_time, points at a previous run's outputs (assembly_time_summary.json or per-run timing_overview.json) to re-plot from existing timing instead of re-running.",
    )
    parser.add_argument(
        "--all-axes",
        action="store_true",
        help="test_convex_decomp data-collection mode: show all THREE principal axes per convex part instead of just the minor axis. Default off (minor axis only — matches the pipeline-wide assumption that the minor PCA axis is the application direction).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=80,
        help="number of worker processes for the sequence planner. "
        "Defaults to None (planner code falls back to its own default). "
        "Used by test_pipeline_batch when scheduling parallel "
        "assemblies — children get the batch's per-assembly slice.",
    )
    parser.add_argument(
        "--user",
        action="store_true",
        help="interactively pick tool/part axes at the start of test_pipeline instead of using the AI selector",
    )
    parser.add_argument(
        "--use-previous-sdf",
        action="store_true",
        help="whether to use previously saved SDFs for faster planning",
    )
    parser.add_argument(
        "--allow-gap",
        action="store_true",
        help="allow small gaps between parts in stability check (useful for IKEA-style assemblies)",
    )
    parser.add_argument(
        "--connect-path",
        action="store_true",
        help="extend disassembly paths to place each part on the ground next to the assembly",
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        default=None,
        help="override the storage/cache directory for the assembly (used by test_render)",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="maximum number of parts in included assemblies (only for --id ranges)",
    )
    parser.add_argument(
        "--min-parts",
        type=int,
        default=None,
        help="minimum number of parts in included assemblies (only for --id ranges)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=6000,
        help="maximum evaluation (feasibility check) budget for sequence planning",
    )
    parser.add_argument(
        "--max-grippers",
        "--max-gripper",
        dest="max_gripper",
        type=int,
        default=3,
        help=(
            "max number of grippers (i.e. parts that may be held fixed during a "
            "single disassembly step). Forwarded to the ASAP planner via "
            "sequence_planner.get_assembly_plans_ASAP -> seq_plan(max_gripper=...). "
            "Higher = more flexibility per step but more search per node."
        ),
    )
    parser.add_argument(
        "--tool-check",
        default=False,
        action="store_true",
        help=(
            "pre-compute per-part tool decisions (static VLM call) and tool/part axes "
            "before the sequence planner runs. Decisions cache to storage_dir/tool_decisions.json "
            "so sequence finding can consume them without per-step VLM calls."
        ),
    )
    parser.add_argument(
        "--seq-tool-check",
        default=False,
        action="store_true",
        help=(
            "enable geometric tool feasibility inside the sequence planner: for every "
            "candidate disassembly step that passes assembly + stability, also require "
            "at least one tool from the scaled tool set to be applicable. No VLM/LLM "
            "calls. Failures show up as 'tool fail' (orange) in the DFA tree plot."
        ),
    )
    parser.add_argument(
        "--seq-optimizer",
        default=None,
        help=(
            "Choose 'divide' to enable the divide optimizer, which splits the assembly into subassemblies."
        ),
    )
    parser.add_argument(
        "--plan-arm",
        dest="plan_arm",
        default=False,
        action="store_true",
        help=(
            "Enable robot-arm planning. During sequence search, candidate disassembly "
            "steps must additionally satisfy IK + arm-vs-{ground,parts,gripper,self} "
            "collision checks via GraspArmPlanner. At render time, each step is "
            "extended with reach (rest -> grasp) and retreat (grasp -> rest) phases "
            "planned by RRT-Connect in joint space (ArmMotionPlanner). Forwards "
            "show_arm=True to play_logged_plan so the rendered GIFs include the arm."
        ),
    )
    parser.add_argument(
        "--gripper-type",
        dest="gripper_type",
        default="rod",
        choices=["panda", "robotiq-85", "robotiq-140", "rod"],
        help=(
            "Contact / gripper model used by the arm planner + renderer. "
            "Default 'rod' is a simple cylinder with a contact point at "
            "the tip (no fingers, no width constraint) — much more "
            "permissive feasibility-wise and approximates tools like "
            "screwdrivers. Settings.contact_model = 'rod' also forces "
            "this regardless of the CLI choice."
        ),
    )
    parser.add_argument(
        "--gripper-scale",
        dest="gripper_scale",
        type=float,
        default=0.4,
        help="Uniform scale applied to gripper + arm meshes (and the IK chain).",
    )
    # endregion
    args = parser.parse_args(sys.argv[1:])

    test_eval = Eval(
        output_dir=output_folder,
        verbose=args.verbose,
        cache=args.cache,
        token_limit=args.token_limit,
    )
    assembly_dir = os.path.join("assets", args.dir)
    _storage_dir = args.storage_dir or None
    if args.id is not None:
        for _id in resolve_ids(
            args.id, assembly_dir, max_parts=args.max_parts, min_parts=args.min_parts
        ):
            test_eval.add_assembly(dir=assembly_dir, id=_id, storage_dir=_storage_dir)
    print(
        f"Initialized evaluation with {len(test_eval.assemblies)} assemblies and {len(test_eval.tools)} {'tool' if len(test_eval.tools) == 1 else 'tools'} available for feedback generation."
    )

    _timings = None
    try:
        _timings = DISPATCH[args.test_type](
            args, test_eval, output_folder, assembly_dir
        )
    finally:
        for ass in test_eval.assemblies:
            ass.create_doc()
        print(
            f"\n\n--- Tokens Used: {test_eval.tokens_used} / {test_eval.token_limit} "
            f"(skipped {test_eval.skipped_llm_calls} LLM calls) ---"
        )
        if _timings:
            print("\n--- Pipeline Timing Summary ---")
            for _step, _elapsed in _timings.items():
                print(f"  {_step:<20}: {_elapsed:6.1f}s")
            print(f"  {'Total':<20}: {sum(_timings.values()):6.1f}s")

        # All argparse attributes - values explicitly passed AND values left at
        # default. vars(args) returns the underlying namespace dict, so any arg
        # added to the parser shows up here without further bookkeeping.
        _cli_args: dict[str, str | int | float | bool | list | None] = {}
        for _k, _v in vars(args).items():
            if isinstance(_v, (str, int, float, bool, type(None))):
                _cli_args[_k] = _v
            elif isinstance(_v, (list, tuple)):
                _cli_args[_k] = list(_v)
            else:
                _cli_args[_k] = str(_v)

        stats = {
            "tokens_used": test_eval.tokens_used,
            "token_limit": test_eval.token_limit,
            "skipped_llm_calls": test_eval.skipped_llm_calls,
            "timings": _timings or {},
            "total_time": sum(_timings.values()) if _timings else 0.0,
            "cli_args": _cli_args,
        }
        _save_dirs = {Path(output_folder).resolve()}
        for ass in test_eval.assemblies:
            _save_dirs.add(Path(ass.storage_dir).resolve())
        for _dir in _save_dirs:
            _dir.mkdir(parents=True, exist_ok=True)
            shutil.copy("settings.py", _dir / "settings.py")
            with open(_dir / "stats.json", "w") as _f:
                json.dump(stats, _f, indent=2)
