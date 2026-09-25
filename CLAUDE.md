# AssembleX — Codebase Map

End-to-end evaluator for **multi-part assembly disassembly planning** + automated
generation of human-facing assembly manuals (instructions, GIFs, feedback).

Pipeline at a glance:
```
raw OBJ files
   │
   ▼
core/preprocess.py ── normalised assembly ──► Assembly / Eval (core/assembly.py)
                                            │
                                            ▼
                              SequencePlanner (core/sequence_planner.py)
                              ├── ATA backend  (ATA/examples/…)
                              └── ASAPx backend (ASAPx/plan_sequence/run_seq_plan.py)
                                            │
                                  tree.pkl + stats.json
                                            │
                                            ▼
                              renderer (play_logged_plan)  ── GIFs / per-step paths
                              feedback_generator.Feedback  ── manuals, instructions
                              tool_analyzer.ToolAnalyzer   ── tool decisions / poses
```

## Top-level entry point: `main.py`

`main.py` is a thin dispatcher: it parses the CLI, builds the shared `Eval`,
and routes the `test_type` positional arg to a handler via the `DISPATCH`
dict (the argparse `choices` are derived from `DISPATCH`, so they can't drift).
Each handler has the uniform signature
`run_<test_type>(args, test_eval, output_folder, assembly_dir)`; only
`run_test_pipeline` returns a `_timings` dict (consumed by the timing summary
in `main.py`'s `finally`). The handlers live in three sibling modules, grouped
by purpose:

- **[run_pipeline.py](run_pipeline.py)** — the product pipeline.
  - `test_pipeline` / `test_pipeline_batch` — full end-to-end on one or many
    assemblies (batch fans out parallel `test_pipeline` subprocesses).
  - `test_render` — re-render an already-planned assembly from its `tree.pkl` +
    `stats.json`.
- **[run_data.py](run_data.py)** — research/data-generation runs.
  - `data_assembly_time` — multi-generator timing benchmark. For each ID, plans
    + renders + arm-pipelines under every entry in `RUNS` and emits per-assembly
    stacked-bar PNGs + a cross-assembly mean chart. Includes `heuristic_trained`
    (reads the Optuna weights) and `heuristic+optimizer` (reuses the heuristic
    tree, applies the divide split, re-plans each subassembly standalone in
    parallel).
  - `train_heuristic_weights` — Optuna study that tunes
    `HeuristicDFASequencePlanner` weights against arm-pipeline `total_s`. Writes
    `assets/heuristic_weights_optuna.json` + history.
  - `data_sequence_runtime` — sequence-finder **compute**-time benchmark
    (as opposed to `data_assembly_time`, which measures predicted *robot*
    time). Plans each assembly once with the heuristic planner under the
    Optuna-trained weights, divide optimizer off, `--plan-arm` forced on and
    rendering off — the same conditions as `assets/results/timing_final` —
    into a fresh per-assembly dir so every run is a cold plan. Records
    wall-clock plus the planner's own timing buckets and emits
    `parts_vs_runtime`, `parts_vs_runtime_breakdown`, `runtime_per_assembly`
    (PNG + PDF) and `sequence_runtime_summary.{json,txt}`. `--data-dir`
    pointing at an earlier run's summary re-plots without re-planning.
    With `--balance-parts N` it searches for N *feasible* assemblies per part
    count (drawing further candidates from the pool whenever one fails) rather
    than testing a fixed N, and only completed runs reach the charts -- an
    aborted plan's wall-clock measures how fast it gave up, not search cost.
  - `data_heuristic_validation`, `data_manual_validation`, `data_validate_cost`
    — offline eval batches.
  - `data_filter_assemblies` — interactive triage that displays an iso render +
    collision graph per assembly and prompts y/N to copy into a "filtered" pool.
    `--allow-gap` swaps the strict trimesh CollisionManager for ASAPx's
    tolerance-aware `get_contact_graph`.
  - `test_convex_decomp` — axis-picker accuracy data-collection (runs the AI
    picker N times per tool, human validates, emits per-tool accuracy plot).
    Named `test_` for historical reasons; it is a data run.
- **[run_debug.py](run_debug.py)** — focused diagnostics / experiments, not part
  of the pipeline or the data outputs.
  - `test_divide_optimizer` — run the DivideOptimizer (subassembly partitioner)
    on a saved tree, verify the top splits, then compare a split-based sequence
    cost against the flat sequence (via `compare.compare_with_split`).
  - `test_param_sweep` — grid sweep over physics params (KN, DAMPING,
    COL_TH_STABLE), re-running planning per cell.
  - `test_collision`, `test_PCA`, `test_tools`, `test_tool_naming`,
    `test_gravity`, `test_collision_resolver`, `test_collision_graph_batch`,
    `test_archive_ASAP` — visual checks and baseline runs.

Shared CLI helpers (`resolve_ids`, `create_output_directory`, the batch-summary
writers) live in **[run_common.py](run_common.py)**. `resolve_ids` also does
the size-based selection for range IDs: `--min-parts` / `--max-parts` bound the
part count, `--balance-parts N` then keeps at most N assemblies per distinct
part count (equal representation per size), and `--sort-by-parts` (implied by
`--balance-parts`) orders the batch by ascending part count instead of by ID.
`candidates_by_part_count` exposes the *whole* pool grouped by size, for
callers that need to keep drawing replacements until N assemblies succeed.

Every subcommand resolves the assembly IDs via `resolve_ids(args.id, dir)`
(supports `"00010-00050"` range strings) and instantiates a shared `Eval`
that tracks an LLM token budget across all assemblies.

`--dir` defaults to `data` (resolved under `assets/`), and an omitted `--id`
falls back to `DEFAULT_ASSEMBLY_ID = "04489"`, the one assembly tracked in the
repository (`assets/data/04489/`, whitelisted in `.gitignore` against the
blanket `assets/*` rule; its `.sdf` caches are not tracked and regenerate on
demand). `wants_default_assembly` gates that fallback: it is skipped for
`NO_ASSEMBLY_TESTS` (`collect_tool_data` / `collect_tool_axes`, which read only
`--data-dir`) and whenever `--data-dir` is given, because `data_assembly_time`
and `data_sequence_runtime` re-plot from it and return before touching
`test_eval.assemblies`. If the id is missing from the resolved dir the run
continues with no assemblies, as before, after printing why.

## Core abstractions

| Class | File | Role |
|---|---|---|
| `Eval` | [core/assembly.py](core/assembly.py) | Owns a batch of `Assembly` instances, the LLM token budget, the loaded tool catalog, and global cache mode. |
| `Assembly` | [core/assembly.py](core/assembly.py) | One physical assembly. Holds `objects` (id → `Object`), the `Simulation`, the per-assembly `storage_dir`, and exposes `planner` / `feedback`. |
| `Object` | [core/models.py](core/models.py) | One part. Mesh, pose, name, color. |
| `Step` | [core/models.py](core/models.py) | One disassembly step in the final sequence — moving part + pose + path + tool decision. |
| `SequencePlanner` | [core/sequence_planner.py](core/sequence_planner.py) | Wraps ATA or ASAPx, calls the chosen backend, persists `log/tree.pkl` + `log/stats.json`, then triggers rendering. |
| `Simulation` | [core/simulation.py](core/simulation.py) | pyvista/redmax surface for collision, gravity, contact-tree extraction. |
| `Feedback` | [core/feedback_generator.py](core/feedback_generator.py) | All LLM/VLM-driven manual generation: per-step instructions, manual SVG/PNG variants, summaries. |
| `ToolAnalyzer` | [core/tool_analyzer.py](core/tool_analyzer.py) | Per-part tool decisions and geometric placement (which screwdriver/wrench, oriented how). |

## The two planner backends

Both produce a **DiGraph tree**: nodes are tuples of remaining part-ids
(root = all parts, leaves = single part); each edge carries a `sim_info`
dict (`feasible`, `part_move`, `pose`, `parts_fix`, `action`, `grasp`,
`dof`, `base_part`).

### ATA — [ATA/](ATA/)
Older C++-backed pipeline. Entry: `ATA/examples/run_multi_plan.py` (called
out-of-process by `SequencePlanner.read_multi_plan`).

### ASAPx — [ASAPx/](ASAPx/)
Active backend. Entry: `ASAPx.plan_sequence.run_seq_plan.seq_plan`
([ASAPx/plan_sequence/run_seq_plan.py](ASAPx/plan_sequence/run_seq_plan.py)).
Composed of three plug-in registries (each is a dict in their package
`__init__.py`):

- **Generators** ([ASAPx/plan_sequence/generator/](ASAPx/plan_sequence/generator/))
  — propose candidate parts to remove. Implementations: `rand`, `heur-vol`,
  `heur-out`, `learn`, `dfa`.
- **Planners** ([ASAPx/plan_sequence/planner/](ASAPx/plan_sequence/planner/))
  — node-selection strategy over the tree. Implementations: `dfs`, `beam`,
  `randseq`, `dfa`, plus the experimental `heuristic`/`llm`/`comparison`/`preference`
  variants configured via `settings.py`.
- **Optimizers** ([ASAPx/plan_sequence/optimizer/](ASAPx/plan_sequence/optimizer/))
  — pick a final sequence from a completed tree.
  - `BaseSequenceOptimizer` — random valid root-to-leaf; also exposes
    `optimize_scored(cost_fn=..., divide_optimizer=...)` which picks the
    minimum-cost valid sequence and surfaces the divide-optimizer's split
    for diagnostic logging.
  - `DivideOptimizer` ([ASAPx/plan_sequence/optimizer/divide.py](ASAPx/plan_sequence/optimizer/divide.py))
    — builds a per-part obstruction graph from DoF traces, runs a DFS over
    canonical `frozenset({S, R})` partitions, and **propagates** every
    initial split forward along a representative disassembly sequence so
    diminished `(S', R')` candidates from later steps also get scored.
    Cuts are scored by `settings.divide_weights = {balance, contact, fragmentation}`.
    `verify_locally_free(top_k, num_proc)` then physically checks the top-k
    by fusing each side into a unified rigid body (`verify_separation`) and
    scanning the 6 world-axis directions. The top verified split is persisted
    into `stats['divide_split']` for the renderer.
  - `split_plan.py` — the **recursive** subassembly plan
    (`prefix -> unified split -> S -> R`, where S and R are planned the same
    way). `build_split_plan` reuses the DivideOptimizer's obstruction graph,
    searching each block under a `restrict_parts` universe, so no re-planning
    is involved. `split_order_constraint` turns a plan into a predicate over
    sequences, `derive_split_sequence` builds the plan's own order, and
    `flatten_split_plan` emits the ordered step list including the `join`
    entries. See "Subassembly plan" below.
  - `compare.py` — `compare_with_split(tree, asset_folder, assembly_dir, split, ...)`
    builds a "split sequence" from a chosen `(S, R)` (prefix taken from the
    original sequence's parts not in `S∪R`, then a unified-split step, then
    independently re-planned S and R sub-sequences in parallel via raw
    `mp.Process` workers using fork context) and returns a fully decomposed
    cost breakdown. Used by `test_divide_optimizer` and the
    `heuristic+optimizer` run of `data_assembly_time`.
  - `weight_trainer.py` — Optuna-backed black-box optimisation of
    `HeuristicDFASequencePlanner` weights against arm-pipeline `total_s`.
    See "Heuristic-weight training" below.
  - `plot_weight_history.py` — diagnostic plot for the training history JSON
    (convergence curve + per-weight sensitivity scatter + per-assembly
    trajectory). Run as `python ASAPx/plan_sequence/optimizer/plot_weight_history.py`.

`seq_plan(...)` runs the chosen generator+planner to build the tree, then
(when `seq_optimizer='divide'`) calls `BaseSequenceOptimizer.optimize_scored`,
which picks the minimum-cost valid sequence; the divide optimizer's split is
passed in for the debug diagnostic only and does **not** steer that choice.
The top verified split is persisted into `stats['divide_split']`, and
`_build_subassembly_plan` then builds the recursive plan on top of it.

### Heuristic cost function — [ASAPx/plan_sequence/planner/heuristic.py](ASAPx/plan_sequence/planner/heuristic.py)
`HeuristicDFASequencePlanner._cost_child` computes
`cost = w · phi` over five features (in `FEATURE_ORDER`):
`contact_distance` (`ln(d+1)` of hops to nearest already-removed part),
`free_dof`, `z_alignment`, `pose_change`, `hold_count`.
Weights are loaded by `_load_weights()` which branches on
`settings.heuristic_weights_source`:
- `"default"` — read `settings.heuristic_weights` (the manually-tuned set).
- `"optuna"` — read `settings.heuristic_weights_optuna_path` (default
  `assets/heuristic_weights_optuna.json`). Falls back to defaults with a
  `WARN` if missing/corrupt, so flipping the switch without a trained file
  doesn't break planning. **The same standalone feature extractor is
  duplicated in `compare.py:CostComputer` — keep the two in sync when
  editing.**

## Subassembly plan — `prefix -> unified split -> S -> R`

Enabled by `--seq-optimizer divide` plus `settings.subassembly_plan` (default
on). Built by `_build_subassembly_plan` in
[ASAPx/plan_sequence/run_seq_plan.py](ASAPx/plan_sequence/run_seq_plan.py),
which runs **after** the flat sequence has been chosen and `divide_split`
persisted, so a failure anywhere in it leaves the run exactly as it was.

1. `build_split_plan` cuts the assembly into `(prefix, S, R)`, then recurses
   into S and R. Each block's cut comes from
   `DivideOptimizer.find_locally_free_subassemblies(restrict_parts=<block>)` —
   the obstruction graph is built once on the full tree and the search universe
   is narrowed per block, so parts outside it are correctly treated as already
   removed. Cuts are then physically verified with `verify_locally_free`.
   A block's `prefix` is the parts in neither side: the DivideOptimizer's
   propagated ("diminished") cuts often only exist after a few parts come off,
   and those parts become the prefix.

   With `settings.subassembly_sweep_states` (default on) the per-block search
   is `sweep_sequence_states` instead: the DFS re-runs at **every prefix state**
   of the block, not just the whole block. This is what finds a subassembly
   locked inside the block — `_propagate_to_subsequent_steps` cannot, because
   it only shrinks cuts that were already free initially, and a cut blocked at
   the initial state is dropped by the DFS before propagation sees it. Both are
   pooled, so the swept candidate set is a strict superset.

   Two things keep this cheap. Scores use a **common basis** (`score_scope`,
   the block's full part set) so cuts found at different states are comparable
   — without it, balance normalised per state makes a `|S|=3 |R|=1` cut at a
   small state outrank real cuts. And the root block carries `divide_split` in
   as `known_verified`, so only candidates that outrank it reach physics.
   `verify_separation` depends only on `(S, R)`, never on the state a cut was
   found in, so nothing about verification scales with the number of states.

   Measured on a 17-part assembly: the sweep adds ~0.4s of graph work against
   ~12 min of sequence planning, and 313 extra candidate cuts. In practice
   those extra cuts rank *below* the initial-state ones (a cut needing a prefix
   has smaller sides, hence lower balance on the common basis) — first
   sweep-only cut lands at rank 16, outside the default `top_k=10`. So today it
   costs nothing and changes nothing on assemblies whose root already splits
   well; it is there for the ones whose root does not. Raising
   `subassembly_verify_top_k` is the knob that actually brings them into play.
   Depth and block size are bounded by `subassembly_max_depth` /
   `subassembly_min_parts`; a side smaller than `MIN_SIDE_PARTS` (2) is never
   called a subassembly.
2. The plan defines a block **order**: prefix before S∪R, S before R, at every
   level. Two ways to realise it, recorded in `stats['split_sequence_source']`:
   - `'tree'` — `BaseSequenceOptimizer.optimize_constrained` found a real
     root-to-leaf path of the tree that respects the order. `stats['sequence']`
     is replaced with it (the old one is kept as `stats['flat_sequence']`), so
     the renderer, per-step poses and the arm pipeline all follow the same
     order the manual tells.
   - `'derived'` — no explored path respects it, which is the common case on
     larger assemblies: a block ordering is a narrow slice of the orderings a
     budget-limited search visits. The order is then assembled from the plan by
     `derive_split_sequence`, and **`stats['sequence']` is left untouched** —
     the renderer keeps following a valid tree path while only the manual reads
     the split order.
3. Persisted: `stats['split_plan']` (nested block dict),
   `stats['split_steps']` (flattened, including the `join` entries), and
   `stats['split_sequence']`.

A `join` step — R separating from S as one rigid body — is **not a tree edge**,
so it has no `Step` and no per-step GIF. It lives only in `split_steps`, with
the verified world-axis separation direction attached as metadata, and the
manual renders its page directly from the meshes: both halves seated, each in
its side colour, with **no** red pre-assembly ghost. (On a per-part step page
the red copy is one part at its starting position and reads clearly; for a
whole subassembly it reads as a third body instead.) Nothing currently draws
`direction`.

Manual page framing: the hue says which side (`subassembly_colors` — green S,
purple R), the tone says how deep (`subassembly_shade_ladder`, alternating
lighter/darker per nesting level, so two nested S blocks are not the same green
twice). One ring per level, outermost = outermost block. Title text uses the
full-strength hue rather than the nesting tone — thin glyphs in a level-1 tone
fall to about 2:1 contrast on white, and the label already spells the depth out
("Subassembly S-S").

Downstream: `SequencePlanner._apply_split_plan` puts the plan on
`assembly.split_plan` / `assembly.split_steps` and tags every `Step` with its
block path (`Step.subassembly`, e.g. `["S", "R"]`). When a plan is present,
`assembly.sequence` is ordered by `split_sequence` — safe because the per-part
artifacts it reads (GIFs, path matrices) are looked up by `obj_id`, not by
position. `_apply_step_details` matches by part id for the same reason.

Caveat: each step's feasibility was verified with the *other* side still
present, which is strictly more constrained than the split narrative, so
collisions are covered. Sub-assembly **stability** in isolation is not
re-checked — `verify_separation` establishes separability, not that each half
stands on its own.

## Storage / outputs

Each assembly has a `storage_dir` (under `assets/output/<timestamp>/<id>/` by
default, or `--storage-dir` override). Inside it:

```
storage_dir/
├── log/
│   ├── tree.pkl              # the planning DiGraph
│   ├── stats.json            # success, sequence, divide_split, timings, cli_args
│                             # + split_plan / split_steps / split_sequence
│                             #   / split_sequence_source / flat_sequence
│                             #   (subassembly plan; see above)
│                             # + timing_breakdown / timing_counts (planner's
│                             #   per-check buckets; worker CPU-seconds)
│   ├── setup.json            # the planner kwargs
│   ├── arm_plans.json        # arm pipeline (when --plan-arm or arm_continuous)
│   ├── timing_overview.json  # per-step + totals timing breakdown (arm pipeline)
│   └── failures.json         # _dump_failure_evidence payload (on partial plans)
├── paths/                    # per-step recorded motion (npy frames)
├── 0_<obj>.gif, …            # primary-view per-step disassembly GIFs
├── 0_<obj>_opposite.gif      # opposite-view per-step GIFs
├── subassembly/              # divide-optimizer renders, one dir per split block
│   └── root[.S[.R]]/         #   split.gif + S_*/R_* internals per block
├── manual/                   # manual pages; join_<i>_manual_offline.png per join
├── sequence_runtime/         # data_sequence_runtime (in the run's output dir)
├── obstruction_graph.png     # test_divide_optimizer diagnostic
├── subassemblies/            # test_divide_optimizer per-partition screenshots
└── assembly_time/<run>/      # per-RUN cache for data_assembly_time (tree.pkl + stats.json)
```

When `--plan-arm` is on (or `arm_continuous=True` in settings), the arm
pipeline also writes `log/arm_plans.json` (per-step in-grasp motion + inter-step
transitions) and `log/timing_overview.json` (`step_disassembly_s`,
`transitions_s`, `base_travel_s`, `reorientation_s`, `hold_s` per step plus
`totals`). When `arm_simplified_mode=True`, Stage 2 RRT-Connect transitions
are skipped and Stage 1 motion time is approximated by a closed-form cost
(`k_dist · d · (1 + k_vol · V)`); part-only GIFs are rendered (no arm overlay).
This is the cheap mode used for benchmarking generator/planner choices.

Caches:
- `assets/assembly_cache/<id>/` — preprocessed mesh + assembly cache (keyed by
  `--cache read|update|new`).
- `<log_dir>/llm_cache/`, `<log_dir>/comparison_cache/`, `<log_dir>/preference_cache/`
  — per-planner LLM/VLM response + render caches keyed by content hash.
- `<storage_dir>/assembly_time/<run_label>/` — per-RUN cache for the
  `data_assembly_time` benchmark (each run gets its own tree.pkl + stats.json).
- `assets/heuristic_weights_optuna.json` — trained heuristic weights (live
  file during training, frozen best at study end).
- `assets/heuristic_weights_optuna_history.json` — per-trial training log
  `[{trial, weights, mean_total_s, per_assembly_total_s, elapsed_s}, ...]`,
  written incrementally so an aborted study leaves usable data.
- `assets/optuna_training/trial_<NNNN>/<id>/` — per-trial render artifacts
  used during weight training.

## Configuration: `settings.py`

Single source of truth for runtime tuning. Notable keys (all already in
[settings.py](settings.py)):
- `LLM_model`, `VLM_model`, `manual_method` — model + manual-backend selection.
- `render_sequence` — global render-on/off switch (when False, planning still
  runs but `_render_plan` does no GIF/path output).
- `n_save_state`, `get_dof`, `skip_stability`, `max_frontier`,
  `no_stable_pose_action` (`exit`/`skip`/`continue`/`ignore_unstable`),
  `interactive_initial_pose`, `debug_stability`, `mark_non_blocking`,
  `filter_below_ground` — ASAPx planner behaviour.
- `max_initial_held_parts` — budget of parts the initial-pose precheck may
  assume are held. Applied *before* `no_stable_pose_action`: when no fully
  self-supporting pose exists, the candidate pose with the fewest falling
  parts is accepted if it needs at most this many held, and those parts are
  ignored in every later stability check. 0 restores strict behaviour. Both
  `plan()` branches route through `_relax_initial_pose_by_held_parts` on the
  planner base class so the serial and parallel-DFA paths can't drift.
- `heuristic_weights`, `llm_planner`, `comparison_planner`, `preference_planner`
  — per-planner configuration dicts.
- `heuristic_weights_source` (`"default"` or `"optuna"`) +
  `heuristic_weights_optuna_path` — switch between manually-tuned and
  Optuna-trained heuristic weights. Train via `train_heuristic_weights`,
  flip to `"optuna"` for inference.
- `divide_weights = {balance, contact, fragmentation}` — DivideOptimizer cut score.
- `divide_split_threshold` — minimum score for accepting a divide split.
- `arm_continuous`, `arm_simplified_mode`, `arm_simplified_k_dist`,
  `arm_simplified_k_vol`, `failed_step_time_multiplier` — arm-pipeline
  behaviour (see "Arm pipeline" below).

## Common workflows

| Goal | Run |
|---|---|
| Plan + render the shipped assembly end-to-end | `python main.py test_pipeline` (same as `--id 04489 --dir data`) |
| Re-render an already planned assembly | `python main.py test_render --id 00100 --storage-dir <path>` |
| Inspect the DivideOptimizer for one tree | `python main.py test_divide_optimizer --id 00100 --storage-dir <path>` |
| Batch validate (no re-planning) | `python main.py data_manual_validation --id 00000-00200` |
| Sweep physics params for a single id | `python main.py test_param_sweep --id 00100 …` |
| Multi-generator timing benchmark | `python main.py data_assembly_time --id 00100-00110` |
| Sequence-finder runtime vs part count | `python main.py data_sequence_runtime --id 00000-20016 --dir data/asap --min-parts 2 --max-parts 20 --balance-parts 3` |
| Train heuristic weights (Optuna) | `python main.py train_heuristic_weights --id 00100-00120 --optuna-trials 30` |
| Plot training history | `python ASAPx/plan_sequence/optimizer/plot_weight_history.py --history assets/heuristic_weights_optuna_history.json --out assets/optuna_training/history.png` |
| Interactive assembly triage | `python main.py data_filter_assemblies --id 00000-00500 [--allow-gap]` |

## Pipeline glue — `SequencePlanner.get_assembly_plans_ASAP`

(in [core/sequence_planner.py](core/sequence_planner.py)). Important quirks:
- ASAPx and ATA both define top-level packages named `assets`, `utils`,
  `plan_sequence`, etc. Before invoking ASAPx, this method **evicts** those
  modules from `sys.modules` and prepends `ASAPx/` to `sys.path` so ASAPx
  re-imports its own copies. Any code that needs to patch `plan_sequence.*`
  module state (e.g. the `test_param_sweep` `KN/DAMPING/COL_TH_STABLE`
  override, see `_setup_param_sweep_imports` in [main.py](main.py)) must
  hold a direct reference to the patched module object — re-importing later
  hits the evicted copy.
- **`settings` must NOT be in the eviction set.** There is exactly one
  `settings.py` in the repo (the root one); neither ATA nor ASAPx ships its
  own, so evicting it cannot resolve a name clash — it only forces a fresh
  re-read from disk, silently discarding every runtime override the caller
  set before planning (`heuristic_weights_source="optuna"`,
  `render_sequence=False`, `debug_stability=False`, …). Anything that flips a
  setting around a `get_assembly_plans` call depends on this.
- `_render_plan` first runs `play_logged_plan` for the flat per-step disassembly,
  then (when `stats['divide_split']` is present) runs
  `play_subassembly_split` to emit the unified-split clip plus each
  subassembly's internal disassembly into `storage_dir/subassembly/`.

## Rendering — [ASAPx/plan_sequence/play_logged_plan.py](ASAPx/plan_sequence/play_logged_plan.py)

- `play_logged_plan(...)` — per-step disassembly worker (`_render_step_worker`),
  optionally parallel via `num_proc` and `utils.parallel.parallel_execute`.
  Each step builds a `MultiPartPathPlanner`, replays the tree's recorded
  `action` to produce a motion, records a GIF, saves per-frame mesh
  matrices via `save_path_all_objects`.
- `_render_unified_split` — fuses S/R into single OBJs in a temp dir
  (`stable_pose.get_combined_mesh`), runs a two-body
  `MultiPartPathPlanner(parts_fix=['S'], part_move='R')`, scans the 6 axis
  directions, records `split.gif`.
- `_render_subassembly_internal` — for each part removed inside a subassembly,
  plans the motion in the **full** assembly (re-using the global plan) and
  replays it in a **reduced** render-sim containing only that subassembly's
  parts, so the other side is invisible.
- `play_subassembly_split(asset_folder, assembly_dir, split, sequence, tree, result_dir, …)`
  — orchestrates the above three for a `stats['divide_split']`.

## Physics core — [ASAPx/plan_sequence/physics_planner.py](ASAPx/plan_sequence/physics_planner.py)

Built on `redmax_py`. Most-used pieces:
- `MultiPartPathPlanner(asset_folder, assembly_dir, parts_fix, part_move, parts_removed, pose, …)`
  — single-step disassembly planner. `plan_path(action)` re-runs `check_success`
  with collision + min_sep stopping; `compute_dof()` probes the 6 world axes
  for free directions; `render(path=…, record_path=…)` replays the recorded
  motion to a GIF.
- `MultiPartStabilityPlanner` / `MultiPartAdaptiveStabilityPlanner` — gravity
  stability checks used by feasibility checks.
- `get_contact_graph(asset_folder, assembly_dir, parts)` — nx contact graph,
  reused by the DivideOptimizer.
- `verify_separation(asset_folder, assembly_dir, parts_S, parts_R)` — physical
  check that R can be lifted off S as a single rigid body.

## Sim string XML — [ASAPx/plan_sequence/sim_string.py](ASAPx/plan_sequence/sim_string.py)

Each redmax sim is built from a generated XML string. The three entry
points used everywhere are `get_contact_sim_string`, `get_path_sim_string`,
`get_stability_sim_string`. Note: `Simulation::set_body_color_map` takes
RGB only (no runtime alpha); transparency can only be set in the XML at
sim construction.

## Arm pipeline — [ASAPx/plan_robot/arm_pipeline.py](ASAPx/plan_robot/arm_pipeline.py)

Runs after sequence planning when `--plan-arm` is on. Two stages:
1. **Stage 1**: in-grasp motion planning for each disassembly step at a
   per-step base chosen from a 4-point circle around the step's centroid.
2. **Stage 2**: RRT-Connect inter-step transitions (prefix → step0,
   step_k → step_{k+1}, step_{N-1} → rest). Base teleports freely between
   steps; the joint configuration must connect.

Output: `log/arm_plans.json` (motion paths) +
`log/timing_overview.json` (`per_step` + `totals` with five components:
`step_disassembly_s, transitions_s, base_travel_s, reorientation_s, hold_s`).
`total_s` is the canonical objective consumed by `data_assembly_time` and
`train_heuristic_weights`.

`arm_simplified_mode=True` (settings) skips Stage 2 entirely and replaces
Stage 1 motion times with a closed-form
`k_dist · d · (1 + k_vol · V)` — cheap mode for benchmarking that still
verifies rod-grasp feasibility along each path.

## Heuristic-weight training — [ASAPx/plan_sequence/optimizer/weight_trainer.py](ASAPx/plan_sequence/optimizer/weight_trainer.py)

Optuna study (TPE sampler — no smoothness assumption, handles the
discontinuous objective well). Per-trial flow:
1. Sample weights from `DEFAULT_SEARCH_SPACE` (all ≥ 0, upper bounds ≈ 2×
   `DEFAULT_WEIGHTS`).
2. Atomic-write to `assets/heuristic_weights_optuna.json`.
3. For each assembly in `test_eval.assemblies`: redirect `ass.storage_dir`
   to a per-trial subdir, force `args.cache="new"`, run `get_assembly_plans`
   under `settings.heuristic_weights_source = "optuna"` (set by the trainer
   for the study's duration), read back `log/timing_overview.json`.
4. Mean `total_s` across assemblies is the objective; failed assemblies
   contribute `None` so partial batches still produce a meaningful number.
5. Append `{trial, weights, mean_total_s, per_assembly_total_s, elapsed_s}`
   to `assets/heuristic_weights_optuna_history.json` after every trial.
6. On study exit (normal or `KeyboardInterrupt`): write the best trial's
   weights back to the weights file as the frozen final state, restore
   `heuristic_weights_source` to its pre-study value.

**Training vs inference switch**: training is via `train_heuristic_weights`
(temporarily flips the setting). Inference is via setting
`heuristic_weights_source = "optuna"` permanently in `settings.py` —
nothing writes the file outside training, so the weights are frozen.
Compare default vs trained head-to-head by running `data_assembly_time`,
which always includes both `heuristic` (default weights) and
`heuristic_trained` (Optuna weights) as separate RUNS.

## Conventions

- **Don't import ATA and ASAPx in the same process without eviction** — both
  ship top-level `assets`/`utils`/`plan_sequence`. See the eviction loop in
  `_render_plan` and `_setup_param_sweep_imports`.
- **`tree.pkl` + `stats.json` are the canonical interchange format** between
  planning and rendering. Anything new should read/write through them.
- **Per-assembly state belongs on `Assembly.storage_dir`**; everything else
  (caches, logs) underneath it.
- **Settings, not flags**, for global behaviour: see `settings.py` keys.
- **Pin matplotlib to `Agg` before importing `pyplot`.** matplotlib defaults
  to an interactive backend (`qtagg`) in the dev environment, and ASAPx's
  `DFASequencePlanner.plot_tree` calls `plt.subplots()` unconditionally at the
  end of planning — under a GUI backend that blocks on the display server and
  deadlocks the planner. `run_data.py` pins it at import time; the backend is
  process-global, so that covers ASAPx's figures too.
- **No emojis, no decorative docs** in code or markdown (per repo style).
- **Plan/render parallelism uses `utils.parallel.parallel_execute`**, which
  treats `num_proc=1` as in-process and supports `terminate_func` for
  early-exit batching (used by the DFA planner and the verify step).
