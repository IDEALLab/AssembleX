# ============================================================================
# Models & LLM/VLM
# ============================================================================
LLM_model = "gpt-5.4"
VLM_model = "gpt-image-1.5"

# Manual generation backend used by Feedback.make_manual(step_idx).
# Options:
#   "offline"   — generate_manual_offline: no VLM/LLM calls; rotation axis
#                 derived from step.pose; instruction taken from the
#                 self.instructions["Steps"] cache.
#   "geometric" — generate_manual_iterative: silhouette/crease SVG + iterative
#                 LLM annotations (the generative backend).
manual_method = "offline"

# Offline-manual base render: when True, intermediate frames from step.matrices
# (between the assembled position and the fully-disassembled position) are
# drawn in purple at low opacity to visualise the insertion path.
manual_show_path_trail = True
# Maximum number of intermediate ghost copies (excluding the blue start and the
# red end). The available frames are uniformly subsampled to this count.
manual_path_trail_max = 7

# Offline-manual base render: decide per step whether the disassembled (initial)
# position and the path trail are worth drawing at all. Four flat-shaded probe
# renders taken from one camera measure how much of the moving part is still
# visible once the rest of the step's assembly is drawn (see
# ManualGenerator._visible_fraction). Below the threshold the part disappears
# into the assembly and the page keeps the red ghost plus the purple path; above
# it the assembled position reads on its own and only that is drawn. Set False
# to draw both positions on every step, as before.
manual_auto_initial_position = True
manual_visibility_threshold = 0.60
# Parts whose unoccluded silhouette covers fewer pixels than this are too small
# for the ratio to mean anything, so they always keep the initial position.
manual_visibility_min_pixels = 200


# ============================================================================
# Pipeline feature toggles
# ============================================================================
angle_ranking = True
part_naming = True
tool_naming = False
tool_assemblability = False

# When True, run DFA / step-level feedback (and leftover-collision feedback)
# even on assemblies that planned successfully. False by default — successful
# runs skip feedback entirely; failure feedback on non-assemblable runs is
# unaffected by this switch.
feedback_on_success = False

# When True and static collisions are detected at pipeline start, attempt to
# nudge each colliding part with the geometric collision resolver before
# planning. When False (default) the collisions are left as-is and surfaced
# to the failure-feedback generator instead. Either way, the user is prompted
# (Press Enter to continue / Ctrl+C to abort) once all collisions have been
# evaluated.
resolve_collisions = False


# ============================================================================
# Sequence planning (ASAPx)
# ============================================================================

# Whether to render the planned disassembly sequence (play_logged_plan: iso1/iso2
# GIFs + path matrices). Set False to skip the post-plan render step entirely.
render_sequence = True

# Number of frames to save per disassembly path step (passed as n_frame to
# save_path_all_objects).
n_save_state = 5

# Number of successful steps required for early termination; use None to
# disable early termination based on success count.
n_success_term = None

# Whether to probe per-edge DoF during planning (needed by the heuristic cost
# function's `free_dof` feature).
get_dof = True

# Skip the multi-part loose-stability check in ASAPx feasibility checks; treats
# every subassembly as if no extra grippers are needed.
skip_stability = False

# Parallel DFA expands up to this many parent nodes per iteration; 1 reproduces
# single-parent DFS.
max_frontier = 4

# Behavior when the pre-flight check finds no self-supporting (parts_fix=[])
# pose for the full assembly:
#   'skip'            — don't run the precheck at all; start with poses=[]
#                       (saves the up-front physics sim)
#   'exit'            — abort sequence planning immediately,
#                       stop_msg='no self-stable initial pose'
#   'continue'        — warn and continue with poses=[] (legacy behavior)
#   'ignore_unstable' — warn, continue with poses=[], and globally ignore the
#                       parts observed falling during the precheck for every
#                       subsequent stability check
no_stable_pose_action = "exit"

# Tolerance applied BEFORE no_stable_pose_action fires. When the precheck finds
# no fully self-supporting pose, accept the best candidate pose that needs at
# most this many parts held (i.e. at most this many parts fall under gravity),
# and treat those parts as held for every subsequent stability check -- the
# operator/robot is assumed to steady them. The pose with the fewest falling
# parts wins; ties break by trimesh probability order. 0 restores the strict
# behavior (any falling part rejects the pose). Most real assemblies in
# data/asap fail the strict check on only one or two parts, so a small budget
# here recovers them without weakening stability checking elsewhere.
max_initial_held_parts = 2

# Per-assembly wall-clock ceiling for data_sequence_runtime, in seconds.
# The parallel search can deadlock on pathological assemblies (parent blocked
# in queue.get() while workers sit idle); without a ceiling a single bad
# assembly stalls a multi-hour batch indefinitely. The assembly is recorded as
# a timeout and the batch moves on. 0 disables the watchdog.
#
# The ceiling scales with assembly size, because legitimate planning cost grows
# steeply with part count -- a flat timeout that is generous at 20 parts would
# cut off honest work at 30. The budget at `ref_parts` is the value below, and
# it is scaled by (n_parts / ref_parts) ** exponent, with the exponent taken
# from the measured wall_s ~ parts^1.45 fit. Never scaled below the base.
seq_runtime_assembly_timeout_s = 10800
seq_runtime_timeout_ref_parts = 20
seq_runtime_timeout_exponent = 1.45

# When True, the planner pauses on the root node, renders one image per
# candidate initial stable pose, and asks the user to pick which one to use
# as the assembly's starting orientation. Falls back to the default
# (highest-probability first) ordering on Enter, invalid input, or
# non-interactive shells.
interactive_initial_pose = False

# When True, MultiPartStabilityPlanner.check_success replays each gravity
# simulation it ran to a GIF under <log_dir>/precheck_stability/pose_XX.gif
# (one per attempted candidate pose during the initial-stable-pose precheck).
# Also enables the per-edge stability-debug GIFs already wired in
# get_stable_plan_1pose_serial. The render reuses the redmax sim's already-
# populated q_his/qdot_his from the forward() loop — no re-simulation, so the
# overhead is roughly the cost of one GIF encode per attempt.
debug_stability = False

# Whether to filter out candidate disassembly actions that would cause parts to
# pass through the ground.
filter_below_ground = True

# Visualization-only: when True, build_obstruction_graph also records explicit
# non-blocking observations (parts seen present at a node where the target DoF
# is free). These are stored on a sibling matrix graph.nodes[p]['non_blocking']
# and only consulted by the visualization functions; the main `obstruction`
# matrix and the sequence-planning logic that reads it are unaffected.
mark_non_blocking = True


# ============================================================================
# Arm planning
# ============================================================================

# When True (and --plan-arm is on at the CLI), run a dedicated two-stage arm
# planner over the full disassembly sequence BEFORE rendering. Each step
# picks its own world-frame arm base from the 4-candidate circle around the
# step's assembly centroid (same selection as the original GraspArmPlanner).
# Stage 1 plans the in-grasp motion at the chosen per-step base. Stage 2
# plans INTER-STEP transitions at the destination step's base — joint
# config flows from step k's end directly to step k+1's start at base_{k+1};
# the base teleports freely between steps (no joint cost). Three transition
# kinds: prefix (rest -> step_0.start), inter (step_k.end -> step_{k+1}.start),
# suffix (step_{N-1}.end -> rest). The transit arm_path lengths are the
# "cost of moving between steps" used by downstream time-estimation logic.
# Result is persisted to <log_dir>/arm_plans.json and the render workers play
# it back. When False, the renderer falls back to the per-step lazy
# GraspArmPlanner path.
arm_continuous = True

# When True, run the arm pipeline in "simplified" mode:
#   - Stage 2 (RRT-Connect inter-step transitions) is skipped entirely
#     (transitions=[]).
#   - Stage 1 (in-grasp motion planning) is replaced by a closed-form cost
#     k_dist · d · (1 + k_vol · V), where
#         d = cartesian length of the disassembly path,
#         V = volume of the part being removed,
#       and k_dist / k_vol are tuned via arm_simplified_k_dist /
#       arm_simplified_k_vol.
#   - Rod-grasp feasibility is still verified along the disassembly path
#     (cheap — no IK, no RRT). Infeasible steps still get a duration; their
#     cost is multiplied by `failed_step_time_multiplier` instead of being
#     imputed from the rest of the sequence's median.
# Renderer falls back to part-only GIFs (no arm overlay) for these runs.
# Use this mode to benchmark generator/planner choices without paying the
# arm-planner's wall-clock cost or noise.
arm_simplified_mode = True

# Distance coefficient in the simplified-mode cost formula
#   duration_s = k_dist · d · (1 + k_vol · V).
# Units: seconds per assembly coordinate unit. Tune so that typical
# disassembly times come out in a reasonable range for the assemblies you're
# benchmarking on.
arm_simplified_k_dist = 1.0

# Volume coefficient in the simplified-mode cost formula. Multiplies the
# part volume V to introduce a small bias against extracting large parts
# (which take longer for an operator to manipulate). Keep small by default
# so distance still dominates the cost.
arm_simplified_k_vol = 0.01

# When True, simplified mode still runs the rod-grasp feasibility check
# (cheap — no IK, no RRT) and multiplies the closed-form cost by
# `failed_step_time_multiplier` for steps where no rod contact survives the
# disassembly path. Set False to skip the check entirely: every step is
# treated as feasible and gets the bare k_dist · d · (1 + k_vol · V) cost.
arm_simplified_check_grasp = True

# Contact model used by the grasp/arm planner.
#   'rod'     — replaces the two-finger gripper with a simple cylindrical rod.
#               A grasp is "feasible" iff the rod tip touches the part surface
#               (along the local outward normal) without the rod body
#               colliding with the still parts or the ground. The contact is
#               then "glued" to the part for the arm's in-grasp motion.
#               Approximates tools like screwdrivers / pokers / magnetic
#               pickers and removes gripper width / finger-collision as
#               failure modes. Default and recommended for arm-planning work.
#   'gripper' — original two-finger antipodal grasp planner. Use when you
#               need realistic pinch-grasp modeling (e.g. lifting an
#               unattached block). Slower and far more finicky.
# Setting this to 'rod' overrides the CLI --gripper-type to 'rod'; setting
# it to 'gripper' lets the CLI choice apply.
contact_model = "rod"

# Wall-clock time budget per step for the arm pipeline's stage 1 — in-grasp
# motion planning via GraspArmPlanner. Once exceeded, the step's per-grasp
# loop bails out without trying more candidates and the step is marked
# infeasible (fail_reason='timeout'). None disables the timeout.
grasp_planner_timeout_s = 300.0

# Wall-clock time budget per transition for the arm pipeline's stage 2 —
# RRT-Connect transit planning via ArmMotionPlanner.plan_with_grasp. Routed
# into the underlying rrt_connect / smooth_path functions' `max_time`
# parameters. None disables.
arm_planner_timeout_s = 600.0


# ============================================================================
# Time-estimate model
#
# All durations in arm_plans.json / timing_overview.json are derived from the
# velocities below. Pure post-hoc — does not influence planning or rendering.
# ============================================================================

# Constant joint-space angular velocity (rad/s) used by arm_pipeline to convert
# planned arm path lengths into time estimates. Each step's and each
# transition's duration is computed as
#     duration_s = sum_i shortest_angular_distance(q_i, q_{i+1}) / velocity
# where the sum is over consecutive waypoints in arm_path_full (active 7 joints
# only, matching the xarm7 hardcoding in interpolate_q). Continuous joints
# (indices 0,2,4,6) use shortest mod-2π distance; bounded joints use raw |Δq|.
# Result is written into arm_plans.json (per-step + per-transition `duration_s`
# fields and a top-level `time_estimate` aggregate).
arm_joint_velocity_rad_s = 0.4

# Linear walking / focus-shift speed (assembly coordinate units per second)
# used by the timing model's `base_travel_s` component. For each consecutive
# step pair (k-1, k), arm_pipeline computes the arc length the operator's
# focus would sweep along the circle of average radius around the assembly
# center (r_avg · |Δθ| between the two parts' centroids) and divides by this
# velocity. Pure geometry — defined regardless of arm-planner success.
arm_base_travel_velocity = 2.0

# Angular speed (rad/s) at which the assembly itself is rotated between two
# steps. The robot planner ignores assembly reorientation, so this is a pure
# penalty: rotation_angle(pose_{k-1}, pose_k) / velocity. Slower than the arm
# since this is a manual / human-mediated rotation.
assembly_reorientation_velocity_rad_s = 0.15

# Time penalty (seconds) added per EXTRA part that must be held during a step
# (beyond the moving gripper itself), summed across all steps in the sequence.
# Per-step hold_s = len(parts_fix) * time_per_held_part_s, written into
# timing_overview.json as a separate component and added to totals.total_s.
# Use 0 to disable the penalty.
time_per_held_part_s = 2.0

# Multiplier applied to the median successful-step time when a step (or a
# stage-2 transition) failed to plan. Models "this step was too hard, costs
# more than average".
failed_step_time_multiplier = 1.5


# ============================================================================
# Physics simulation
# ============================================================================

# Presets for the physics-sim params swept by test_param_sweep (see
# main._setup_param_sweep_imports). KN and DAMPING live in
# ASAPx/plan_sequence/sim_string.py; COL_TH_STABLE lives in
# ASAPx/plan_sequence/physics_planner.py.
#   'original' — values from the upstream archive/ASAP codebase
#   'new'      — current ASAPx module defaults
sim_param_presets = {
    "original": {"KN": 1e3, "DAMPING": 1e3, "COL_TH_STABLE": 0.00},
    "new": {"KN": 1e5, "DAMPING": 1e4, "COL_TH_STABLE": 0.002},
}
sim_param_preset = "new"


# ============================================================================
# Planner-specific configs
# ============================================================================

# Config for the 'llm' frontier-selection planner (LLMDFASequencePlanner).
llm_planner = {
    "l": 4,  # number of random candidates shown to the LLM per step
    "render_size": (
        512,
        512,
    ),  # pyvista off-screen PNG size for each candidate / history image
    "cache_dir": "llm_cache",  # subdir under the planner log_dir for cached renders + LLM responses
    "model": None,  # overrides LLM_model when set; otherwise falls back to LLM_model
}

# Config for the 'comparison' planner (ComparisonDFASequencePlanner).
comparison_planner = {
    # Constituent selectors invoked at every step. Supported names:
    #   'heuristic', 'first' (or 'dfa'), 'random', 'llm'
    #   'gen:<name>'  — wrap a candidate-part generator as a selector
    #                   (gen:heur-out, gen:heur-vol, gen:learn, gen:rand, gen:dfa)
    "planners": ["heuristic", "random", "gen:heur-out"],
    "l": None,  # optional sub-sample size; None = use all feasible children
    "use_vlm_meta": True,  # if True, also ask a VLM to choose among the constituents' picks
    "use_vlm_for_progress": True,  # if True (and use_vlm_meta=True), the VLM meta-pick drives
    # forward exploration instead of the uniform-random pick.
    # Falls back to random whenever the VLM didn't return a valid
    # pick (e.g. <2 distinct constituents responded, API failure).
    "vlm_model": None,  # overrides LLM_model for the VLM meta-selector and 'llm' constituent
    "render_size": (512, 512),
    "cache_dir": "comparison_cache",  # subdir under log_dir for cached renders + LLM responses
}

# Config for the 'preference' planner (PreferenceLearningDFASequencePlanner).
# Online PA-perceptron learning of heuristic_weights from interactive human
# picks. Requires settings.max_frontier = 1. Learns over the 3 features in
# FEATURE_ORDER (contact_distance, free_dof, z_alignment, pose_change,
# hold_count).
preference_planner = {
    "epsilon": 0.5,  # active-query threshold: query only when the cost gap between
    # the cheapest two candidates is below this (model is uncertain)
    "l": 10,  # max candidates rendered/scored per step (random subset if more)
    "render_size": (512, 512),  # pyvista off-screen PNG size for each candidate image
    "cache_dir": "preference_cache",  # subdir under log_dir for cached candidate renders
    "weights_path": "assets/preference_weights.json",  # stable cross-run learned-weight store
    "log_path": "assets/preference_decisions.jsonl",  # PA-update audit log (append-only)
}


# ============================================================================
# Divide optimizer
# ============================================================================

# Weights for DivideOptimizer.find_locally_free_subassemblies cut score.
divide_weights = {
    "balance": 0.6,  # rewards even-sized splits min(|S|,|R|)/n_parts
    "contact": 0.7,  # penalises edges crossing the cut, normalised by total contact edges
    "fragmentation": 1.0,  # penalises cuts that split either side into multiple disconnected pieces
}

# Minimum DivideOptimizer score above which the integrated sequence pipeline
# treats a verified (S, R) split as "acceptable" and reports the resulting
# sub-sequences in debug output. The full sequence is always returned.
divide_split_threshold = 0.1


# ============================================================================
# Optuna-trained heuristic weights
# ============================================================================

# Which weight set HeuristicDFASequencePlanner._load_weights returns:
#   "default" — read from `heuristic_weights` above (the manually-tuned set).
#   "optuna"  — read from the on-disk Optuna-trained file at
#               `heuristic_weights_optuna_path` (set to "default" if missing).
# Stored separately from `heuristic_weights` so default and trained weights
# can be compared head-to-head without overwriting either.
# Inference: keep this at "optuna" and run normal commands — weights are
# frozen (the planner only reads the file).
# Training: launched via `python main.py train_heuristic_weights --id <range>`;
# the training loop writes the file iteratively, then writes the best trial's
# weights at the end of the study.
heuristic_weights_source = "default"
heuristic_weights_optuna_path = "assets/heuristic_weights_optuna.json"


# ============================================================================
# Visualization — assembly mesh colors
# ============================================================================
color_scheme = "distinctipy"  # default, distinctipy, max_contrast
brightness = 0.5
alpha = 0.4
emphasize_moving = (
    True  # whether to visually emphasize moving parts in sim visualizations
)
bleak = (
    0.85  # how strongly non-moving parts are pulled toward gray in sim visualizations
)
boost = 1.3  # slight saturation boost for moving parts in sim visualizations
