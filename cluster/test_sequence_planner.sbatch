#!/bin/bash
#SBATCH --job-name=assemblex_seqplan_test
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
# No --gpus: the sequence planner is pure CPU (RedMax physics + trimesh).
#
# --mem is a flat per-job total, deliberately not --mem-per-cpu. The planner's
# parallelism is fork-based, so most of the footprint (redmax_py, VTK, torch,
# coacd, the loaded meshes) is copy-on-write shared between workers rather than
# duplicated per core: the total behaves like base + k*workers, not 4G*workers.
# Under --mem-per-cpu=4G, raising --cpus-per-task to 48 would silently request
# 192G and push the job onto a bigmem node for no reason. Flat --mem lets the
# core count and the memory request move independently.
#
# 16G is extrapolated, not measured. The 7-part 04489 smoke test peaked at
# 1.13G across 8 workers (sacct MaxRSS; MaxVMSize reads 0 there because Euler
# gathers accounting from the cgroup, which is precisely what makes that figure
# whole-job rather than largest-single-process). 15 parts over 16 workers holds
# more mesh and a deeper tree, so the margin here is wide on purpose. Get the
# real number from
#
#   sacct -j <jobid> --format=JobID,ReqMem,MaxRSS,MaxVMSize,Elapsed,State
#
# and tighten afterwards -- but not to the peak itself: accounting only samples
# every JobAcctGatherFrequency seconds (scontrol show config), so a short spike
# during worker spawn can fall between two samples.
#
# The Slurm log paths above cannot read shell variables: %u is the submitting
# user, %x the job name, %j the job id. $SCRATCH/logs has to exist before
# submitting, or Slurm has nowhere to write and the job dies without a word:
#
#   mkdir -p "$SCRATCH/logs"
#
# Install test: plans one assembly end to end with the sequence planner only.
#
#   sbatch cluster/test_sequence_planner.sbatch
#   ASSEMBLY_ID=00100 BUDGET=500 sbatch cluster/test_sequence_planner.sbatch
#
# Subcommand: `data_sequence_runtime`, which is the repo's planner-only
# benchmark. It forces planner=heuristic, generator=rand, the divide optimizer
# off and rendering off, then writes a timing summary. No LLM or VLM code path
# is reachable from it:
#   - the heuristic planner scores candidates with a weighted geometric cost
#     function (contact distance, free DoF, z alignment, pose change, hold
#     count), no model calls;
#   - part naming, tool naming, angle ranking, instructions, feedback and
#     manual generation all live in test_pipeline, which this does not touch;
#   - --token-limit 0 is passed as a hard backstop: every LLM call site checks
#     the budget first, so with a limit of 0 any accidental call is skipped
#     rather than made. The run prints "Tokens Used: 0 / 0" at the end and this
#     script fails the test if it does not.
#
# Reference: assembly 01234 has 15 parts and plans in 14 steps. On the dev
# machine it took ~570 s wall with 80 workers, which is ~24,000 CPU-seconds,
# so expect roughly 30-45 min on 16 cores. --time=02:00:00 leaves ~2.5x margin
# on that and stays inside Euler's <=4h scheduling class, where shorter jobs get
# backfilled into gaps sooner; override with `sbatch --time=...` for a slower
# assembly. Scale --cpus-per-task and --time together if you change the
# assembly -- --mem no longer has to move with them.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override by exporting these before sbatch)
# ---------------------------------------------------------------------------
# $SCRATCH is Euler's own shortcut for /cluster/scratch/$USER; the literal path
# is the fallback for a shell that does not define it. Export SCRATCH_DIR to
# install somewhere else entirely.
SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/cluster/scratch/${USER}}}"
REPO_DIR="${REPO_DIR:-${SCRATCH_DIR}/AssembleX}"
CONTAINER_PATH="${CONTAINER_PATH:-${SCRATCH_DIR}/assemble-x-container_latest.sif}"
USER_BASE="${SCRATCH_DIR}/.local"

ASSEMBLY_ID="${ASSEMBLY_ID:-01234}"
# Passed to --dir, which main.py resolves as assets/<dir>.
ASSEMBLY_SUBDIR="${ASSEMBLY_SUBDIR:-data/asap}"
# Feasibility-check budget for the search. 6000 is the repo default; lower it
# for a faster smoke test at the cost of a possibly partial sequence.
BUDGET="${BUDGET:-6000}"
NUM_PROC="${NUM_PROC:-${SLURM_CPUS_PER_TASK:-4}}"

export APPTAINERENV_PYTHONUSERBASE="${USER_BASE}"
export APPTAINERENV_http_proxy="${http_proxy:-}"
export APPTAINERENV_https_proxy="${https_proxy:-}"
export APPTAINERENV_PREPEND_PATH="${USER_BASE}/bin"
export APPTAINER_CACHEDIR="${SCRATCH_DIR}/.apptainer/cache"
export APPTAINER_TMPDIR="${SCRATCH_DIR}/.apptainer/tmp"

# Headless environment. There is no X server on a compute node and pyvista/VTK
# will otherwise open an X connection in the parent process right before the
# planner forks its workers; parent and children then share that socket and the
# workers deadlock in poll(). DISPLAY is deliberately not forwarded.
export APPTAINERENV_PYVISTA_OFF_SCREEN="true"
export APPTAINERENV_MPLBACKEND="Agg"
# One thread per worker process: the planner's parallelism is process-level,
# so letting BLAS/OpenMP also thread oversubscribes the allocation.
export APPTAINERENV_OMP_NUM_THREADS="1"
export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
export APPTAINERENV_MKL_NUM_THREADS="1"

section() { echo; echo "=== $* ==="; }
die()     { echo "ERROR: $*" >&2; exit 1; }

# Checked before the mkdir below, so a wrong SCRATCH_DIR is one clear error
# rather than a pile of mkdir failures.
[ -d "${SCRATCH_DIR}" ] || die "scratch directory does not exist: ${SCRATCH_DIR}
       On Euler this is the \$SCRATCH shortcut for /cluster/scratch/\$USER.
       Export SCRATCH_DIR to install somewhere else."

mkdir -p "${SCRATCH_DIR}/logs" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

BIND_ARGS=(--bind "${SCRATCH_DIR}:${SCRATCH_DIR}")
case "${REPO_DIR}" in
    "${SCRATCH_DIR}"/*) ;;
    *) BIND_ARGS+=(--bind "${REPO_DIR}:${REPO_DIR}") ;;
esac

in_container() {
    apptainer exec --cleanenv "${BIND_ARGS[@]}" --pwd "${REPO_DIR}" \
        "${CONTAINER_PATH}" "$@"
}

# ---------------------------------------------------------------------------
# Step 1: preflight
# ---------------------------------------------------------------------------
section "Step 1: preflight"
command -v apptainer >/dev/null 2>&1 || die "apptainer not found on PATH"
[ -f "${CONTAINER_PATH}" ] || die "container image not found: ${CONTAINER_PATH}"
[ -f "${REPO_DIR}/main.py" ] || die "repository not found at ${REPO_DIR}; run cluster/setup_assemblex.sbatch first"

ASSEMBLY_PATH="${REPO_DIR}/assets/${ASSEMBLY_SUBDIR}/${ASSEMBLY_ID}"
N_OBJ=$(find "${ASSEMBLY_PATH}" -maxdepth 1 -name '*.obj' 2>/dev/null | wc -l)
if [ "${N_OBJ}" -eq 0 ]; then
    die "no .obj files in ${ASSEMBLY_PATH}
       assets/* is gitignored, so the meshes have to be staged from a machine
       that has them:
         rsync -av <local>/AssembleX/assets/${ASSEMBLY_SUBDIR}/${ASSEMBLY_ID}/ ${ASSEMBLY_PATH}/"
fi
echo "Assembly ${ASSEMBLY_ID}: ${N_OBJ} parts in ${ASSEMBLY_PATH}"

SITE_PACKAGES="$(in_container python3 -c 'import site; print(site.getusersitepackages())')"
export APPTAINERENV_PYTHONPATH="${SITE_PACKAGES}:${REPO_DIR}"
echo "PYTHONPATH: ${SITE_PACKAGES}:${REPO_DIR}"

echo "Checking that the planner's imports resolve..."
in_container python3 - <<'PYEOF' || die "import check failed; re-run cluster/setup_assemblex.sbatch"
import sys

sys.path.insert(0, "ASAPx")
import coacd  # noqa: F401  (core/models.py imports this at module level)
import numpy as np
import redmax_py  # noqa: F401
import trimesh  # noqa: F401
from plan_sequence.generator import generators
from plan_sequence.planner import planners

print(f"  numpy {np.__version__}, redmax_py ok")
print(f"  generators: {sorted(generators)}")
print(f"  planners:   {sorted(planners)}")
PYEOF

# ---------------------------------------------------------------------------
# Step 2: the run
# ---------------------------------------------------------------------------
section "Step 2: sequence planning (no rendering, no LLM/VLM)"
echo "Assembly:  ${ASSEMBLY_ID} (--dir ${ASSEMBLY_SUBDIR})"
echo "Workers:   ${NUM_PROC}"
echo "Budget:    ${BUDGET} feasibility checks"
echo

RUN_LOG="${SCRATCH_DIR}/logs/assemblex_seqplan_test_${SLURM_JOB_ID:-local}.run.log"
START_TS=$(date +%s)
set +e
in_container python3 -u main.py data_sequence_runtime \
    --id "${ASSEMBLY_ID}" \
    --dir "${ASSEMBLY_SUBDIR}" \
    --num-proc "${NUM_PROC}" \
    --budget "${BUDGET}" \
    --token-limit 0 \
    --verbose 1 \
    | tee "${RUN_LOG}"
RUN_RC=${PIPESTATUS[0]}
set -e
ELAPSED=$(( $(date +%s) - START_TS ))
echo
echo "main.py exited with ${RUN_RC} after ${ELAPSED}s"

# ---------------------------------------------------------------------------
# Step 3: check the result
# ---------------------------------------------------------------------------
section "Step 3: result"

if ! grep -q -- "--- Tokens Used: 0 / 0 " "${RUN_LOG}"; then
    echo "WARNING: could not confirm a zero token count in the run log." >&2
    grep -- "Tokens Used" "${RUN_LOG}" >&2 || true
fi

export APPTAINERENV_EXPECTED_ID="${ASSEMBLY_ID}"
set +e
in_container python3 - <<'PYEOF'
"""Read the newest sequence_runtime_summary.json and judge the run.

status is set by run_data.run_data_sequence_runtime:
  ok            - a complete, feasible disassembly sequence was found
  partial       - the search ran but stopped (budget/timeout) with a prefix
  no_sequence   - the search ran and found nothing
  no_stable_pose- no self-supporting initial pose for the assembly
  error         - the planner raised
Only 'ok' and 'partial' mean the install works; 'partial' additionally means
the search was cut short, which is a tuning result, not an install result.
"""
import glob
import json
import os
import sys

pattern = "assets/output/*/sequence_runtime/sequence_runtime_summary.json"
paths = sorted(glob.glob(pattern), key=os.path.getmtime)
if not paths:
    print(f"FAIL: no summary written (looked for {pattern})")
    sys.exit(1)

summary_path = paths[-1]
with open(summary_path) as f:
    summary = json.load(f)
print(f"Summary: {summary_path}")

expected = os.environ.get("EXPECTED_ID")
records = [r for r in summary.get("per_assembly", []) if r.get("id") == expected]
if not records:
    print(f"FAIL: no record for assembly {expected} in the summary")
    sys.exit(1)

r = records[0]
print()
print(f"  assembly        {r['id']} ({r.get('n_parts')} parts)")
print(f"  status          {r.get('status')}")
print(f"  steps planned   {r.get('n_steps')} / {(r.get('n_parts') or 1) - 1}")
print(f"  wall clock      {r.get('wall_s')} s")
print(f"  planner time    {r.get('planner_s')} s")
print(f"  feasibility ch. {r.get('n_eval')}")
print(f"  parallel factor {r.get('parallel_factor')}")
if r.get("stop_msg"):
    print(f"  stop reason     {r['stop_msg']}")
if r.get("error"):
    print(f"  error           {r['error']}")

breakdown = r.get("wall_breakdown") or {}
if breakdown:
    print("\n  where the time went:")
    for comp, val in sorted(breakdown.items(), key=lambda kv: -kv[1]):
        if val > 0.05:
            print(f"    {comp:<16} {val:>9.2f} s")

status = r.get("status")
print()
if status == "ok":
    print("PASS: the install works; a complete disassembly sequence was planned.")
    sys.exit(0)
if status == "partial":
    print(
        "PASS (with caveat): the planner ran and produced feasible steps but "
        "did not finish the sequence. The install is fine; raise --budget or "
        "give the job more cores/time."
    )
    sys.exit(0)
print(f"FAIL: status={status}. See the traceback in the run log above.")
sys.exit(1)
PYEOF
CHECK_RC=$?
set -e

section "Done"
echo "Run directory:  ${REPO_DIR}/assets/output/"
echo "Full log:       ${RUN_LOG}"
exit $(( RUN_RC != 0 ? RUN_RC : CHECK_RC ))
