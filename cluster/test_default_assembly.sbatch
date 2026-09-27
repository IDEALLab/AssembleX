#!/bin/bash
#SBATCH --job-name=assemblex_default_test
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
# No --gpus: the sequence planner is pure CPU (RedMax physics + trimesh).
#
# The Slurm log paths above cannot read shell variables: %u is the submitting
# user, %x the job name, %j the job id. $SCRATCH/logs has to exist before
# submitting, or Slurm has nowhere to write and the job dies without a word:
#
#   mkdir -p "$SCRATCH/logs"
#
# Fast install test on the assembly that ships with the repository.
#
#   sbatch cluster/test_default_assembly.sbatch
#   BUDGET=1000 sbatch cluster/test_default_assembly.sbatch
#
# Same subcommand and the same pass/fail criteria as
# cluster/test_sequence_planner.sbatch; see its header for why
# data_sequence_runtime cannot reach an LLM or VLM code path. This is the
# smaller, quicker sibling: run it first after an install, and reach for the
# 15-part 01234 run only once this one passes.
#
# Two things make it the better smoke test:
#   - 7 parts instead of 15. The search space is a fraction of the size, so it
#     finishes in minutes on 8 cores rather than 30-45 on 16.
#   - the meshes are tracked in git (assets/data/04489 is whitelisted against
#     the blanket assets/* rule in .gitignore), so nothing has to be staged by
#     hand. Every other assembly has to be rsync'd in from a machine that has
#     the dataset.
#
# 04489 is also main.py's default --id / --dir, so `python main.py
# test_pipeline` with no arguments plans this same assembly. Both are still
# passed explicitly below: the result check needs the id to find its record in
# the summary, and being explicit keeps the override variables working.

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

# The assembly shipped with the repository. Both values are main.py's own
# defaults (DEFAULT_ASSEMBLY_ID / DEFAULT_ASSEMBLY_DIR).
ASSEMBLY_ID="${ASSEMBLY_ID:-04489}"
# Passed to --dir, which main.py resolves as assets/<dir>.
ASSEMBLY_SUBDIR="${ASSEMBLY_SUBDIR:-data}"
# Feasibility-check budget for the search. 6000 is the repo default and is far
# more than 7 parts need, so hitting it means something is wrong rather than
# the assembly being large.
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
# /cluster/home is read-only from a compute node, so matplotlib cannot write its
# default cache there and falls back to a fresh temp dir on every import. Point
# it at scratch instead: one less warning, and the font cache survives.
export APPTAINERENV_MPLCONFIGDIR="${SCRATCH_DIR}/.cache/matplotlib"
# A segfault in a native extension (redmax_py, VTK, torch) otherwise kills the
# interpreter with no traceback at all -- the shell only reports the signal.
# With this set, Python prints the C-level stack, naming the import or call that
# died, to stderr (the .err log).
export APPTAINERENV_PYTHONFAULTHANDLER="1"
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

mkdir -p "${SCRATCH_DIR}/logs" "${SCRATCH_DIR}/.cache/matplotlib" \
         "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

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
    # This assembly is tracked, so an empty directory means the checkout is
    # behind rather than the meshes needing to be staged.
    die "no .obj files in ${ASSEMBLY_PATH}
       ${ASSEMBLY_ID} is tracked in the repository, so this is a stale or
       incomplete checkout. Update it:
         cd ${REPO_DIR} && git pull && git submodule update --init --recursive ASAPx
       If a submodule fetch fails there, the pull did not apply: push the
       submodule commit the superproject points at, then pull again."
fi
echo "Assembly ${ASSEMBLY_ID}: ${N_OBJ} parts in ${ASSEMBLY_PATH}"
[ "${N_OBJ}" -eq 7 ] || echo "NOTE: expected 7 parts for the shipped 04489, found ${N_OBJ}."

SITE_PACKAGES="$(in_container python3 -c 'import site; print(site.getusersitepackages())')"
export APPTAINERENV_PYTHONPATH="${SITE_PACKAGES}:${REPO_DIR}"
echo "PYTHONPATH: ${SITE_PACKAGES}:${REPO_DIR}"

echo "Checking that the planner's imports resolve..."
# The import order here deliberately mirrors the real run: main.py pulls in
# core.simulation (redmax_py) long before SequencePlanner imports ASAPx's
# generator registry, which is what brings in torch, torch_geometric and
# matplotlib. Loading redmax_py's native extension before torch's is the order
# that matters for a native-library clash, so the check is only worth anything
# if it reproduces it. If this segfaults, re-run it with torch imported first:
# if that passes, the load order is the fault and the fix belongs in the
# pipeline, not in this script.
in_container python3 -X faulthandler - <<'PYEOF' || die "import check failed; re-run cluster/setup_assemblex.sbatch"
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

RUN_LOG="${SCRATCH_DIR}/logs/assemblex_default_test_${SLURM_JOB_ID:-local}.run.log"
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
Only 'ok' and 'partial' mean the install works; on 7 parts, anything short of
'ok' is worth reading the run log over -- 'partial' here is not the ordinary
budget-exhaustion result it would be on a large assembly.
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
        "did not finish the sequence. The install is fine, but 7 parts should "
        "not exhaust the budget -- check the stop reason above before moving "
        "on to a larger assembly."
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
echo "Next:           sbatch cluster/test_sequence_planner.sbatch  (15 parts, 16 cores)"
exit $(( RUN_RC != 0 ? RUN_RC : CHECK_RC ))
