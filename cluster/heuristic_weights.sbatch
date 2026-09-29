#!/bin/bash
#SBATCH --job-name=assemblex_hw
#SBATCH --output=/cluster/scratch/%u/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/%x_%A_%a.err
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
# No --gpus: planning and the simplified arm pipeline are pure CPU.
#
# One worker of a heuristic-weight run, in one of four phases:
#   baselines  training split, reference weights, uncached (main.py
#              train_heuristic_weights --optuna-trials 0);
#   train      the Optuna trials (main.py train_heuristic_weights);
#   eval_ref   test split, reference weights and gen:heur-out, uncached and
#              independent of training (main.py data_heuristic_weights_eval
#              --eval-reference-only);
#   eval       test split, trained weights (EVAL_SPLIT=1: also trained weights +
#              subassembly plan), then the summary (main.py
#              data_heuristic_weights_eval [--eval-split]).
# Submit through cluster/heuristic_weights_submit.sh, which sets the variables
# below, sizes each phase and chains them; by hand it works the same way:
#
#   PHASE=train RUN_NAME=my_run IDS=04600,03094 sbatch --array=0-2 cluster/heuristic_weights.sbatch
#
# Several array tasks of one phase cooperate on one run directory
# (assets/optuna_runs/$RUN_NAME: the study, weights, history, summary):
# training workers share an Optuna journal study (constant-liar TPE, so they
# sample different points). Every planned run (baselines, trial runs, eval
# runs) and the candidate-check cache live in the store shared by all runs
# (STORE_DIR, assets/optuna_store), each claimed through a lock file, so the
# workers split the work and a later run reuses it. --optuna-trials is the
# study-wide total, so all workers stop together.
#
# Resources (measured on the 2026-09-27 test run; the defaults below are only
# for a bare sbatch, the submit script sets each phase):
#   - memory per CPU, never per job (Euler's submit filter rejects --mem);
#     1G per CPU, because the planner forks its workers and shares most of the
#     footprint copy-on-write (base + k*workers): 32-core tasks peaked at
#     4.1-6.1G on 8-16-part assemblies. Below ~4 cores raise --mem-per-cpu.
#   - cores by phase: uncached planning (baselines, eval_ref) keeps 16-32
#     workers busy (one frontier batch is ~16-38 checks wide); trials replay
#     93-100% of their physics from the cache after ~12 trials and used only
#     17-19% of 32 cores, so training runs on narrow tasks, more of them.
#
# Time: the Python side gets a budget = the job's remaining time minus
# MARGIN_S (read from squeue at start). No trial or run starts past it, and a
# trial that would run past it ends early, so the job finishes on its own and
# the weights file always holds the best trial so far.
#
# The Slurm log paths above cannot read shell variables: %u is the submitting
# user, %x the job name, %A the array job id, %a the task id. $SCRATCH/logs has
# to exist before submitting (the submit script creates it).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (set by the submit script; override by exporting)
# ---------------------------------------------------------------------------
PHASE="${PHASE:?set PHASE to baselines, train, eval_ref or eval}"
RUN_NAME="${RUN_NAME:?set RUN_NAME (the run directory name under assets/optuna_runs)}"
# Comma-separated assembly ids: the training split for baselines / train, the
# test split for eval_ref / eval.
IDS="${IDS:?set IDS to a comma-separated list of assembly ids}"
ASSEMBLY_SUBDIR="${ASSEMBLY_SUBDIR:-data/asap}"
N_TRIALS="${N_TRIALS:-60}"
# Feasibility-check budget per plan; must match between train and eval (it is
# part of every stored run's fingerprint).
BUDGET="${BUDGET:-6000}"
# Seconds kept free at the end of the job for the final writes.
MARGIN_S="${MARGIN_S:-1800}"
# PHASE=eval: 1 also plans the test split with the trained weights plus the
# recursive subassembly plan and compares it (--eval-split).
EVAL_SPLIT="${EVAL_SPLIT:-0}"
# PHASE=eval: N > 0 also plans the set N times with random decisions and with
# the reference weights without sequence selection (--eval-random N).
EVAL_RANDOM="${EVAL_RANDOM:-0}"
# PHASE=eval: the summary goes to <run>/eval_<EVAL_LABEL>/.
EVAL_LABEL="${EVAL_LABEL:-trained}"
# PHASE=eval: 1 also plans the reference weights without sequence selection.
EVAL_REFERENCE_FIRST="${EVAL_REFERENCE_FIRST:-0}"
# PHASE=eval: 1 = a task ends once nothing is left for it to claim instead of
# waiting for runs other tasks hold (wide arrays; a summary pass follows).
EVAL_NO_WAIT="${EVAL_NO_WAIT:-0}"

SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/cluster/scratch/${USER}}}"
REPO_DIR="${REPO_DIR:-${SCRATCH_DIR}/AssembleX}"
CONTAINER_PATH="${CONTAINER_PATH:-${SCRATCH_DIR}/assemble-x-container_latest.sif}"
USER_BASE="${SCRATCH_DIR}/.local"
NUM_PROC="${NUM_PROC:-${SLURM_CPUS_PER_TASK:-4}}"
# Relative to REPO_DIR, which is the working directory inside the container.
RUN_DIR="assets/optuna_runs/${RUN_NAME}"
# Every planned run and the candidate-check cache, shared by all runs: a new
# run reuses whatever an earlier one planned under identical conditions.
STORE_DIR="${STORE_DIR:-assets/optuna_store}"
# Train phase: earlier run directories (comma-separated, relative to the repo)
# whose best weights start the study.
WARM_START="${WARM_START:-}"

export APPTAINERENV_PYTHONUSERBASE="${USER_BASE}"
export APPTAINERENV_http_proxy="${http_proxy:-}"
export APPTAINERENV_https_proxy="${https_proxy:-}"
export APPTAINERENV_PREPEND_PATH="${USER_BASE}/bin"
export APPTAINER_CACHEDIR="${SCRATCH_DIR}/.apptainer/cache"
export APPTAINER_TMPDIR="${SCRATCH_DIR}/.apptainer/tmp"

# Headless: no X on a compute node, and VTK opening one in the parent right
# before the planner forks deadlocks the workers. Training and eval render no
# media (settings.heuristic_training["render_gifs"]), but the guard stays.
export APPTAINERENV_PYVISTA_OFF_SCREEN="true"
export APPTAINERENV_MPLBACKEND="Agg"
# /cluster/home is read-only from a compute node.
export APPTAINERENV_MPLCONFIGDIR="${SCRATCH_DIR}/.cache/matplotlib"
# A segfault in redmax_py or VTK otherwise dies without a traceback.
export APPTAINERENV_PYTHONFAULTHANDLER="1"
# Process-level parallelism only: threaded BLAS would oversubscribe the task.
export APPTAINERENV_OMP_NUM_THREADS="1"
export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
export APPTAINERENV_MKL_NUM_THREADS="1"

section() { echo; echo "=== $* ==="; }
die()     { echo "ERROR: $*" >&2; exit 1; }

[ -d "${SCRATCH_DIR}" ] || die "scratch directory does not exist: ${SCRATCH_DIR}"
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

# Seconds left in this job, from squeue's %L ([D-]HH:MM:SS, MM:SS or SS).
job_seconds_left() {
    local left days=0 h=0 m=0 s=0
    left="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null || true)"
    [ -n "${left}" ] && [ "${left}" != "INVALID" ] && [ "${left}" != "UNLIMITED" ] || return 1
    if [[ "${left}" == *-* ]]; then days="${left%%-*}"; left="${left#*-}"; fi
    IFS=: read -r -a parts <<< "${left}"
    case "${#parts[@]}" in
        3) h="${parts[0]}"; m="${parts[1]}"; s="${parts[2]}" ;;
        2) m="${parts[0]}"; s="${parts[1]}" ;;
        1) s="${parts[0]}" ;;
    esac
    echo $(( 10#${days} * 86400 + 10#${h} * 3600 + 10#${m} * 60 + 10#${s} ))
}

# ---------------------------------------------------------------------------
# Step 1: preflight
# ---------------------------------------------------------------------------
section "Step 1: preflight (${PHASE}, task ${SLURM_ARRAY_TASK_ID:-0} of array ${SLURM_ARRAY_JOB_ID:-none})"
command -v apptainer >/dev/null 2>&1 || die "apptainer not found on PATH"
[ -f "${CONTAINER_PATH}" ] || die "container image not found: ${CONTAINER_PATH}"
[ -f "${REPO_DIR}/main.py" ] || die "repository not found at ${REPO_DIR}; run cluster/setup_assemblex.sbatch first"
grep -q "data_heuristic_weights_eval" "${REPO_DIR}/main.py" \
    && grep -q "def evaluate_heuristic_weights" "${REPO_DIR}/ASAPx/plan_sequence/optimizer/weight_trainer.py" \
    || die "the checkout at ${REPO_DIR} predates the heuristic-weight evaluation code;
       update it (git pull && git submodule update --init) and resubmit"

missing=()
IFS=, read -r -a id_list <<< "${IDS}"
for id in "${id_list[@]}"; do
    n=$(find "${REPO_DIR}/assets/${ASSEMBLY_SUBDIR}/${id}" -maxdepth 1 -name '*.obj' 2>/dev/null | wc -l)
    [ "${n}" -gt 0 ] || missing+=("${id}")
done
if [ "${#missing[@]}" -gt 0 ]; then
    die "no meshes for ${#missing[@]} assemblies: ${missing[*]}
       assets/* is gitignored; stage them from a machine that has the dataset,
       e.g. from the repository root there:
         rsync -av --relative $(printf './assets/%s/%s ' "${ASSEMBLY_SUBDIR}" "${missing[@]}")\\
               euler:${REPO_DIR}/"
fi
echo "${#id_list[@]} assemblies staged under assets/${ASSEMBLY_SUBDIR}: ${IDS}"

SITE_PACKAGES="$(in_container python3 -c 'import site; print(site.getusersitepackages())')"
export APPTAINERENV_PYTHONPATH="${SITE_PACKAGES}:${REPO_DIR}"

in_container python3 - <<'PYEOF' || die "import check failed; re-run cluster/setup_assemblex.sbatch (optuna is one of its optional packages)"
import sys

sys.path.insert(0, "ASAPx")
import optuna
import redmax_py  # noqa: F401
from scipy.stats import wilcoxon  # noqa: F401  (WilcoxonPruner, eval statistics)
from plan_sequence.planner import planners

assert "heuristic" in planners
print(f"  optuna {optuna.__version__}, redmax_py ok, scipy ok")
PYEOF

if SECONDS_LEFT="$(job_seconds_left)"; then
    TIME_BUDGET_S=$(( SECONDS_LEFT - MARGIN_S ))
else
    TIME_BUDGET_S="${TIME_BUDGET_S:-$(( 6 * 3600 ))}"
    echo "could not read the job's remaining time from squeue; using ${TIME_BUDGET_S}s"
fi
[ "${TIME_BUDGET_S}" -gt 600 ] || die "time budget ${TIME_BUDGET_S}s is too small; raise --time or lower MARGIN_S"
# PHASE=eval, EVAL_SUMMARY_ONLY=1: plan nothing, only write the summary from
# what the store holds (the last pass of a campaign).
[ "${EVAL_SUMMARY_ONLY:-0}" = "1" ] && TIME_BUDGET_S=1
echo "time budget for Python: ${TIME_BUDGET_S}s  (job left ${SECONDS_LEFT:-?}s, margin ${MARGIN_S}s)"
echo "run directory: ${REPO_DIR}/${RUN_DIR}"
echo "store:         ${REPO_DIR}/${STORE_DIR}"
echo "workers: ${NUM_PROC}   planner budget: ${BUDGET}"

# ---------------------------------------------------------------------------
# Step 2: the run
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --id "${IDS}"
    --dir "${ASSEMBLY_SUBDIR}"
    --optuna-dir "${RUN_DIR}"
    --optuna-store "${STORE_DIR}"
    --optuna-timeout "${TIME_BUDGET_S}"
    --num-proc "${NUM_PROC}"
    --budget "${BUDGET}"
    --token-limit 0
)
case "${PHASE}" in
    baselines)
        section "Step 2: training baselines (reference weights, uncached)"
        CMD=(main.py train_heuristic_weights "${COMMON_ARGS[@]}"
             --optuna-resume --optuna-trials 0)
        ;;
    train)
        section "Step 2: training (study-wide target ${N_TRIALS} trials)"
        CMD=(main.py train_heuristic_weights "${COMMON_ARGS[@]}"
             --optuna-resume --optuna-trials "${N_TRIALS}")
        [ -n "${WARM_START}" ] && CMD+=(--optuna-warm-start "${WARM_START}")
        ;;
    eval_ref)
        section "Step 2: test-split baselines (reference weights and gen:heur-out)"
        CMD=(main.py data_heuristic_weights_eval "${COMMON_ARGS[@]}" --eval-reference-only)
        ;;
    eval)
        section "Step 2: evaluation on the held-out split"
        [ -f "${REPO_DIR}/${RUN_DIR}/heuristic_weights.json" ] \
            || die "no trained weights at ${RUN_DIR}/heuristic_weights.json (did training finish a trial?)"
        CMD=(main.py data_heuristic_weights_eval "${COMMON_ARGS[@]}")
        [ "${EVAL_SPLIT}" = "1" ] && CMD+=(--eval-split)
        [ "${EVAL_RANDOM}" -gt 0 ] && CMD+=(--eval-random "${EVAL_RANDOM}")
        [ "${EVAL_REFERENCE_FIRST}" = "1" ] && CMD+=(--eval-reference-first)
        [ "${EVAL_NO_WAIT}" = "1" ] && CMD+=(--eval-no-wait)
        CMD+=(--eval-label "${EVAL_LABEL}")
        ;;
    *) die "PHASE must be baselines, train, eval_ref or eval, not ${PHASE}" ;;
esac

START_TS=$(date +%s)
set +e
in_container python3 -u "${CMD[@]}"
RUN_RC=$?
set -e
ELAPSED=$(( $(date +%s) - START_TS ))
echo
echo "main.py exited with ${RUN_RC} after ${ELAPSED}s"

# ---------------------------------------------------------------------------
# Step 3: report
# ---------------------------------------------------------------------------
section "Step 3: report"
count_runs() { grep -l "\"label\": \"$1\"" "${REPO_DIR}/${STORE_DIR}"/runs/*/*.json 2>/dev/null | wc -l; }
case "${PHASE}" in
    baselines|eval_ref)
        echo "store (all runs): $(count_runs baseline) + $(count_runs reference) reference, $(count_runs heur-out) heur-out"
        ;;
    train)
        if [ -f "${REPO_DIR}/${RUN_DIR}/heuristic_weights.json" ]; then
            echo "best weights so far: $(cat "${REPO_DIR}/${RUN_DIR}/heuristic_weights.json" | tr -d '\n ')"
        else
            echo "no trial has planned every training assembly yet"
        fi
        echo "history: ${REPO_DIR}/${RUN_DIR}/history.json"
        ;;
    eval)
        SUMMARY="${REPO_DIR}/${RUN_DIR}/eval_${EVAL_LABEL}/summary.txt"
        [ -f "${SUMMARY}" ] && echo "summary: ${SUMMARY}" || echo "no summary written"
        ;;
esac
echo "memory actually used (compare with ReqMem before the full run):"
echo "  sacct -j ${SLURM_JOB_ID:-<jobid>} --format=JobID,ReqMem,MaxRSS,Elapsed,State"
exit "${RUN_RC}"
