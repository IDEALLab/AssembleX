#!/bin/bash
#SBATCH --job-name=assemblex_store
#SBATCH --output=/cluster/scratch/%u/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#
# One worker of a maintenance pass over the heuristic-weight result store
# (assets/optuna_store), in one of three phases (cluster/store_tool.py):
#   import  copy the runs of every directory under assets/optuna_runs made
#           before the store into it (one task; never overwrites);
#   select  for every stored heuristic run made before sequence selection,
#           store the run the current default (min_cost) would have made --
#           same tree, the cheapest sequence re-timed -- so every later run
#           reuses it (array; tasks share the work through per-run locks);
#   report  per run directory: training history re-scored with selected
#           sequences, best trial and weights, store-wide time ratios.
# Submit through cluster/store_maintenance_submit.sh, which chains the three.
# Every phase is safe to repeat and resumes where an earlier one stopped.

set -euo pipefail

PHASE="${PHASE:?set PHASE to import, select or report}"
ASSEMBLY_SUBDIR="${ASSEMBLY_SUBDIR:-data/asap}"
STORE_DIR="${STORE_DIR:-assets/optuna_store}"
# Seconds kept free at the end of the job (select stops starting runs then).
MARGIN_S="${MARGIN_S:-900}"

SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/cluster/scratch/${USER}}}"
REPO_DIR="${REPO_DIR:-${SCRATCH_DIR}/AssembleX}"
CONTAINER_PATH="${CONTAINER_PATH:-${SCRATCH_DIR}/assemble-x-container_latest.sif}"
USER_BASE="${SCRATCH_DIR}/.local"
NUM_PROC="${NUM_PROC:-${SLURM_CPUS_PER_TASK:-4}}"

export APPTAINERENV_PYTHONUSERBASE="${USER_BASE}"
export APPTAINERENV_PREPEND_PATH="${USER_BASE}/bin"
export APPTAINER_CACHEDIR="${SCRATCH_DIR}/.apptainer/cache"
export APPTAINER_TMPDIR="${SCRATCH_DIR}/.apptainer/tmp"
export APPTAINERENV_PYVISTA_OFF_SCREEN="true"
export APPTAINERENV_MPLBACKEND="Agg"
export APPTAINERENV_MPLCONFIGDIR="${SCRATCH_DIR}/.cache/matplotlib"
export APPTAINERENV_PYTHONFAULTHANDLER="1"
export APPTAINERENV_OMP_NUM_THREADS="1"
export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
export APPTAINERENV_MKL_NUM_THREADS="1"

die() { echo "ERROR: $*" >&2; exit 1; }
mkdir -p "${SCRATCH_DIR}/logs" "${SCRATCH_DIR}/.cache/matplotlib" \
         "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
BIND_ARGS=(--bind "${SCRATCH_DIR}:${SCRATCH_DIR}")
case "${REPO_DIR}" in
    "${SCRATCH_DIR}"/*) ;;
    *) BIND_ARGS+=(--bind "${REPO_DIR}:${REPO_DIR}") ;;
esac
in_container() {
    apptainer exec --cleanenv "${BIND_ARGS[@]}" --pwd "${REPO_DIR}" "${CONTAINER_PATH}" "$@"
}
[ -f "${CONTAINER_PATH}" ] || die "container image not found: ${CONTAINER_PATH}"
[ -f "${REPO_DIR}/cluster/store_tool.py" ] || die "${REPO_DIR} predates cluster/store_tool.py; git pull"
SITE_PACKAGES="$(in_container python3 -c 'import site; print(site.getusersitepackages())')"
export APPTAINERENV_PYTHONPATH="${SITE_PACKAGES}:${REPO_DIR}"

RUN_DIRS=()
for d in "${REPO_DIR}"/assets/optuna_runs/*/; do
    [ -d "${d}" ] && RUN_DIRS+=("assets/optuna_runs/$(basename "${d}")")
done
echo "phase ${PHASE}, task ${SLURM_ARRAY_TASK_ID:-0}; store ${STORE_DIR}; ${#RUN_DIRS[@]} run directories"

TOOL=(python3 -u cluster/store_tool.py)
case "${PHASE}" in
    import)
        CMD=("${TOOL[@]}" import "${RUN_DIRS[@]}" --store "${STORE_DIR}"
             --dataset-dir "assets/${ASSEMBLY_SUBDIR}")
        ;;
    select)
        left="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null || true)"
        budget=""
        if [[ "${left}" =~ ^(([0-9]+)-)?(([0-9]+):)?([0-9]+):([0-9]+)$ ]]; then
            secs=$(( 10#${BASH_REMATCH[2]:-0} * 86400 + 10#${BASH_REMATCH[4]:-0} * 3600
                     + 10#${BASH_REMATCH[5]} * 60 + 10#${BASH_REMATCH[6]} ))
            budget="--time-budget $(( secs - MARGIN_S ))"
        fi
        # shellcheck disable=SC2206
        CMD=("${TOOL[@]}" select --store "${STORE_DIR}" --num-proc "${NUM_PROC}" ${budget})
        ;;
    report)
        CMD=("${TOOL[@]}" report "${RUN_DIRS[@]}" --store "${STORE_DIR}")
        ;;
    *) die "PHASE must be import, select or report, not ${PHASE}" ;;
esac
in_container "${CMD[@]}"
