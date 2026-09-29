#!/bin/bash
# Is the heuristic better than chance? Plans every assembly of an earlier run
# (its training and test ids) with random decisions -- the same DFA search,
# the next frontier drawn at random (planner dfa-random) -- N_SEEDS times,
# and compares with the runs already stored: the heuristic with the reference
# weights (with and without sequence selection), the trained weights and
# gen:heur-out. From a login node in the repository:
#
#   bash cluster/random_baseline_submit.sh                       # the split test run
#   RUN_NAME=overnight_20260927 N_SEEDS=5 bash cluster/random_baseline_submit.sh
#
# One array of cluster/heuristic_weights.sbatch in the eval phase with
# EVAL_RANDOM; everything but the random runs (and runs the store lacks, e.g.
# heur-out on training assemblies) comes from the result store, so run the
# store maintenance first (cluster/store_maintenance_submit.sh). Summary:
# assets/optuna_runs/$RUN_NAME/eval_random/summary.txt. Resubmitting continues
# where a job stopped.
#
# Time: random plans are mostly cold (little of their physics is cached):
# ~5-15 min each on 32 cores; 21 assemblies x 3 seeds on 6 tasks ~2-3 h.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/cluster/scratch/${USER}}}"
RUN_NAME="${RUN_NAME:-split_test_20260928_1223}"
N_SEEDS="${N_SEEDS:-3}"
WORKERS="${WORKERS:-6}"
CPUS="${CPUS:-32}"
LIMIT="${LIMIT:-04:00:00}"
MEM_PER_CPU="${MEM_PER_CPU:-2G}"
RUN_PATH="${REPO_DIR}/assets/optuna_runs/${RUN_NAME}"

[ -f "${RUN_PATH}/heuristic_weights.json" ] || { echo "ERROR: no ${RUN_PATH}/heuristic_weights.json" >&2; exit 1; }
if [ -z "${IDS:-}" ]; then
    # The run's training and test ids, from its run_info.txt.
    IDS="$(grep -oE 'ids [0-9,]+' "${RUN_PATH}/run_info.txt" | cut -d' ' -f2 | paste -sd, - \
           | tr ',' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)"
fi
[ -n "${IDS}" ] || { echo "ERROR: no ids found in ${RUN_PATH}/run_info.txt; set IDS" >&2; exit 1; }

active="$(squeue -u "${USER}" -h -o '%.200j' 2>/dev/null | grep -E "hw_random_${RUN_NAME}\$" || true)"
[ -z "${active}" ] || { echo "ERROR: a random-baseline job for ${RUN_NAME} is already queued or running" >&2; exit 1; }
mkdir -p "${SCRATCH_DIR}/logs"

job=$(env PHASE=eval RUN_NAME="${RUN_NAME}" IDS="${IDS}" EVAL_RANDOM="${N_SEEDS}" EVAL_LABEL=random \
      sbatch --parsable --job-name="hw_random_${RUN_NAME}" --array="0-$(( WORKERS - 1 ))" \
      --cpus-per-task="${CPUS}" --mem-per-cpu="${MEM_PER_CPU}" --time="${LIMIT}" \
      "${REPO_DIR}/cluster/heuristic_weights.sbatch")

cat <<EOF
random baseline for ${RUN_NAME}: job ${job} (${WORKERS} x ${CPUS} cores, ${LIMIT}), ${N_SEEDS} seeds
ids: ${IDS}
Monitor:
  squeue -u ${USER}
  grep -h "random#" ${SCRATCH_DIR}/logs/hw_random_${RUN_NAME}_${job}_*.out
Result:
  cat ${RUN_PATH}/eval_random/summary.txt
EOF
