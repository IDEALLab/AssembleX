#!/bin/bash
# Bring the heuristic-weight result store up to date on Euler, from a login
# node in the repository:
#
#   bash cluster/store_maintenance_submit.sh
#
# Three chained jobs of cluster/store_maintenance.sbatch:
#   1. import  every run directory under assets/optuna_runs into the store
#              (runs made before the store; skips what is already there);
#   2. select  every stored heuristic run made before sequence selection gets
#              its min_cost counterpart -- the stored tree's cheapest sequence,
#              re-timed only when it differs -- under the fingerprint current
#              runs use, so all later training, evaluation and warm starts
#              reuse it instead of replanning;
#   3. report  per run directory: sequence_selection_report.txt (history
#              re-scored, best trial, store-wide selected / first time ratios)
#              and heuristic_weights_min_cost.json.
# All three are safe to repeat: resubmitting continues where a job stopped
# (e.g. at its time limit), and a store already up to date costs a few minutes.
#
# Sizing (override by exporting): ~1300 stored runs of 8-16 parts, ~5-30 s
# each to re-time on 10 cores -> ~1 h on 8 tasks.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/cluster/scratch/${USER}}}"
SELECT_WORKERS="${SELECT_WORKERS:-8}"
SELECT_CPUS="${SELECT_CPUS:-10}"
SELECT_TIME="${SELECT_TIME:-04:00:00}"
MEM_PER_CPU="${MEM_PER_CPU:-2G}"
export STORE_DIR="${STORE_DIR:-assets/optuna_store}"
export ASSEMBLY_SUBDIR="${ASSEMBLY_SUBDIR:-data/asap}"

SBATCH_SCRIPT="${REPO_DIR}/cluster/store_maintenance.sbatch"
mkdir -p "${SCRATCH_DIR}/logs"

active="$(squeue -u "${USER}" -h -o '%.200j' 2>/dev/null | grep -E '^ *store_(import|select|report)$' || true)"
if [ -n "${active}" ]; then
    echo "ERROR: store maintenance jobs are already queued or running; wait for them." >&2
    exit 1
fi

submit() {
    local phase="$1"
    shift
    env PHASE="${phase}" sbatch --parsable --job-name="store_${phase}" \
        --mem-per-cpu="${MEM_PER_CPU}" "$@" "${SBATCH_SCRIPT}"
}
imp=$(submit import --array=0 --cpus-per-task=1 --time=01:00:00)
sel=$(submit select --array="0-$(( SELECT_WORKERS - 1 ))" --cpus-per-task="${SELECT_CPUS}" \
      --time="${SELECT_TIME}" --dependency="afterok:${imp}")
rep=$(submit report --array=0 --cpus-per-task=1 --time=00:30:00 --dependency="afterany:${sel}")

cat <<EOF
import ${imp}  ->  select ${sel} (${SELECT_WORKERS} x ${SELECT_CPUS} cores)  ->  report ${rep}
Monitor:
  squeue -u ${USER}
  grep -h "^\[select\] done" ${SCRATCH_DIR}/logs/store_select_${sel}_*.out
Results:
  cat ${REPO_DIR}/assets/optuna_runs/*/sequence_selection_report.txt
If select ran out of time (its report says "left for the next run"), just
submit this script again.
EOF
