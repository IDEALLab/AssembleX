#!/bin/bash
# Submit a heuristic-weight train + held-out test run on Euler. Run it on a
# login node (with bash, not sbatch -- it submits the jobs itself):
#
#   bash $SCRATCH/AssembleX/cluster/heuristic_weights_submit.sh
#   RUN_NAME=full_v1 TRAIN_IDS=... TEST_IDS=... bash .../heuristic_weights_submit.sh
#
# Four arrays of cluster/heuristic_weights.sbatch, each sized for its work
# (measurements from the 2026-09-27 test run, 32-core tasks):
#
#   baselines  wide, from the start: the training split planned once with the
#              reference weights, uncached (the parallel search keeps ~16-38
#              workers busy).
#   train      narrow, after baselines: the Optuna trials. After ~12 trials
#              93-100% of the physics replays from the shared cache, and 32-core
#              tasks used only 17-19% of their cores, so many 8-core tasks run
#              more trials for the same cores.
#   eval_ref   wide, from the start, alongside training: the test split planned
#              with the reference weights and by gen:heur-out, uncached and
#              independent of training.
#   eval       medium, after train and eval_ref: the test split with the trained
#              weights (~60% cache hits) and the summary.
#
# Results land in $REPO_DIR/assets/optuna_runs/$RUN_NAME/:
#   heuristic_weights.json, history.json      trained weights, per-trial log
#   eval_trained/summary.{txt,json}           the test result
#   run_info.txt                              what was submitted, from which commit
#
# A run name is used once: submitting again while its jobs are queued or
# running is refused (a second submission doubles every phase on the same
# study), and so is reusing a finished run's directory unless RESUME=1.
#
# Defaults: the overnight test split. Train: one assembly per size 8-13
#   04600 (8) 03094 (9) 01350 (10) 02024 (11) 00675 (12) 01629 (13)
# Test: one per size 5-16 plus three, disjoint from train
#   05018 (5) 02741 (6) 05468 (7) 01082 (8) 01538 (9) 04383 (10) 03936 (10)
#   07604 (11) 00752 (12) 06245 (13) 07264 (13) 00664 (14) 05573 (14)
#   01234 (15) 07151 (16)
#
# Sizing a run (Euler, measured): an uncached plan takes ~440 s at 5-9 parts,
# ~830 s at 10-14, ~1750 s at 15-19, ~4400 s at 20-26 (medians; the 90th
# percentile is 2-3x). baselines ~ sum over the training split; eval_ref ~1.8x
# the sum over the test split (reference + heur-out); eval ~0.5x of it;
# a trial ~0.35x the training sum for the first ~10 trials, then less as the
# cache fills. Set the *_TIME limits from that; every phase stops starting new
# work 30 min before its limit and the weights file always holds the best
# trial so far.
#
# The checkout on Euler must include the code this calls:
#   cd $SCRATCH/AssembleX && git pull && git submodule update --init

set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/cluster/scratch/${USER}}}"
REPO_DIR="${REPO_DIR:-${SCRATCH_DIR}/AssembleX}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M)}"

TRAIN_IDS="${TRAIN_IDS:-04600,03094,01350,02024,00675,01629}"
TEST_IDS="${TEST_IDS:-05018,02741,05468,01082,01538,04383,03936,07604,00752,06245,07264,00664,05573,01234,07151}"
# Study-wide trial cap; the train time limit may end training first.
N_TRIALS="${N_TRIALS:-60}"
# 1: the eval phase also plans every test assembly with the trained weights plus
# the recursive subassembly plan (--seq-optimizer divide), timed as the plan is
# carried out, and compares it with the other planners.
EVAL_SPLIT="${EVAL_SPLIT:-0}"

# workers x cores x time limit per phase
BASE_WORKERS="${BASE_WORKERS:-3}";   BASE_CPUS="${BASE_CPUS:-32}";   BASE_TIME="${BASE_TIME:-03:00:00}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"; TRAIN_CPUS="${TRAIN_CPUS:-8}";  TRAIN_TIME="${TRAIN_TIME:-06:00:00}"
EREF_WORKERS="${EREF_WORKERS:-3}";   EREF_CPUS="${EREF_CPUS:-32}";   EREF_TIME="${EREF_TIME:-04:00:00}"
EVAL_WORKERS="${EVAL_WORKERS:-3}";   EVAL_CPUS="${EVAL_CPUS:-16}";   EVAL_TIME="${EVAL_TIME:-03:00:00}"
# 1G per CPU (forked workers share most of their memory); raise it for tasks
# below ~4 cores, where the fixed base footprint dominates.
MEM_PER_CPU="${MEM_PER_CPU:-1G}"

SBATCH_SCRIPT="${REPO_DIR}/cluster/heuristic_weights.sbatch"
[ -f "${SBATCH_SCRIPT}" ] || { echo "ERROR: ${SBATCH_SCRIPT} not found" >&2; exit 1; }
RUN_PATH="${REPO_DIR}/assets/optuna_runs/${RUN_NAME}"

# Refuse a second submission of the same run.
active="$(squeue -u "${USER}" -h -o '%.200j' 2>/dev/null | awk '{print $1}' | grep -E "_${RUN_NAME}\$" || true)"
if [ -n "${active}" ]; then
    echo "ERROR: jobs of run ${RUN_NAME} are already queued or running:" >&2
    squeue -u "${USER}" -o '%.18i %.9P %.40j %.2t %.10M %R' | grep -E "_${RUN_NAME}( |\$)" >&2 || true
    echo "       Submitting again would double every phase on the same study." >&2
    exit 1
fi
if [ -e "${RUN_PATH}/run_info.txt" ] && [ "${RESUME:-0}" != "1" ]; then
    echo "ERROR: ${RUN_PATH} already holds a run. Pick another RUN_NAME, or set" >&2
    echo "       RESUME=1 to extend it (study, stored runs and cache are reused; if an" >&2
    echo "       earlier job was killed, first: find ${RUN_PATH} -name '*.lock' -delete)." >&2
    exit 1
fi
mkdir -p "${SCRATCH_DIR}/logs" "${RUN_PATH}"

# submit PHASE IDS WORKERS CPUS TIME [sbatch args...] -> job id. IDS holds
# commas, which sbatch --export would split, so it goes through the
# environment (sbatch exports it by default).
submit() {
    local phase="$1" ids="$2" workers="$3" cpus="$4" limit="$5"
    shift 5
    env PHASE="${phase}" RUN_NAME="${RUN_NAME}" IDS="${ids}" N_TRIALS="${N_TRIALS}" \
        EVAL_SPLIT="${EVAL_SPLIT}" \
        sbatch --parsable \
            --job-name="hw_${phase}_${RUN_NAME}" \
            --array="0-$(( workers - 1 ))" \
            --cpus-per-task="${cpus}" --mem-per-cpu="${MEM_PER_CPU}" --time="${limit}" \
            "$@" "${SBATCH_SCRIPT}"
}

base_job=$(submit baselines "${TRAIN_IDS}" "${BASE_WORKERS}" "${BASE_CPUS}" "${BASE_TIME}")
eref_job=$(submit eval_ref "${TEST_IDS}" "${EREF_WORKERS}" "${EREF_CPUS}" "${EREF_TIME}")
# afterany, not afterok: a task that ends unusually still leaves its stored
# runs, and the next phase computes whatever is missing.
train_job=$(submit train "${TRAIN_IDS}" "${TRAIN_WORKERS}" "${TRAIN_CPUS}" "${TRAIN_TIME}" \
    --dependency="afterany:${base_job}")
eval_job=$(submit eval "${TEST_IDS}" "${EVAL_WORKERS}" "${EVAL_CPUS}" "${EVAL_TIME}" \
    --dependency="afterany:${train_job}:${eref_job}")

{
    echo "run:        ${RUN_NAME}"
    echo "submitted:  $(date -Iseconds) by ${USER}"
    echo "commit:     $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "ASAPx:      $(git -C "${REPO_DIR}/ASAPx" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "baselines:  job ${base_job}, ${BASE_WORKERS} x ${BASE_CPUS} cores, ${BASE_TIME}"
    echo "train:      job ${train_job}, ${TRAIN_WORKERS} x ${TRAIN_CPUS} cores, ${TRAIN_TIME}, ids ${TRAIN_IDS}"
    echo "eval_ref:   job ${eref_job}, ${EREF_WORKERS} x ${EREF_CPUS} cores, ${EREF_TIME}"
    echo "eval:       job ${eval_job}, ${EVAL_WORKERS} x ${EVAL_CPUS} cores, ${EVAL_TIME}, ids ${TEST_IDS}"
    echo "n_trials:   ${N_TRIALS}   mem per cpu: ${MEM_PER_CPU}   eval_split: ${EVAL_SPLIT}"
} | tee -a "${RUN_PATH}/run_info.txt"

cat <<EOF

Monitor:
  squeue -u ${USER}
  grep -h "trial .* complete\|trial .* pruned\|sim cache:" ${SCRATCH_DIR}/logs/hw_train_${RUN_NAME}_${train_job}_*.out
Result (after the eval job):
  cat ${RUN_PATH}/eval_trained/summary.txt
Resource use, for sizing the next run:
  sacct -j ${base_job},${train_job},${eref_job},${eval_job} --units=G \\
    --format=JobID%18,JobName%30,State,Elapsed,Timelimit,AllocCPUS,TotalCPU,ReqMem,MaxRSS
EOF
