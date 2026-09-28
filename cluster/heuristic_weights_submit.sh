#!/bin/bash
# Overnight test run of heuristic-weight training + held-out evaluation on a
# small asap subset (~10 h wall), ahead of the full-dataset run. Run on an
# Euler login node from anywhere:
#
#   bash $SCRATCH/AssembleX/cluster/heuristic_weights_overnight.sh
#   RUN_NAME=retry1 bash .../heuristic_weights_overnight.sh    # a second run
#
# It submits two arrays of cluster/heuristic_weights.sbatch:
#   train  TRAIN_WORKERS x 32 cores, 7 h: baselines for the training split,
#          then Optuna trials until the time budget (or N_TRIALS) is used up;
#   eval   TEST_WORKERS x 32 cores, 3 h (inside Euler's <=4 h class, which is
#          backfilled sooner), starting when every train task has ended: each
#          test assembly planned three ways -- heuristic DFA planner with the
#          reference weights, with the trained weights, and the gen:heur-out
#          baseline -- then the paired comparisons trained vs reference,
#          trained vs heur-out and heur-out vs reference.
# Results land in $REPO_DIR/assets/optuna_runs/$RUN_NAME/:
#   heuristic_weights.json, history.json      trained weights, per-trial log
#   eval_trained/summary.{txt,json}           the test result
#   run_info.txt                              what was submitted, from which commit
#
# Split (sizes in parts; all planned successfully in the earlier
# data_sequence_runtime runs). Estimated cold planning time on 32 cores is
# max(dev-machine wall, CPU-seconds / 30):
#   train (6)  one assembly per size 8-13, typical rather than extreme cost:
#              04600 (8) 03094 (9) 01350 (10) 02024 (11) 00675 (12) 01629 (13)
#              ~45 min for one cold pass over all six. The baselines are that
#              pass split over the workers (~20 min); a trial replays most
#              physics from the shared cache, so it costs a fraction of it.
#              Expect ~30-50 trials in the ~6.5 h budget; the log of the first
#              few trials (hits vs simulated per plan) is the number to read
#              for sizing the full run.
#   test (15)  one per size 5-16 plus three, disjoint from train:
#              05018 (5) 02741 (6) 05468 (7) 01082 (8) 01538 (9) 04383 (10)
#              03936 (10) 07604 (11) 00752 (12) 06245 (13) 07264 (13)
#              00664 (14) 05573 (14) 01234 (15) 07151 (16)
#              ~1.5 h cold; the three runs per assembly cost ~2 cold plans (the
#              later two replay much of the first's physics from the cache),
#              ~3 h of work over three workers, ~1-1.5 h wall.
# Testing on 5-7 and 14-16 parts, sizes training never saw, is deliberate:
# it shows whether the weights transfer across assembly size, the question
# the full run has to answer.
#
# Staging: assets/* is gitignored, so the meshes must be on Euler. From the
# repository root on a machine that has the dataset:
#   rsync -av --relative ./assets/data/asap/{04600,03094,...} euler:$SCRATCH/AssembleX/
# (the job preflight lists any that are missing, with the exact command).
# The checkout on Euler must include the training code this script calls:
#   cd $SCRATCH/AssembleX && git pull && git submodule update --init

set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/cluster/scratch/${USER}}}"
REPO_DIR="${REPO_DIR:-${SCRATCH_DIR}/AssembleX}"
RUN_NAME="${RUN_NAME:-overnight_$(date +%Y%m%d)}"

TRAIN_IDS="${TRAIN_IDS:-04600,03094,01350,02024,00675,01629}"
TEST_IDS="${TEST_IDS:-05018,02741,05468,01082,01538,04383,03936,07604,00752,06245,07264,00664,05573,01234,07151}"

TRAIN_WORKERS="${TRAIN_WORKERS:-3}"
TEST_WORKERS="${TEST_WORKERS:-3}"
CPUS="${CPUS:-32}"
TRAIN_TIME="${TRAIN_TIME:-07:00:00}"
TEST_TIME="${TEST_TIME:-03:00:00}"
# Study-wide cap; the time budget normally ends training first.
N_TRIALS="${N_TRIALS:-60}"

SBATCH_SCRIPT="${REPO_DIR}/cluster/heuristic_weights.sbatch"
[ -f "${SBATCH_SCRIPT}" ] || { echo "ERROR: ${SBATCH_SCRIPT} not found" >&2; exit 1; }
RUN_PATH="${REPO_DIR}/assets/optuna_runs/${RUN_NAME}"
if [ -e "${RUN_PATH}" ]; then
    echo "NOTE: ${RUN_PATH} exists; this submission extends that run (the study, baselines"
    echo "      and cache are reused). Use a new RUN_NAME for a fresh run. If an earlier job"
    echo "      was killed, delete its leftover locks first:"
    echo "        find ${RUN_PATH} -name '*.lock' -delete"
fi
mkdir -p "${SCRATCH_DIR}/logs" "${RUN_PATH}"

# IDS holds commas, which sbatch --export would split, so pass everything
# through the environment (sbatch exports it by default).
train_job=$(env PHASE=train RUN_NAME="${RUN_NAME}" IDS="${TRAIN_IDS}" N_TRIALS="${N_TRIALS}" \
    sbatch --parsable \
        --job-name="hw_train_${RUN_NAME}" \
        --array="0-$(( TRAIN_WORKERS - 1 ))" \
        --cpus-per-task="${CPUS}" --mem-per-cpu=1G --time="${TRAIN_TIME}" \
        "${SBATCH_SCRIPT}")
# afterany, not afterok: a train task that ends unusually still leaves the
# best weights so far, and the eval phase checks that a weights file exists.
eval_job=$(env PHASE=eval RUN_NAME="${RUN_NAME}" IDS="${TEST_IDS}" \
    sbatch --parsable \
        --job-name="hw_eval_${RUN_NAME}" \
        --dependency="afterany:${train_job}" \
        --array="0-$(( TEST_WORKERS - 1 ))" \
        --cpus-per-task="${CPUS}" --mem-per-cpu=1G --time="${TEST_TIME}" \
        "${SBATCH_SCRIPT}")

{
    echo "run:        ${RUN_NAME}"
    echo "submitted:  $(date -Iseconds) by ${USER}"
    echo "commit:     $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "ASAPx:      $(git -C "${REPO_DIR}/ASAPx" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "train:      job ${train_job}, ${TRAIN_WORKERS} x ${CPUS} cores, ${TRAIN_TIME}, ids ${TRAIN_IDS}"
    echo "eval:       job ${eval_job}, ${TEST_WORKERS} x ${CPUS} cores, ${TEST_TIME}, ids ${TEST_IDS}"
    echo "n_trials:   ${N_TRIALS}"
} | tee -a "${RUN_PATH}/run_info.txt"

cat <<EOF

Monitor:
  squeue -u ${USER}
  tail -f ${SCRATCH_DIR}/logs/hw_train_${RUN_NAME}_${train_job}_0.out
  grep -h "trial .* complete\|trial .* pruned\|sim cache:" ${SCRATCH_DIR}/logs/hw_train_${RUN_NAME}_${train_job}_*.out
Result (after the eval job):
  cat ${RUN_PATH}/eval_trained/summary.txt
Memory used, to size the full run:
  sacct -j ${train_job},${eval_job} --format=JobID,ReqMem,MaxRSS,Elapsed,State
EOF
