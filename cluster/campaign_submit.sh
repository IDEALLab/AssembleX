#!/bin/bash
# Data-collection campaign: plan every sampled assembly with
#   reference          heuristic, reference weights (sequence selection on)
#   trained            heuristic, trained weights (sequence selection on)
#   heur-out           the gen:heur-out baseline
#   trained+split      trained weights + subassembly plan; its summary also
#                      gives the time with 2 workers (trained+split-2w)
#   random             random decisions (dfa-random), N_SEEDS seeds
# From a login node in the repository:
#
#   bash cluster/campaign_submit.sh
#   MAX_PARTS=25 RUN_NAME=campaign_upto25 bash cluster/campaign_submit.sh   # next size step
#
# Assemblies: every staged one with MIN_PARTS..MAX_PARTS parts except the
# training assemblies of the weights under test, smallest size band first and
# in random order within a band (cluster/sample_ids.py). Everything goes
# through the result store, so what earlier runs planned is reused and a
# larger MAX_PARTS later only plans the new assemblies.
#
# Jobs: ROUNDS chained arrays of WORKERS x CPUS (each <= 4 h, the next starts
# when the last task of the previous one ends), then one summary pass. Tasks
# take whole assemblies in order and end when nothing is left to claim, so a
# round with no work left costs a few minutes. Resubmitting (same RUN_NAME,
# RESUME=1) continues where the last round stopped.
#
# Sizing: ~10-15 core-hours per assembly up to 20 parts (all five planners,
# 1 random seed), so ~150 assemblies take ~2000 core-hours: 48 x 16 cores
# (768, 4x the 192 of the last run) about 4-5 h, i.e. two rounds.
#
# Larger assemblies: plans grow steeply with size (cold reference plan on 16
# cores, 2026-09-29 campaign: median 7 min at 5-9 parts, 47 min at 18-20, the
# longest 176 min at 16 parts), so past 20 parts give each task a longer
# LIMIT than a plan can take, a per-plan RUN_TIMEOUT (seconds; a plan over it
# is killed with its workers and stored as 'timeout', so a worker killed for
# memory no longer hangs its plan until the job limit) and more memory:
#   MAX_PARTS=30 LIMIT=24:00:00 RUN_TIMEOUT=43200 MEM_PER_CPU=4G ROUNDS=2 \
#       bash cluster/campaign_submit.sh
#
# Testing weights that are still being trained: submit into the training
# run's own directory and let the first round wait for the training job
# (AFTER=<job id>, afterany); the weights file is then read when the rounds
# start, and the training assemblies are excluded from its run_info.txt:
#   RUN_NAME=<train run> WEIGHTS_RUN=<train run> RESUME=1 AFTER=<train job> \
#       bash cluster/campaign_submit.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/cluster/scratch/${USER}}}"
RUN_NAME="${RUN_NAME:-campaign_$(date +%Y%m%d_%H%M)}"
ASSEMBLY_SUBDIR="${ASSEMBLY_SUBDIR:-data/asap}"
MIN_PARTS="${MIN_PARTS:-5}"
MAX_PARTS="${MAX_PARTS:-20}"
SEED="${SEED:-0}"
N_SEEDS="${N_SEEDS:-1}"
# The weights under test and the assemblies they were trained on (excluded).
WEIGHTS_RUN="${WEIGHTS_RUN:-split_test_20260928_1223}"
TRAINED_WEIGHTS="${TRAINED_WEIGHTS:-assets/optuna_runs/${WEIGHTS_RUN}/heuristic_weights.json}"
WORKERS="${WORKERS:-48}"
CPUS="${CPUS:-16}"
LIMIT="${LIMIT:-04:00:00}"
ROUNDS="${ROUNDS:-3}"
MEM_PER_CPU="${MEM_PER_CPU:-2G}"
# Per-plan wall-clock limit in seconds (empty = none).
RUN_TIMEOUT="${RUN_TIMEOUT:-}"
# Job id(s) the first round waits for (afterany), e.g. the training job.
AFTER="${AFTER:-}"

RUN_PATH="${REPO_DIR}/assets/optuna_runs/${RUN_NAME}"
SBATCH_SCRIPT="${REPO_DIR}/cluster/heuristic_weights.sbatch"

active="$(squeue -u "${USER}" -h -o '%.200j' 2>/dev/null | grep -E "hw_(campaign|summary)_${RUN_NAME}\$" || true)"
[ -z "${active}" ] || { echo "ERROR: campaign jobs of ${RUN_NAME} are already queued or running" >&2; exit 1; }
if [ -e "${RUN_PATH}/run_info.txt" ] && [ "${RESUME:-0}" != "1" ]; then
    echo "ERROR: ${RUN_PATH} exists; pick another RUN_NAME or set RESUME=1 to continue it" >&2
    exit 1
fi
# With AFTER the weights may not exist yet: the job it waits for writes them.
[ -n "${AFTER}" ] || [ -f "${REPO_DIR}/${TRAINED_WEIGHTS}" ] || { echo "ERROR: no ${TRAINED_WEIGHTS}" >&2; exit 1; }
mkdir -p "${RUN_PATH}" "${SCRATCH_DIR}/logs"
if [ -f "${REPO_DIR}/${TRAINED_WEIGHTS}" ] && ! [ "${REPO_DIR}/${TRAINED_WEIGHTS}" -ef "${RUN_PATH}/heuristic_weights.json" ]; then
    cp "${REPO_DIR}/${TRAINED_WEIGHTS}" "${RUN_PATH}/heuristic_weights.json"
fi

if [ ! -f "${RUN_PATH}/ids.txt" ]; then
    train_ids="$(grep -oE '^train: .*ids [0-9,]+' "${REPO_DIR}/assets/optuna_runs/${WEIGHTS_RUN}/run_info.txt" \
                 | grep -oE '[0-9,]+$' | tail -1 || true)"
    python3 "${REPO_DIR}/cluster/sample_ids.py" --dataset-dir "${REPO_DIR}/assets/${ASSEMBLY_SUBDIR}" \
        --min-parts "${MIN_PARTS}" --max-parts "${MAX_PARTS}" --exclude "${train_ids}" \
        --seed "${SEED}" --out "${RUN_PATH}/ids.txt"
fi
IDS="$(cat "${RUN_PATH}/ids.txt")"
N_IDS="$(tr ',' '\n' <<< "${IDS}" | grep -c .)"

submit() {  # submit NAME SUMMARY_ONLY [sbatch args...] -> job id
    local name="$1" summary_only="$2"
    shift 2
    env PHASE=eval RUN_NAME="${RUN_NAME}" IDS="${IDS}" ASSEMBLY_SUBDIR="${ASSEMBLY_SUBDIR}" \
        EVAL_SPLIT=1 EVAL_SPLIT_REPLAN="${SPLIT_REPLAN:-0}" EVAL_RANDOM="${N_SEEDS}" \
        EVAL_LABEL=campaign EVAL_NO_WAIT=1 \
        EVAL_SUMMARY_ONLY="${summary_only}" EVAL_RUN_TIMEOUT="${RUN_TIMEOUT}" \
        sbatch --parsable --job-name="hw_${name}_${RUN_NAME}" --mem-per-cpu="${MEM_PER_CPU}" \
        "$@" "${SBATCH_SCRIPT}"
}
jobs=()
dep=()
[ -z "${AFTER}" ] || dep=(--dependency="afterany:${AFTER}")
for r in $(seq 1 "${ROUNDS}"); do
    j=$(submit "campaign" 0 --array="0-$(( WORKERS - 1 ))" --cpus-per-task="${CPUS}" \
        --time="${LIMIT}" ${dep[@]+"${dep[@]}"})
    jobs+=("${j}")
    dep=(--dependency="afterany:${j}")
done
summary=$(submit "summary" 1 --array=0 --cpus-per-task=2 --time=01:00:00 ${dep[@]+"${dep[@]}"})

{
    echo "run:        ${RUN_NAME} (campaign)"
    echo "submitted:  $(date -Iseconds) by ${USER}"
    echo "commit:     $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "ASAPx:      $(git -C "${REPO_DIR}/ASAPx" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "assemblies: ${N_IDS} with ${MIN_PARTS}-${MAX_PARTS} parts (seed ${SEED}; ids.txt, ids_manifest.json)"
    echo "weights:    ${TRAINED_WEIGHTS}"
    echo "rounds:     ${jobs[*]} (${WORKERS} x ${CPUS} cores, ${LIMIT} each), summary ${summary}"
    echo "random:     ${N_SEEDS} seed(s) per assembly"
    echo "limits:     task ${LIMIT}, per plan ${RUN_TIMEOUT:-none} s, ${MEM_PER_CPU} per core"
    echo "after:      ${AFTER:-none}"
} | tee -a "${RUN_PATH}/run_info.txt"

cat <<EOF

Monitor:
  squeue -u ${USER}
  grep -ch "^\[optuna\] .* ok " ${SCRATCH_DIR}/logs/hw_campaign_${RUN_NAME}_*.out | paste -sd+ | bc   # runs finished
Result (after job ${summary}):
  cat ${RUN_PATH}/eval_campaign/summary.txt
EOF
