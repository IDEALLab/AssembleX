#!/bin/bash
# Test run with subassembly splitting, the same size as the 2026-09-27 test
# run (6 training, 15 test assemblies). Run it on an Euler login node:
#
#   bash $SCRATCH/AssembleX/cluster/heuristic_weights_split_test.sh
#
# It trains the heuristic weights exactly like that run, then evaluates four
# planners on every test assembly:
#   reference      heuristic DFA planner, reference weights
#   trained        the same planner, trained weights
#   heur-out       the gen:heur-out baseline
#   trained+split  trained weights + the recursive subassembly plan (the
#                  DivideOptimizer's cuts, searched at every state of the
#                  sequence), timed as the plan is carried out: prefix parts
#                  off the whole, R lifted off S as one body, S and R taken
#                  apart on their own. Every step is re-checked in that
#                  context; a plan that cannot be carried out as told (e.g. a
#                  half that does not stand on its own) falls back to the flat
#                  time and is reported.
# and reports trained vs reference, trained vs heur-out, and trained+split vs
# each of the other three, plus on how many assemblies a plan was found and
# executable. Everything else -- phases, resources, guards, outputs -- is
# cluster/heuristic_weights_submit.sh; any of its variables can be overridden
# here the same way (e.g. EVAL_TIME=05:00:00 bash ...).
#
# Split. Train: the 2026-09-27 training set (one assembly per size 8-13).
# Test: 15 assemblies of 8-16 parts, disjoint from train -- a subassembly needs
# two sides of at least 2 parts, and blocks under 4 parts are not split, so the
# 5-7-part assemblies of the last test set would mostly have no plan. 11 of the
# 15 are from the last test set, so the reference / trained / heur-out columns
# can be checked against it; all planned successfully before.
#   01082 (8) 01895 (8) 01538 (9) 04383 (10) 03936 (10) 07604 (11) 04205 (11)
#   00752 (12) 03008 (12) 06245 (13) 07264 (13) 00664 (14) 05573 (14)
#   01234 (15) 07151 (16)
#
# Time: training as the 2026-09-27 run. The eval phase adds, per test
# assembly, the divide optimizer and its physical verification of the top
# cuts, the recursive plan's own verification per block, and the per-context
# checks of the timing -- ~5-15 min each on 16 cores by estimate, hence 4 h
# instead of 3 for that phase. The first run of this measures it; read
# sacct afterwards (the submit script prints the command).

set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/cluster/scratch/${USER}}}"
REPO_DIR="${REPO_DIR:-${SCRATCH_DIR}/AssembleX}"

export RUN_NAME="${RUN_NAME:-split_test_$(date +%Y%m%d_%H%M)}"
export TRAIN_IDS="${TRAIN_IDS:-04600,03094,01350,02024,00675,01629}"
export TEST_IDS="${TEST_IDS:-01082,01895,01538,04383,03936,07604,04205,00752,03008,06245,07264,00664,05573,01234,07151}"
export EVAL_SPLIT=1
export EVAL_TIME="${EVAL_TIME:-04:00:00}"

exec bash "${REPO_DIR}/cluster/heuristic_weights_submit.sh"
