"""Maintenance of the heuristic-weight result store (assets/optuna_store).

    python cluster/store_tool.py import <run dir> [<run dir> ...] [--dataset-dir assets/data/asap]
        Copy the planned runs of run directories made before the store into it
        (weight_trainer.import_run_into_store). Never overwrites.
    python cluster/store_tool.py select [--time-budget S] [--num-proc N]
        For every stored heuristic run made before sequence selection existed,
        store the run the current default (settings.sequence_selection =
        "min_cost") would have made, under the fingerprint a current run gets,
        so later training, evaluation and warm starts reuse it
        (weight_trainer.derive_selected_runs). Several workers can run at once.
    python cluster/store_tool.py report <run dir> [<run dir> ...]
        What selection changes: the run's training history re-scored, the best
        trial and its weights (<run>/heuristic_weights_min_cost.json), and the
        store-wide time ratio selected / first (weight_trainer.selection_report).

All of them are safe to repeat. --store overrides the store location. Run
through cluster/store_maintenance_submit.sh on Euler.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('import', 'select', 'report'))
    parser.add_argument('run_dirs', nargs='*')
    parser.add_argument('--store', default=None)
    parser.add_argument('--dataset-dir', default=os.path.join('assets', 'data', 'asap'))
    parser.add_argument('--time-budget', type=float, default=None, help='select: seconds')
    parser.add_argument('--num-proc', type=int, default=10, help='select: arm-pipeline workers')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path[:0] = [os.path.join(ROOT, 'ASAPx'), ROOT]
    from plan_sequence.optimizer import weight_trainer as wt

    if args.command == 'import':
        for run_dir in args.run_dirs:
            wt.import_run_into_store(run_dir, args.dataset_dir, store=args.store)
    elif args.command == 'select':
        deadline = None if args.time_budget is None else time.time() + args.time_budget
        wt.derive_selected_runs(store=args.store, deadline=deadline, num_proc=args.num_proc)
    else:
        for run_dir in args.run_dirs:
            wt.selection_report(run_dir, store=args.store)


if __name__ == '__main__':
    main()
