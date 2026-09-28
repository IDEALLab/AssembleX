"""Copy the planned runs of heuristic-weight run directories made before the
shared store into it, so later runs reuse them (weight_trainer.import_run_into_store).

    python cluster/import_optuna_run.py assets/optuna_runs/<run> [<run> ...] \
        [--dataset-dir assets/data/asap] [--store assets/optuna_store]

Safe to repeat: existing store entries are never overwritten, and the run
directories are only read.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', nargs='+')
    parser.add_argument('--dataset-dir', default=os.path.join('assets', 'data', 'asap'))
    parser.add_argument('--store', default=None)
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path[:0] = [os.path.join(ROOT, 'ASAPx'), ROOT]
    from plan_sequence.optimizer.weight_trainer import import_run_into_store
    for run_dir in args.run_dirs:
        import_run_into_store(run_dir, args.dataset_dir, store=args.store)


if __name__ == '__main__':
    main()
