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
    python cluster/store_tool.py split-features [--out <csv>]
        One row per stored subassembly (divide) run: what its plan looks like
        before any timing (cut sizes, balance, prefix, depth, divide score, the
        removal steps two or more workers could save) next to the times the
        timing model gave the plan and the run's flat sequence. The data for a
        split decision that does not use the timing model. Standard library
        only, so it runs on a login node.
    python cluster/store_tool.py repick <run dir> [--tolerance 0.15] [--out <json>]
        Pick a multi-objective run's weights again from its history.json, e.g.
        with another pareto_tolerance (weight_trainer.repick_run); writes
        --out (default <run dir>/heuristic_weights.json) and its _pareto.json.
        No optuna needed, runs on a login node.

All of them are safe to repeat. --store overrides the store location. Run
through cluster/store_maintenance_submit.sh on Euler.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLIT_FEATURE_COLUMNS = (
    'id', 'n_parts', 'label', 'weights_key', 'record', 'status', 'split', 'source',
    # plan structure, known before timing
    'root_prefix', 'root_S', 'root_R', 'root_score', 'root_balance', 'root_balance_sr',
    'n_splits', 'max_depth', 'steps_saved_2w', 'steps_saved_par',
    # timing model
    'flat_s', 'split_1w_s', 'split_2w_s', 'split_par_s',
)


def _plan_features(plan, n_parts):
    """Structure of a split_plan (weight_trainer stats['split_plan']): the
    outermost cut, how many cuts and how deep, and the removal steps (joins
    counted as steps) that 2 workers / one worker per split save against doing
    everything in sequence -- the parallel_makespan / parallel_2 recursion over
    step counts instead of seconds."""
    def seq(b):  # steps done one after another
        if b['kind'] != 'split':
            return len(b['parts']) - 1
        return len(b['prefix']) + 1 + seq(b['S']) + seq(b['R'])

    def par(b):  # every split's halves at once
        if b['kind'] != 'split':
            return len(b['parts']) - 1
        return len(b['prefix']) + 1 + max(par(b['S']), par(b['R']))

    def splits(b, depth):
        if b['kind'] != 'split':
            return 0, depth
        s1, d1 = splits(b['S'], depth + 1)
        s2, d2 = splits(b['R'], depth + 1)
        return 1 + s1 + s2, max(d1, d2)

    if not plan or plan.get('kind') != 'split':
        return {}
    S, R = len(plan['split']['S']), len(plan['split']['R'])
    n_splits, depth = splits(plan, 0)
    total = seq(plan)
    two = len(plan['prefix']) + 1 + max(seq(plan['S']), seq(plan['R']))
    return {'root_prefix': len(plan['prefix']), 'root_S': S, 'root_R': R,
            'root_score': plan['split'].get('score'),
            'root_balance': min(S, R) / n_parts, 'root_balance_sr': min(S, R) / (S + R),
            'n_splits': n_splits, 'max_depth': depth,
            'steps_saved_2w': (total - two) / total, 'steps_saved_par': (total - par(plan)) / total}


def split_features(store, out):
    rows = []
    for path in sorted(glob.glob(os.path.join(store, 'runs', '*', '*.json'))):
        if path.endswith(('assembly.json', '.weights.json')):
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        fp = rec.get('fingerprint') or {}
        if fp.get('seq_optimizer') != 'divide':
            continue
        row = {'id': rec.get('id'), 'n_parts': rec.get('n_parts'), 'label': rec.get('label'),
               'weights_key': hashlib.sha1(json.dumps(fp.get('weights'), sort_keys=True).encode()).hexdigest()[:8],
               'record': os.path.relpath(path, store), 'status': rec.get('status'),
               'split': rec.get('split'), 'source': rec.get('source'),
               'flat_s': rec.get('flat_total_s'),
               'split_1w_s': rec.get('total_s') if rec.get('split') == 'used' else None,
               'split_2w_s': rec.get('parallel2_total_s'), 'split_par_s': rec.get('parallel_total_s')}
        stats_path = os.path.join(path[:-len('.json')] + '_run', 'log', 'stats.json')
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            row.update(_plan_features(stats.get('split_plan'), rec.get('n_parts') or len(stats.get('sequence') or [])))
        except (OSError, json.JSONDecodeError, KeyError, ZeroDivisionError):
            pass
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SPLIT_FEATURE_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in SPLIT_FEATURE_COLUMNS})
    with_plan = sum(r.get('root_S') is not None for r in rows)
    timed = sum(r.get('split') == 'used' and r.get('status') == 'ok' for r in rows)
    print(f'{len(rows)} divide runs, {with_plan} with a plan, {timed} timed ok -> {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('import', 'select', 'report', 'split-features', 'repick'))
    parser.add_argument('run_dirs', nargs='*')
    parser.add_argument('--store', default=None)
    parser.add_argument('--dataset-dir', default=os.path.join('assets', 'data', 'asap'))
    parser.add_argument('--time-budget', type=float, default=None, help='select: seconds')
    parser.add_argument('--num-proc', type=int, default=10, help='select: arm-pipeline workers')
    parser.add_argument('--out', default=os.path.join('assets', 'optuna_runs', 'split_features.csv'),
                        help='split-features: output CSV; repick: output weights file')
    parser.add_argument('--tolerance', type=float, default=None,
                        help='repick: pareto_tolerance (default: settings.heuristic_training)')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'split-features':
        split_features(args.store or os.path.join('assets', 'optuna_store'), args.out)
        return
    out_given = '--out' in sys.argv
    sys.path[:0] = [os.path.join(ROOT, 'ASAPx'), ROOT]
    if args.command == 'repick':
        # weight_trainer by file: the package __init__ pulls in networkx and
        # the physics, which a login node's python does not have.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'weight_trainer', os.path.join(ROOT, 'ASAPx', 'plan_sequence', 'optimizer', 'weight_trainer.py'))
        wt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wt)
    else:
        from plan_sequence.optimizer import weight_trainer as wt

    if args.command == 'import':
        for run_dir in args.run_dirs:
            wt.import_run_into_store(run_dir, args.dataset_dir, store=args.store)
    elif args.command == 'repick':
        for run_dir in args.run_dirs:
            wt.repick_run(run_dir, tolerance=args.tolerance, out=args.out if out_given else None)
    elif args.command == 'select':
        deadline = None if args.time_budget is None else time.time() + args.time_budget
        wt.derive_selected_runs(store=args.store, deadline=deadline, num_proc=args.num_proc)
    else:
        for run_dir in args.run_dirs:
            wt.selection_report(run_dir, store=args.store)


if __name__ == '__main__':
    main()
