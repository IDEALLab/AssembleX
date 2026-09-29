"""Pick the assemblies of a data-collection campaign from the staged dataset.

    python3 cluster/sample_ids.py [--dataset-dir assets/data/asap] [--min-parts 5]
        [--max-parts 20] [--band 5] [--exclude <ids>] [--seed 0] [--out <file>]

Every assembly under the dataset directory with min..max parts (part count =
number of .obj files), minus --exclude (e.g. the training assemblies of the
weights under test, which would be scored in-sample), in size bands of
--band parts (5-9, 10-14, 15-19, 20...), smallest band first and in a seeded
random order within each band. A campaign processes ids in this order, so it
works its way up the sizes and a run cut short still covers the smaller ones
at random. Prints the count per part number; writes the comma-separated ids to
--out and a manifest (id, parts, band) next to it. Standard library only, so
it runs on a login node.
"""
import argparse
import json
import os
import random
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-dir', default=os.path.join('assets', 'data', 'asap'))
    ap.add_argument('--min-parts', type=int, default=5)
    ap.add_argument('--max-parts', type=int, default=20)
    ap.add_argument('--band', type=int, default=5)
    ap.add_argument('--exclude', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    exclude = {x for x in a.exclude.split(',') if x}
    parts = {}
    for name in sorted(os.listdir(a.dataset_dir)):
        path = os.path.join(a.dataset_dir, name)
        if os.path.isdir(path):
            n = sum(f.endswith('.obj') for f in os.listdir(path))
            if a.min_parts <= n <= a.max_parts and name not in exclude:
                parts[name] = n
    rng = random.Random(a.seed)
    bands = {}
    for aid, n in parts.items():
        bands.setdefault((n - a.min_parts) // a.band, []).append(aid)
    order = []
    for b in sorted(bands):
        ids = sorted(bands[b])
        rng.shuffle(ids)
        order += ids

    counts = Counter(parts.values())
    print(f'{len(order)} assemblies with {a.min_parts}-{a.max_parts} parts in {a.dataset_dir}'
          + (f' ({len(exclude)} excluded)' if exclude else ''))
    print('  parts: ' + '  '.join(f'{n}:{counts[n]}' for n in sorted(counts)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        f.write(','.join(order) + '\n')
    lo = a.min_parts
    with open(os.path.splitext(a.out)[0] + '_manifest.json', 'w') as f:
        json.dump([{'id': aid, 'parts': parts[aid],
                    'band': f'{lo + (parts[aid] - lo) // a.band * a.band}-'
                            f'{lo + (parts[aid] - lo) // a.band * a.band + a.band - 1}'}
                   for aid in order], f, indent=1)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
