#!/usr/bin/env python3
"""Reproduce the revised submission Tables 1 and 2, using the original engines.

The repository's preprint calls these Tables 5 and 7. Existing defaults and
published reproduction recipes remain unchanged. Outputs include table1.tsv
and table2.tsv aliases; smoke mode uses reduced experiments, not paper values.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', choices=['1', '2', 'both'], default='both')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--out', type=Path, default=Path('output/submission_tables'))
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error('--jobs must be positive')
    root = Path(__file__).resolve().parent.parent
    out = args.out.resolve()

    def run(script, *options):
        subprocess.run([sys.executable, str(root/'scripts'/script), *map(str, options)],
                       cwd=root, check=True)

    if args.table in ('1', 'both'):
        target = out/'table1'
        options = ['--d', 10000, '--n-values', 1000, '--l-max', 80,
                   '--include-zero', '--extra-baselines', '--jobs', args.jobs,
                   '--transient-cache', '--checkpoint',
                   '--out', target]
        if args.smoke:
            options += ['--trials', 1, '--targets', 'uniform,zipf_2']
        run('benchmark_experiment.py', *options)
        run('benchmark_report.py', '--results', target/'results.json')
        shutil.copyfile(target/'table5.tsv', target/'table1.tsv')
    if args.table in ('2', 'both'):
        run('get_kjv.py')
        target = out/'table2'
        checkpoints = '10000' if args.smoke else '10000,30000,100000,300000,all'
        run('unigram_experiment.py', '--corpus', root/'data/kjv.txt', '--d', 100000,
            '--checkpoints', checkpoints, '--include-zero', '--jobs', args.jobs,
            '--out', target/'lsa')
        run('bible_baselines_experiment.py', '--corpus', root/'data/kjv.txt',
            '--d', 100000, '--checkpoints', checkpoints, '--extra-baselines',
            '--out', target/'baselines')
        run('bible_report.py', '--unigram', target/'lsa/results.json',
            '--baselines', target/'baselines/results.json', '--out', target)
        shutil.copyfile(target/'table7.tsv', target/'table2.tsv')


if __name__ == '__main__':
    main()
