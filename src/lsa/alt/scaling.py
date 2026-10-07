"""Retained spectrum, factorial scaling, and depth-coefficient experiments.

Raw draws precede summaries. Each profile is evaluated at every declared depth;
the depth average is the equal prior over precisely that grid, including zero.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, logsumexp

from .artifacts import read_json, write_json
from .benchmark import jsonable

LOG2 = math.log(2)
C_STAR = 1 / (1 - float(np.euler_gamma))


def _validate_config(kind, config):
    """Reject undeclared rules and grids before drawing or fitting profiles."""
    def integer(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f'{name} must be an integer')
        if value < minimum:
            raise ValueError(f'{name} must be >= {minimum}')

    def unique(values, name):
        if not values or len(set(values)) != len(values):
            raise ValueError(f'{name} must be nonempty and unique')

    integer(config['seed'], 'seed', 0)
    integer(config['trials'], 'trials')
    integer(config.get('trial_start', 0), 'trial_start', 0)
    unique(config['alphas'], 'alphas')
    if any(not math.isfinite(a) or a < 0 for a in config['alphas']):
        raise ValueError('alphas must be finite and nonnegative')
    if kind == 'spectrum':
        if not config['panels']:
            raise ValueError('spectrum needs at least one panel')
        identities = []
        for panel in config['panels']:
            for name in ('d', 'n'):
                integer(panel[name], name)
            integer(panel['max_depth'], 'max_depth', 0)
            unique(panel['show_depths'], 'show_depths')
            for depth in panel['show_depths']:
                integer(depth, 'show_depth', 0)
                if depth > panel['max_depth']:
                    raise ValueError('displayed depth exceeds evaluated depth cap')
            identities.append((panel['d'], panel['n']))
        unique(identities, 'spectrum panels')
    elif kind == 'factorial':
        for key in ('ds', 'ns', 'table_ds', 'table_ns'):
            unique(config[key], key)
            for value in config[key]:
                integer(value, key)
        if len(config['ds']) < 2 or len(config['ns']) < 2:
            raise ValueError('factorial slope fits need at least two d and two N values')
        if not set(config['table_ds']) <= set(config['ds']) or not set(config['table_ns']) <= set(config['ns']):
            raise ValueError('table settings must be present in the evaluated factorial grid')
        if config['depth_rule'] == 'fixed':
            integer(config['fixed_depth'], 'fixed_depth', 0)
        elif config['depth_rule'] != 'round_c_star_log_d':
            raise ValueError('unknown factorial depth rule')
    elif kind == 'depth_scaling':
        for name in ('d', 'n'):
            integer(config[name], name)
        unique(config['c_values'], 'c_values')
        if any(not math.isfinite(c) or c <= 0 for c in config['c_values']):
            raise ValueError('depth coefficients must be finite and positive')
        if config['c_values'] != sorted(config['c_values']):
            raise ValueError('depth coefficients must be ordered')
        if config['heuristic_offset'] != 'alpha_log2_e_minus_2':
            raise ValueError('unknown depth heuristic offset; no implicit substitution')
    else:
        raise ValueError(f'unknown scaling experiment {kind}')
    if 'cell_ids' in config:
        unique(config['cell_ids'], 'cell_ids')
        for cell_id in config['cell_ids']:
            integer(cell_id, 'cell_id', 0)
        if max(config['cell_ids']) >= sum(1 for _ in cells(kind, config)):
            raise ValueError('cell_ids must refer to the complete original scaling grid')


def zipf(d, alpha):
    logp = -float(alpha) * np.log(np.arange(1, d + 1, dtype=float))
    return np.exp(logp - logsumexp(logp))


def expected_discovered(p, n):
    with np.errstate(divide='ignore'):
        return float(np.sum(-np.expm1(n * np.log1p(-p))))


def rho(c):
    if c >= C_STAR:
        return 1.0
    if c <= 0:
        raise ValueError('depth coefficient must be positive')
    y, upper = 1 / c, 10.0
    while digamma(1 + upper) < y:
        upper *= 2
    s = brentq(lambda t: digamma(1 + t) - y, 1e-9, upper, xtol=1e-12)
    return float(c * (s * y - gammaln(1 + s)))


def cells(kind, config):
    """The complete original grid, independent of optional shard selectors."""
    if kind == 'spectrum':
        for panel in config['panels']:
            for alpha in config['alphas']:
                yield {**panel, 'alpha': alpha, 'depths': list(range(panel['max_depth'] + 1))}
    elif kind == 'factorial':
        for d in config['ds']:
            depth = (max(1, round(C_STAR * math.log(d)))
                     if config['depth_rule'] == 'round_c_star_log_d' else config['fixed_depth'])
            for n in config['ns']:
                for alpha in config['alphas']:
                    yield {'d': d, 'n': n, 'alpha': alpha, 'depths': [depth]}
    elif kind == 'depth_scaling':
        depths = sorted({max(1, round(c * math.log(config['d']))) for c in config['c_values']})
        for alpha in config['alphas']:
            yield {'d': config['d'], 'n': config['n'], 'alpha': alpha, 'depths': depths}
    else:
        raise ValueError(f'unknown scaling experiment {kind}')


def selected_cells(kind, config):
    """Return (global cell ID, cell) in the full grid's canonical order.

    A shard keeps the full panels/ds/ns/alphas/c_values and sets ``cell_ids``;
    filtering those grid fields would define a different sampling protocol.
    """
    selected = config.get('cell_ids')
    return [(cell_id, cell) for cell_id, cell in enumerate(cells(kind, config))
            if selected is None or cell_id in selected]


def run_scaling(kind, config, output_dir, *, evaluator, batch_size=None):
    """Evaluate a full grid or explicit cell/global-trial shard.

    ``trial_start`` defaults to zero and ``trials`` is this shard's length.
    Earlier draws are consumed without evaluation to preserve the existing
    sequential multinomial stream seeded by [seed, global cell ID].
    """
    _validate_config(kind, config)
    if batch_size is not None and (
        isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
    ):
        raise ValueError('batch_size must be a positive integer')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'config.json', {'kind': kind, **config})
    write_json(root / 'execution.json', {'batch_size': batch_size})
    start = config.get('trial_start', 0)
    for cell_id, cell in selected_cells(kind, config):
        p = zipf(cell['d'], cell['alpha'])
        entropy = float(-np.dot(p[p > 0], np.log2(p[p > 0])))
        np.savez_compressed(root / f'target-{cell_id:04d}.npz', probabilities=p)
        rng = np.random.default_rng([config['seed'], cell_id])
        for _ in range(start):
            rng.multinomial(cell['n'], p)
        k_n = expected_discovered(p, cell['n'])
        # Persist the entire common sample set first. A failed numerical
        # evaluation leaves the exact offending profile available for diagnosis
        # and an independently implemented evaluator.
        samples_path = root / f'samples-{cell_id:04d}.jsonl.gz'
        with gzip.open(samples_path, 'xt', encoding='utf8') as samples:
            for trial in range(start, start + config['trials']):
                m = rng.multinomial(cell['n'], p)
                occupied = np.flatnonzero(m)
                profile = tuple(sorted(map(int, m[occupied]), reverse=True))
                draw = {**cell, 'cell_id': cell_id, 'trial': trial,
                        'occupied_labels': occupied.tolist(), 'counts': m[occupied].tolist(),
                        'profile': profile}
                samples.write(json.dumps(draw, allow_nan=False) + '\n')
        path = root / f'trials-{cell_id:04d}.jsonl.gz'
        with gzip.open(samples_path, 'rt', encoding='utf8') as samples, gzip.open(path, 'xt', encoding='utf8') as stream:
            draws = (json.loads(line) for line in samples)
            if batch_size is None:
                evaluated = (
                    (draw, evaluator.evidence_at_depths(cell['d'], tuple(draw['profile']), cell['depths']))
                    for draw in draws
                )
            else:
                from .batch_depth import iter_evidence_batches

                evaluated = iter_evidence_batches(
                    evaluator, ((draw, draw['profile']) for draw in draws),
                    d=cell['d'], depths=cell['depths'], chunk_size=batch_size,
                )
            for draw, result in evaluated:
                profile = tuple(draw['profile'])
                logs = np.asarray(result.log_evidence)
                if logs.shape != (len(cell['depths']),) or np.any(~np.isfinite(logs)):
                    raise ArithmeticError('nonfinite or incorrectly shaped depth evidence')
                if tuple(result.depths) != tuple(cell['depths']):
                    raise ArithmeticError('evaluator returned a different depth grid/order')
                regret = -logs / (cell['n'] * LOG2) - entropy
                k = len(profile)
                naming = float((gammaln(cell['d'] + 1) - gammaln(k + 1)
                                - gammaln(cell['d'] - k + 1)) / LOG2)
                row = {**draw, 'sample_source': samples_path.name,
                       'entropy_bits': entropy, 'expected_discovered': k_n,
                       'discovered': k, 'naming_bits': naming,
                       'log_evidence_nats': logs.tolist(), 'regret_bits': regret.tolist(),
                       'mixture_regret_bits': float(-(logsumexp(logs) - math.log(len(logs)))
                                                   / (cell['n'] * LOG2) - entropy),
                       'diagnostics': result.diagnostics}
                stream.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
    summary = summarize_scaling(root)
    write_json(root / 'summary.json', summary)
    return summary


def summarize_scaling(root):
    root = Path(root)
    config = read_json(root / 'config.json')
    _validate_config(config['kind'], config)
    expected = selected_cells(config['kind'], config)
    files = sorted(root.glob('trials-*.jsonl.gz'))
    expected_names = [f'trials-{cell_id:04d}.jsonl.gz' for cell_id, _ in expected]
    if [path.name for path in files] != expected_names:
        raise ValueError('incomplete or unexpected scaling cell files')
    result = []
    start = config.get('trial_start', 0)
    for path, (cell_id, cell) in zip(files, expected, strict=True):
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != config['trials'] or sorted(r['trial'] for r in rows) != list(range(start, start + config['trials'])):
            raise ValueError(f'incomplete trial group {path.name}')
        rows.sort(key=lambda row: row['trial'])
        for row in rows:
            if row['cell_id'] != cell_id or any(row[key] != value for key, value in cell.items()):
                raise ValueError(f'trial metadata differs from declared cell in {path.name}')
            if len(row['profile']) != row['discovered'] or sum(row['profile']) != cell['n']:
                raise ValueError(f'trial counts differ from declared sample size in {path.name}')
        values = np.asarray([r['regret_bits'] for r in rows])
        mixture = np.asarray([r['mixture_regret_bits'] for r in rows])
        if values.shape != (config['trials'], len(cell['depths'])) or np.any(~np.isfinite(values)) or np.any(~np.isfinite(mixture)):
            raise ValueError(f'invalid regret arrays in {path.name}')
        r = rows[0]
        means = values.mean(axis=0)
        se = (values.std(axis=0, ddof=1) / math.sqrt(len(rows))).tolist() if len(rows) > 1 else [None] * len(cell['depths'])
        summary = {k: r[k] for k in ('cell_id', 'd', 'n', 'alpha', 'depths',
                                    'entropy_bits', 'expected_discovered')}
        summary.update(regret_mean_bits=means.tolist(), regret_se_bits=se,
                       mixture_mean_bits=float(mixture.mean()),
                       mixture_se_bits=float(mixture.std(ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else None,
                       naming_mean_bits=float(np.mean([r['naming_bits'] for r in rows])),
                       trials=len(rows))
        result.append(summary)
    return {'config': config, 'cells': result}


def report_scaling(source, output_dir):
    """Derive existing manuscript assets from saved trials; no sampling here."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    summary = summarize_scaling(source)
    if len(summary['cells']) != sum(1 for _ in cells(summary['config']['kind'], summary['config'])):
        raise ValueError('manuscript reports require the complete scaling cell grid; merge shards first')
    write_json(root / 'summary.json', summary)
    config, rows = summary['config'], summary['cells']
    if config['kind'] == 'spectrum':
        fig, axes = plt.subplots(1, len(config['panels']), squeeze=False,
                                 figsize=(4.1 * len(config['panels']), 3.7))
        for ax, panel in zip(axes[0], config['panels']):
            group = sorted((r for r in rows if r['d'] == panel['d'] and r['n'] == panel['n']),
                           key=lambda r: r['alpha'])
            x = [r['alpha'] for r in group]
            for depth in panel['show_depths']:
                ix = group[0]['depths'].index(depth)
                y = np.array([r['regret_mean_bits'][ix] for r in group])
                se = np.array([r['regret_se_bits'][ix] for r in group])
                ax.errorbar(x, np.where(y > 1e-14, y, np.nan), yerr=se,
                            lw=1, capsize=1.5, label=f'L = {depth}')
            ax.errorbar(x, [r['mixture_mean_bits'] for r in group],
                        yerr=[r['mixture_se_bits'] for r in group], color='black',
                        lw=2, label=f"mixture, L = 0,…,{panel['max_depth']}")
            ax.set(yscale='log', xlabel='Zipf exponent α of the target',
                   title=f"d = {panel['d']:,}, N = {panel['n']:,}")
            ax.legend(fontsize=6)
        axes[0, 0].set_ylabel('online regret [bits/symbol]')
        fig.tight_layout()
        fig.savefig(root / 'complexity_spectrum_L0.pdf', bbox_inches='tight')
        plt.close(fig)
    elif config['kind'] == 'depth_scaling':
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
        claims = []
        for r in rows:
            cs = np.asarray(config['c_values'])
            ix = [r['depths'].index(max(1, round(c * math.log(r['d'])))) for c in cs]
            measured = np.asarray(r['regret_mean_bits'])[ix] * r['n'] / r['expected_discovered']
            x = r['naming_mean_bits'] / r['expected_discovered']
            predicted = np.asarray([rho(c) * x + r['alpha'] / LOG2 - 2 for c in cs])
            color = {2: 'tab:blue', 3: 'tab:orange', 4: 'tab:green'}.get(r['alpha'])
            for ax, c_min in zip(axes, (0, .75)):
                keep = cs >= c_min
                ax.plot(cs[keep], measured[keep], 'o', ms=3.5, color=color,
                        label=f"measured, α = {r['alpha']:g}")
                ax.plot(cs[keep], predicted[keep], '--', lw=1.1, color=color,
                        label=f"prediction, α = {r['alpha']:g}")
            best = int(np.argmin(measured))
            at_star = int(np.argmin(np.abs(cs - C_STAR)))
            claims.append({'alpha': r['alpha'], 'best_tested_c': float(cs[best]),
                           'cost_at_nearest_c_star': float(measured[at_star]),
                           'minimum_cost': float(measured[best]),
                           'nearest_c_star_grid_point': float(cs[at_star]),
                           'offset_bits': r['alpha'] / LOG2 - 2})
        for ax, title in zip(axes, ('Full depth range', 'Logarithmic-depth range')):
            ax.axvline(C_STAR, color='.4', ls=':', lw=1)
            ax.set(xlabel='depth coefficient c', title=title)
        axes[0].set_ylabel('cost per discovered symbol [bits]')
        axes[1].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(root / 'depth_scaling.pdf', bbox_inches='tight')
        plt.close(fig)
        write_json(root / 'depth-claims.json', claims)
    else:
        alphabet, data = [], []
        for alpha in config['alphas']:
            for n in config['table_ns']:
                group = sorted((r for r in rows if r['alpha'] == alpha and r['n'] == n),
                               key=lambda r: r['d'])
                x = [r['naming_mean_bits'] / r['expected_discovered'] for r in group]
                y = [r['n'] * r['regret_mean_bits'][0] / r['expected_discovered'] for r in group]
                slope, offset = np.polyfit(x, y, 1)
                alphabet.append({'alpha': alpha, 'n': n, 'slope': float(slope),
                                     'offset_bits': float(offset),
                                     'slope_one_offset_bits': float(np.mean(np.subtract(y, x)))})
            for d in config['table_ds']:
                group = sorted((r for r in rows if r['alpha'] == alpha and r['d'] == d),
                               key=lambda r: r['n'])
                y = np.asarray([r['regret_mean_bits'][0] for r in group])
                if np.any(y <= 0):
                    data.append({'alpha': alpha, 'd': d, 'status': 'nonpositive_sample_mean', 'exponent': None})
                else:
                    slope, _ = np.polyfit(np.log([r['n'] for r in group]), np.log(y), 1)
                    data.append({'alpha': alpha, 'd': d, 'status': 'defined', 'exponent': float(-slope)})
        write_json(root / 'scaling-tables.json', {'alphabet': alphabet, 'data': data})
        # Preserve the manuscript's two juxtaposed tables, with one row per
        # alpha rather than introducing separate rows for every N/d setting.
        def tex_integer(value):
            exponent = round(math.log10(value))
            return f'10^{{{exponent}}}' if value == 10 ** exponent else str(value)

        table_ns, table_ds = config['table_ns'], config['table_ds']
        alphabet_lookup = {(r['alpha'], r['n']): r for r in alphabet}
        data_lookup = {(r['alpha'], r['d']): r for r in data}
        lines = ['% Labels: tab:alphabet and tab:data. Generated from saved trial records.',
                 '% Slopes and intercepts are free OLS fits; offsets are in bits.',
                 r'\begin{tabular}{c' + 'r' * (2 * len(table_ns)) + '}',
                 r'\toprule',
                 '& ' + rf'\multicolumn{{{len(table_ns)}}}{{c}}{{slope}} & '
                 + rf'\multicolumn{{{len(table_ns)}}}{{c}}{{offset $B$ (bits)}}' + r' \\',
                 r'$\alpha$ & ' + ' & '.join(
                     f'$N={tex_integer(n)}$' for n in [*table_ns, *table_ns]) + r' \\',
                 r'\midrule']
        for alpha in config['alphas']:
            row_cells = [alphabet_lookup[alpha, n] for n in table_ns]
            values = [f"{r['slope']:.2f}" for r in row_cells] + [f"{r['offset_bits']:.2f}" for r in row_cells]
            lines.append(f'{alpha:g} & ' + ' & '.join(values) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\hfill',
                  r'\begin{tabular}{cc' + 'r' * len(table_ds) + '}', r'\toprule',
                  '& asymptotic & ' + rf'\multicolumn{{{len(table_ds)}}}{{c}}{{measured exponent}}' + r' \\',
                  r'$\alpha$ & $1-1/\alpha$ & ' + ' & '.join(f'$d={tex_integer(d)}$' for d in table_ds) + r' \\',
                  r'\midrule']
        for alpha in config['alphas']:
            values = []
            for d in table_ds:
                exponent = data_lookup[alpha, d]['exponent']
                values.append(f'{exponent:.3f}' if exponent is not None else 'n.a.')
            lines.append(f'{alpha:g} & {1-1/alpha:.3f} & ' + ' & '.join(values) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        (root / 'scaling-tables.tex').write_text('\n'.join(lines) + '\n')
    return summary
