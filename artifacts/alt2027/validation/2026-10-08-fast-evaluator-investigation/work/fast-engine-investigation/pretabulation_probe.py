"""Bounded diagnostic only: reuse accurate high-depth columns by interpolation.

No experiment source, protocol, production store, or production output is changed.
"""
from pathlib import Path
import hashlib
import json
import math
import platform
import statistics
import time

import numpy as np
from scipy.special import gammaln
from lsa.alt._vendor.pmwm import _runtime, mellin, universal_tables as ut

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'pretabulation-probe-001'
OUT.mkdir(exist_ok=False)
settings_token = _runtime._settings.set({'PMM_BUILD_EXACT': '1'})
rng = np.random.default_rng(2026100801)
rows = []
try:
    for L in (54, 80, 138):
        for r in (0, 3, 32, 1000):
            # The column spans an outer-plot range; this diagnostic checks
            # interpolation on off-grid queries, not a production acceptance gate.
            u0 = math.floor((-60.0 - L * math.log(r + 1.0)) / ut.H) * ut.H
            count = int(round((80.0 - u0) / ut.H)) + 1
            grid = u0 + np.arange(count, dtype=float) * ut.H
            start = time.perf_counter()
            vals = mellin.exact_log_phi_column(float(r), L, grid,
                oversample=8.0, series_tolerance=1e-13, contour_tail_nats=40.0)
            build_seconds = time.perf_counter() - start
            # Query positions cover the entire column, its transition, and edges.
            query = np.sort(np.concatenate([
                rng.uniform(grid[0] + .08, grid[-1] - .08, 1600),
                grid[0] + np.array([0., .001, .011, .021, .079]),
                grid[-1] - np.array([0., .001, .011, .021, .079]),
            ]))
            start = time.perf_counter()
            reference = mellin.exact_log_phi_column(float(r), L, query,
                oversample=8.0, series_tolerance=1e-13, contour_tail_nats=40.0)
            direct_seconds = time.perf_counter() - start
            times = []
            for _ in range(7):
                start = time.perf_counter()
                interpolated = ut._interp_column(grid, vals, query)
                times.append(time.perf_counter() - start)
            interpolation_seconds = statistics.median(times)
            delta = np.abs(interpolated - reference)
            f = OUT / f'L{L}-r{r}.npz'
            np.savez_compressed(f, grid=grid, values=vals, query=query,
                                direct=reference, interpolated=interpolated)
            row = {
                'L': L, 'r': r, 'grid_spacing': ut.H,
                'grid_bounds': [float(grid[0]), float(grid[-1])],
                'stored_values': len(grid), 'query_values': len(query),
                'one_time_build_seconds': build_seconds,
                'direct_query_seconds': direct_seconds,
                'interpolation_query_median_seconds': interpolation_seconds,
                'query_speed_ratio': direct_seconds / interpolation_seconds,
                'max_log_kernel_difference_nats': float(delta.max()),
                'worst_query_u': float(query[int(delta.argmax())]),
                'artifact': f.name, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
finally:
    _runtime._settings.reset(settings_token)

summary = {
    'purpose': 'Diagnostic only; interpolation overhead and off-grid agreement on 12 high-depth columns',
    'rows': rows,
    'source_sha256': {str(Path(m.__file__)): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (mellin, ut)},
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'runtime': {'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__},
    'seed': 2026100801,
    'limits': [
        'Direct-column comparison checks added interpolation discrepancy, not independent absolute accuracy.',
        'Kernel-only query timing excludes store build and downstream profile integration.',
        'No count-anchor interpolation or end-to-end prediction validation is performed here.',
        'This diagnostic does not authorize production admission.',
    ],
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
