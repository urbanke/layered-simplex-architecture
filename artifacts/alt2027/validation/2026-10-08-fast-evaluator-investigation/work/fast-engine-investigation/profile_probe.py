"""Diagnostic comparison of direct and reusable high-depth kernel columns.

Uses a private in-process provider replacement, never production admission.
The same profiles, depth grid and integration settings are used for both paths.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import time

import numpy as np
from scipy.special import gammaln
from lsa.alt.depth import DepthEvaluator, StoreConfig
from lsa.alt._vendor.pmwm import universal_tables as ut

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / 'github-cleanup'
OUT = ROOT / 'profile-probe-002'
OUT.mkdir(exist_ok=False)
cfg = json.loads((REPO / 'output/alt2027/engine-candidate.json').read_text())
cache = {}
build_seconds = 0.0
old_provider = ut._saddle_row

def pretabulated(L, r, u):
    global build_seconds
    key = (int(L), int(r))
    if key not in cache:
        lo = math.floor((-60 - L * math.log(r + 1)) / ut.H) * ut.H
        size = int(round((80 - lo) / ut.H)) + 1
        grid = lo + np.arange(size) * ut.H
        start = time.perf_counter()
        vals = old_provider(L, r, grid)
        build_seconds += time.perf_counter() - start
        cache[key] = grid, vals
    grid, vals = cache[key]
    u = np.asarray(u)
    if np.max(u) > grid[-1] + 1e-9:
        raise ValueError('diagnostic lookup exceeds prepared upper bound')
    out = np.full(u.shape, L * float(gammaln(r + 1.0)))
    use = u >= grid[0]
    # Below the column: the first omitted relative moment term <=exp(-60).
    if np.any(~use):
        assert np.max(u[~use] + L * math.log(r + 1)) <= -60 + 1e-9
    out[use] = ut._interp_column(grid, vals, u[use])
    return out

def loss_changes(a, b, p, counts):
    classes = np.asarray(a.counts)
    weights = np.asarray([p[counts == c].sum() for c in classes])
    mult = np.asarray(a.multiplicities)
    ac = a.component_probabilities / (a.component_probabilities @ mult)[:, None]
    bc = b.component_probabilities / (b.component_probabilities @ mult)[:, None]
    am = a.mixture_probabilities / (a.mixture_probabilities @ mult)
    bm = b.mixture_probabilities / (b.mixture_probabilities @ mult)
    return {
        'component_kl_change_bits': (np.log2(ac / bc) @ weights).tolist(),
        'mixture_kl_change_bits': float(np.log2(am / bm) @ weights),
        'component_total_evidence_change_bits': ((b.component_log_evidence - a.component_log_evidence) / math.log(2)).tolist(),
        'direct_raw_mass_discrepancy': a.diagnostics['maximum_normalization_error'],
        'candidate_raw_mass_discrepancy': b.diagnostics['maximum_normalization_error'],
        'max_absolute_component_probability_change': float(np.max(np.abs(a.component_probabilities - b.component_probabilities))),
    }

rows = []
with DepthEvaluator(mode='store', store=StoreConfig(**cfg['store']),
                    prediction_tolerance=cfg['prediction_tolerance']) as evaluator:
    for case, d, n, alpha in [('zipf1p5-n1000',10000,1000,1.5), ('zipf5-n20000',10000,20000,5.0)]:
        rng = np.random.default_rng(2026100802)
        p = np.arange(1, d+1, dtype=float) ** -alpha
        p /= p.sum()
        counts = rng.multinomial(n, p)
        parts = tuple(sorted((int(x) for x in counts if x), reverse=True))
        start = time.perf_counter()
        direct = evaluator.prediction_by_count(d, parts, depths=[0,1,54,80,138])
        direct_time = time.perf_counter() - start
        before = build_seconds
        ut._saddle_row = pretabulated
        try:
            start = time.perf_counter()
            cold = evaluator.prediction_by_count(d, parts, depths=[0,1,54,80,138])
            cold_time = time.perf_counter() - start
            start = time.perf_counter()
            warm = evaluator.prediction_by_count(d, parts, depths=[0,1,54,80,138])
            warm_time = time.perf_counter() - start
        finally:
            ut._saddle_row = old_provider
        row = {
            'case':case,'d':d,'N':n,'alpha':alpha,'depths':[0,1,54,80,138],
            'observed_count_classes':len(Counter(parts)),'seed':2026100802,
            'direct_seconds':direct_time,'cold_candidate_seconds':cold_time,
            'additional_column_build_seconds':build_seconds-before,
            'warm_candidate_seconds':warm_time,'warm_speed_ratio':direct_time/warm_time,
            'comparison':loss_changes(direct,warm,p,counts),
            'cold_warm_identical':bool(np.array_equal(cold.component_log_evidence,warm.component_log_evidence)
                                      and np.array_equal(cold.component_probabilities,warm.component_probabilities)),
        }
        f=OUT/(case+'.npz')
        np.savez_compressed(f, target=p,counts=counts,classes=direct.counts,
            direct_log_evidence=direct.component_log_evidence,candidate_log_evidence=warm.component_log_evidence,
            direct_component_predictions=direct.component_probabilities,
            candidate_component_predictions=warm.component_probabilities,
            direct_mixture=direct.mixture_probabilities,candidate_mixture=warm.mixture_probabilities)
        row['artifact_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
        rows.append(row)
        print(json.dumps(row),flush=True)

report={
 'purpose':'Diagnostic only; two profile comparisons using unchanged sample/model/integration definitions',
 'rows':rows,'prepared_columns':len(cache),'column_bytes':sum(g.nbytes+v.nbytes for g,v in cache.values()),
 'cumulative_build_seconds':build_seconds,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'production_eligible':False,
 'limits':['Selected five-depth diagnostic; not full 81/139-depth benchmark or production gate.',
           'Cached values prepared for actual count rows; no count-anchor interpolation is exercised.',
           'Warm timing excludes one-time preparation, which is reported separately.',
           'Shares the accurate direct kernel evaluator; compares added interpolation effects.'],
}
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
