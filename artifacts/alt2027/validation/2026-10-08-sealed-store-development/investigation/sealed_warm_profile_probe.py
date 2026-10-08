"""Read-only repeated serving and phase timing on the saved paired pilot draws."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import json,time,hashlib,gzip,platform
from pathlib import Path
import numpy as np
from lsa.alt.depth import DepthEvaluator, StoreConfig
repo=Path('/Users/ruediger/Documents/Codex/2026-10-06/i-ha/work/github-cleanup')
base=repo/'output/alt2027/sealed-profile-pilot-20261008-001'
config=json.loads((repo/'output/alt2027/engine-sealed-pilot-20261008-001.json').read_text())
summary=json.loads((base/'data/summary.json').read_text())
paths=[repo/'src/lsa/alt/depth.py',repo/'src/lsa/alt/sealed_tables.py',repo/'src/lsa/alt/_vendor/pmwm/layered.py']
source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
report={'scope':'single-process, single-thread, read-only warm serving of exactly saved two pilot profiles; outer settings unchanged','platform':platform.platform(),'source_sha256':source,'pilot_summary_sha256':hashlib.sha256((base/'data/summary.json').read_bytes()).hexdigest(),'cases':[]}
start=time.perf_counter()
with DepthEvaluator(mode='store',store=StoreConfig(**config['store']),prediction_tolerance=config['prediction_tolerance']) as engine:
    report['engine_open_seconds']=time.perf_counter()-start
    stats={}
    def wrap(name):
        original=getattr(engine._store,name)
        def measured(*args,**kwargs):
            t=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                row=stats.setdefault(name,{'calls':0,'seconds':0.})
                row['calls']+=1;row['seconds']+=time.perf_counter()-t
        setattr(engine._store,name,measured)
    for method in ('_load_level','_check_open','_anchor_values','log_phi_matrix'):wrap(method)
    for i,old in enumerate(summary['cases']):
        with np.load(base/f'data/case-{i:03d}.npz') as saved:counts=saved['counts']
        with gzip.open(base/f'data/evaluation-{i:03d}.json.gz','rt') as f:pilot=json.load(f)['candidate']
        parts=tuple(int(x) for x in counts if x)
        result={'id':old['id'],'sample_sha256':hashlib.sha256((base/f'data/case-{i:03d}.npz').read_bytes()).hexdigest(),'runs':[]}
        first=None
        for repeat in range(3):
            stats.clear();t=time.perf_counter()
            got=engine.prediction_by_count(old['case']['d'],parts,depths=old['case']['depths'])
            sec=time.perf_counter()-t
            values=[got.component_log_evidence,got.component_probabilities,got.mixture_probabilities,got.posterior]
            if first is None:first=[x.copy() for x in values]
            row={'repeat':repeat,'seconds':sec,'phase_inclusive_timings':dict(stats),'identical_to_first':all(np.array_equal(a,b) for a,b in zip(values,first)),'identical_to_saved_pilot':all(np.array_equal(x,np.asarray(pilot[key])) for x,key in zip(values,['component_log_evidence','component_probabilities','mixture_probabilities','posterior']))}
            result['runs'].append(row);print(json.dumps({'case':old['id'],**row}),flush=True)
        report['cases'].append(result)
report['sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source.items())
report['timing_note']='instrumented method totals are inclusive/overlap; do not add _check_open/_load_level/_anchor_values to log_phi_matrix'
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
