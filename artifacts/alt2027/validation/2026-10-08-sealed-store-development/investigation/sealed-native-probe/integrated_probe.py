"""Read-only actual production API parity/timing after explicit local compilation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,platform,time
from pathlib import Path
from collections import defaultdict
from statistics import median
import numpy as np
from lsa.alt.sealed_tables import SealedKernelTables
from lsa.alt.sealed_native import NativeInterpolator
from lsa.alt.depth import DepthEvaluator,StoreConfig
from lsa.alt.sealed_profile_validation import compare_profiles
REPO=Path('/Users/ruediger/Documents/Codex/2026-10-06/i-ha/work/github-cleanup')
HERE=Path(__file__).resolve().parent
PILOT=REPO/'output/alt2027/sealed-profile-pilot-20261008-001'
KERNEL=REPO/'output/alt2027/sealed-pilot-validation-20261008-001/data/resolved-cases.json'
OPTIONS=json.loads((REPO/'output/alt2027/engine-sealed-pilot-20261008-001.json').read_text())
BINARY=HERE/'production-native-001.so'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact(a,b):return np.asarray(a).shape==np.asarray(b).shape and np.asarray(a).dtype==np.asarray(b).dtype and np.asarray(a).tobytes()==np.asarray(b).tobytes()
sources=['src/lsa/alt/sealed_tables.py','src/lsa/alt/sealed_native.py','src/lsa/alt/sealed_interp.c','src/lsa/alt/depth.py']
report={'scope':'actual explicit production NativeInterpolator, SealedKernelTables and DepthEvaluator APIs; no source/store mutation','platform':platform.platform(),'source_sha256':{p:sha(REPO/p) for p in sources},'script_sha256':sha(__file__),'kernel_cases_sha256':sha(KERNEL),'kernels':[],'profiles':[]}
native=NativeInterpolator(BINARY,sha(BINARY));report['native_identity']=native.identity
report['store_plan_sha256']=sha(Path(OPTIONS['store']['path'])/'plan.json')
report['store_manifest_sha256']=sha(Path(OPTIONS['store']['path'])/'manifest.json')
def save():(HERE/'integrated-results.json').write_text(json.dumps(report,indent=2)+'\n')
groups=defaultdict(list)
for c in json.loads(KERNEL.read_text()):groups[c['depth']].append(c)
with SealedKernelTables(OPTIONS['store']['path']) as python_table, SealedKernelTables(OPTIONS['store']['path'],native_interpolator=native) as native_table:
    for L,cases in groups.items():
        rs=sorted({c['r'] for c in cases});u=sorted({c['u'] for c in cases})
        py=python_table.log_phi_matrix(L,rs,u);c=native_table.log_phi_matrix(L,rs,u)
        pys=np.array([python_table.log_phi(L,x['r'],[x['u']])[0] for x in cases]);cs=np.array([native_table.log_phi(L,x['r'],[x['u']])[0] for x in cases])
        row={'L':L,'declared_cases':len(cases),'batched_values':int(py.size),'matrix_byte_identical':exact(py,c),'scalar_byte_identical':exact(pys,cs),'maximum_difference_nats':float(max(np.abs(py-c).max(),np.abs(pys-cs).max()))}
        report['kernels'].append(row);save();print(json.dumps(row),flush=True)
python_options=dict(OPTIONS['store']);native_options={**python_options,'interpolation_backend':'native','native_library_path':str(BINARY),'native_library_sha256':sha(BINARY)}
with DepthEvaluator(mode='store',store=StoreConfig(**python_options),prediction_tolerance=OPTIONS['prediction_tolerance']) as python_engine, DepthEvaluator(mode='store',store=StoreConfig(**native_options),prediction_tolerance=OPTIONS['prediction_tolerance']) as native_engine:
    report['engine_native_identity']=native_engine.configuration.get('sealed_native_identity')
    summary=json.loads((PILOT/'data/summary.json').read_text())
    for i,case in enumerate(summary['cases']):
        with np.load(PILOT/f'data/case-{i:03d}.npz') as data:counts=data['counts'];target=data['target']
        parts=tuple(int(x) for x in counts if x);d=case['case']['d'];depths=case['case']['depths']
        ref=python_engine.prediction_by_count(d,parts,depths=depths);native_engine.prediction_by_count(d,parts,depths=depths)
        mass=[float(target[counts==c].sum()) for c in ref.counts]
        result={'id':case['id'],'sample_sha256':sha(PILOT/f'data/case-{i:03d}.npz'),'depths':depths,'runs':[]}
        for rep in range(3):
            for tag,engine in [('python',python_engine),('native',native_engine)]:
                before=time.perf_counter();value=engine.prediction_by_count(d,parts,depths=depths);seconds=time.perf_counter()-before
                row={'implementation':tag,'repeat':rep,'seconds':seconds,'byte_identical':all(exact(getattr(ref,k),getattr(value,k)) for k in ('component_log_evidence','component_probabilities','mixture_probabilities','posterior')),'metrics':compare_profiles(ref,value,n=case['case']['n'],class_mass=mass,multiplicities=ref.multiplicities)}
                result['runs'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
        result['median_seconds']={tag:median(x['seconds'] for x in result['runs'] if x['implementation']==tag) for tag in ['python','native']}
        result['speedup']=result['median_seconds']['python']/result['median_seconds']['native'];report['profiles'].append(result);save()
report['source_unchanged']={p:sha(REPO/p)==v for p,v in report['source_sha256'].items()}
report['all_byte_identical']=all(x['matrix_byte_identical'] and x['scalar_byte_identical'] for x in report['kernels']) and all(x['byte_identical'] for p in report['profiles'] for x in p['runs'])
save();print(json.dumps({'all_byte_identical':report['all_byte_identical'],'sources_unchanged':all(report['source_unchanged'].values()),'medians':[p['median_seconds'] for p in report['profiles']]}),flush=True)
