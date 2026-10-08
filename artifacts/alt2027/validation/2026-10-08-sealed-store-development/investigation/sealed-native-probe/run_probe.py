"""Standalone native diagnostic: no production source/table mutation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import ctypes,hashlib,importlib.util,json,math,platform,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.special import loggamma
from lsa.alt.depth import DepthEvaluator,StoreConfig
from lsa.alt import sealed_tables as production_reader
from lsa.alt.sealed_profile_validation import compare_profiles

HERE=Path(__file__).resolve().parent
REPO=Path('/Users/ruediger/Documents/Codex/2026-10-06/i-ha/work/github-cleanup')
PILOT=REPO/'output/alt2027/sealed-profile-pilot-20261008-001'
KERNEL=REPO/'output/alt2027/sealed-pilot-validation-20261008-001/data/resolved-cases.json'
OPTIONS=json.loads((REPO/'output/alt2027/engine-sealed-pilot-20261008-001.json').read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
copy_hash=digest(HERE/'reader.py')
if copy_hash!=digest(REPO/'src/lsa/alt/sealed_tables.py'):raise RuntimeError('reader changed before probe')
spec=importlib.util.spec_from_file_location('lsa.alt._sealed_native_reference',HERE/'reader.py')
reader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reader;spec.loader.exec_module(reader)
Reader=reader.SealedKernelTables
original_anchor=Reader._anchor_values
lib=ctypes.CDLL(str(HERE/'interpolate.so'))
fn=lib.sealed_interp_column
P=ctypes.c_void_p;I=ctypes.c_int64;D=ctypes.c_double
fn.argtypes=[P,I,P,I,D,D,D,D,D,P,P];fn.restype=ctypes.c_int

def native_anchor(self,L,r,u,columns,data):
    col=columns[r]
    vals=data[col.offset:col.offset+col.length]
    query=np.ascontiguousarray(u,dtype=np.float64)
    out=np.empty(len(query),dtype=np.float64)
    failure=np.array([-1],dtype=np.int64)
    status=fn(vals.ctypes.data,len(vals),query.ctypes.data,len(query),col.u_min,
              self.grid_step,self.maximum_u,L*math.log(r+1.),L*float(loggamma(r+1.)),
              out.ctypes.data,failure.ctypes.data)
    if status:raise reader.SealedTableError(f'native interpolation rejected status={status}, L={L},r={r},query={int(failure[0])}')
    return out

def compare(a,b):
    a,b=np.asarray(a),np.asarray(b)
    difference=np.abs(a-b)
    return {'bit_identical':bool(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()),'values':int(a.size),'unequal_values':int(np.count_nonzero(a!=b)),'maximum_absolute_difference_nats':float(difference.max(initial=0.))}

def save():
    (HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')

report={'scope':'DIAGNOSTIC ONLY: copied final Python reader with per-anchor C vector interpolation replacing only _anchor_values; same physical float64 nodes, centered values and Neumaier summation; no approximate fallback','platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'compiler':'Apple clang21.0.0 clang-2100.3.34.2','flags':['-O3','-std=c11','-ffp-contract=off','-fno-fast-math','-fPIC','-shared'],'source_sha256':{'copied_reader':copy_hash,'native_c':digest(HERE/'interpolate.c'),'library':digest(HERE/'interpolate.so'),'probe_script':digest(__file__),'depth.py':digest(REPO/'src/lsa/alt/depth.py')},'store_plan_sha256':digest(Path(OPTIONS['store']['path'])/'plan.json'),'store_manifest_sha256':digest(Path(OPTIONS['store']['path'])/'manifest.json'),'kernel_cases_sha256':digest(KERNEL),'kernel_groups':[],'profiles':[]}
cases=json.loads(KERNEL.read_text());groups=defaultdict(list)
for case in cases:groups[case['depth']].append(case)
# Validate both scalar and rectangular batched call shapes. Each query retains
# exactly the originally declared float64 u value and count/depth identity.
with Reader(OPTIONS['store']['path']) as table:
    for L,group in groups.items():
        rs=sorted({c['r'] for c in group});us=sorted({c['u'] for c in group})
        result={'depth':L,'declared_cases':len(group),'batched_shape':[len(rs),len(us)]}
        start=time.perf_counter();Reader._anchor_values=original_anchor
        py_matrix=table.log_phi_matrix(L,rs,us)
        py_scalar=np.array([table.log_phi(L,c['r'],[c['u']])[0] for c in group])
        result['python_seconds']=time.perf_counter()-start
        start=time.perf_counter();Reader._anchor_values=native_anchor
        c_matrix=table.log_phi_matrix(L,rs,us)
        c_scalar=np.array([table.log_phi(L,c['r'],[c['u']])[0] for c in group])
        result['native_seconds']=time.perf_counter()-start
        result['matrix_comparison']=compare(py_matrix,c_matrix)
        result['scalar_comparison']=compare(py_scalar,c_scalar)
        ri={r:i for i,r in enumerate(rs)};ui={u:i for i,u in enumerate(us)}
        select=np.array([c_matrix[ri[c['r']],ui[c['u']]] for c in group])
        result['native_scalar_vs_matrix']=compare(c_scalar,select)
        report['kernel_groups'].append(result);save();print(json.dumps(result),flush=True)
        if not result['matrix_comparison']['bit_identical'] or not result['scalar_comparison']['bit_identical']:
            np.savez_compressed(HERE/f'kernel-differences-{L}.npz',python_matrix=py_matrix,native_matrix=c_matrix,python_scalar=py_scalar,native_scalar=c_scalar,rs=rs,us=us)
Reader._anchor_values=original_anchor
# Keep the copied reader active inside the normal evaluator; only its private
# u-stencil is substituted during timed native repetitions. Construction and
# verification are excluded from warm timings and remain visible separately.
production_class=production_reader.SealedKernelTables
production_reader.SealedKernelTables=Reader
summary=json.loads((PILOT/'data/summary.json').read_text())
try:
    start=time.perf_counter()
    with DepthEvaluator(mode='store',store=StoreConfig(**OPTIONS['store']),prediction_tolerance=OPTIONS['prediction_tolerance']) as engine:
        report['engine_open_seconds']=time.perf_counter()-start
        for index,case in enumerate(summary['cases']):
            with np.load(PILOT/f'data/case-{index:03d}.npz') as sample:
                counts=sample['counts'];target=sample['target']
            parts=tuple(int(x) for x in counts if x)
            d=case['case']['d'];depths=case['case']['depths']
            Reader._anchor_values=original_anchor
            start=time.perf_counter();baseline=engine.prediction_by_count(d,parts,depths=depths)
            result={'id':case['id'],'sample_sha256':digest(PILOT/f'data/case-{index:03d}.npz'),'depths':depths,'first_python_seconds':time.perf_counter()-start,'warm_runs':[]}
            class_mass=[float(target[counts==c].sum()) for c in baseline.counts]
            for repeat in range(3):
                for tag,implementation in [('python',original_anchor),('native',native_anchor)]:
                    Reader._anchor_values=implementation
                    start=time.perf_counter();value=engine.prediction_by_count(d,parts,depths=depths)
                    seconds=time.perf_counter()-start
                    arrays=('component_log_evidence','component_probabilities','mixture_probabilities','posterior')
                    row={'implementation':tag,'repeat':repeat,'seconds':seconds,'bit_identical_to_python':all(np.asarray(getattr(value,key)).tobytes() == np.asarray(getattr(baseline,key)).tobytes() for key in arrays),'measurements':compare_profiles(baseline,value,n=case['case']['n'],class_mass=class_mass,multiplicities=baseline.multiplicities)}
                    result['warm_runs'].append(row)
                    print(json.dumps({'profile':case['id'],'implementation':tag,'repeat':repeat,'seconds':seconds,'bit_identical':row['bit_identical_to_python']}),flush=True)
            report['profiles'].append(result);save()
finally:
    Reader._anchor_values=original_anchor
    production_reader.SealedKernelTables=production_class
report['production_reader_unchanged']=digest(REPO/'src/lsa/alt/sealed_tables.py')==copy_hash
report['depth_source_unchanged']=digest(REPO/'src/lsa/alt/depth.py')==report['source_sha256']['depth.py']
report['declared_kernel_cases']=len(cases)
report['all_kernel_scalar_values_identical']=all(x['scalar_comparison']['bit_identical'] for x in report['kernel_groups'])
report['all_kernel_batched_values_identical']=all(x['matrix_comparison']['bit_identical'] for x in report['kernel_groups'])
report['all_profile_values_identical']=all(x['bit_identical_to_python'] for p in report['profiles'] for x in p['warm_runs'])
save()
print(json.dumps({k:report[k] for k in ['declared_kernel_cases','all_kernel_scalar_values_identical','all_kernel_batched_values_identical','all_profile_values_identical','production_reader_unchanged','depth_source_unchanged']}),flush=True)
