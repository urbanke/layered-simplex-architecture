import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PMM_KERNEL']='1'
os.environ['PMM_PHI_LADDER_EVERY']='1'
os.environ['PMM_PHI_LADDER_DEGREE']='11'
os.environ['PMM_PHI_SADDLE_MIN_L']='54'
import hashlib,json,math,platform,shutil,statistics,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
UP=Path('/Users/ruediger/Projects/product_model_with_memory')
ALT=Path('/Users/ruediger/Documents/Codex/2026-10-06/i-ha/work/github-cleanup')
COPIES=BASE/'copies'
files=[]
def copy(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    files.append({'source':str(source),'copy':str(target),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
for name in ('kernel.py','_kernel.c','mellin.py','universal_tables.py'):
    copy(UP/'src/product_model_with_memory'/name,COPIES/'product_model_with_memory'/name)
(COPIES/'product_model_with_memory/__init__.py').write_text('')
for name in ('kernel.py','mellin.py','universal_tables.py','_runtime.py'):
    copy(ALT/'src/lsa/alt/_vendor/pmwm'/name,COPIES/'lsa/alt/_vendor/pmwm'/name)
for name in ('lsa','lsa/alt','lsa/alt/_vendor','lsa/alt/_vendor/pmwm'):
    (COPIES/name/'__init__.py').write_text('')
sys.path.insert(0,str(COPIES))
import numpy as np
import scipy
from scipy.special import loggamma
import product_model_with_memory.kernel as K
import product_model_with_memory.universal_tables as U
from lsa.alt._vendor.pmwm import universal_tables as V, _runtime
if K.interp_column is None: raise RuntimeError('isolated native compile/load failed')
store=UP/'tables/anchors_prod'
# Only already-existing sealed files are read, never provision/build columns.
su=U.UniversalTables(store,read_only=True)
sv=V.UniversalTables(store,read_only=True)
for mod in (U,V):
    mod._SADDLE_MIN_L=54; mod._LADDER_F=0.;mod._LADDER_EVERY=1;mod._LADDER_DEGREE=11
    mod.SERIES_TAIL_NATS=math.inf
count={'calls':0}
def spy(*args):
    count['calls']+=1
    return K.interp_column(*args)
U._INTERP=spy
V._INTERP=spy
rs=[0,1,2,3,17,251,256,300,4001,67465,200000]
report={'description':'DIAGNOSTIC ONLY: stored row interpolation, no model evaluation, single thread, read-only production store', 'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'numpy':np.__version__,'scipy':scipy.__version__,'sources':files,'compiled_library':str(K._library_path()),'compiled_library_sha256':hashlib.sha256(K._library_path().read_bytes()).hexdigest(),'flags':K.CFLAGS,'rs':rs,'repeats':3,'cases':[]}
def timed(fn):
    vals=[]; result=None
    for _ in range(3):
        t=time.perf_counter();result=fn();vals.append(time.perf_counter()-t)
    return result,{'seconds':vals,'median_seconds':statistics.median(vals)}
def compare(a,b):
    d=np.abs(a-b)
    pos=np.unravel_index(d.argmax(),d.shape)
    return {'bit_identical':bool(np.array_equal(a,b)),'unequal_values':int(np.count_nonzero(a!=b)),'total_values':int(a.size),'max_absolute_difference_nats':float(d[pos]),'worst_index':[int(x) for x in pos]}
for L in (2,26,53):
    lo=min(-70.,-L*math.log(max(rs)+1.)-40.)
    u=np.linspace(lo,35.,math.ceil((35.-lo)/.02)+1)
    case={'L':L,'grid_min':float(u[0]),'grid_max':float(u[-1]),'grid_points':len(u),'grid_step':float(u[1]-u[0])}
    # Compare copied frozen ALT route with native symbol available/on vs disabled.
    token=_runtime._settings.set({'PMM_INTERP_KERNEL':'0'})
    a,case['alt_ladder_native_off']=timed(lambda:sv.log_phi_matrix(L,rs,u))
    _runtime._settings.reset(token)
    token=_runtime._settings.set({'PMM_INTERP_KERNEL':'1'})
    count['calls']=0
    b,case['alt_ladder_native_on']=timed(lambda:sv.log_phi_matrix(L,rs,u))
    case['alt_ladder_native_on']['native_calls']=count['calls']
    case['alt_toggle_values']=compare(a,b)
    _runtime._settings.reset(token)
    count['calls']=0
    c,case['upstream_ladder_native_on']=timed(lambda:su.log_phi_matrix(L,rs,u))
    case['upstream_ladder_native_on']['native_calls']=count['calls']
    case['upstream_vs_alt']=compare(a,c)
    per_r=[sv.ladder_anchors_for(L,r) for r in rs]
    anchors=sorted(set(x for row in per_r for x in row))
    case['unique_anchor_count']=len(anchors)
    case['anchors']=anchors
    # Compare existing batched native route against existing batched NumPy route.
    V._LADDER_EVERY=0
    token=_runtime._settings.set({'PMM_INTERP_KERNEL':'0'})
    pure,case['anchor_matrix_numpy']=timed(lambda:sv.log_phi_matrix(L,anchors,u))
    _runtime._settings.reset(token)
    token=_runtime._settings.set({'PMM_INTERP_KERNEL':'1'})
    count['calls']=0
    native,case['anchor_matrix_native']=timed(lambda:sv.log_phi_matrix(L,anchors,u))
    case['anchor_matrix_native']['native_calls']=count['calls']
    case['native_vs_batched_numpy']=compare(pure,native)
    V._LADDER_EVERY=1
    # Prototype ONLY in benchmark function: keep identical count interpolation arithmetic;
    # obtain each unique anchor through native matrix instead of repeated scalar method.
    def prototype():
        saved=V._LADDER_EVERY
        V._LADDER_EVERY=0
        try: mat=sv.log_phi_matrix(L,anchors,u)
        finally: V._LADDER_EVERY=saved
        raw=dict(zip(anchors,mat));resid={x:raw[x]-L*float(loggamma(x+1.)) for x in anchors}
        out=np.empty((len(rs),len(u)))
        for m,(r,anch) in enumerate(zip(rs,per_r)):
            base=L*float(loggamma(r+1.))
            if len(anch)==1: out[m]=raw[anch[0]];continue
            xs=np.log(np.asarray(anch,dtype=np.float64)+1.)
            w=np.ones(len(anch))
            for i in range(len(anch)):
                for k in range(len(anch)):
                    if k!=i:w[i]/=xs[i]-xs[k]
            coeff=w/(math.log(r+1.)-xs)
            Y=np.array([resid[x] for x in anch])
            out[m]=(coeff[:,None]*Y).sum(0)/coeff.sum()+base
        return out
    proto,case['prototype_batched_anchor_ladder']=timed(prototype)
    case['prototype_vs_active_alt']=compare(a,proto)
    case['prototype_vs_active_alt']['worst_r']=rs[case['prototype_vs_active_alt']['worst_index'][0]]
    case['prototype_vs_active_alt']['worst_u']=float(u[case['prototype_vs_active_alt']['worst_index'][1]])
    _runtime._settings.reset(token)
    report['cases'].append(case)
    (BASE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(case),flush=True)
su.close();sv.close()
# Hash all store bytes actually used and record final source equality.
used=['manifest.json','anchors.json']+[f'level_{L:02d}.{suffix}' for L in (2,26,53) for suffix in ('bin','index.json')]
report['store_files']=[{'path':str(store/name),'sha256':hashlib.sha256((store/name).read_bytes()).hexdigest()} for name in used]
report['source_copies_identical_at_end']=all(hashlib.sha256(Path(x['source']).read_bytes()).hexdigest()==x['sha256'] for x in files)
report['benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(BASE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
