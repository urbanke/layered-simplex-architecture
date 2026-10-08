import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['PMM_KERNEL']='1'
import sys,math,time,json,statistics,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'copies'))
import numpy as np
from product_model_with_memory.kernel import interp_column as native
from lsa.alt._vendor.pmwm import universal_tables as V
V._SADDLE_MIN_L=54;V._LADDER_EVERY=1;V._LADDER_F=0;V._LADDER_DEGREE=11;V.SERIES_TAIL_NATS=math.inf
store=V.UniversalTables('/Users/ruediger/Projects/product_model_with_memory/tables/anchors_prod',read_only=True)
original=V._interp_column
calls=0
def exact_scalar_native(grid, vals, u):
    global calls
    # Replicate the active scalar method's coordinate, clamping and weights exactly.
    s=(u-grid[0])/V.H
    i=np.clip(np.floor(s).astype(np.int64)-(V._STENCIL//2-1),0,len(grid)-V._STENCIL)
    x=s-i
    dx=x[:,None]-np.arange(V._STENCIL)[None,:]
    exact=np.abs(dx)<1e-12
    dx=np.where(exact,1.,dx)
    w=V._BARY_W[None,:]/dx
    wsum=w.sum(axis=1)
    hit=np.ascontiguousarray(exact.any(axis=1),dtype=np.uint8)
    hitcol=np.ascontiguousarray(np.argmax(exact,axis=1),dtype=np.int64)
    row=np.empty(len(u));todo=np.empty(len(u),dtype=np.uint8)
    # Passing i0=0 permits the existing C stencil to consume these exact scalar starts.
    native(vals.ctypes.data,len(vals),0,u.ctypes.data,len(u),float(grid[0]),float(grid[-1]),i.ctypes.data,w.ctypes.data,wsum.ctypes.data,hit.ctypes.data,hitcol.ctypes.data,row.ctypes.data,todo.ctypes.data)
    calls+=1
    if todo.any():raise RuntimeError('unexpected scalar-native leftover')
    return row
rs=[0,1,2,3,17,251,256,300,4001,67465,200000]
report={'description':'DIAGNOSTIC ONLY: existing C stencil with weights/coordinate arithmetic exactly matching active scalar path; single thread; 3 repeats; no source edits','cases':[]}
for L in (2,26,53):
    lo=min(-70.,-L*math.log(max(rs)+1.)-40.)
    u=np.linspace(lo,35.,math.ceil((35.-lo)/.02)+1)
    timings={};values={}
    for tag,fn in [('scalar_numpy',original),('scalar_native_exact_coordinates',exact_scalar_native)]:
        V._interp_column=fn
        samples=[]
        for _ in range(3):
            start=time.perf_counter();values[tag]=store.log_phi_matrix(L,rs,u);samples.append(time.perf_counter()-start)
        timings[tag]={'seconds':samples,'median_seconds':statistics.median(samples)}
    a,b=values.values()
    case={'L':L,'points':len(u),'timings':timings,'bit_identical':bool(np.array_equal(a,b)),'unequal_values':int(np.count_nonzero(a!=b)),'max_absolute_difference_nats':float(np.abs(a-b).max()),'value_count':a.size,'native_calls_cumulative':calls}
    report['cases'].append(case)
    print(json.dumps(case),flush=True)
V._interp_column=original
store.close()
report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report['parent_manifest_sha256']=hashlib.sha256((BASE/'results.json').read_bytes()).hexdigest()
(BASE/'scalar-exact-results.json').write_text(json.dumps(report,indent=2)+'\n')
