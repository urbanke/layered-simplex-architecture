"""Diagnostic only: dense aligned/fixed count-coordinate handover sweep."""
from pathlib import Path
import ast,json,math,time
import numpy as np
from scipy.special import loggamma
from lsa.alt.sealed_tables import SealedKernelTables
from lsa.alt._vendor.pmwm import mellin,_runtime
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'count-hybrid-sweep-001';OUT.mkdir(exist_ok=False)
tables=SealedKernelTables(ROOT.parent/'sealed-kernel-pilot-20261008-001')
source=(ROOT/'count_coordinate_probe.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='precise_anchor')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'count_coordinate_probe.py'),'exec'))
token=_runtime._settings.set({'PMM_BUILD_EXACT':'1'})
rows=[];summary=[];tick=time.perf_counter()
try:
 for L in (54,138):
  all_a=np.array(tables._anchors[L]);logs=np.log(all_a+1.)
  spacing=float(np.median(np.diff(np.log(all_a[all_a>256]+1.))))
  origin=math.log(int(all_a[-20])+1.)
  columns,data=tables._load_level(L)
  for r in (266,1000,20000,990031):
   v=np.arange(-30.137,100.0,.237)
   u=v-L*math.log(r+1.)
   normal=np.arange(-79.913,80,.617)
   u=np.sort(np.unique(np.r_[u,normal]));v=u+L*math.log(r+1.)
   ref=mellin.exact_log_phi_column(float(r),L,u,oversample=16,series_tolerance=1e-14,contour_tail_nats=50)
   first=math.floor((math.log(r+1.)-origin)/spacing)-5
   targets=origin+(first+np.arange(12))*spacing
   anchors=all_a[[int(np.argmin(abs(logs-t))) for t in targets]]
   x=np.log(anchors.astype(np.longdouble)+1);xq=np.log(np.longdouble(r)+1)
   weights=np.ones(12,dtype=np.longdouble)
   for j in range(12):
    for k in range(12):
     if j!=k:weights[j]/=x[j]-x[k]
   coeff=weights/(xq-x);coeff/=coeff.sum()
   fixed=np.array([precise_anchor(L,int(a),u,columns,data) for a in anchors])
   fixed=np.asarray((coeff[:,None]*fixed).sum(0),dtype=float)
   shifted=[u+L*math.log((r+1.)/(int(a)+1.)) for a in anchors]
   mask=np.array([max(z[i] for z in shifted)<=80 for i in range(len(u))])
   aligned=np.full(len(u),np.nan)
   raw=np.array([precise_anchor(L,int(a),z[mask],columns,data) for a,z in zip(anchors,shifted)])
   resid=raw-np.array([L*float(loggamma(int(a)+1.)) for a in anchors],dtype=np.longdouble)[:,None]
   aligned[mask]=np.asarray((coeff[:,None]*resid).sum(0)+L*float(loggamma(r+1.)),dtype=float)
   result={'depth':L,'r':r,'cases':len(u),'anchors':anchors.tolist(),'hybrids':{}}
   for cap in (5,10,15,20,30,40,60,80):
    candidate=np.where((v<=cap)&mask,aligned,fixed)
    error=candidate-ref
    worst=int(np.argmax(abs(error)))
    result['hybrids'][str(cap)]={'max_error_nats':float(error[worst]),'worst_u':float(u[worst]),'worst_v':float(v[worst]),'refined_reference_ulp':float(np.spacing(abs(ref[worst]))),'over_3e9':int((abs(error)>3e-9).sum())}
   for i in range(len(u)):
    rows.append({'depth':L,'r':r,'u':float(u[i]),'v':float(v[i]),'reference':float(ref[i]),'reference_ulp':float(np.spacing(abs(ref[i]))),'fixed_error':float(fixed[i]-ref[i]),'aligned_error':float(aligned[i]-ref[i]) if mask[i] else None})
   summary.append(result);print(json.dumps(result),flush=True)
finally:_runtime._settings.reset(token)
(OUT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(OUT/'summary.json').write_text(json.dumps({'seconds':time.perf_counter()-tick,'cases':len(rows),'rows':summary,'production_eligible':False},indent=2)+'\n')
tables.close()
