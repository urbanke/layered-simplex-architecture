"""Separate u polynomial truncation from pre-existing saved node errors."""
from pathlib import Path
import json,time,math,hashlib
import numpy as np
import mpmath as mp
from lsa.alt.sealed_tables import SealedKernelTables
from lsa.alt.kernel_validation import high_precision_log_phi
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'count-payload-decomposition-001';OUT.mkdir(exist_ok=False)
mp.mp.dps=75

def exact(x):
 n,d=float(x).as_integer_ratio();return mp.mpf(n)/d

def coeff(xs,q):
 ts=[1/mp.fprod(x-z for j,z in enumerate(xs) if i!=j)/(q-x) for i,x in enumerate(xs)]
 return [t/mp.fsum(ts) for t in ts]

def serial(x):
 if isinstance(x,mp.mpf):return mp.nstr(x,65)
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [serial(v) for v in x]
 return x

parent=json.loads((ROOT/'count-error-decomposition-001/summary.json').read_text())
rows=[]
with SealedKernelTables(ROOT.parent/'sealed-kernel-pilot-20261008-001') as tab:
 for previous in parent['rows']:
  t=time.perf_counter();L,r,u=previous['depth'],previous['r'],previous['u'];cols,data=tab._load_level(L)
  anchorrows=[]
  for ap in previous['anchors']:
   a=ap['r'];col=cols[a];vals=data[col.offset:col.offset+col.length]
   start=int(np.clip(math.floor((u-col.u_min)/tab.grid_step)-3,0,len(vals)-8))
   inds=np.arange(start,start+8);nodes=col.u_min+tab.grid_step*inds.astype(np.float64)
   uc=coeff([exact(x) for x in nodes],exact(u));nt=[]
   for node,i in zip(nodes,inds):
    truth=mp.mpf(high_precision_log_phi(a,L,float(node),dps=60,tail_digits=80)['log_phi_nats'])
    nt.append({'u':node,'stored':vals[i],'truth':truth,'stored_error':exact(vals[i])-truth})
   truepoly=mp.fsum(c*x['truth'] for c,x in zip(uc,nt))
   stored=mp.fsum(c*exact(x['stored']) for c,x in zip(uc,nt))
   anchorrows.append({'r':a,'coefficient':mp.mpf(ap['count_coefficient']),'nodes':nt,'stored_error_at_query':stored-truepoly,'u_polynomial_truncation':truepoly-mp.mpf(ap['reference60'])})
   print(json.dumps(serial({'depth':L,'r':a,'stored_error_at_query':anchorrows[-1]['stored_error_at_query'],'u_polynomial_truncation':anchorrows[-1]['u_polynomial_truncation']})),flush=True)
  row={'depth':L,'r':r,'u':u,'stored_payload_error_propagation':mp.fsum(a['coefficient']*a['stored_error_at_query'] for a in anchorrows),'u_polynomial_truncation_propagation':mp.fsum(a['coefficient']*a['u_polynomial_truncation'] for a in anchorrows),'anchors':anchorrows,'seconds':time.perf_counter()-t}
  rows.append(serial(row));print(json.dumps(serial({k:v for k,v in row.items() if k!='anchors'})),flush=True)
(OUT/'summary.json').write_text(json.dumps({'production_eligible':False,'method':'75-digit exact-binary64 interpolation;60-digit independent contour at all192 saved float64 grid points','parent_reader_sha256':parent['reader_sha256'],'rows':rows},indent=2)+'\n')
