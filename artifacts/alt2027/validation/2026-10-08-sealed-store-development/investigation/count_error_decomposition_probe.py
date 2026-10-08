"""Attribute remaining errors without modifying the sealed reader or store."""
from pathlib import Path
import hashlib,json,time,math
import numpy as np
import mpmath as mp
from lsa.alt.sealed_tables import SealedKernelTables,_count_coefficients,_count_interpolate
from lsa.alt.kernel_validation import high_precision_log_phi
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'count-error-decomposition-001';OUT.mkdir(exist_ok=False)
STORE=ROOT.parent/'sealed-kernel-pilot-20261008-001'
READER=ROOT.parent/'github-cleanup/src/lsa/alt/sealed_tables.py'
mp.mp.dps=75

def exact(x):
    if isinstance(x,(float,np.floating)):
        n,d=float(x).as_integer_ratio();return mp.mpf(n)/d
    return mp.mpf(x)

def interp(xs,ys,q):
    if q in xs:return ys[xs.index(q)]
    ts=[1/mp.fprod(x-z for j,z in enumerate(xs) if i!=j)/(q-x) for i,x in enumerate(xs)]
    coeff=[t/mp.fsum(ts) for t in ts]
    return mp.fsum(c*y for c,y in zip(coeff,ys)),coeff

def serial(x):
    if isinstance(x,mp.mpf):return mp.nstr(x,65)
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x
refs=json.loads((ROOT/'count-remaining-reference-001/summary.json').read_text())['rows']
rows=[]
with SealedKernelTables(STORE) as tables:
 for L,r,u in [(2,1000000,-.2573),(138,990031,-4.013)]:
    t=time.perf_counter();cols,data=tables._load_level(L);anchors=tables.ladder_anchors_for(L,r)
    truth=mp.mpf(next(x for x in refs if x['case']['depth']==L)['reference60']['log_phi_nats'])
    xs=[mp.log(mp.mpf(a+1)/(r+1)) for a in anchors]
    fp_anchor=[];mp_payload=[];true_anchor=[];anchor_rows=[]
    for a in anchors:
        col=cols[a];vals=data[col.offset:col.offset+col.length]
        start=int(np.clip(math.floor((u-col.u_min)/tables.grid_step)-3,0,len(vals)-8))
        inds=np.arange(start,start+8);nodes=col.u_min+tables.grid_step*inds.astype(np.float64)
        pval,uc=interp([exact(x) for x in nodes],[exact(x) for x in vals[inds]],exact(u))
        fval=float(tables._anchor_values(L,a,np.array([u]),cols,data)[0])
        aval=mp.mpf(high_precision_log_phi(a,L,u,dps=60,tail_digits=80)['log_phi_nats'])
        fp_anchor.append(exact(fval));mp_payload.append(pval);true_anchor.append(aval)
        anchor_rows.append({'r':a,'fp_value':fval,'mp_payload_interpolation':pval,'reference60':aval,'fp_u_arithmetic_error':exact(fval)-pval,'mp_payload_minus_truth':pval-aval})
    fpanchors_mpcount,c=interp(xs,fp_anchor,mp.mpf(0))
    mppayload_mpcount,_=interp(xs,mp_payload,mp.mpf(0))
    trueanchors_mpcount,_=interp(xs,true_anchor,mp.mpf(0))
    current=exact(float(tables.log_phi(L,r,u)[0]))
    for i,ar in enumerate(anchor_rows):
        ar['count_coefficient']=c[i]
        ar['payload_weighted_error']=c[i]*(mp_payload[i]-true_anchor[i])
    row={'depth':L,'r':r,'u':u,'reader':current,'truth':truth,
         'current_minus_truth':current-truth,
         'fpanchors_mpcount_minus_truth':fpanchors_mpcount-truth,
         'mppayload_mpcount_minus_truth':mppayload_mpcount-truth,
         'trueanchors_mpcount_minus_truth':trueanchors_mpcount-truth,
         'count_fp_arithmetic':current-fpanchors_mpcount,
         'u_fp_arithmetic_propagation':fpanchors_mpcount-mppayload_mpcount,
         'payload_and_u_truncation_propagation':mppayload_mpcount-trueanchors_mpcount,
         'count_polynomial_truncation':trueanchors_mpcount-truth,
         'nearest_float_if_all_mp':float(mppayload_mpcount),
         'anchors':anchor_rows,'seconds':time.perf_counter()-t}
    rows.append(serial(row));print(json.dumps(serial({k:v for k,v in row.items() if k!='anchors'})),flush=True)
(OUT/'summary.json').write_text(json.dumps({'production_eligible':False,'method':'75-digit interpolation of exact binary64 physical nodes and payloads;60-digit independent anchor references','reader_sha256':hashlib.sha256(READER.read_bytes()).hexdigest(),'rows':rows},indent=2)+'\n')
