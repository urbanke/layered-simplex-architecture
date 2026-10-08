"""Diagnostic only: compare count stencils/coordinates against saved references."""
from pathlib import Path
import collections, hashlib, json, math, time
import numpy as np
from scipy.special import loggamma
from lsa.alt.sealed_tables import SealedKernelTables

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'count-coordinate-002'
OUT.mkdir(exist_ok=False)
repo=ROOT.parent/'github-cleanup'
source=repo/'output/alt2027/sealed-pilot-validation-20261008-001/data/rows.jsonl'
rows=[json.loads(line) for line in source.read_text().splitlines()]
store=ROOT.parent/'sealed-kernel-pilot-20261008-001'
tables=SealedKernelTables(store)
groups=collections.defaultdict(list)
for row in rows: groups[(row['depth'],row['r'])].append(row)
result=[]
start=time.perf_counter()
for (L,r),cases in groups.items():
    u=np.array([c['u'] for c in cases])
    ref=np.array([c['refined_reference_log_phi_nats'] for c in cases])
    all_a=np.array(tables._anchors[L])
    columns,data=tables._load_level(L)
    exact=r in columns
    for degree in (5,7,9,11):
      m=degree+1
      for stencil in ('adjacent','log-balanced'):
        if exact:
            anchors=np.array([r])
        elif stencil=='adjacent':
            j=np.searchsorted(all_a,r)
            lo=max(0,min(j-m//2,len(all_a)-m))
            anchors=all_a[lo:lo+m]
        else:
            spacing=float(np.median(np.diff(np.log(all_a[all_a>256]+1.))))
            origin=math.log(int(all_a[-20])+1.)
            first=math.floor((math.log(r+1.)-origin)/spacing)-(m//2-1)
            targets=origin+(first+np.arange(m))*spacing
            logs=np.log(all_a+1.)
            chosen=[int(np.argmin(np.abs(logs-target))) for target in targets]
            assert len(set(chosen))==m,(L,r,degree,chosen)
            anchors=all_a[chosen]
        x=np.log(anchors.astype(np.longdouble)+1)
        xq=np.log(np.longdouble(r)+1)
        weights=np.ones(len(anchors),dtype=np.longdouble)
        if not exact:
            for i in range(len(anchors)):
                for j in range(len(anchors)):
                    if i!=j: weights[i]/=x[i]-x[j]
            coeff=weights/(xq-x)
            coeff/=coeff.sum()
        else: coeff=weights
        for coordinate in ('fixed','aligned'):
            grids=[u if coordinate=='fixed' else u+L*math.log((r+1.)/(int(a)+1.)) for a in anchors]
            covered=np.array([max(ua[k] for ua in grids)<=80 for k in range(len(u))])
            if not covered.any(): continue
            anchor_values=np.array([tables._anchor_values(L,int(a),ua[covered],columns,data) for a,ua in zip(anchors,grids)])
            for trend in ('raw','gamma-residual'):
                vals=anchor_values.astype(np.longdouble)
                if trend=='gamma-residual':
                    vals-=np.array([L*float(loggamma(int(a)+1.)) for a in anchors],dtype=np.longdouble)[:,None]
                interp=(coeff[:,None]*vals).sum(axis=0)
                if trend=='gamma-residual': interp+=L*float(loggamma(r+1.))
                interp=np.asarray(interp,dtype=float)
                key=f'{coordinate}/{trend}/{stencil}/degree{degree}'
                at=0
                for k,c in enumerate(cases):
                    if not covered[k]: continue
                    err=float(interp[at]-ref[k]);at+=1
                    result.append({'variant':key,'depth':L,'r':r,'u':c['u'],'case_id':c['case_id'],'exact_anchor':exact,'error_nats':err,'reference_nats':float(ref[k]),'reference_ulp_nats':float(np.spacing(abs(ref[k]))),'anchors':[int(a) for a in anchors]})
summary={}
for variant in sorted({r['variant'] for r in result}):
    cases=[r for r in result if r['variant']==variant]
    domains={
      'all':cases,
      'nonanchor':[r for r in cases if not r['exact_anchor']],
      'u_ge_minus80':[r for r in cases if r['u']>=-80],
      'r_le_1000':[r for r in cases if r['r']<=1000],
      'r_le_1000_u_ge_minus80':[r for r in cases if r['r']<=1000 and r['u']>=-80],
      'representable_gate':[r for r in cases if r['reference_ulp_nats']<=3e-9],
    }
    summary[variant]={}
    for name,vals in domains.items():
      if not vals:continue
      worst=max(vals,key=lambda r:abs(r['error_nats']))
      summary[variant][name]={'cases':len(vals),'over_3e9':sum(abs(r['error_nats'])>3e-9 for r in vals),'worst':worst}
(OUT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
(OUT/'summary.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seconds':time.perf_counter()-start,'variants':summary,'production_eligible':False,'limits':['Uses saved same-provider refined double references','No production source modified','Aligned cases requiring any anchor query above80 excluded rather than extrapolated','Longdouble coefficient arithmetic used diagnostically; actual stored payload remains float64']},indent=2)+'\n')
for key,v in summary.items():
 print(key,'all',v['all']['worst']['error_nats'],'u>=-80',v['u_ge_minus80']['worst']['error_nats'],'r<=1000',v['r_le_1000']['worst']['error_nats'],'small-normal',v['r_le_1000_u_ge_minus80']['worst']['error_nats'],flush=True)
tables.close()
