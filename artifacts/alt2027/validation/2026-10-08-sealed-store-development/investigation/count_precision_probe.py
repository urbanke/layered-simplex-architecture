"""Diagnostic: distinguish count interpolation error from double reference error."""
from pathlib import Path
from decimal import Decimal,localcontext
import json,time
from lsa.alt.kernel_validation import high_precision_log_phi
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'count-precision-001';OUT.mkdir(exist_ok=False)
source=ROOT.parent/'github-cleanup/output/alt2027/sealed-pilot-validation-20261008-001/data/rows.jsonl'
original=[json.loads(s) for s in source.read_text().splitlines()]
variants=[json.loads(s) for s in (ROOT/'count-coordinate-003/rows.jsonl').read_text().splitlines()]
result=[]
for L,r,u in [(138,990031,10.007),(53,990031,79.993),(138,20000,-1366.6811820774915)]:
 available=[c for c in original if c['depth']==L and c['r']==r]
 case=min(available,key=lambda c:abs(c['u']-u));u=case['u']
 start=time.perf_counter();exact_u=str(Decimal.from_float(float(u)))
 references=[high_precision_log_phi(r,L,exact_u,dps=dps,tail_digits=dps+20) for dps in (45,60)]
 with localcontext() as context:
  context.prec=80
  reference=Decimal(references[-1]['log_phi_nats'])
  values={'old_reader':case['scalar_log_phi_nats'],'direct_double':case['direct_reference_log_phi_nats'],'refined_double':case['refined_reference_log_phi_nats']}
  for c in variants:
   if c['case_id']==case['case_id']:
    values[c['variant']]=c['reference_nats']+c['error_nats']
  errors={k:str(Decimal.from_float(v)-reference) for k,v in values.items()}
  row={'depth':L,'r':r,'u':u,'exact_binary_u_decimal':exact_u,'case_id':case['case_id'],'high_precision':references,'precision_delta':str(Decimal(references[0]['log_phi_nats'])-reference),'errors_nats':errors,'seconds':time.perf_counter()-start}
 result.append(row);print(json.dumps(row),flush=True)
(OUT/'summary.json').write_text(json.dumps({'rows':result,'production_eligible':False},indent=2)+'\n')
