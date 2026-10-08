"""Independent references for the three remaining nominal-gate diagnostic cases."""
from pathlib import Path
from decimal import Decimal,localcontext
import hashlib,json,time
from lsa.alt.kernel_validation import high_precision_log_phi
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'count-remaining-reference-001';OUT.mkdir(exist_ok=False)
source=ROOT.parent/'github-cleanup/output/alt2027/sealed-reader-corrected-diagnostic-20261008-001.json'
parent=json.loads(source.read_text())
result=[]
for L,r,u in [(2,1000000,-.2573),(80,1000000,-.2573),(138,990031,-4.013)]:
 row=next(c for c in parent['rows'] if c['depth']==L and c['r']==r and c['u']==u)
 start=time.perf_counter()
 ref45=high_precision_log_phi(r,L,float(u),dps=45,tail_digits=65)
 ref60=high_precision_log_phi(r,L,float(u),dps=60,tail_digits=80)
 with localcontext() as c:
  c.prec=90
  truth=Decimal(ref60['log_phi_nats'])
  answer={'case':row,'reference45':ref45,'reference60':ref60,'precision_delta_nats':str(Decimal(ref45['log_phi_nats'])-truth),'reader_error_nats':str(Decimal.from_float(row['value'])-truth),'old_double_reference_error_nats':str(Decimal.from_float(row['reference'])-truth),'nearest_binary64':float(truth),'reader_error_to_nearest_binary64_nats':row['value']-float(truth),'seconds':time.perf_counter()-start}
  answer['passed_nominal_3e9']=abs(Decimal(answer['reader_error_nats']))<=Decimal('3e-9')
 result.append(answer);print(json.dumps(answer),flush=True)
(OUT/'summary.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reader_sha256':parent['source_sha256'],'rows':result,'production_eligible':False},indent=2)+'\n')
