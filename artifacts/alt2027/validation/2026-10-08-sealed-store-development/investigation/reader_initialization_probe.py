"""Read-only serving initialization timings; no numerical source/store changes."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import json,time,platform,hashlib
from pathlib import Path
from lsa.alt import depth
from lsa.alt.depth import DepthEvaluator, StoreConfig
repo=Path('/Users/ruediger/Documents/Codex/2026-10-06/i-ha/work/github-cleanup')
output=Path(__file__).with_suffix('.json')
options=json.loads((repo/'output/alt2027/engine-sealed-pilot-20261008-001.json').read_text())
config=StoreConfig(**options['store'])
real_hash=depth._hash_file
stats={'hash_calls':0,'hash_bytes':0}
def counted(path):
    stats['hash_calls']+=1;stats['hash_bytes']+=Path(path).stat().st_size
    return real_hash(path)
depth._hash_file=counted
report={'scope':'Single-thread read-only pilot store initialization; OS page cache was not cleared; no full-production timing projection','platform':platform.platform(),'store':str(config.path),'payload_bytes':sum(p.stat().st_size for p in Path(config.path).glob('level_*.bin')),'source_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'src/lsa/alt/depth.py',repo/'src/lsa/alt/sealed_tables.py']},'runs':[]}
for repeat in range(3):
    stats.update(hash_calls=0,hash_bytes=0)
    begin=time.perf_counter()
    engine=DepthEvaluator(mode='store',store=config,prediction_tolerance=options['prediction_tolerance'])
    row={'repeat':repeat,'open_seconds':time.perf_counter()-begin,**stats,'levels':[]}
    for L in engine._store.coverage_depths:
        begin=time.perf_counter();engine._store.ensure_columns(L,[0]);first=time.perf_counter()-begin
        begin=time.perf_counter();engine._store.ensure_columns(L,[0]);second=time.perf_counter()-begin
        row['levels'].append({'depth':L,'first_level_load_seconds':first,'same_reader_second_check_seconds':second,'bytes':(Path(config.path)/f'level_{L:03d}.bin').stat().st_size})
    row['total_first_level_load_seconds']=sum(x['first_level_load_seconds'] for x in row['levels'])
    row['total_same_reader_second_check_seconds']=sum(x['same_reader_second_check_seconds'] for x in row['levels'])
    engine.close();report['runs'].append(row)
    print(json.dumps(row),flush=True)
report['sources_unchanged']=all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in report['source_sha256'].items())
output.write_text(json.dumps(report,indent=2)+'\n')
