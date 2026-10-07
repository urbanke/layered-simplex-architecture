"""Time exact late-shard RNG replay only, without evaluating evidence."""
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time
import numpy as np
from lsa.alt.artifacts import environment, sha256, source_identity, utc_now, write_json
from lsa.alt.scaling import cells, zipf

base = Path('/scratch/urbanke/lsa-alt2027')
repo = base / 'checkouts/f0a5ebd0a57d46caeb9bf45407c062f37b8d64bc'
out = base / 'runs/throughput-f0a5ebd-20261007-001/sampling-late-offsets'
allocated = subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).splitlines()
assert socket.gethostname().split('.')[0] in {n.split('.')[0] for n in allocated}
out.mkdir(parents=True, exist_ok=False)
protocol = json.loads((repo / 'experiments/alt2027/protocol.json').read_text())
write_json(out / 'request.json', {'purpose':'validation','production_admission':False,
    'source':source_identity(repo),'driver_sha256':sha256(__file__),'environment':environment(),
    'job_id':os.environ['SLURM_JOB_ID'],'started_utc':utc_now(),
    'note':'Exact declared scaling multinomial streams; timings only, no numerical model evaluation.'})
rows=[]
for name, cell_id, start, count in [('spectrum',22,980,20),('spectrum',27,980,20),('factorial',96,4500,500)]:
    config=protocol['experiments'][name]
    cell=list(cells(name,config))[cell_id]
    begun=time.perf_counter()
    p=zipf(cell['d'],cell['alpha'])
    target_seconds=time.perf_counter()-begun
    rng=np.random.default_rng([config['seed'],cell_id])
    begun=time.perf_counter()
    for _ in range(start): rng.multinomial(cell['n'],p)
    replay_seconds=time.perf_counter()-begun
    begun=time.perf_counter()
    for _ in range(count):
        m=rng.multinomial(cell['n'],p)
        occupied=np.flatnonzero(m)
        profile=tuple(sorted(map(int,m[occupied]),reverse=True))
    generation_seconds=time.perf_counter()-begun
    path=out/f'{name}-{cell_id}-trial-{start+count-1}.npz'
    np.savez_compressed(path,counts=m)
    row={'experiment':name,'cell_id':cell_id,'cell':cell,'trial_start':start,'trials':count,
         'target_seconds':target_seconds,'replay_seconds':replay_seconds,
         'generation_and_profile_seconds':generation_seconds,
         'last_count_file':path.name,'last_count_file_sha256':sha256(path)}
    rows.append(row)
    print(json.dumps(row),flush=True)
write_json(out/'result.json',{'status':'passed','production_admission':False,'rows':rows,
    'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'finished_utc':utc_now()})
