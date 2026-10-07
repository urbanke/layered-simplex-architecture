"""Synthetic gate checks only: contains no scientific experiment evidence."""
import copy
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace

from lsa.alt.artifacts import canonical_hash, read_json, sha256, source_identity, write_json
from lsa.alt.distributed import engine_options, power_options
from lsa.alt.scaling import cells

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location('scheduling_assessment', HERE/'scheduling-assessment.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
plan = read_json(REPO/'output/alt2027/scitas-setup-20261007/production-plan-9ec82bc-block20.json')
source = source_identity(REPO)
assert source['commit'] == plan['source_commit'] and source['tree_sha256'] == plan['source_tree_sha256']
# Other agents are archiving artifacts in this shared checkout. Synthetic
# fixtures use its identical frozen computational tree as a clean test source.
source['dirty'] = False
gate.source_identity = lambda repo: copy.deepcopy(source)
options = engine_options(read_json(REPO/'output/alt2027/scitas-setup-20261007/jed-engine.json'))
environment = {'platform':'Linux synthetic gate fixture', 'machine':platform.machine(),
               'python':platform.python_version(),
               'packages':{k:importlib.metadata.version(k) for k in ('numpy','scipy','mpmath')},
               'threads':{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}}
runtime = {'python':platform.python_version(),'numpy':environment['packages']['numpy'],
           'scipy':environment['packages']['scipy'],'system':'Linux','machine':platform.machine()}
gate.platform.system = lambda: 'Linux'

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))

def run_record(root, protocol, experiment, engine, files):
    dump(root/'protocol.json', protocol)
    for name, value in files.items():
        path=root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value) if isinstance(value,str) else dump(path,value)
    record={'status':'complete','purpose':'validation','experiment':experiment,
            'source':copy.deepcopy(source),'environment':copy.deepcopy(environment),
            'protocol_sha256':canonical_hash(protocol),'engine':engine,
            'outputs':{str(p.relative_to(root)):{'sha256':sha256(p),'bytes':p.stat().st_size}
                       for p in root.rglob('*') if p.is_file()}}
    dump(root/'manifest.json', record)

def fixture(root, power_seconds=100):
    pilot=root/'pilots'
    times={'primary-zipf1p5-20':3600,'primary-uniform-20':600,
           'spectrum-d1e6-alpha1p5-20':600,'depth-alpha2-20':105,
           'factorial-d1e5-n1e4-alpha1p5-100':16,'bible-full':1200}
    for name,(experiment,selectors) in gate.CASES.items():
        protocol=copy.deepcopy(plan['protocol'])
        config=copy.deepcopy(protocol['experiments'][experiment]);config.update(selectors)
        if experiment.startswith('benchmark_'):config['sampling_n_values']=config['n_values']
        protocol['experiments']={experiment:config};protocol['status']='implementation'
        files={}
        if experiment=='benchmark_primary':
            rows=[{'n':n,'samples':list(range(20)),'preparation':{'seconds':times[name]/10}} for n in config['n_values']]
            files['data/batching.jsonl']=''.join(json.dumps(r)+'\n' for r in rows)
        run_record(pilot/name/'run',protocol,experiment,{'depth':{**options,'runtime':runtime}},files)
        dump(pilot/name/'pilot-request.json',{'source':source,'cpu_count':1,'batch_size':20,'purpose':'validation'})
        dump(pilot/name/'pilot-result.json',{'status':'passed','error':None,'wall_seconds':times[name],'peak_rss_kib':1024**2})
    late=pilot/'sampling-late-offsets';rows=[]
    dump(late/'request.json',{'source':source,'environment':environment,'purpose':'validation'})
    for name,cell_id,start,trials in [('spectrum',22,980,20),('spectrum',27,980,20),('factorial',96,4500,500)]:
        path=late/f'{name}-{cell_id}.npz';path.write_bytes(b'SYNTHETIC TEST FIXTURE; NOT NUMERICAL EVIDENCE')
        rows.append({'experiment':name,'cell_id':cell_id,'trial_start':start,'trials':trials,
                     'cell':list(cells(name,plan['protocol']['experiments'][name]))[cell_id],
                     'last_count_file':path.name,'last_count_file_sha256':sha256(path),'replay_seconds':35})
    dump(late/'result.json',{'status':'passed','rows':rows})
    benchmark=plan['protocol']['experiments']['benchmark_powers']
    config={'default_settings':power_options({}),'targets':benchmark['targets'],
            'powers':benchmark['powers'],'d':benchmark['d'],'n':benchmark['n_values'][0]}
    files={'data/config.json':config,'data/summary.json':{'status':'passed','source_unchanged':True}}
    for target in benchmark['targets']:
        files[f'data/{target}/components.jsonl']=''.join(json.dumps({'setting':'default','power':w,'status':'passed','seconds':power_seconds/81})+'\n' for w in benchmark['powers'])
    run_record(root/'power',plan['protocol'],'power_validation',{},files)
    return SimpleNamespace(plan=REPO/'output/alt2027/scitas-setup-20261007/production-plan-9ec82bc-block20.json',
                           repo=REPO,pilot_root=pilot,power_root=root/'power')

checks=[]
for rate,expected in [(100,4),(600,8),(1000,16),(1800,None)]:
    with tempfile.TemporaryDirectory(prefix='lsa-resource-gate-fixture-') as tmp:
        args=fixture(Path(tmp),rate);report={'input_files':{}}
        gate.assess(args,report)
        assert report['selected_nodes']==expected,(rate,report['selected_nodes'],report['projections'])
        assert report['status']==('passed' if expected else 'pending')
        assert report['plan_sha256']==canonical_hash(plan)
        for projection in report['projections']:
            assert all(c['conservative_bound_hours']>=c['simulated_queue_hours_with_reserve'] for c in projection['controllers'])
        checks.append({'case':f'power_seconds_{rate}','selected_nodes':expected,'status':'passed',
                       'projected_hours':{p['nodes']:p['conservative_makespan_hours'] for p in report['projections']}})
for label in ('source_guard','power_runtime_guard','late_threads_guard'):
    with tempfile.TemporaryDirectory(prefix='lsa-resource-gate-fixture-') as tmp:
        args=fixture(Path(tmp))
        if label=='source_guard':
            path=args.pilot_root/'primary-zipf1p5-20/run/manifest.json';value=read_json(path)
            value['source']['files']['src/lsa/alt/depth.py']='0'*64
        elif label=='power_runtime_guard':
            path=args.power_root/'manifest.json';value=read_json(path);value['environment']['packages']['mpmath']='0.0.0'
        else:
            path=args.pilot_root/'sampling-late-offsets/request.json';value=read_json(path);value['environment']['threads']['OMP_NUM_THREADS']='2'
        dump(path,value)
        try:gate.assess(args,{'input_files':{}})
        except ValueError as error:checks.append({'case':label,'status':'passed','expected_rejection':str(error)})
        else:raise AssertionError(label+' unexpectedly passed')
record={'purpose':'synthetic scheduling gate checks, not scientific evidence','status':'passed',
        'script_sha256':sha256(HERE/'scheduling-assessment.py'),'checks':checks}
write_json(HERE/'synthetic-checks.json',record)
print(json.dumps(record,indent=2))
