"""Run admitted, resource-checked ALT production once its dependencies succeed."""
from pathlib import Path
import json
import os
import re
import socket
import subprocess
import sys
from lsa.alt.artifacts import canonical_hash, read_json, sha256, source_identity, utc_now, write_json
from lsa.alt.distributed import validate_plan

BASE = Path('/scratch/urbanke/lsa-alt2027')
COMMIT = '0399cb28b973e5550fcbcbce082e8d36c95c2e1e'
REPO = BASE / 'checkouts' / COMMIT
SETUP = BASE / 'setup'
PLAN = SETUP / 'production-plan-0399cb2-nsplit.json'
ADMISSION = BASE / 'runs/admission-0399cb2-production-001'
RESOURCES = BASE / 'runs/scheduling-0399cb2-production-001'
OUT = BASE / 'runs/production-0399cb2-001'
GATE = BASE / 'runs/production-launch-0399cb2-001'
PYTHON = '/home/urbanke/venvs/alt2027-py313-20261007/bin/python'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NUM_NODES') == '1'
        and os.environ.get('SLURM_NTASKS') == '1' and os.environ.get('SLURM_CPUS_PER_TASK') == '1',
        'Admission gate requires one allocated node, task and CPU')
    require(os.path.samefile(sys.executable, PYTHON) and sys.prefix != sys.base_prefix,
        'Explicit pinned ALT virtualenv required')
    nodes = subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']], text=True).splitlines()
    require(socket.gethostname().split('.')[0] in {x.split('.')[0] for x in nodes},
        'Current host is outside the compute-node allocation')
    input_manifest = SETUP / 'handoff-inputs-0399cb2.json'
    handoff = read_json(input_manifest)
    expected_inputs = {
        'launch-when-ready-0399cb2.py', 'scheduling-assessment-nsplit.py',
        'jed-admission-and-launch-0399cb2.sbatch', 'jed-production-0399cb2.sbatch',
        'jed-postprocess-0399cb2.sbatch', 'production-plan-0399cb2-nsplit.json',
        'jed-engine.json', 'evidence-index-0399cb2.json',
    }
    require(handoff['source_commit'] == COMMIT and set(handoff['files_sha256']) == expected_inputs,
        'Handoff manifest must bind this exact source and all eight operational inputs')
    for name, expected in handoff['files_sha256'].items():
        require(sha256(SETUP/name) == expected, f'Frozen handoff input changed: {name}')
    source = source_identity(REPO)
    require(source['commit'] == COMMIT and not source['dirty'], 'Expected clean frozen source required')
    plan = validate_plan(read_json(PLAN))
    require(plan['source_commit'] == COMMIT and plan['source_tree_sha256'] == source['tree_sha256'],
        'Plan source differs from frozen checkout')
    require(plan['purpose'] == 'production' and plan['protocol']['status'] == 'frozen',
        'Frozen production plan required')
    require(plan.get('split_benchmark_n') is True and len(plan['jobs']) == 7754
        and (plan['block_size'], plan['factorial_block_size'], plan['batch_size']) == (20, 500, 20),
        'Exact 7754-job primary-n split required, with unchanged trial blocks and batching')
    require(read_json(REPO/'experiments/alt2027/protocol.json') == plan['protocol'],
        'Committed protocol differs from production plan')
    GATE.mkdir(parents=True, exist_ok=False)
    write_json(GATE/'started.json', {'started_utc':utc_now(), 'job_id':os.environ['SLURM_JOB_ID'], 'source':source,
        'plan_sha256':canonical_hash(plan), 'plan_file_sha256':sha256(PLAN),
        'launcher_sha256':sha256(__file__), 'scheduling_driver_sha256':sha256(SETUP/'scheduling-assessment-nsplit.py'),
        'array_driver_sha256':sha256(SETUP/'jed-production-0399cb2.sbatch'),
        'postprocess_driver_sha256':sha256(SETUP/'jed-postprocess-0399cb2.sbatch')})
    subprocess.run([PYTHON,str(REPO/'scripts/alt_admission.py'),'assemble','--repo',str(REPO),
        '--protocol',str(REPO/'experiments/alt2027/protocol.json'),'--engine-config',str(SETUP/'jed-engine.json'),
        '--evidence',str(SETUP/'evidence-index-0399cb2.json'),'--batch-size','20','--out',str(ADMISSION)], check=True)
    certificate = read_json(ADMISSION/'calibration.json')
    require(certificate['status'] == 'passed' and certificate['required_checks_complete'],
        'Passed complete numerical admission required')
    require(certificate['source_tree_sha256'] == plan['source_tree_sha256'], 'Certificate source mismatch')
    require(certificate['protocol_sha256'] == plan['protocol_sha256'], 'Certificate protocol mismatch')
    subprocess.run([PYTHON,str(SETUP/'scheduling-assessment-nsplit.py'),'--repo',str(REPO),'--plan',str(PLAN),
        '--pilot-root',str(BASE/'runs/throughput-f0a5ebd-20261007-001'),
        '--nsplit-root',str(BASE/'runs/throughput-nsplit-f0a5ebd-20261007-001'),
        '--power-root',str(BASE/'runs/calibration-f0a5ebd0a57d46caeb9bf45407c062f37b8d64bc-001/power'),
        '--out',str(RESOURCES)], check=True)
    resource_report = read_json(RESOURCES/'assessment.json')
    require(resource_report['status'] == 'passed' and resource_report['plan_sha256'] == canonical_hash(plan),
        'Passed scheduling assessment for this exact plan required')
    count = resource_report['selected_nodes']
    require(type(count) is int and count in (4,8,16), 'Assessed node count must be 4, 8 or 16')
    require(not OUT.exists(), 'Production root already exists; review it before any resubmission')
    require(source_identity(REPO) == source, 'Source changed during admission/scheduling assessment')
    require(read_json(input_manifest) == handoff and all(
        sha256(SETUP/name) == expected for name, expected in handoff['files_sha256'].items()),
        'Frozen handoff inputs changed during admission/scheduling assessment')
    environment = {key: value for key, value in os.environ.items()
        if not key.startswith(('SLURM_', 'SBATCH_', 'SRUN_'))}
    environment.update(ALT_REPO=str(REPO), ALT_EXPECTED_COMMIT=COMMIT, ALT_PLAN=str(PLAN), ALT_OUT=str(OUT),
        ALT_PYTHON=PYTHON, ALT_ENGINE_CONFIG=str(SETUP/'jed-engine.json'), ALT_CALIBRATION=str(ADMISSION/'calibration.json'))
    environment.pop('ALT_POWER_SETTINGS', None)
    command = ['sbatch','--parsable',f'--array=0-{count-1}%{count}',str(SETUP/'jed-production-0399cb2.sbatch')]
    write_json(GATE/'submission-intent.json', {'utc':utc_now(),'command':command,'nodes':count,'workers_per_node':72,
        'memory_per_node':'440G','walltime_per_node':'24:00:00','output':str(OUT),
        'certificate_sha256':sha256(ADMISSION/'calibration.json'),'resource_assessment_sha256':sha256(RESOURCES/'assessment.json')})
    result = subprocess.run(command,env=environment,capture_output=True,text=True)
    (GATE/'scheduler.stderr').write_text(result.stderr)
    (GATE/'scheduler.stdout').write_text(result.stdout)
    result.check_returncode()
    job = result.stdout.strip().split(';')[0]
    require(re.fullmatch(r'[0-9]+',job), 'Unknown submission outcome: inspect scheduler output; do not resubmit blindly')
    write_json(GATE/'submitted.json', {'status':'submitted','job_id':job,'nodes':count,'workers':72*count,
        'submitted_utc':utc_now(),'output':str(OUT),'source_commit':COMMIT,'plan_sha256':canonical_hash(plan)})
    post_command = ['sbatch','--parsable',f'--dependency=afterok:{job}',str(SETUP/'jed-postprocess-0399cb2.sbatch')]
    write_json(GATE/'postprocess-intent.json', {'utc':utc_now(),'command':post_command,'production_array_job':job})
    post = subprocess.run(post_command,env=environment,capture_output=True,text=True)
    (GATE/'postprocess-scheduler.stderr').write_text(post.stderr)
    (GATE/'postprocess-scheduler.stdout').write_text(post.stdout)
    post.check_returncode()
    post_job = post.stdout.strip().split(';')[0]
    require(re.fullmatch(r'[0-9]+',post_job), 'Unknown postprocessing submission outcome: inspect saved scheduler output')
    write_json(GATE/'postprocess-submitted.json', {'status':'submitted','job_id':post_job,
        'after_successful_production_array_job':job,'submitted_utc':utc_now()})
    print(json.dumps({'production_array_job':job,'nodes':count,'workers':72*count,'postprocess_job':post_job}), flush=True)

if __name__ == '__main__':
    main()
