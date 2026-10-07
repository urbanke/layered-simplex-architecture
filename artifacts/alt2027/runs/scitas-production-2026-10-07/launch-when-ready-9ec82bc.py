"""Run admitted, resource-checked ALT production once its dependencies succeed."""
from pathlib import Path
import json
import os
import re
import socket
import subprocess
import sys
from lsa.alt.artifacts import canonical_hash, read_json, sha256, source_identity, utc_now, write_json

BASE = Path('/scratch/urbanke/lsa-alt2027')
COMMIT = '9ec82bcef549ad593dee5450e974d83de44ddc77'
REPO = BASE / 'checkouts' / COMMIT
SETUP = BASE / 'setup'
PLAN = SETUP / 'production-plan-9ec82bc-block20.json'
ADMISSION = BASE / 'runs/admission-9ec82bc-production-001'
RESOURCES = BASE / 'runs/scheduling-9ec82bc-production-001'
OUT = BASE / 'runs/production-9ec82bc-001'
GATE = BASE / 'runs/production-launch-9ec82bc-001'
PYTHON = '/home/urbanke/venvs/alt2027-py313-20261007/bin/python'

def main():
    nodes = subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']], text=True).splitlines()
    assert socket.gethostname().split('.')[0] in {x.split('.')[0] for x in nodes}
    input_manifest = SETUP / 'handoff-inputs-9ec82bc.json'
    for name, expected in read_json(input_manifest)['files_sha256'].items():
        assert sha256(SETUP/name) == expected, f'Frozen handoff input changed: {name}'
    source = source_identity(REPO)
    assert source['commit'] == COMMIT and not source['dirty']
    plan = read_json(PLAN)
    assert plan['source_commit'] == COMMIT and plan['source_tree_sha256'] == source['tree_sha256']
    assert plan['purpose'] == 'production' and plan['protocol']['status'] == 'frozen'
    GATE.mkdir(parents=True, exist_ok=False)
    write_json(GATE/'started.json', {'started_utc':utc_now(), 'job_id':os.environ['SLURM_JOB_ID'], 'source':source,
        'plan_sha256':canonical_hash(plan), 'plan_file_sha256':sha256(PLAN),
        'launcher_sha256':sha256(__file__), 'scheduling_driver_sha256':sha256(SETUP/'scheduling-assessment.py'),
        'array_driver_sha256':sha256(SETUP/'jed-production-9ec82bc.sbatch'),
        'postprocess_driver_sha256':sha256(SETUP/'jed-postprocess-9ec82bc.sbatch')})
    subprocess.run([PYTHON,str(REPO/'scripts/alt_admission.py'),'assemble','--repo',str(REPO),
        '--protocol',str(REPO/'experiments/alt2027/protocol.json'),'--engine-config',str(SETUP/'jed-engine.json'),
        '--evidence',str(SETUP/'evidence-index-9ec82bc.json'),'--batch-size','20','--out',str(ADMISSION)], check=True)
    certificate = read_json(ADMISSION/'calibration.json')
    assert certificate['status'] == 'passed' and certificate['required_checks_complete']
    assert certificate['source_tree_sha256'] == plan['source_tree_sha256']
    assert certificate['protocol_sha256'] == plan['protocol_sha256']
    subprocess.run([PYTHON,str(SETUP/'scheduling-assessment.py'),'--repo',str(REPO),'--plan',str(PLAN),
        '--pilot-root',str(BASE/'runs/throughput-f0a5ebd-20261007-001'),
        '--power-root',str(BASE/'runs/calibration-f0a5ebd0a57d46caeb9bf45407c062f37b8d64bc-001/power'),
        '--out',str(RESOURCES)], check=True)
    resource_report = read_json(RESOURCES/'assessment.json')
    assert resource_report['status'] == 'passed' and resource_report['plan_sha256'] == canonical_hash(plan)
    count = resource_report['selected_nodes']
    assert type(count) is int and count in (4,8,16)
    assert not OUT.exists(), 'Production root already exists; review it before any resubmission'
    environment = {key: value for key, value in os.environ.items()
        if not key.startswith(('SLURM_', 'SBATCH_', 'SRUN_'))}
    environment.update(ALT_REPO=str(REPO), ALT_EXPECTED_COMMIT=COMMIT, ALT_PLAN=str(PLAN), ALT_OUT=str(OUT),
        ALT_PYTHON=PYTHON, ALT_ENGINE_CONFIG=str(SETUP/'jed-engine.json'), ALT_CALIBRATION=str(ADMISSION/'calibration.json'))
    environment.pop('ALT_POWER_SETTINGS', None)
    command = ['sbatch','--parsable',f'--array=0-{count-1}%{count}',str(SETUP/'jed-production-9ec82bc.sbatch')]
    write_json(GATE/'submission-intent.json', {'utc':utc_now(),'command':command,'nodes':count,'workers_per_node':72,
        'memory_per_node':'440G','walltime_per_node':'24:00:00','output':str(OUT),
        'certificate_sha256':sha256(ADMISSION/'calibration.json'),'resource_assessment_sha256':sha256(RESOURCES/'assessment.json')})
    result = subprocess.run(command,env=environment,capture_output=True,text=True)
    (GATE/'scheduler.stderr').write_text(result.stderr)
    (GATE/'scheduler.stdout').write_text(result.stdout)
    result.check_returncode()
    job = result.stdout.strip().split(';')[0]
    assert re.fullmatch(r'[0-9]+',job), 'Unknown submission outcome: inspect scheduler output; do not resubmit blindly'
    write_json(GATE/'submitted.json', {'status':'submitted','job_id':job,'nodes':count,'workers':72*count,
        'submitted_utc':utc_now(),'output':str(OUT),'source_commit':COMMIT,'plan_sha256':canonical_hash(plan)})
    post_command = ['sbatch','--parsable',f'--dependency=afterok:{job}',str(SETUP/'jed-postprocess-9ec82bc.sbatch')]
    write_json(GATE/'postprocess-intent.json', {'utc':utc_now(),'command':post_command,'production_array_job':job})
    post = subprocess.run(post_command,env=environment,capture_output=True,text=True)
    (GATE/'postprocess-scheduler.stderr').write_text(post.stderr)
    (GATE/'postprocess-scheduler.stdout').write_text(post.stdout)
    post.check_returncode()
    post_job = post.stdout.strip().split(';')[0]
    assert re.fullmatch(r'[0-9]+',post_job), 'Unknown postprocessing submission outcome: inspect saved scheduler output'
    write_json(GATE/'postprocess-submitted.json', {'status':'submitted','job_id':post_job,
        'after_successful_production_array_job':job,'submitted_utc':utc_now()})
    print(json.dumps({'production_array_job':job,'nodes':count,'workers':72*count,'postprocess_job':post_job}), flush=True)

if __name__ == '__main__':
    main()
