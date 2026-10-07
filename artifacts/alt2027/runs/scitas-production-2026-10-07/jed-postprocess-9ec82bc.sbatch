#!/usr/bin/env bash
# Queue only with --dependency=afterok:PRODUCTION_ARRAY_JOB_ID.
#SBATCH --job-name=lsa-alt-postprocess
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/urbanke/alt2027-logs/postprocess-%j.out
#SBATCH --export=ALL

set -euo pipefail
alt_base=/scratch/urbanke/lsa-alt2027
alt_commit=9ec82bcef549ad593dee5450e974d83de44ddc77
alt_repo="$alt_base/checkouts/$alt_commit"
alt_python=/home/urbanke/venvs/alt2027-py313-20261007/bin/python
export PYTHONPATH="$alt_repo/src" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export MPLCONFIGDIR="$alt_base/setup/matplotlib"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 "$alt_python" -u - <<'PY'
"""Immutable postprocessing of the exact admitted ALT production campaign."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from lsa.alt.artifacts import canonical_hash, read_json, sha256, source_identity, utc_now, write_json
from lsa.alt.distributed import validate_plan

BASE = Path('/scratch/urbanke/lsa-alt2027')
COMMIT = '9ec82bcef549ad593dee5450e974d83de44ddc77'
REPO = BASE / 'checkouts' / COMMIT
PLAN = BASE / 'setup/production-plan-9ec82bc-block20.json'
SCRIPT = BASE / 'setup/jed-postprocess-9ec82bc.sbatch'
PYTHON = '/home/urbanke/venvs/alt2027-py313-20261007/bin/python'
CAMPAIGN = BASE / 'runs/production-9ec82bc-001'
MERGED = BASE / 'runs/production-9ec82bc-001-merged'
REPORT = BASE / 'runs/production-9ec82bc-001-report'
RECORDS = BASE / 'runs/production-9ec82bc-001-postprocess'
EXPERIMENTS = {
    'architecture', 'spectrum', 'benchmark_primary', 'benchmark_powers',
    'bible', 'bible_secondary', 'factorial', 'depth_scaling', 'validation',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_command(records, name, command):
    """Every attempted CLI command retains its log and exact outcome."""
    print(json.dumps({'step': name, 'status': 'started', 'utc': utc_now()}), flush=True)
    write_json(records / f'{name}-started.json', {'started_utc': utc_now(), 'command': command})
    with (records / f'{name}.log').open('x') as log:
        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False)
    write_json(records / f'{name}-finished.json', {
        'finished_utc': utc_now(), 'exit_code': result.returncode,
        'log_sha256': sha256(records / f'{name}.log'),
    })
    print(json.dumps({'step': name, 'exit_code': result.returncode, 'utc': utc_now()}), flush=True)
    result.check_returncode()


def checked_manifest(path, experiment, plan):
    record = read_json(path / 'manifest.json')
    require(record['status'] == 'complete' and record['purpose'] == 'production',
            f'{experiment}: completed production Run required')
    require(record['experiment'] == experiment, f'{experiment}: wrong Run type')
    require(record['source']['commit'] == COMMIT and
            record['source']['tree_sha256'] == plan['source_tree_sha256'],
            f'{experiment}: source identity mismatch')
    require(record['protocol_sha256'] == plan['protocol_sha256'],
            f'{experiment}: protocol identity mismatch')
    return {'path': str(path), 'status': record['status'],
            'manifest_sha256': sha256(path / 'manifest.json'),
            'output_files': len(record['outputs'])}


def main():
    # Existing directories require explicit inspection; never merge over a partial run.
    RECORDS.mkdir(parents=True, exist_ok=False)
    outcome = {'status': 'failed', 'started_utc': utc_now(), 'source_commit': COMMIT,
               'campaign': str(CAMPAIGN), 'merged': str(MERGED), 'report': str(REPORT),
               'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': socket.gethostname()}
    initial_source = None
    input_hashes = {}
    code = 1
    try:
        require(os.environ.get('SLURM_JOB_ID'), 'compute-node Slurm allocation required')
        require(os.environ.get('SLURM_JOB_NUM_NODES') == '1' and
                os.environ.get('SLURM_NTASKS') == '1' and
                os.environ.get('SLURM_CPUS_PER_TASK') == '8',
                'postprocessing requires one node, one task and eight CPUs')
        nodes = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], text=True).splitlines()
        require(socket.gethostname().split('.')[0] in {node.split('.')[0] for node in nodes},
                'current host is outside the compute-node allocation')
        require(os.path.samefile(sys.executable, PYTHON) and sys.prefix != sys.base_prefix,
                'explicit pinned ALT virtualenv required')
        initial_source = source_identity(REPO)
        require(initial_source['commit'] == COMMIT and not initial_source['dirty'],
                'expected clean frozen source required')
        plan = validate_plan(read_json(PLAN))
        require(plan['source_commit'] == COMMIT and
                plan['source_tree_sha256'] == initial_source['tree_sha256'],
                'plan source identity differs from frozen checkout')
        require(plan['purpose'] == 'production' and plan['protocol']['status'] == 'frozen',
                'frozen production plan required')
        require(len(plan['jobs']) == 3904 and set(plan['protocol']['experiments']) == EXPERIMENTS,
                'expected exact 3904-job, nine-experiment campaign')
        require(read_json(REPO / 'experiments/alt2027/protocol.json') == plan['protocol'],
                'committed protocol differs from the production plan')
        require(canonical_hash(read_json(CAMPAIGN / 'plan.json')) == canonical_hash(plan),
                'campaign plan differs from the scheduled plan')
        require({entry.name for entry in (CAMPAIGN / 'jobs').iterdir()} ==
                {job['id'] for job in plan['jobs']}, 'campaign has missing or unexpected jobs')
        require(not MERGED.exists() and not REPORT.exists(),
                'merged/report output already exists; inspect before restarting')
        inputs = [PLAN, SCRIPT, CAMPAIGN / 'plan.json', CAMPAIGN / 'runtime.json',
                  CAMPAIGN / 'admission.json', CAMPAIGN / 'parent-calibration.json']
        input_hashes = {str(path): sha256(path) for path in inputs}
        outcome.update(plan_sha256=canonical_hash(plan), expected_jobs=3904)
        write_json(RECORDS / 'started.json', {
            **outcome, 'status': 'started', 'source': initial_source,
            'input_files_sha256': input_hashes,
            'python': sys.version, 'python_executable': sys.executable,
            'resources': {'cpus': 8, 'memory': '48G', 'walltime': '04:00:00'},
            'threads': {key: os.environ.get(key) for key in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')},
        })
        run_command(RECORDS, 'merge', [PYTHON, str(REPO / 'scripts/alt_distributed.py'),
            'merge', '--repo', str(REPO), '--plan', str(PLAN), '--runs', str(CAMPAIGN),
            '--out', str(MERGED)])
        merge_index = read_json(MERGED / 'merge.json')
        require(set(merge_index['experiments']) == EXPERIMENTS and
                sum(row['jobs'] for row in merge_index['experiments'].values()) == 3904,
                'merged campaign coverage differs from all 3904 planned jobs')
        manifests = {}
        for name in sorted(EXPERIMENTS):
            run_command(RECORDS, f'verify-{name}', [PYTHON, str(REPO / 'scripts/alt_experiments.py'),
                '--repo', str(REPO), 'verify', str(MERGED / name)])
            manifests[name] = checked_manifest(MERGED / name, name, plan)
            require(manifests[name]['manifest_sha256'] == merge_index['experiments'][name]['manifest_sha256'],
                    f'{name}: manifest differs from merge index')
        run_command(RECORDS, 'report', [PYTHON, str(REPO / 'scripts/alt_experiments.py'),
            '--repo', str(REPO), 'report', '--protocol', str(REPO / 'experiments/alt2027/protocol.json'),
            '--purpose', 'production', '--sources', str(MERGED), '--out', str(REPORT)])
        run_command(RECORDS, 'verify-report', [PYTHON, str(REPO / 'scripts/alt_experiments.py'),
            '--repo', str(REPO), 'verify', str(REPORT)])
        manifests['reports'] = checked_manifest(REPORT, 'reports', plan)
        write_json(RECORDS / 'verified-manifests.json', manifests)
        write_json(RECORDS / 'report-assets.json', read_json(REPORT / 'manifest.json')['outputs'])
        outcome.update(status='complete', verified_runs=10, merged_jobs=3904,
                       verified_manifests_sha256=sha256(RECORDS / 'verified-manifests.json'),
                       report_assets_sha256=sha256(RECORDS / 'report-assets.json'),
                       merge_index_sha256=sha256(MERGED / 'merge.json'))
        code = 0
    except Exception as error:
        outcome['error'] = {'type': type(error).__name__, 'message': str(error)}
        if isinstance(error, subprocess.CalledProcessError):
            code = error.returncode if error.returncode > 0 else 128 - error.returncode
        print(json.dumps({'status': 'failed', 'error': outcome['error']}), file=sys.stderr, flush=True)
    finally:
        try:
            unchanged = initial_source is not None and source_identity(REPO) == initial_source and all(
                sha256(path) == digest for path, digest in input_hashes.items())
        except Exception:
            unchanged = False
        if not unchanged:
            outcome.update(status='failed', identity_error='source/input identity changed or could not be verified')
            code = 1
        outcome.update(finished_utc=utc_now(), exit_code=code, source_and_inputs_unchanged=unchanged)
        write_json(RECORDS / 'outcome.json', outcome)
    print(json.dumps(outcome), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
PY
