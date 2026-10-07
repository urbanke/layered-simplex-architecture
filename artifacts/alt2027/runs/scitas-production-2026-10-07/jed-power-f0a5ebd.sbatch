#!/usr/bin/env bash
#SBATCH --job-name=lsa-alt-check
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/urbanke/alt2027-logs/calibration-%j.out
set -euo pipefail
alt_suite=${1:?suite required}
alt_base=/scratch/urbanke/lsa-alt2027
alt_commit=f0a5ebd0a57d46caeb9bf45407c062f37b8d64bc
alt_repo="$alt_base/checkouts/$alt_commit"
alt_python=/home/urbanke/venvs/alt2027-py313-20261007/bin/python
alt_engine="$alt_base/setup/jed-engine.json"
alt_out="$alt_base/runs/calibration-$alt_commit-001/$alt_suite"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$alt_repo/src"
export MPLCONFIGDIR="$alt_base/setup/matplotlib"
test "$(git -C "$alt_repo" rev-parse HEAD)" = "$alt_commit"
test -z "$(git -C "$alt_repo" status --porcelain)"
alt_config="$alt_repo/experiments/alt2027/$alt_suite-validation.json"
alt_options=()
case "$alt_suite" in
  kernel) alt_config="$alt_base/setup/jed-kernel-validation.json" ;;
  prior) ;;
  chain) alt_config="$alt_repo/experiments/alt2027/bible-chain-validation.json"; alt_options=(--engine-config "$alt_engine") ;;
  power) alt_options=(--workers 11) ;;
  depth) alt_options=(--workers 27 --engine-config "$alt_engine") ;;
  *) exit 2 ;;
esac
"$alt_python" - "$alt_repo" "$alt_base" "$alt_suite" "$alt_out" <<'PY'
import json,os,socket,subprocess,sys
from pathlib import Path
from lsa.alt.artifacts import write_json,sha256,source_identity,environment,utc_now
repo,base,suite,out=map(Path,sys.argv[1:])
allocated=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).splitlines()
assert socket.gethostname().split('.')[0] in {n.split('.')[0] for n in allocated}
required=27 if str(suite)=='depth' else 11 if str(suite)=='power' else 1
assert int(os.environ['SLURM_CPUS_PER_TASK'])>=required
prior=list((base/'runs/preflight-cc1524a636a5f1996b4fbff1668c477ed0e989b8/launches').glob('*-preflight.json'))
assert len(prior)==1 and json.loads(prior[0].read_text())['status']=='passed'
if str(suite)=='kernel':
    manifest=json.loads((base/'setup/probe-exact-subset.manifest.json').read_text())
    store=base/'reference/probe-exact-subset-20261007'
    for name,record in manifest['files'].items():
        assert sha256(store/name)==record['sha256'] and (store/name).stat().st_size==record['bytes']
records=base/'runs/calibration-f0a5ebd0a57d46caeb9bf45407c062f37b8d64bc-001/launches'
records.mkdir(parents=True,exist_ok=True)
write_json(records/f'{suite}-{os.environ["SLURM_JOB_ID"]}.json',{'started_utc':utc_now(),'suite':str(suite),'output':str(out),'source':source_identity(repo),'environment':environment(),'job_id':os.environ['SLURM_JOB_ID'],'allocated_nodes':allocated,'cpus':required,'driver_sha256':sha256(base/'setup/jed-power-f0a5ebd.sbatch'),'passed_preflight_sha256':sha256(prior[0])})
PY
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" \
  "$alt_python" -u "$alt_repo/scripts/alt_experiments.py" calibrate \
  --suite "$alt_suite" --config "$alt_config" --out "$alt_out" "${alt_options[@]}"
