#!/usr/bin/env bash
#SBATCH --job-name=lsa-throughput
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/urbanke/alt2027-logs/throughput-%A-%a.out
set -euo pipefail
alt_base=/scratch/urbanke/lsa-alt2027
alt_repo="$alt_base/checkouts/f0a5ebd0a57d46caeb9bf45407c062f37b8d64bc"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$alt_repo/src"
export MPLCONFIGDIR="$alt_base/setup/matplotlib"
srun --ntasks=1 --cpus-per-task=1 /home/urbanke/venvs/alt2027-py313-20261007/bin/python -u \
  "$alt_base/setup/throughput-f0a5ebd-20261007/pilot.py" "${SLURM_ARRAY_TASK_ID:?}"
