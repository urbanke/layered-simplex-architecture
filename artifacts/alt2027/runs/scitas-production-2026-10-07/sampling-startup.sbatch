#!/usr/bin/env bash
#SBATCH --job-name=lsa-sampling-time
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/urbanke/alt2027-logs/sampling-time-%j.out
set -euo pipefail
alt_base=/scratch/urbanke/lsa-alt2027
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$alt_base/checkouts/f0a5ebd0a57d46caeb9bf45407c062f37b8d64bc/src"
srun --ntasks=1 --cpus-per-task=1 /home/urbanke/venvs/alt2027-py313-20261007/bin/python -u \
  "$alt_base/setup/throughput-f0a5ebd-20261007/sampling-startup.py"
