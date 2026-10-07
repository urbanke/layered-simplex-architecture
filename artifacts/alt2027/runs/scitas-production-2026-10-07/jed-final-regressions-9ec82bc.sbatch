#!/usr/bin/env bash
#SBATCH --job-name=lsa-alt-regression
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/urbanke/alt2027-logs/regressions-%j.out
set -euo pipefail
alt_base=/scratch/urbanke/lsa-alt2027
alt_commit=9ec82bcef549ad593dee5450e974d83de44ddc77
alt_repo="$alt_base/checkouts/$alt_commit"
alt_python=/home/urbanke/venvs/alt2027-py313-20261007/bin/python
export PYTHONPATH="$alt_repo/src" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$alt_base/setup/matplotlib"
test "$(git -C "$alt_repo" rev-parse HEAD)" = "$alt_commit"
test -z "$(git -C "$alt_repo" status --porcelain)"
srun --ntasks=1 --cpus-per-task=4 "$alt_python" "$alt_repo/scripts/alt_admission.py" regressions --repo "$alt_repo" --out "$alt_base/runs/regressions-$alt_commit-001"
