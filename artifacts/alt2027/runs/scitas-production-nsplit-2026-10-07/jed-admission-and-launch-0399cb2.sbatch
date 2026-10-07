#!/usr/bin/env bash
#SBATCH --job-name=lsa-alt-admit-launch
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/urbanke/alt2027-logs/admission-launch-%j.out
set -euo pipefail
alt_base=/scratch/urbanke/lsa-alt2027
alt_commit=0399cb28b973e5550fcbcbce082e8d36c95c2e1e
alt_repo="$alt_base/checkouts/$alt_commit"
alt_python=/home/urbanke/venvs/alt2027-py313-20261007/bin/python
export PYTHONPATH="$alt_repo/src" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export MPLCONFIGDIR="$alt_base/setup/matplotlib"
test "$(git -C "$alt_repo" rev-parse HEAD)" = "$alt_commit"
test -z "$(git -C "$alt_repo" status --porcelain)"
srun --ntasks=1 --cpus-per-task=1 "$alt_python" "$alt_base/setup/launch-when-ready-0399cb2.py"
