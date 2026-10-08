#!/usr/bin/env bash
# Full store identity plus tiny numerical checks; not production calibration.
#SBATCH --job-name=lsa-alt-preflight
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/urbanke/alt2027-logs/preflight-%j.out
#SBATCH --export=ALL

set -euo pipefail
: "${ALT_REPO:?absolute clean checkout required}"
: "${ALT_EXPECTED_COMMIT:?full audited commit required}"
: "${ALT_PLAN:?absolute validation-purpose plan required}"
: "${ALT_OUT:?absolute preflight output root required}"
: "${ALT_PYTHON:?absolute Linux virtualenv Python required}"
: "${ALT_ENGINE_CONFIG:?absolute store engine configuration required}"
alt_arguments=(--repo "$ALT_REPO" --expected-commit "$ALT_EXPECTED_COMMIT"
  --plan "$ALT_PLAN" --out "$ALT_OUT" --python "$ALT_PYTHON"
  --engine-config "$ALT_ENGINE_CONFIG" --workers 1
  --worker-index 0 --worker-count 1 --host-profile scitas --preflight)
if [[ -n "${ALT_POWER_SETTINGS:-}" ]]; then alt_arguments+=(--power-settings "$ALT_POWER_SETTINGS"); fi
srun --nodes=1 --ntasks=1 --cpus-per-task=2 \
  bash "$ALT_REPO/cluster/alt2027/run_worker.sh" "${alt_arguments[@]}"
