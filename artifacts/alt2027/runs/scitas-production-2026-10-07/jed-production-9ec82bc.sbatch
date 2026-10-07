#!/usr/bin/env bash
# Submit from a configured Jed login shell; all numerical work runs on a node.
# Override --array explicitly for the chosen number of independent node workers.
#SBATCH --job-name=lsa-alt
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=/scratch/urbanke/alt2027-logs/alt-%A_%a.out
#SBATCH --export=ALL

set -euo pipefail
# Stop the sibling controllers if this controller reports a failure.
# Immutable failed attempts and already completed jobs remain in the output root.
alt_stop_array_on_error() {
  local alt_exit=$?
  trap - ERR
  echo "Controller failed with status $alt_exit; cancelling this production array." >&2
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    scancel "$SLURM_ARRAY_JOB_ID" || true
  fi
  exit "$alt_exit"
}
trap alt_stop_array_on_error ERR
: "${ALT_REPO:?absolute clean checkout required}"
: "${ALT_EXPECTED_COMMIT:?full audited commit required}"
: "${ALT_PLAN:?absolute immutable plan required}"
: "${ALT_OUT:?absolute shared output root required}"
: "${ALT_PYTHON:?absolute calibrated Linux virtualenv Python required}"
: "${ALT_ENGINE_CONFIG:?absolute store engine configuration required}"
: "${SLURM_ARRAY_TASK_ID:?submit as a contiguous zero-based array}"
: "${SLURM_ARRAY_TASK_COUNT:?missing array size}"
[[ "${SLURM_ARRAY_TASK_MIN:?}" == 0 && \
   "$((SLURM_ARRAY_TASK_MAX + 1))" == "$SLURM_ARRAY_TASK_COUNT" ]] || {
  echo 'Use a contiguous array 0..K-1; sparse arrays change work assignment.' >&2
  exit 2
}
[[ "${SLURM_CPUS_PER_TASK:?}" == 72 && "${SLURM_NTASKS:?}" == 1 && \
   "${SLURM_JOB_NUM_NODES:?}" == 1 ]] || {
  echo 'This template requires one node, one task and 72 CPUs per array element.' >&2
  exit 2
}
# The module environment must already match the explicitly chosen virtualenv.
# No automatic module, dependency, calibration, or scientific-plan changes.
alt_arguments=(--repo "$ALT_REPO" --expected-commit "$ALT_EXPECTED_COMMIT"
  --plan "$ALT_PLAN" --out "$ALT_OUT" --python "$ALT_PYTHON"
  --engine-config "$ALT_ENGINE_CONFIG" --workers 72
  --worker-index "$SLURM_ARRAY_TASK_ID" --worker-count "$SLURM_ARRAY_TASK_COUNT"
  --host-profile scitas)
if [[ -n "${ALT_CALIBRATION:-}" ]]; then alt_arguments+=(--calibration "$ALT_CALIBRATION"); fi
if [[ -n "${ALT_POWER_SETTINGS:-}" ]]; then alt_arguments+=(--power-settings "$ALT_POWER_SETTINGS"); fi
srun --nodes=1 --ntasks=1 --cpus-per-task=72 \
  bash "$ALT_REPO/cluster/alt2027/run_worker.sh" "${alt_arguments[@]}"
