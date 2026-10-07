#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 || $# -gt 9 ]]; then
  echo 'Usage: bash cluster/alt2027/laptop.sh REPO_ABS EXPECTED_COMMIT PLAN_ABS OUT_ABS VENV_PYTHON_ABS ENGINE_CONFIG_ABS [WORKERS=10] [CALIBRATION_ABS] [POWER_SETTINGS_ABS]' >&2
  exit 2
fi
alt_wrapper_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
alt_arguments=(--repo "$1" --expected-commit "$2" --plan "$3" --out "$4"
  --python "$5" --engine-config "$6" --workers "${7:-10}"
  --worker-index 0 --worker-count 1 --host-profile laptop)
if [[ -n "${8:-}" ]]; then alt_arguments+=(--calibration "$8"); fi
if [[ -n "${9:-}" ]]; then alt_arguments+=(--power-settings "$9"); fi
exec bash "$alt_wrapper_dir/run_worker.sh" "${alt_arguments[@]}"
