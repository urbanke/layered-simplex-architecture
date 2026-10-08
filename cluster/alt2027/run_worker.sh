#!/usr/bin/env bash
set -euo pipefail

# No implicit interpreter: use the exact virtualenv supplied by the caller.
alt_python=''
alt_previous=''
for alt_argument in "$@"; do
  if [[ "$alt_previous" == '--python' ]]; then alt_python="$alt_argument"; fi
  alt_previous="$alt_argument"
done
[[ "$alt_python" == /* && -x "$alt_python" ]] || {
  echo 'Supply --python /absolute/path/to/venv/bin/python.' >&2
  exit 2
}
alt_wrapper_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONNOUSERSITE=1
unset PYTHONPATH
exec "$alt_python" "$alt_wrapper_dir/launch.py" "$@"
