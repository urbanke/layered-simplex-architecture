#!/usr/bin/env bash
# Submit explicitly with measured --cpus-per-task, --mem, --time and --output.
# This wrapper never creates a plan, changes a source store, or submits a job.
#SBATCH --job-name=lsa-kernel-store
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL

set -euo pipefail
: "${ALT_REPO:?absolute clean checkout required}"
: "${ALT_EXPECTED_COMMIT:?full audited commit required}"
: "${ALT_PYTHON:?absolute Linux virtualenv Python required}"
: "${ALT_KERNEL_STORE:?absolute new store directory containing plan.json required}"
: "${ALT_KERNEL_PLAN_SHA256:?SHA256 of the frozen plan.json required}"
: "${ALT_BUILD_WORKERS:?explicit process limit required}"
: "${SLURM_JOB_ID:?run only in a Slurm allocation}"
: "${SLURM_CPUS_PER_TASK:?request --cpus-per-task explicitly}"
[[ "$ALT_PYTHON" == /* && -x "$ALT_PYTHON" ]] || {
  echo 'ALT_PYTHON must name an absolute executable virtualenv interpreter.' >&2
  exit 2
}
[[ "$ALT_BUILD_WORKERS" =~ ^[1-9][0-9]*$ && "$SLURM_CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || {
  echo 'Worker and allocated CPU counts must be positive integers.' >&2
  exit 2
}
(( ALT_BUILD_WORKERS <= SLURM_CPUS_PER_TASK && ALT_BUILD_WORKERS <= 72 )) || {
  echo 'Requested workers exceed allocated CPUs or the Jed 72-process cap.' >&2
  exit 2
}
[[ "${ALT_BUILD_SEAL:-0}" == 0 || "${ALT_BUILD_SEAL:-0}" == 1 ]] || {
  echo 'ALT_BUILD_SEAL must be 0 or 1.' >&2
  exit 2
}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONNOUSERSITE=1
export PYTHONPATH="$ALT_REPO/src"

# The guard and build both execute on an allocated compute node. The numerical
# command is a separate script, allowing multiprocessing's spawn start method.
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --kill-on-bad-exit=1 "$ALT_PYTHON" -u - <<'PY'
import json
import os
import re
import resource
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


def checked(args):
    return subprocess.check_output(args, text=True).strip()


def absolute(name):
    raw = os.environ[name]
    if not Path(raw).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(raw).resolve(strict=True)


repo = absolute("ALT_REPO")
store = absolute("ALT_KERNEL_STORE")
python = Path(os.environ["ALT_PYTHON"])
expected_commit = os.environ["ALT_EXPECTED_COMMIT"]
expected_plan = os.environ["ALT_KERNEL_PLAN_SHA256"]
if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
    raise ValueError("ALT_EXPECTED_COMMIT must be a full lowercase commit SHA")
if not re.fullmatch(r"[0-9a-f]{64}", expected_plan):
    raise ValueError("ALT_KERNEL_PLAN_SHA256 must be a full lowercase file SHA256")
if sys.platform != "linux" or sys.prefix == sys.base_prefix:
    raise ValueError("use the explicit Linux virtualenv, not a base interpreter")
if not os.path.samefile(sys.executable, python):
    raise ValueError("running interpreter differs from ALT_PYTHON")
if (int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1
        or int(os.environ.get("SLURM_NTASKS", "0")) != 1):
    raise ValueError("one node and one Slurm task are required")
if os.environ.get("SLURM_ARRAY_TASK_ID"):
    raise ValueError("use one level-parallel builder, not a Slurm array")
nodes = checked(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]).splitlines()
if socket.gethostname().split(".")[0] not in {node.split(".")[0] for node in nodes}:
    raise ValueError("current host is outside this allocation's compute nodes")

os.chdir(repo)
from lsa.alt.artifacts import environment, source_identity
from lsa.alt import sealed_store_build as builder

if Path(builder.__file__).resolve() != repo / "src/lsa/alt/sealed_store_build.py":
    raise ValueError("builder did not load from the selected checkout")


def frozen_inputs():
    source = source_identity(repo)
    if source["dirty"] or source["commit"] != expected_commit:
        raise ValueError("checkout is dirty or differs from ALT_EXPECTED_COMMIT")
    if builder.sha256(store / "plan.json") != expected_plan:
        raise ValueError("plan.json differs from its explicit launch hash")
    plan = builder.validate_plan(json.loads((store / "plan.json").read_text()))
    if plan["source_sha256"] != builder.source_identity():
        raise ValueError("plan was frozen with different builder/numerical source")
    return source


source = frozen_inputs()
command = [str(python), str(repo / "scripts/alt_build_kernel_store.py"), "build",
           "--store", str(store), "--workers", os.environ["ALT_BUILD_WORKERS"]]
if os.environ.get("ALT_BUILD_LEVELS"):
    command += ["--levels", os.environ["ALT_BUILD_LEVELS"]]
if os.environ.get("ALT_BUILD_SEAL", "0") == "1":
    command += ["--seal"]

launch = store / "launches" / uuid.uuid4().hex
launch.mkdir(parents=True)


def write_new(name, value):
    with (launch / name).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


slurm_keys = (
    "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION", "SLURM_JOB_QOS", "SLURM_CPUS_PER_TASK",
    "SLURM_NTASKS", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_STEP_ID",
    "SLURM_RESTART_COUNT", "LOADEDMODULES", "_LMFILES_",
)
write_new("started.json", {
    "purpose": "offline-kernel-store-construction", "started_utc": now(),
    "command": command, "plan_sha256": expected_plan, "source": source,
    "runtime": environment(), "allocated_nodes": nodes,
    "scheduler": {name: os.environ.get(name) for name in slurm_keys},
    "timing_scope": {
        "command_seconds": "builder subprocess, including checkpoint verification and optional sealing",
        "seconds": "builder subprocess plus final source and plan checks",
        "cpu_seconds": "child-process CPU difference across builder and final input checks",
    },
})
print(json.dumps({"launch_record": str(launch), "command": command}), flush=True)
start = time.perf_counter()
usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
command_seconds = None
status = 1
error = None
try:
    status = subprocess.run(command, check=False).returncode
    command_seconds = time.perf_counter() - start
    if frozen_inputs() != source:
        raise ValueError("source identity changed during construction")
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}"
    status = status or 1
finally:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write_new("exit.json", {
        "finished_utc": now(), "returncode": status, "error": error,
        "seconds": time.perf_counter() - start,
        "command_seconds": command_seconds,
        "children_user_cpu_seconds": usage.ru_utime - usage_start.ru_utime,
        "children_system_cpu_seconds": usage.ru_stime - usage_start.ru_stime,
        "children_reported_maxrss_kib": usage.ru_maxrss,
        "memory_note": "Lifetime Linux child maximum, including preflight; not simultaneous aggregate worker RSS",
    })
if error:
    print(error, file=sys.stderr)
raise SystemExit(status if status >= 0 else 128 - status)
PY
