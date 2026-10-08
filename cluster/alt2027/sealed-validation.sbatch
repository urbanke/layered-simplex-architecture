#!/usr/bin/env bash
# Explicit validation only: never submits jobs or installs packages.
# Supply --cpus-per-task, --mem, --time and --output at submission.
# Modes: setup, sealed_store, sealed_profile, kernel, prior, power, depth,
# chain, regressions. setup builds a NEW native binary and configures a NEW
# engine file. Every other mode runs one complete immutable suite at ALT_OUT.
# Companion launch records are written to ALT_OUT.launch (also NEW).
# Required: ALT_REPO ALT_EXPECTED_COMMIT ALT_PYTHON ALT_VALIDATION_MODE
# ALT_ENGINE_CONFIG ALT_OUT ALT_WORKERS. Optional ALT_VALIDATION_CONFIG selects
# an explicit suite specification; otherwise the committed specification is used.
# setup additionally requires ALT_KERNEL_STORE ALT_STORE_SPEC ALT_NATIVE_BINARY;
# ALT_NATIVE_CC defaults to cc. sealed_profile requires ALT_LEGACY_ENGINE_CONFIG.
#SBATCH --job-name=lsa-sealed-check
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL

set -euo pipefail
: "${ALT_REPO:?absolute clean checkout required}"
: "${ALT_EXPECTED_COMMIT:?full audited commit required}"
: "${ALT_PYTHON:?absolute Linux virtualenv interpreter required}"
: "${ALT_VALIDATION_MODE:?explicit validation mode required}"
: "${ALT_ENGINE_CONFIG:?absolute engine configuration path required}"
: "${ALT_OUT:?absolute NEW output root required}"
: "${ALT_WORKERS:?explicit process count required}"
: "${SLURM_JOB_ID:?run only in a Slurm allocation}"
: "${SLURM_CPUS_PER_TASK:?request --cpus-per-task explicitly}"
[[ "$ALT_PYTHON" == /* && -x "$ALT_PYTHON" ]] || {
  echo 'ALT_PYTHON must name an absolute executable virtualenv interpreter.' >&2
  exit 2
}
[[ "$ALT_WORKERS" =~ ^[1-9][0-9]*$ && "$SLURM_CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || {
  echo 'Worker and allocated CPU counts must be positive integers.' >&2
  exit 2
}
(( ALT_WORKERS <= SLURM_CPUS_PER_TASK && ALT_WORKERS <= 72 )) || {
  echo 'Requested workers exceed allocated CPUs or the Jed 72-process cap.' >&2
  exit 2
}
case "$ALT_VALIDATION_MODE" in
  sealed_store|power|depth) ;;
  setup|sealed_profile|kernel|prior|chain|regressions)
    (( ALT_WORKERS == 1 )) || { echo 'This mode is serial; set ALT_WORKERS=1.' >&2; exit 2; } ;;
  *) echo 'Unknown ALT_VALIDATION_MODE.' >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONNOUSERSITE=1
export PYTHONPATH="$ALT_REPO/src"

# The inline process only guards/records execution. Numerical drivers are real
# script subprocesses, so multiprocessing spawn can import their main modules.
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
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(name, *, exists=True):
    raw = os.environ[name]
    path = Path(raw)
    require(path.is_absolute(), f"{name} must be an absolute path")
    if not exists:
        require(not path.is_symlink(), f"{name} cannot be a symlink output")
    return path.resolve(strict=exists)


def checked(args):
    return subprocess.check_output(args, text=True).strip()


repo = absolute("ALT_REPO")
python = Path(os.environ["ALT_PYTHON"])
require(python.is_absolute() and python.is_file() and os.access(python, os.X_OK),
        "ALT_PYTHON must be an absolute executable")
# Preserve the virtualenv symlink path: resolving it would run the base Python.
mode = os.environ["ALT_VALIDATION_MODE"]
out = absolute("ALT_OUT", exists=False)
launch = Path(str(out) + ".launch")
engine_path = absolute("ALT_ENGINE_CONFIG", exists=mode != "setup")
expected = os.environ["ALT_EXPECTED_COMMIT"]
workers = int(os.environ["ALT_WORKERS"])
require(re.fullmatch(r"[0-9a-f]{40}", expected), "full lowercase commit SHA required")
require(sys.platform == "linux" and sys.prefix != sys.base_prefix,
        "the explicit Linux virtualenv is required")
require(os.path.samefile(sys.executable, python), "interpreter differs from ALT_PYTHON")
require(int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) == 1
        and int(os.environ.get("SLURM_NTASKS", "0")) == 1,
        "one allocated node and one Slurm task are required")
require(not os.environ.get("SLURM_ARRAY_TASK_ID"), "validation wrapper is not an array")
cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
require(1 <= workers <= min(cpus, 72), "worker count exceeds allocation or cap")
require(mode in ("sealed_store", "power", "depth") or workers == 1,
        "this mode is serial")
if hasattr(os, "sched_getaffinity"):
    require(len(os.sched_getaffinity(0)) >= workers,
            "process CPU affinity is smaller than requested worker count")
nodes = checked(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]).splitlines()
require(socket.gethostname().split(".")[0] in {n.split(".")[0] for n in nodes},
        "current host is outside the allocated compute nodes")
require(not out.exists() and not out.is_symlink(), "ALT_OUT already exists")
require(not launch.exists() and not launch.is_symlink(), "launch record already exists")
require(not out.is_relative_to(repo), "keep validation output outside the clean checkout")
os.chdir(repo)
from lsa.alt import artifacts
from lsa.alt.depth import DepthEvaluator, StoreConfig

require(Path(artifacts.__file__).resolve() == repo / "src/lsa/alt/artifacts.py",
        "ALT package did not load from selected checkout")
source = artifacts.source_identity(repo)
require(not source["dirty"] and source["commit"] == expected,
        "checkout is dirty or differs from ALT_EXPECTED_COMMIT")

spec_names = {
    "sealed_store": "sealed-store-validation.json",
    "sealed_profile": "sealed-profile-validation.json",
    "kernel": "kernel-validation.json", "prior": "prior-validation.json",
    "power": "power-validation.json", "depth": "depth-validation.json",
    "chain": "bible-chain-validation.json",
}
inputs = {}
config = None
if mode in spec_names:
    config = (absolute("ALT_VALIDATION_CONFIG")
              if os.environ.get("ALT_VALIDATION_CONFIG")
              else (repo / "experiments/alt2027" / spec_names[mode]).resolve(strict=True))
    inputs[str(config)] = artifacts.sha256(config)
else:
    require(not os.environ.get("ALT_VALIDATION_CONFIG"),
            "ALT_VALIDATION_CONFIG does not apply to this mode")


def engine_identity(path, *, sealed):
    options = artifacts.read_json(path)
    require(options.get("mode") == "store", "validation requires a store engine")
    store = options["store"]
    require(store.get("format", "legacy") == ("sealed" if sealed else "legacy"),
            "engine store format differs from selected route")
    if store.get("interpolation_backend", "python") == "native":
        library = Path(store["native_library_path"])
        require(library.is_absolute(), "native library path must be absolute")
        from lsa.alt.sealed_native import NativeInterpolator, manifest_path
        native = NativeInterpolator(library, store["native_library_sha256"])
        native.check_unchanged()
        for item in (library, manifest_path(library)):
            inputs.setdefault(str(item), artifacts.sha256(item))
        if os.environ.get("ALT_NATIVE_BINARY"):
            require(library.resolve() == absolute("ALT_NATIVE_BINARY"),
                    "ALT_NATIVE_BINARY differs from engine selection")
    with DepthEvaluator(mode="store", store=StoreConfig(**store),
                        prediction_tolerance=options.get("prediction_tolerance", 1e-3)) as engine:
        return engine.configuration


native_binary = store = spec = None
legacy_path = None
identity = legacy_identity = None
if mode == "setup":
    require(not engine_path.exists() and not engine_path.is_symlink(),
            "setup engine output already exists")
    store = absolute("ALT_KERNEL_STORE")
    spec = absolute("ALT_STORE_SPEC")
    native_binary = absolute("ALT_NATIVE_BINARY", exists=False)
    require(not native_binary.is_relative_to(repo), "native binary must stay outside Git")
    require(not native_binary.exists() and not native_binary.is_symlink()
            and not Path(str(native_binary) + ".json").exists(),
            "setup native binary/manifest already exists")
    specification = artifacts.read_json(spec)
    require(specification.get("settings", {}).get("format") == "sealed",
            "setup requires an explicit pinned sealed-store specification")
    require(specification["settings"].get("saddle_min_depth") is None,
            "sealed setup cannot use the legacy direct route")
    for path in (spec, store / "plan.json", store / "manifest.json"):
        inputs[str(path)] = artifacts.sha256(path)
else:
    inputs[str(engine_path)] = artifacts.sha256(engine_path)
    identity = engine_identity(engine_path, sealed=True)
    if mode == "sealed_profile":
        legacy_path = absolute("ALT_LEGACY_ENGINE_CONFIG")
        inputs[str(legacy_path)] = artifacts.sha256(legacy_path)
        legacy_identity = engine_identity(legacy_path, sealed=False)

# This sibling holds orchestration logs; suite-owned ALT_OUT remains absent.
launch.mkdir(parents=True, exist_ok=False)
os.environ["MPLCONFIGDIR"] = str(launch / "matplotlib")


def write_new(name, value):
    with (launch / name).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check_inputs():
    require(artifacts.source_identity(repo) == source, "source changed during validation")
    for path, digest in inputs.items():
        require(artifacts.sha256(path) == digest, f"frozen input changed: {path}")


base = [str(python), "-u"]
experiments = str(repo / "scripts/alt_experiments.py")
command = None
if mode == "setup":
    command = base + [str(repo / "scripts/alt_build_sealed_native.py"),
                      "--out", str(native_binary), "--cc", os.environ.get("ALT_NATIVE_CC", "cc")]
elif mode == "sealed_store":
    command = base + [str(repo / "scripts/alt_validate_kernel_store.py"),
                      "--repo", str(repo), "--engine-config", str(engine_path),
                      "--config", str(config), "--workers", str(workers), "--out", str(out)]
elif mode == "sealed_profile":
    command = base + [str(repo / "scripts/alt_validate_sealed_profiles.py"),
                      "--repo", str(repo), "--config", str(config),
                      "--candidate-engine", str(engine_path),
                      "--legacy-engine", str(legacy_path), "--out", str(out)]
elif mode == "regressions":
    command = base + [str(repo / "scripts/alt_admission.py"), "regressions",
                      "--repo", str(repo), "--out", str(out)]
else:
    command = base + [experiments, "--repo", str(repo), "calibrate", "--suite", mode,
                      "--config", str(config), "--engine-config", str(engine_path),
                      "--out", str(out)]
    if mode in ("power", "depth"):
        command += ["--workers", str(workers)]

scheduler_keys = (
    "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION",
    "SLURM_JOB_QOS", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU", "SLURM_STEP_ID", "SLURM_RESTART_COUNT", "LOADEDMODULES", "_LMFILES_",
)
write_new("started.json", {
    "purpose": "sealed-engine-setup" if mode == "setup" else "validation",
    "mode": mode, "started_utc": now(), "source": source,
    "wrapper_sha256": artifacts.sha256(repo / "cluster/alt2027/sealed-validation.sbatch"),
    "python": str(python), "command": command, "output": str(out),
    "configuration_path": str(config) if config else None,
    "engine_path": str(engine_path), "engine_identity": identity,
    "legacy_engine_path": str(legacy_path) if legacy_path else None,
    "legacy_engine_identity": legacy_identity, "inputs_sha256": inputs,
    "workers": workers, "allocated_cpus": cpus, "allocated_nodes": nodes,
    "environment": artifacts.environment(),
    "scheduler": {key: os.environ.get(key) for key in scheduler_keys},
})
print(json.dumps({"mode": mode, "launch_record": str(launch), "command": command}), flush=True)
start = time.perf_counter()
usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
status, error = 1, None
commands = []
try:
    def execute(args):
        before = time.perf_counter()
        value = subprocess.run(args, check=False).returncode
        commands.append({"command": args, "returncode": value,
                         "seconds": time.perf_counter() - before})
        require(value == 0, f"validation subprocess exited {value}")

    execute(command)
    if mode == "setup":
        from lsa.alt.sealed_native import manifest_path
        native_sha = artifacts.sha256(native_binary)
        configure = base + [experiments, "--repo", str(repo), "configure-engine",
                            "--store-directory", str(store), "--store-spec", str(spec),
                            "--native-library-path", str(native_binary),
                            "--native-library-sha256", native_sha,
                            "--out", str(launch / "engine-config.json")]
        # Configure privately, then publish exclusively. Concurrent setup jobs
        # cannot overwrite the same selected engine output.
        execute(configure)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(launch / "engine-config.json", engine_path)
        identity = engine_identity(engine_path, sealed=True)
        out.mkdir(parents=True, exist_ok=False)
        artifacts.write_json(out / "result.json", {
            "status": "passed", "purpose": "engine-setup-not-calibration",
            "engine_path": str(engine_path), "engine_sha256": artifacts.sha256(engine_path),
            "engine_identity": identity, "native_binary_path": str(native_binary),
            "native_binary_sha256": native_sha,
            "native_manifest_sha256": artifacts.sha256(manifest_path(native_binary)),
            "store_spec_path": str(spec), "store_spec_sha256": artifacts.sha256(spec),
        })
        inputs[str(engine_path)] = artifacts.sha256(engine_path)
    else:
        artifacts.verify_run(out)
        require(engine_identity(engine_path, sealed=True) == identity,
                "engine identity changed during validation")
        if legacy_path:
            require(engine_identity(legacy_path, sealed=False) == legacy_identity,
                    "legacy engine changed during cross-provider validation")
    check_inputs()
    status = 0
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}"
finally:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write_new("exit.json", {
        "finished_utc": now(), "returncode": status, "error": error,
        "seconds": time.perf_counter() - start, "commands": commands,
        "children_user_cpu_seconds": usage.ru_utime - usage_start.ru_utime,
        "children_system_cpu_seconds": usage.ru_stime - usage_start.ru_stime,
        "children_reported_maxrss_kib": usage.ru_maxrss,
        "memory_note": "Linux child maximum, not simultaneous aggregate worker RSS",
    })
if error:
    print(error, file=sys.stderr)
raise SystemExit(status)
PY
