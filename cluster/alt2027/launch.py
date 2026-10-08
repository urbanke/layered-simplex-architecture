"""Explicit launch guards and immutable resource records; no scheduler calls to submit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import sysconfig
import uuid
from datetime import UTC, datetime
from pathlib import Path

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "PYTHONNOUSERSITE": "1",
}
RECORDED_ENV = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_ARRAY_TASK_MIN",
    "SLURM_ARRAY_TASK_MAX",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_QOS",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_CPUS_PER_TASK",
    "SLURM_NTASKS",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_RESTART_COUNT",
    "SLURM_STEP_ID",
    "LOADEDMODULES",
    "_LMFILES_",
)


def checked(command, *, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def scheduler_check(args, purpose):
    hostname = socket.gethostname().split(".")[0]
    if args.host_profile == "laptop":
        if args.workers > 10:
            raise ValueError("laptop is capped at 10 worker processes")
        if os.environ.get("SLURM_JOB_ID") or hostname.lower().startswith("jed"):
            raise ValueError("use the SCITAS profile on the cluster")
        return None
    required = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")
    if any(not os.environ.get(name) for name in required):
        raise ValueError("SCITAS work requires a compute-node Slurm allocation")
    # salloc can set JOB_ID while leaving the shell on the login node. Verify
    # actual node membership instead of accepting JOB_ID alone.
    nodes = checked(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    ).splitlines()
    if hostname not in {node.split(".")[0] for node in nodes}:
        raise ValueError("current host is outside the allocated compute nodes")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if (
        int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1
        or int(os.environ.get("SLURM_NTASKS", "0")) != 1
    ):
        raise ValueError("one node and one Slurm task are required per controller")
    if args.workers > min(cpus, 72):
        raise ValueError("SCITAS workers exceed allocated CPUs or the 72-process cap")
    job = checked(["scontrol", "--oneliner", "show", "job", os.environ["SLURM_JOB_ID"]])
    qos = re.search(r"(?:^|\s)QOS=(\S+)", job)
    if purpose == "production" and (qos is None or qos.group(1) == "debug"):
        raise ValueError("production requires a known non-debug Slurm QOS")
    return {"allocated_nodes": nodes, "job_description": job}


def check_plan(specification, source):
    """Bind preflight context as well as campaign launches to the current source."""
    from lsa.alt.artifacts import canonical_hash

    if source["dirty"]:
        raise ValueError("launch requires a clean source identity")
    purpose = specification.get("purpose")
    if purpose not in ("validation", "production", "smoke"):
        raise ValueError("plan must declare its purpose")
    protocol = specification.get("protocol")
    if not isinstance(protocol, dict) or specification.get(
        "protocol_sha256"
    ) != canonical_hash(protocol):
        raise ValueError("plan protocol hash mismatch")
    if (
        specification.get("source_commit") != source["commit"]
        or specification.get("source_tree_sha256") != source["tree_sha256"]
    ):
        raise ValueError("plan was prepared for a different source identity")
    if purpose == "production" and protocol.get("status") != "frozen":
        raise ValueError("production requires a frozen protocol")
    return purpose


def check_numerical_inputs(specification, engine, power):
    from lsa.alt.artifacts import canonical_hash
    from lsa.alt.distributed import engine_options, power_options, validate_plan

    validate_plan(specification)
    if specification["engine_options_sha256"] != canonical_hash(engine_options(engine)):
        raise ValueError("engine configuration differs from the plan")
    if specification["power_settings_sha256"] != canonical_hash(power_options(power)):
        raise ValueError("power settings differ from the plan")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "expected-commit", "plan", "out", "python", "engine-config"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--power-settings")
    for name in ("workers", "worker-index", "worker-count"):
        parser.add_argument(f"--{name}", required=True, type=int)
    parser.add_argument("--host-profile", required=True, choices=("laptop", "scitas"))
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        raise ValueError("expected commit must be a full lowercase 40-digit Git SHA")
    for name in (
        "repo",
        "plan",
        "out",
        "python",
        "engine_config",
        "calibration",
        "power_settings",
    ):
        if (
            getattr(args, name) is not None
            and not Path(getattr(args, name)).is_absolute()
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be an absolute path")
    if (
        args.workers < 1
        or args.worker_count < 1
        or not 0 <= args.worker_index < args.worker_count
    ):
        raise ValueError("invalid worker count/index")
    if sys.prefix == sys.base_prefix:
        raise ValueError(
            "the supplied interpreter must belong to an explicit virtualenv"
        )
    if (
        not os.path.samefile(args.python, sys.executable)
        or Path(args.python).parent.parent.resolve() != Path(sys.prefix).resolve()
    ):
        raise ValueError("running interpreter differs from --python")
    repo, plan, out = (
        Path(args.repo).resolve(),
        Path(args.plan).resolve(strict=True),
        Path(args.out).resolve(),
    )
    if Path(__file__).resolve() != repo / "cluster/alt2027/launch.py":
        raise ValueError("launcher must belong to the supplied --repo checkout")
    if checked(["git", "rev-parse", "--show-toplevel"], cwd=repo) != str(repo):
        raise ValueError("--repo must name the Git repository root")
    actual = checked(["git", "rev-parse", "HEAD"], cwd=repo)
    if actual != args.expected_commit:
        raise ValueError(f"expected commit {args.expected_commit}, found {actual}")
    if checked(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo):
        raise ValueError("launch requires a clean committed checkout")
    if out == repo:
        raise ValueError("output root must not be the repository root")
    if (
        out.is_relative_to(repo)
        and subprocess.run(
            [
                "git",
                "check-ignore",
                "--quiet",
                "--no-index",
                str(out.relative_to(repo)),
            ],
            cwd=repo,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("an output root inside the checkout must be Git-ignored")
    os.environ.update(THREAD_ENV)
    os.environ["PYTHONPATH"] = str(repo / "src")
    sys.path.insert(0, str(repo / "src"))
    from lsa.alt.artifacts import source_identity

    source = source_identity(repo)
    specification = json.loads(plan.read_text())
    purpose = check_plan(specification, source)
    if args.preflight and purpose != "validation":
        raise ValueError("tiny preflight requires a validation-purpose plan")
    if args.preflight and (
        args.workers != 1 or args.worker_count != 1 or args.worker_index != 0
    ):
        raise ValueError("preflight uses exactly one worker/controller")
    if purpose == "production" and not args.calibration:
        raise ValueError("production requires --calibration")
    scheduler = scheduler_check(args, purpose)
    inputs = {
        "plan": plan,
        "engine_config": Path(args.engine_config).resolve(strict=True),
    }
    for name in ("calibration", "power_settings"):
        if getattr(args, name):
            inputs[name] = Path(getattr(args, name)).resolve(strict=True)
    fingerprints = {name: sha256(path) for name, path in inputs.items()}
    engine = json.loads(inputs["engine_config"].read_text())
    power = (
        json.loads(inputs["power_settings"].read_text())
        if "power_settings" in inputs
        else {}
    )
    if engine.get("store") and not Path(engine["store"]["path"]).is_absolute():
        raise ValueError("engine store path must be absolute")
    check_numerical_inputs(specification, engine, power)
    launch_id = f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))}-{args.worker_index}-{uuid.uuid4().hex}"
    launches = out / "launches"
    launches.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        "-u",
        str(repo / "scripts/alt_distributed.py"),
        "work",
        "--repo",
        str(repo),
        "--plan",
        str(plan),
        "--out",
        str(out),
        "--workers",
        str(args.workers),
        "--worker-index",
        str(args.worker_index),
        "--worker-count",
        str(args.worker_count),
        "--host-profile",
        args.host_profile,
        "--engine-config",
        str(inputs["engine_config"]),
    ]
    for name in ("calibration", "power_settings"):
        if name in inputs:
            command.extend([f"--{name.replace('_', '-')}", str(inputs[name])])
    record = {
        "launch_id": launch_id,
        "started_utc": datetime.now(UTC).isoformat(),
        "mode": "tiny_environment_preflight"
        if args.preflight
        else "distributed_worker",
        "expected_commit": args.expected_commit,
        "actual_commit": actual,
        "source_tree_sha256": source["tree_sha256"],
        "source_files_sha256": source["files"],
        "repo": str(repo),
        "plan": str(plan),
        "plan_sha256": sha256(plan),
        "input_files": {
            name: {"path": str(path), "sha256": fingerprints[name]}
            for name, path in inputs.items()
        },
        "protocol_sha256": specification["protocol_sha256"],
        "purpose": purpose,
        "host_profile": args.host_profile,
        "workers": args.workers,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_compiler": platform.python_compiler(),
        "python_build": platform.python_build(),
        "python_build_compiler": sysconfig.get_config_var("CC"),
        "virtualenv_supplied": args.python,
        "packages": dict(
            sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            )
        ),
        "environment": {
            name: os.environ.get(name)
            for name in (*RECORDED_ENV, *THREAD_ENV, "PYTHONPATH")
        },
        "scheduler": scheduler,
        "worker_command": None if args.preflight else command,
        "wrapper_sha256": {
            str(p.relative_to(repo)): sha256(p)
            for p in sorted((repo / "cluster/alt2027").glob("*"))
            if p.is_file()
        },
    }
    write_new(launches / f"{launch_id}.json", record)
    status = 1
    error = None
    try:
        if args.preflight:
            from preflight import reference_preflight, store_preflight

            result = store_preflight(repo, engine, reference_preflight())
            write_new(launches / f"{launch_id}-preflight.json", result)
            status = 0 if result["status"] == "passed" else 1
        else:
            status = subprocess.call(command, cwd=repo)
            if status < 0:
                status = 128 - status
    except Exception as exception:  # noqa: BLE001 - preserve failures in immutable exit metadata
        error = {"type": type(exception).__name__, "message": str(exception)}
        print(f"launch failed: {exception}", file=sys.stderr)
    except KeyboardInterrupt:
        status = 130
        error = {"type": "KeyboardInterrupt", "message": "launch interrupted"}
    finally:
        try:
            final_source = source_identity(repo)
            unchanged = (
                not final_source["dirty"]
                and final_source["commit"] == actual
                and final_source["tree_sha256"] == source["tree_sha256"]
                and all(
                    sha256(path) == fingerprints[name] for name, path in inputs.items()
                )
            )
        except Exception as exception:  # noqa: BLE001 - an unverifiable identity fails closed
            unchanged = False
            error = error or {
                "type": type(exception).__name__,
                "message": str(exception),
            }
        if not unchanged:
            status = 1
            print(
                "launch failed: source or input identity changed during execution",
                file=sys.stderr,
            )
        write_new(
            launches / f"{launch_id}-exit.json",
            {
                "exit_code": status,
                "finished_utc": datetime.now(UTC).isoformat(),
                "source_and_inputs_unchanged": unchanged,
                "error": error,
            },
        )
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"launch refused: {error}", file=sys.stderr)
        sys.exit(2)
