"""Deterministic, resumable execution of disjoint ALT experiment blocks."""

from __future__ import annotations

import argparse
import copy
import fcntl
import os
import platform
import re
import socket
import subprocess
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from .artifacts import (
    canonical_hash,
    read_json,
    sha256,
    source_identity,
    utc_now,
    verify_run,
    write_json,
)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def engine_options(options):
    """Fully bind numeric defaults while allowing a host-local store path."""
    from .depth import StoreConfig

    if set(options) - {"mode", "store", "prediction_tolerance"}:
        raise ValueError("unknown engine options")
    mode = options.get("mode", "reference")
    if mode not in ("reference", "store"):
        raise ValueError("invalid engine mode")
    store = options.get("store")
    if (mode == "store") != bool(store):
        raise ValueError("store mode requires explicit store settings")
    if store:
        store = asdict(StoreConfig(**store))
        store.pop("path")
        if store["native_library_path"] is not None:
            store["native_library_path"] = "/host-local-native"
    return {
        "mode": mode,
        "store": store,
        "prediction_tolerance": options.get("prediction_tolerance", 1e-3),
    }


def power_options(options):
    from .powers import PowerSettings

    return asdict(PowerSettings(**options))


def build_jobs(protocol, block_size, factorial_block_size, split_benchmark_n=False):
    from .scaling import cells

    positive_integer(block_size, "block size")
    positive_integer(factorial_block_size, "factorial block size")
    if not isinstance(split_benchmark_n, bool):
        raise TypeError("split_benchmark_n must be a boolean")
    jobs = []
    # Immutable JSON writers sort object keys. Job order must survive a saved
    # plan's round trip, independently of the input protocol's key order.
    for name, config in sorted(protocol["experiments"].items()):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("unsafe experiment name")
        if any(k in config for k in ("trial_start", "cell_ids", "sampling_n_values")):
            raise ValueError("parent protocol must contain the complete unsharded grid")
        if name.startswith("benchmark_"):
            selectors = [
                {"targets": [target], "sampling_n_values": config["n_values"]}
                for target in config["targets"]
            ]
            if name == "benchmark_primary" and split_benchmark_n:
                # The sampler still consumes the entire original n grid for
                # each global trial, preserving targets and paired counts.
                selectors = [
                    {**selector, "n_values": [n]}
                    for selector in selectors
                    for n in config["n_values"]
                ]
        elif name in ("spectrum", "factorial", "depth_scaling"):
            selectors = [{"cell_ids": [i]} for i, _ in enumerate(cells(name, config))]
        elif name in ("architecture", "bible", "bible_secondary", "validation"):
            jobs.append(
                {"id": name, "experiment": name, "config": copy.deepcopy(config)}
            )
            continue
        else:
            raise ValueError(f"unsupported experiment {name}")
        trials = positive_integer(config["trials"], "trials")
        size = factorial_block_size if name == "factorial" else block_size
        for index, selector in enumerate(selectors):
            for start in range(0, trials, size):
                local = copy.deepcopy(config)
                local.update(
                    copy.deepcopy(selector),
                    trial_start=start,
                    trials=min(size, trials - start),
                )
                jobs.append(
                    {
                        "id": f"{name}-{index:04d}-{start:07d}",
                        "experiment": name,
                        "config": local,
                    }
                )
    return jobs


def make_plan(
    protocol,
    *,
    repo,
    purpose,
    engine=None,
    power=None,
    block_size=100,
    factorial_block_size=500,
    batch_size=20,
    split_benchmark_n=False,
):
    if purpose not in ("smoke", "validation", "production"):
        raise ValueError("invalid purpose")
    positive_integer(batch_size, "batch size")
    source = source_identity(repo)
    if purpose == "production" and (
        source["dirty"] or protocol.get("status") != "frozen"
    ):
        raise ValueError("production plans require frozen protocol and clean source")
    return {
        "schema_version": 1,
        "purpose": purpose,
        "protocol": copy.deepcopy(protocol),
        "protocol_sha256": canonical_hash(protocol),
        "source_tree_sha256": source["tree_sha256"],
        "source_commit": source["commit"],
        "engine_options_sha256": canonical_hash(
            engine_options(engine or {"mode": "reference"})
        ),
        "power_settings_sha256": canonical_hash(power_options(power or {})),
        "block_size": block_size,
        "factorial_block_size": factorial_block_size,
        "batch_size": batch_size,
        "split_benchmark_n": split_benchmark_n,
        "jobs": build_jobs(
            protocol, block_size, factorial_block_size, split_benchmark_n
        ),
    }


def validate_plan(plan):
    if plan["schema_version"] != 1 or plan["purpose"] not in (
        "smoke",
        "validation",
        "production",
    ):
        raise ValueError("unsupported plan schema or purpose")
    if plan["protocol_sha256"] != canonical_hash(plan["protocol"]):
        raise ValueError("parent protocol hash mismatch")
    if plan["purpose"] == "production" and plan["protocol"].get("status") != "frozen":
        raise ValueError("production protocol must be frozen")
    positive_integer(plan["batch_size"], "batch size")
    expected = build_jobs(
        plan["protocol"],
        plan["block_size"],
        plan["factorial_block_size"],
        plan.get("split_benchmark_n", False),
    )
    if canonical_hash(plan["jobs"]) != canonical_hash(expected):
        raise ValueError("jobs differ from the exact declared partition")
    for name in (
        "source_tree_sha256",
        "engine_options_sha256",
        "power_settings_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", plan[name]):
            raise ValueError(f"invalid {name}")
    if not re.fullmatch(r"[0-9a-f]{40,64}", plan["source_commit"]):
        raise ValueError("invalid source commit")
    return plan


def job_protocol(plan, job):
    protocol = copy.deepcopy(plan["protocol"])
    protocol["experiments"][job["experiment"]] = copy.deepcopy(job["config"])
    protocol["execution_shard"] = {
        "parent_protocol_sha256": plan["protocol_sha256"],
        "job_id": job["id"],
        "plan_sha256": canonical_hash(plan),
    }
    return protocol


def derive_calibration(plan, job, calibration):
    """Only an exact deterministic subset inherits a parent's certificate."""
    validate_plan(plan)
    if job not in plan["jobs"]:
        raise ValueError("unknown execution shard")
    if (
        calibration.get("status") != "passed"
        or not calibration.get("required_checks_complete")
        or calibration.get("protocol_sha256") != plan["protocol_sha256"]
        or calibration.get("source_tree_sha256") != plan["source_tree_sha256"]
        or job["experiment"] not in calibration.get("covered_experiments", [])
    ):
        raise ValueError("parent calibration does not cover this plan")
    result = copy.deepcopy(calibration)
    result["protocol_sha256"] = canonical_hash(job_protocol(plan, job))
    result["derivation"] = {
        "type": "exact_deterministic_subset",
        "job_id": job["id"],
        "parent_calibration_sha256": canonical_hash(calibration),
        "parent_protocol_sha256": plan["protocol_sha256"],
        "plan_sha256": canonical_hash(plan),
    }
    return result


def validate_saved_admission(plan, job, record, calibration):
    if plan["purpose"] != "production":
        return
    if calibration is None:
        raise ValueError("saved production campaign lacks calibration")
    derive_calibration(plan, job, calibration)
    if calibration.get("engine_sha256_by_experiment", {}).get(
        job["experiment"]
    ) != canonical_hash(record["engine"]):
        raise ValueError("saved job engine is not covered by parent calibration")
    depth = record["engine"].get("depth")
    if depth is not None:
        from .depth import _json_hash

        if calibration.get("configuration_sha256") != _json_hash(depth):
            raise ValueError("saved depth configuration is not covered by calibration")


def _same_or_create(path, value):
    temporary = path.with_name(f".{path.name}-{uuid4().hex}")
    write_json(temporary, value)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if canonical_hash(read_json(path)) != canonical_hash(value):
                raise ValueError(f"immutable specification differs: {path}") from None
    finally:
        temporary.unlink()


def runtime_identity(plan):
    import mpmath
    import numpy
    import scipy

    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "mpmath": mpmath.__version__,
        "engine_options_sha256": plan["engine_options_sha256"],
        "power_settings_sha256": plan["power_settings_sha256"],
    }


def validate_record_runtime(record, runtime):
    env = record["environment"]
    reported_system = (
        "Darwin"
        if env["platform"].startswith(("macOS", "Darwin"))
        else env["platform"].split("-")[0]
    )
    if (
        env["python"].split()[0] != runtime["python"]
        or env["machine"] != runtime["machine"]
        or reported_system != runtime["system"]
        or any(env["packages"][k] != runtime[k] for k in ("numpy", "scipy", "mpmath"))
    ):
        raise ValueError("saved run runtime differs from campaign runtime")


def validate_record_engine(plan, experiment, engine, *, runtime=None):
    """Check scientific settings again when resuming or collecting saved jobs."""
    has_depth = experiment not in ("architecture", "bible_secondary")
    if not isinstance(engine, dict):
        raise ValueError("missing saved engine identity")  # noqa: TRY004
    expected_keys = {"depth"}
    depth = engine.get("depth")
    if has_depth:
        if not isinstance(depth, dict):
            raise ValueError("saved engine lacks depth configuration")
        options = {key: depth[key] for key in ("mode", "store", "prediction_tolerance")}
        if options["store"] is not None:
            options["store"] = {**options["store"], "path": "/host-local-store"}
        if canonical_hash(engine_options(options)) != plan["engine_options_sha256"]:
            raise ValueError("saved depth engine differs from plan settings")
        vendor = Path(__file__).parent / "_vendor" / "pmwm"
        expected = {
            "implementation_sha256": sha256(Path(__file__).with_name("depth.py")),
            "vendor_provenance_sha256": sha256(vendor / "provenance.json"),
            "vendor_source_sha256": {
                p.name: sha256(p) for p in sorted(vendor.glob("*.py"))
            },
            "native_kernel": False,
            "depth_truncation": False,
        }
        if (depth.get("store") or {}).get("format") == "sealed":
            expected["sealed_provider_sha256"] = sha256(
                Path(__file__).with_name("sealed_tables.py")
            )
            if depth["store"].get("interpolation_backend") == "native":
                identity = depth.get("sealed_native_identity", {})
                if (identity.get("binary_sha256") != depth["store"]["native_library_sha256"]
                        or identity.get("source_sha256") != sha256(Path(__file__).with_name("sealed_interp.c"))
                        or identity.get("wrapper_sha256") != sha256(Path(__file__).with_name("sealed_native.py"))):
                    raise ValueError("saved native interpolation identity differs from current source or pin")
        if any(depth.get(key) != value for key, value in expected.items()):
            raise ValueError("saved depth implementation differs from current source")
        if runtime is not None and depth.get("runtime") != {
            key: runtime[key]
            for key in ("python", "numpy", "scipy", "system", "machine")
        }:
            raise ValueError("saved depth runtime differs from campaign runtime")
    elif depth is not None:
        raise ValueError("unexpected depth engine for experiment")
    if experiment in (
        "benchmark_primary",
        "benchmark_powers",
        "spectrum",
        "factorial",
        "depth_scaling",
    ):
        expected_keys.add("execution")
        if engine.get("execution") != {
            "batch_size": plan["batch_size"],
            "batch_implementation_sha256": sha256(
                Path(__file__).with_name("batch_depth.py")
            ),
        }:
            raise ValueError("saved batch execution differs from plan")
    if experiment == "benchmark_powers":
        expected_keys.add("power")
        power = engine.get("power", {})
        if canonical_hash(power_options(power.get("settings", {}))) != plan[
            "power_settings_sha256"
        ] or power.get("implementation_sha256") != sha256(
            Path(__file__).with_name("powers.py")
        ):
            raise ValueError("saved power engine differs from plan")
    if set(engine) != expected_keys:
        raise ValueError("unexpected saved engine fields")


def _verify_job(plan, job, path):
    record = verify_run(path)
    if (
        record["protocol_sha256"] != canonical_hash(job_protocol(plan, job))
        or record["purpose"] != plan["purpose"]
        or record["experiment"] != job["experiment"]
        or record["source"]["tree_sha256"] != plan["source_tree_sha256"]
        or record["source"]["commit"] != plan["source_commit"]
    ):
        raise ValueError(f"completed job identity mismatch: {job['id']}")
    runtime = runtime_identity(plan)
    validate_record_engine(plan, job["experiment"], record["engine"], runtime=runtime)
    validate_record_runtime(record, runtime)
    return record


def _execute_job(plan, job, root, repo, engine_config, power_settings, calibration):
    from .cli import run_one

    root, repo = Path(root), Path(repo)
    final = root / "jobs" / job["id"]
    if final.exists():
        record = _verify_job(plan, job, final)
        validate_saved_admission(
            plan, job, record, read_json(calibration) if calibration else None
        )
        return {"id": job["id"], "status": "verified_existing"}
    # An atomic claim prevents accidental simultaneous launchers from duplicating work.
    # Advisory locks are released by the OS if a worker is killed. Keep the lock
    # inode: unlinking it would allow a second concurrent lock on a new inode.
    claim = (root / "claims" / job["id"]).open("a")
    try:
        fcntl.flock(claim.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if final.exists():
            record = _verify_job(plan, job, final)
            validate_saved_admission(
                plan, job, record, read_json(calibration) if calibration else None
            )
            return {"id": job["id"], "status": "verified_existing"}
        spec = root / "specs" / job["id"]
        spec.mkdir(exist_ok=True)
        protocol = job_protocol(plan, job)
        protocol_path = spec / "protocol.json"
        _same_or_create(protocol_path, protocol)
        local_calibration = None
        if calibration:
            local_calibration = spec / "calibration.json"
            _same_or_create(
                local_calibration, derive_calibration(plan, job, read_json(calibration))
            )
        args = SimpleNamespace(
            repo=repo,
            purpose=plan["purpose"],
            protocol=protocol_path,
            engine_config=engine_config,
            power_settings=power_settings,
            calibration=local_calibration,
            batch_size=plan["batch_size"],
        )
        attempt = root / "attempts" / f"{job['id']}-{uuid4().hex}"
        run_one(job["experiment"], protocol, args, attempt)
        record = _verify_job(plan, job, attempt)
        validate_saved_admission(
            plan, job, record, read_json(calibration) if calibration else None
        )
        attempt.rename(final)
        return {"id": job["id"], "status": "complete"}
    finally:
        claim.close()


def check_resources(workers, host_profile):
    positive_integer(workers, "workers")
    if host_profile == "laptop":
        if os.environ.get("SLURM_JOB_ID") or socket.gethostname().lower().startswith(
            "jed"
        ):
            raise ValueError("use the SCITAS profile on the cluster")
        limit = 10
    elif host_profile == "scitas":
        if not all(
            os.environ.get(k)
            for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")
        ):
            raise ValueError("SCITAS workers require a Slurm allocation")
        nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            text=True,
        ).splitlines()
        if socket.gethostname().split(".")[0] not in {n.split(".")[0] for n in nodes}:
            raise ValueError("current host is outside the allocated compute nodes")
        if (
            int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1
            or int(os.environ.get("SLURM_NTASKS", "0")) != 1
        ):
            raise ValueError("each controller requires one Slurm node and task")
        limit = min(72, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    else:
        raise ValueError("unknown host profile")
    if workers > limit:
        raise ValueError(f"worker count exceeds {host_profile} limit {limit}")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(key) != "1":
            raise ValueError(f"{key}=1 is required")


def work(
    plan,
    out,
    *,
    repo,
    workers,
    worker_index=0,
    worker_count=1,
    host_profile="laptop",
    engine_config=None,
    power_settings=None,
    calibration=None,
):
    validate_plan(plan)
    check_resources(workers, host_profile)
    positive_integer(worker_count, "worker count")
    if (
        isinstance(worker_index, bool)
        or not isinstance(worker_index, int)
        or not 0 <= worker_index < worker_count
    ):
        raise ValueError("worker index is outside the worker count")
    source = source_identity(repo)
    if (
        source["tree_sha256"] != plan["source_tree_sha256"]
        or source["commit"] != plan["source_commit"]
        or (plan["purpose"] == "production" and source["dirty"])
    ):
        raise ValueError("worker source differs from the plan")
    options = read_json(engine_config) if engine_config else {"mode": "reference"}
    powers = read_json(power_settings) if power_settings else {}
    if (
        canonical_hash(engine_options(options)) != plan["engine_options_sha256"]
        or canonical_hash(power_options(powers)) != plan["power_settings_sha256"]
    ):
        raise ValueError("worker numerical settings differ from the plan")
    if plan["purpose"] == "production" and not calibration:
        raise ValueError("production worker requires parent calibration")
    assigned = [
        job for i, job in enumerate(plan["jobs"]) if i % worker_count == worker_index
    ]
    if calibration:
        for job in assigned:
            derive_calibration(plan, job, read_json(calibration))
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "attempts", "claims", "specs", "workers"):
        (root / name).mkdir(exist_ok=True)
    _same_or_create(root / "plan.json", plan)
    _same_or_create(root / "runtime.json", runtime_identity(plan))
    parent_calibration = read_json(calibration) if calibration else None
    _same_or_create(
        root / "admission.json",
        {
            "calibration_sha256": canonical_hash(parent_calibration)
            if calibration
            else None,
            "purpose": plan["purpose"],
        },
    )
    if parent_calibration is not None:
        calibration = root / "parent-calibration.json"
        _same_or_create(calibration, parent_calibration)
    run_id = uuid4().hex
    record = {
        "started_utc": utc_now(),
        "host": socket.gethostname(),
        "worker_index": worker_index,
        "worker_count": worker_count,
        "processes": workers,
        "plan_sha256": canonical_hash(plan),
        "assigned_jobs": [job["id"] for job in assigned],
    }
    write_json(root / "workers" / f"{run_id}-started.json", record)
    results, failures = [], []
    queue = iter(assigned)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def submit():
            job = next(queue, None)
            if job is not None:
                future = pool.submit(
                    _execute_job,
                    plan,
                    job,
                    root,
                    repo,
                    engine_config,
                    power_settings,
                    calibration,
                )
                pending[future] = job["id"]
            return job is not None

        for _ in range(workers):
            if not submit():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job_id = pending.pop(future)
                try:
                    results.append(future.result())
                except Exception as error:  # noqa: BLE001 -- persist every worker failure
                    failures.append(
                        {
                            "id": job_id,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
            if not failures:
                for _ in done:
                    submit()
    record.update(
        finished_utc=utc_now(),
        results=results,
        failures=failures,
        status="failed" if failures else "complete",
    )
    write_json(root / "workers" / f"{run_id}-finished.json", record)
    if failures:
        raise RuntimeError(
            f"{len(failures)} job(s) failed; inspect immutable worker and attempt records"
        )
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument(
        "--purpose", choices=("smoke", "validation", "production"), required=True
    )
    prepare.add_argument("--block-size", type=int, default=100)
    prepare.add_argument("--factorial-block-size", type=int, default=500)
    prepare.add_argument("--batch-size", type=int, default=20)
    prepare.add_argument(
        "--split-benchmark-n",
        action="store_true",
        help="split primary benchmark jobs by sample size, preserving common samples",
    )
    worker = sub.add_parser("work")
    worker.add_argument("--workers", type=int, required=True)
    worker.add_argument("--worker-index", type=int, default=0)
    worker.add_argument("--worker-count", type=int, default=1)
    worker.add_argument("--host-profile", choices=("laptop", "scitas"), required=True)
    worker.add_argument("--calibration", type=Path)
    merger = sub.add_parser("merge")
    merger.add_argument("--runs", type=Path, required=True)
    for command in (worker, merger):
        command.add_argument("--plan", type=Path, required=True)
    for command in (prepare, worker):
        command.add_argument("--engine-config", type=Path)
        command.add_argument("--power-settings", type=Path)
    for command in (prepare, worker, merger):
        command.add_argument(
            "--repo", type=Path, default=Path(__file__).resolve().parents[3]
        )
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = make_plan(
            read_json(args.protocol),
            repo=args.repo,
            purpose=args.purpose,
            engine=read_json(args.engine_config)
            if args.engine_config
            else {"mode": "reference"},
            power=read_json(args.power_settings) if args.power_settings else {},
            block_size=args.block_size,
            factorial_block_size=args.factorial_block_size,
            batch_size=args.batch_size,
            split_benchmark_n=args.split_benchmark_n,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, result)
        print(
            f"Prepared {len(result['jobs'])} disjoint jobs; plan {canonical_hash(result)}"
        )
    elif args.command == "work":
        work(
            read_json(args.plan),
            args.out,
            repo=args.repo,
            workers=args.workers,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            host_profile=args.host_profile,
            engine_config=args.engine_config,
            power_settings=args.power_settings,
            calibration=args.calibration,
        )
    else:
        from .distributed_merge import merge_campaign

        merge_campaign(read_json(args.plan), args.runs, args.out, repo=args.repo)
    return 0
