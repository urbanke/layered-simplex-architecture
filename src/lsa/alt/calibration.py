"""Reproducible entry points for the five declared numerical check suites."""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

from .artifacts import Run, canonical_hash, read_json, verify_run, write_json

SUITES = ("kernel", "prior", "depth", "power", "chain")


def depth_shards(config, workers):
    """Partition cases without changing their predeclared random draws."""
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not config["cases"]:
        raise ValueError("depth calibration requires nonempty cases")
    ids = [c["id"] for c in config["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("depth calibration case IDs must be unique")
    from .benchmark import TARGET_IDS

    resolved = deepcopy(config)
    for index, case in enumerate(resolved["cases"]):
        if case["kind"] == "synthetic":
            case.setdefault(
                "seed_coordinates",
                [resolved["seed"], TARGET_IDS.index(case["target"]), index],
            )
    count = min(workers, len(ids))
    return [dict(resolved, cases=resolved["cases"][i::count]) for i in range(count)]


def _depth_worker(job):
    from .depth_validation import run_depth_validation

    config, out, engine, repo = job
    return run_depth_validation(config, out, engine_config=engine, repo=repo)


def run_parallel_depth(config, out, *, engine_config, repo, workers):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    shards = depth_shards(config, workers)
    write_json(out / "config.json", config)
    write_json(out / "resolved-shards.json", shards)
    jobs = [
        (shard, out / f"shard-{i}", engine_config, repo)
        for i, shard in enumerate(shards)
    ]
    if len(jobs) == 1:
        results = [_depth_worker(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            results = list(pool.map(_depth_worker, jobs))
    cases = {case["id"]: case for result in results for case in result["cases"]}
    ordered = [cases[case["id"]] for case in config["cases"]]
    result = {
        "status": "passed"
        if all(r["status"] == "passed" for r in results)
        else "failed",
        "config_sha256": canonical_hash(config),
        "workers": len(jobs),
        "cases": ordered,
        "scope": "declared finite cases under outer-grid/window refinement",
    }
    write_json(out / "summary.json", result)
    return result


def kernel_status(summary, out, config):
    # The original stored-row tolerance and the stricter new direct low-count
    # tolerance stay separate, as declared before the profile comparisons.
    count = summary["cases"]
    rows = [
        json.loads(line) for line in (Path(out) / "rows.jsonl").read_text().splitlines()
    ]
    special = read_json(Path(out) / "special-functions.json")

    def within(value, tolerance):
        return math.isfinite(value) and abs(value) <= tolerance

    sealed_ids = {s["id"] for s in config.get("stores", [])
                  if s.get("format") == "sealed"}

    return (
        "passed"
        if (
            count > 0
            and len(rows) == count
            and summary["source_unchanged"]
            and summary["reference_converged_cases"] == count
            and summary["direct_column_nominal_passes"] == count
            and len(special) == len(config["special_function_cases"]) > 0
            and all(
                within(r["batched_direct_column_error_nats"], r["tolerance_nats"])
                and within(r["direct_column_refined_error_nats"], r["tolerance_nats"])
                for r in rows
            )
            and all(
                sealed_ids <= r["stores"].keys()
                and all(
                    r["stores"][name]["status"] == "evaluated"
                    and within(r["stores"][name]["error_nats"], r["tolerance_nats"])
                    and within(r["stores"][name]["matrix_error_nats"], r["tolerance_nats"])
                    for name in sealed_ids
                )
                for r in rows
            )
            and all(
                value["status"] == "unavailable"
                or (
                    value["status"] == "evaluated"
                    and within(value["error_nats"], config["nominal_tolerance_nats"])
                )
                for r in rows
                for value in r["stores"].values()
            )
            and all(
                within(r["meijer_error_nats"], config["reference_convergence_nats"])
                and within(
                    r.get("recursion_error_nats", 0),
                    config["reference_convergence_nats"],
                )
                for r in special
            )
        )
        else "failed"
    )


def execute(suite, config, out, *, repo, engine_config=None, workers=None):
    if suite == "kernel":
        from .kernel_validation import expand_cases, run_kernel_validation

        config = deepcopy(config)
        if engine_config and engine_config.get("store", {}).get("format") == "sealed":
            if engine_config.get("mode") != "store":
                raise ValueError("sealed kernel calibration requires mode=store")
            cases = expand_cases(config)
            config["stores"] = [{
                "id": "sealed_candidate", "format": "sealed",
                "path": engine_config["store"]["path"],
                "files_sha256": engine_config["store"]["files_sha256"],
                "depths": sorted({c["depth"] for c in cases}),
                "counts": sorted({c["r"] for c in cases}),
            }]
        result = run_kernel_validation(config, out)
        return {"status": kernel_status(result, out, config), "measurements": result}
    if suite == "prior":
        from .prior_validation import run_prior_validation

        return run_prior_validation(config, out)
    if suite == "power":
        from .power_validation import run_power_validation

        if workers is not None:
            config = dict(config, workers=workers)
        return run_power_validation(config, out)
    if suite == "depth":
        from .depth_validation_assessment import assess_depth_run

        result = run_parallel_depth(
            config,
            out,
            engine_config=engine_config,
            repo=repo,
            workers=workers if workers is not None else config.get("workers", 1),
        )
        assessed = assess_depth_run(
            out,
            Path(out).parent / "depth-assessment",
            engine_config=engine_config,
            run_supplemental=True,
        )
        return {
            "status": assessed["status"],
            "profiles": result,
            "assessment": assessed,
        }
    if suite == "chain":
        from .chain_validation import run_chain_validation
        from .depth import DepthEvaluator, StoreConfig

        with DepthEvaluator(
            mode=engine_config["mode"],
            store=StoreConfig(**engine_config["store"]),
            prediction_tolerance=engine_config.get("prediction_tolerance", 1e-3),
        ) as evaluator:
            return run_chain_validation(config, out, evaluator=evaluator, repo=repo)
    raise ValueError(f"unknown calibration suite: {suite}")


def run_suite(suite, config_path, out, *, repo, engine_path=None, workers=None):
    if suite not in SUITES:
        raise ValueError(f"unknown calibration suite: {suite}")
    if suite in ("depth", "chain") and engine_path is None:
        raise ValueError(f"{suite} calibration requires --engine-config")
    if workers is not None and (workers < 1 or suite not in ("depth", "power")):
        raise ValueError("--workers applies to depth/power and must be positive")
    config = read_json(config_path)
    inputs = [config_path]
    engine = read_json(engine_path) if engine_path else None
    if engine_path:
        inputs.append(engine_path)
    if suite == "chain":
        inputs.extend(Path(repo) / config[k] for k in ("corpus", "manifest"))
    elif suite == "depth":
        inputs.extend(
            {
                Path(repo) / case[k]
                for case in config["cases"]
                if case["kind"] != "synthetic"
                for k in ("corpus", "manifest")
            }
        )
    specification = {"suite": suite, "config": config, "workers_override": workers}
    with Run(
        out,
        repo=repo,
        protocol=specification,
        experiment=f"calibration_{suite}",
        purpose="validation",
        engine_identity=engine,
        inputs=inputs,
    ) as run:
        result = execute(
            suite,
            config,
            run.path / "data",
            repo=repo,
            engine_config=engine,
            workers=workers,
        )
        write_json(run.path / "result.json", result)
        if result["status"] != "passed":
            raise ArithmeticError(f"{suite} calibration has failing declared checks")
    verify_run(out)
    return result
