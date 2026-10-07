"""Assemble production admission from the existing immutable validation suites.

This module runs no new numerical acceptance test and changes no tolerance.
Source reuse is an explicit, conservative file comparison, retained in the output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .artifacts import (
    Run,
    canonical_hash,
    read_json,
    sha256,
    source_identity,
    verify_run,
    write_json,
)
from .calibration import SUITES, kernel_status
from .distributed import (
    engine_options,
    power_options,
    validate_record_engine,
    validate_record_runtime,
)

SPECIFICATIONS = {
    "kernel": "kernel-validation.json",
    "prior": "prior-validation.json",
    "depth": "depth-validation.json",
    "power": "power-validation.json",
    "chain": "bible-chain-validation.json",
}
REGRESSION_COMMANDS = {
    "pytest": ["-m", "pytest", "-q"],
    "appendix_full": ["scripts/validate_appendix_c.py"],
}
ORCHESTRATION_FILES = {
    "src/lsa/alt/admission.py",
    "scripts/alt_admission.py",
    "tests/test_alt_admission.py",
}
POWER_FILES = {"src/lsa/alt/powers.py", "src/lsa/alt/power_validation.py"}


def normalized_specification(suite, config):
    result = copy.deepcopy(config)
    if suite in ("depth", "power"):
        result.pop("workers", None)
    if suite == "kernel":
        for store in result["stores"]:
            store.pop("path")
    return result


def source_reuse(suite, saved, current, repo):
    """Reject all unlisted scientific changes, including new/deleted modules."""
    if saved.get("dirty"):
        raise ValueError("validation source was not a clean frozen checkout")
    before, after = saved["files"], current["files"]
    changes = []
    for name in sorted(before.keys() | after.keys()):
        if before.get(name) == after.get(name):
            continue
        reason = None
        if name.endswith(".md") or name.startswith("manuscript/"):
            reason = "documentation_only"
        elif suite != "regressions" and name.startswith("cluster/alt2027/"):
            reason = "launch_wrapper_not_executed_by_this_numerical_suite"
        elif suite != "regressions" and name in ORCHESTRATION_FILES:
            reason = "admission_orchestration_only"
        elif suite != "regressions" and name.startswith("tests/"):
            reason = "test_source_not_executed_by_this_numerical_suite"
        elif suite not in ("power", "regressions") and name in POWER_FILES:
            reason = "powered_model_not_executed_by_this_suite"
        elif (
            name == "experiments/alt2027/protocol.json"
            and before.get(name)
            and after.get(name)
        ):
            original = subprocess.check_output(
                ["git", "show", f"{saved['commit']}:{name}"],
                cwd=repo,
            )
            if hashlib.sha256(original).hexdigest() != before[name]:
                raise ValueError(
                    "saved protocol source does not match its recorded commit"
                )
            old_protocol, new_protocol = (
                json.loads(original),
                read_json(Path(repo) / name),
            )
            old_status, new_status = (
                old_protocol.pop("status"),
                new_protocol.pop("status"),
            )
            if (
                old_protocol == new_protocol
                and old_status in ("implementation", "frozen")
                and new_status == "frozen"
            ):
                reason = "protocol_status_only_freeze"
        if reason is None:
            raise ValueError(
                f"{suite} evidence needs rerun after source change: {name}"
            )
        changes.append(
            {
                "path": name,
                "before_sha256": before.get(name),
                "after_sha256": after.get(name),
                "reason": reason,
            }
        )
    return {
        "original_commit": saved["commit"],
        "original_tree_sha256": saved["tree_sha256"],
        "current_tree_sha256": current["tree_sha256"],
        "permitted_changes": changes,
    }


def suite_result(suite, root, result, specification):
    """Check completion using existing suite results, never relaxed tolerances."""
    if result.get("status") != "passed":
        raise ValueError(f"{suite} result is not passed")
    if suite == "kernel":
        from .kernel_validation import expand_cases

        if read_json(root / "data/resolved-cases.json") != expand_cases(specification):
            raise ValueError("kernel resolved cases differ from the declared grid")
        if kernel_status(
            result["measurements"], root / "data", specification
        ) != "passed" or result["measurements"]["cases"] != len(
            expand_cases(specification)
        ):
            raise ValueError("kernel measured checks are incomplete or failed")
    elif suite == "prior":
        if not result.get("declared_suite_complete") or result.get("failed") != 0:
            raise ValueError("prior declared suite is incomplete")
    elif suite == "depth":
        assessment = result["assessment"]
        expected = {case["id"] for case in specification["cases"]}
        cases = assessment["cases"]
        if (
            assessment.get("status") != "passed"
            or not assessment.get("source_unchanged")
            or assessment.get("pending_case_ids")
            or len(cases) != len(expected)
            or {case["id"] for case in cases} != expected
            or any(case["status"] != "passed" for case in cases)
            or assessment.get("original_config_sha256") != canonical_hash(specification)
        ):
            raise ValueError(
                "depth assessment has missing, pending or failed declared cases"
            )
    elif suite == "power":
        profiles = result["profiles"]
        if (
            not result.get("source_unchanged")
            or len(profiles) != len(specification["targets"])
            or {row["target"] for row in profiles} != set(specification["targets"])
            or any(row["status"] != "passed" or row["failures"] for row in profiles)
        ):
            raise ValueError("power suite has missing or failed target profiles")
    elif suite == "chain" and (
        result["steps"] != specification["steps"]
        or result["depths"] != specification["depths"]
        or result["d"] != specification["d"]
    ):
        raise ValueError("chain suite does not cover its declared prefix/depths")


def read_evidence(suite, root, *, repo, current):
    root = Path(root)
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("validation artifacts must not contain symlinks")
    root = root.resolve()
    record = verify_run(root)
    expected_experiment = (
        "production_regressions" if suite == "regressions" else f"calibration_{suite}"
    )
    if record["purpose"] != "validation" or record["experiment"] != expected_experiment:
        raise ValueError(f"wrong validation Run type for {suite}")
    reuse = source_reuse(suite, record["source"], current, repo)
    result = read_json(root / "result.json")
    specification = read_json(root / "protocol.json")
    if suite == "regressions":
        if (
            specification != {"checks": REGRESSION_COMMANDS}
            or result.get("status") != "passed"
            or set(result["checks"]) != set(REGRESSION_COMMANDS)
            or any(row["exit_code"] != 0 for row in result["checks"].values())
        ):
            raise ValueError("full pytest and full Appendix C checks are required")
    else:
        declared = read_json(Path(repo) / "experiments/alt2027" / SPECIFICATIONS[suite])
        if specification["suite"] != suite or normalized_specification(
            suite, specification["config"]
        ) != normalized_specification(suite, declared):
            raise ValueError(
                f"{suite} cases/settings differ from the committed specification"
            )
        suite_result(suite, root, result, specification["config"])
    return {
        "path": str(root),
        "manifest_sha256": sha256(root / "manifest.json"),
        "result_sha256": sha256(root / "result.json"),
        "source_reuse": reuse,
        "record": record,
        "result": result,
        "specification": specification,
    }


def experiment_engines(protocol, depth, powers, batch_size, repo):
    result = {}
    for name in protocol["experiments"]:
        engine = {
            "depth": None if name in ("architecture", "bible_secondary") else depth
        }
        if name in (
            "benchmark_primary",
            "benchmark_powers",
            "spectrum",
            "factorial",
            "depth_scaling",
        ):
            engine["execution"] = {
                "batch_size": batch_size,
                "batch_implementation_sha256": sha256(
                    Path(repo) / "src/lsa/alt/batch_depth.py"
                ),
            }
        if name == "benchmark_powers":
            engine["power"] = {
                "settings": powers,
                "implementation_sha256": sha256(Path(repo) / "src/lsa/alt/powers.py"),
            }
        result[name] = engine
    return result


def assemble(*, repo, protocol, engine, evidence, batch_size=20, powers=None):
    """Return a pending/failed assessment, or a runner-compatible certificate."""
    from .depth import _json_hash

    repo = Path(repo).resolve()
    current = source_identity(repo)
    powers = power_options(powers or {})
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch size must be a positive integer")
    gates, accepted = {}, {}
    gates["source"] = {
        "status": "pending" if current["dirty"] else "passed",
        "reason": "clean final committed source required",
    }
    gates["protocol"] = {
        "status": "passed"
        if protocol.get("status") == "frozen"
        and protocol == read_json(repo / "experiments/alt2027/protocol.json")
        else "pending",
        "reason": "supplied protocol must equal the committed frozen protocol",
    }
    candidate = read_json(repo / "experiments/alt2027/store-candidate.json")
    if (
        engine.get("mode") != "store"
        or engine.get("store", {}).get("files_sha256") != candidate["files_sha256"]
    ):
        raise ValueError("production requires the complete pinned store identity")
    unknown = set(evidence) - {*SUITES, "regressions"}
    if unknown:
        raise ValueError(f"unknown evidence keys: {sorted(unknown)}")
    for suite in (*SUITES, "regressions"):
        path = evidence.get(suite)
        if path is None or not (Path(path) / "manifest.json").exists():
            gates[suite] = {
                "status": "pending",
                "reason": "completed immutable Run missing",
            }
            continue
        try:
            accepted[suite] = read_evidence(suite, path, repo=repo, current=current)
            gates[suite] = {
                "status": "passed",
                **{
                    key: accepted[suite][key]
                    for key in (
                        "path",
                        "manifest_sha256",
                        "result_sha256",
                        "source_reuse",
                    )
                },
            }
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            subprocess.CalledProcessError,
        ) as error:
            gates[suite] = {"status": "failed", "reason": str(error)}
    engines, runtime, depth = None, None, None
    if "chain" in accepted:
        try:
            chain = accepted["chain"]
            depth = read_json(Path(chain["path"]) / "data/engine.json")
            runtime = {
                **depth["runtime"],
                "mpmath": chain["record"]["environment"]["packages"]["mpmath"],
            }
            binding = {
                "engine_options_sha256": canonical_hash(engine_options(engine)),
                "power_settings_sha256": canonical_hash(powers),
                "batch_size": batch_size,
            }
            engines = experiment_engines(protocol, depth, powers, batch_size, repo)
            for name, configuration in engines.items():
                validate_record_engine(binding, name, configuration, runtime=runtime)
            for suite, item in accepted.items():
                validate_record_runtime(item["record"], runtime)
                if suite in ("chain", "depth") and engine_options(
                    item["record"]["engine"]
                ) != engine_options(engine):
                    raise ValueError(
                        f"{suite} numerical engine differs from requested production engine"
                    )
            if (
                "power" in accepted
                and power_options(
                    accepted["power"]["specification"]["config"]["default_settings"]
                )
                != powers
            ):
                raise ValueError(
                    "production power settings differ from the validated nominal settings"
                )
            gates["engine_and_runtime"] = {"status": "passed", "runtime": runtime}
        except (ValueError, OSError, KeyError, TypeError) as error:
            gates["engine_and_runtime"] = {"status": "failed", "reason": str(error)}
    else:
        gates["engine_and_runtime"] = {
            "status": "pending",
            "reason": "verified chain engine identity required",
        }
    if source_identity(repo) != current:
        gates["source"] = {
            "status": "failed",
            "reason": "source changed during assembly",
        }
    status = (
        "failed"
        if any(g["status"] == "failed" for g in gates.values())
        else (
            "pending"
            if any(g["status"] != "passed" for g in gates.values())
            else "passed"
        )
    )
    assessment = {
        "schema_version": 1,
        "status": status,
        "required_checks_complete": status == "passed",
        "source_commit": current["commit"],
        "source_tree_sha256": current["tree_sha256"],
        "protocol_sha256": canonical_hash(protocol),
        "batch_size": batch_size,
        "gates": gates,
        "scope": "The committed finite validation cases, full regression checks and exact production identities; no whole-domain numerical error bound is asserted.",
    }
    if status == "passed":
        assessment.update(
            covered_experiments=list(protocol["experiments"]),
            configuration_sha256=_json_hash(depth),
            engine_sha256_by_experiment={
                name: canonical_hash(value) for name, value in engines.items()
            },
            runtime=runtime,
        )
    return assessment


def run_regressions(repo, out):
    repo = Path(repo).resolve()
    if source_identity(repo)["dirty"]:
        raise ValueError("regression evidence requires a clean committed source")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["PYTHONPATH"] = str(repo / "src")
    checks = {}
    with Run(
        out,
        repo=repo,
        protocol={"checks": REGRESSION_COMMANDS},
        experiment="production_regressions",
        purpose="validation",
    ) as run:
        for name, arguments in REGRESSION_COMMANDS.items():
            with (run.path / f"{name}.log").open("x") as log:
                command = [sys.executable, *arguments]
                code = subprocess.call(
                    command, cwd=repo, stdout=log, stderr=subprocess.STDOUT
                )
            checks[name] = {"command": command, "exit_code": code}
        result = {
            "status": "passed"
            if all(r["exit_code"] == 0 for r in checks.values())
            else "failed",
            "checks": checks,
        }
        write_json(run.path / "result.json", result)
        if result["status"] != "passed":
            raise ArithmeticError("production regression checks failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    regression = commands.add_parser("regressions")
    assembly = commands.add_parser("assemble")
    for command in (regression, assembly):
        command.add_argument("--repo", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    for name in ("protocol", "engine-config", "evidence"):
        assembly.add_argument(f"--{name}", type=Path, required=True)
    assembly.add_argument("--power-settings", type=Path)
    assembly.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "regressions":
        run_regressions(args.repo, args.out)
        return 0
    paths = read_json(args.evidence)
    evidence = {
        suite: (args.evidence.parent / path).resolve() for suite, path in paths.items()
    }
    result = assemble(
        repo=args.repo,
        protocol=read_json(args.protocol),
        engine=read_json(args.engine_config),
        evidence=evidence,
        batch_size=args.batch_size,
        powers=read_json(args.power_settings) if args.power_settings else {},
    )
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "assessment.json", result)
    write_json(
        args.out / "evidence-index.json",
        {key: str(value) for key, value in evidence.items()},
    )
    if result["status"] == "passed":
        write_json(args.out / "calibration.json", result)
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return {"passed": 0, "pending": 2, "failed": 1}[result["status"]]
