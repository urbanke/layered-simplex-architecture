"""Assemble production admission from the existing immutable validation suites.

This module runs no new numerical acceptance test and changes no tolerance.
Source reuse is an explicit, conservative file comparison, retained in the output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    "sealed_store": "sealed-store-validation.json",
    "sealed_profile": "sealed-profile-validation.json",
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
EXECUTION_FILES = {"src/lsa/alt/distributed.py"}


def normalized_specification(suite, config):
    result = copy.deepcopy(config)
    if suite in ("depth", "power"):
        result.pop("workers", None)
    if suite == "kernel":
        for store in result["stores"]:
            store.pop("path", None)
    return result


def _configuration_options(configuration):
    """Extract path-independent options from a complete evaluator identity."""
    options = {
        key: copy.deepcopy(configuration[key])
        for key in ("mode", "store", "prediction_tolerance")
    }
    if options["store"] is not None:
        options["store"]["path"] = "/host-local-store"
    return engine_options(options)


def _within(value, tolerance):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and abs(value) <= tolerance
    )


def _sealed_plan(root, pins=None):
    """Check bound metadata, complete production coverage and every manifest pin."""
    plan_path, manifest_path = root / "store-plan.json", root / "store-manifest.json"
    plan, manifest = read_json(plan_path), read_json(manifest_path)
    if pins is not None and (
        sha256(plan_path) != pins.get("plan.json")
        or sha256(manifest_path) != pins.get("manifest.json")
        or manifest.get("files")
        != {k: v for k, v in pins.items() if k != "manifest.json"}
    ):
        raise ValueError(
            "sealed metadata or complete manifest pins differ from production store"
        )
    required_levels = list(range(2, 139))
    if (
        plan.get("format") != "lsa-sealed-kernels-v1"
        or manifest.get("format") != "lsa-sealed-kernels-v1"
        or manifest.get("sealed") is not True
        or manifest.get("plan_sha256") != sha256(plan_path)
        or plan.get("levels") != required_levels
        or manifest.get("levels") != required_levels
        or plan.get("support_max_count", -1) < 1045889
        or plan.get("u_max", -math.inf) < 80
        or plan.get("left_drop", -math.inf) < 60
        or plan.get("interpolation_degree") != 7
        or plan.get("count_degree") != 11
    ):
        raise ValueError(
            "sealed production store requires complete depths 2..138, count 1045889 and u<=80 coverage"
        )
    files = manifest["files"]
    required_files = {"plan.json"} | {
        f"level_{L:03d}.{suffix}"
        for L in required_levels
        for suffix in ("bin", "index.json")
    }
    if not required_files <= files.keys():
        raise ValueError("sealed manifest omits planned level files")
    if set(plan["anchors"]) != {str(L) for L in required_levels}:
        raise ValueError("sealed plan anchor levels are incomplete")
    for anchors in plan["anchors"].values():
        if (
            anchors != sorted(set(anchors))
            or anchors[:257] != list(range(257))
            or anchors[-1] < plan["support_max_count"]
        ):
            raise ValueError(
                "sealed plan must retain the complete dense floor and upper count coverage"
            )
    return plan


def _sealed_declared_cases(plan, specification):
    from .sealed_store_validation import resolve_validation_cases

    metadata = SimpleNamespace(
        coverage_depths=tuple(plan["levels"]),
        supported_count=plan["support_max_count"],
        maximum_u=plan["u_max"],
        grid_step=plan["grid_step"],
        plan=plan,
        count_anchors=lambda L: tuple(plan["anchors"][str(L)]),
        column_metadata=lambda L, r: {
            "u_min": math.floor(
                (-plan["left_drop"] - L * math.log(r + 1)) / plan["grid_step"]
            )
            * plan["grid_step"]
        },
    )
    return resolve_validation_cases(metadata, specification)


def _validate_sealed_backend(backend, engine):
    """Bind the executed interpolation provider to its source and binary pins."""
    requested = engine["store"].get("interpolation_backend", "python")
    if backend.get("interpolation_backend") != requested:
        raise ValueError("measured sealed interpolation backend differs from engine")
    identity = backend.get("native_identity")
    if requested == "python":
        if identity is not None or engine.get("sealed_native_identity") is not None:
            raise ValueError("Python sealed evidence must not claim a native identity")
        return
    from .sealed_native import ABI_VERSION, COMPILE_FLAGS, FORMAT, SOURCE_PATH

    if (
        requested != "native"
        or not isinstance(identity, dict)
        or identity.get("format") != FORMAT
        or identity.get("abi_version") != ABI_VERSION
        or identity.get("flags") != list(COMPILE_FLAGS)
        or identity.get("binary_sha256") != engine["store"].get("native_library_sha256")
        or identity.get("source_sha256") != sha256(SOURCE_PATH)
        or identity.get("wrapper_sha256")
        != sha256(Path(__file__).with_name("sealed_native.py"))
        or (
            engine.get("sealed_native_identity") is not None
            and identity != engine["sealed_native_identity"]
        )
    ):
        raise ValueError(
            "measured sealed native identity differs from engine/source pins"
        )


def sealed_store_result(root, result, specification):
    """Recalculate nominal/qualified classes from exact stored reference decimals."""
    from .sealed_store_validation import (
        accuracy_contract,
        assess_measurement,
        finalize_measurement,
        summarize_measurements,
    )

    plan = _sealed_plan(root)
    cases, unavailable = _sealed_declared_cases(plan, specification)
    rows = [
        json.loads(line) for line in (root / "data/rows.jsonl").read_text().splitlines()
    ]
    if (
        specification.get("accuracy_contract") != accuracy_contract()
        or result.get("accuracy_contract") != accuracy_contract()
        or read_json(root / "data/config.json") != specification
        or unavailable
        or result.get("unavailable_strata") != []
        or read_json(root / "data/unavailable-strata.json") != []
        or read_json(root / "data/resolved-cases.json") != cases
        or result.get("resolved_cases_sha256") != canonical_hash(cases)
        or result.get("config_sha256") != canonical_hash(specification)
        or result.get("cases") != len(cases)
        or result.get("passed_cases") != len(cases)
        or result.get("failed_cases") != 0
        or result.get("unresolved_reference_cases") != 0
        or result.get("status") != "passed"
        or not result.get("source_unchanged")
        or not result.get("store_unchanged")
        or result.get("reader_backend_unchanged") is not True
        or result.get("reader_backend") != read_json(root / "data/reader-backend.json")
        or result.get("gates_nats") != {"general": 3e-9, "counts_0_to_3": 1e-11}
        or len(rows) != len(cases)
    ):
        raise ValueError(
            "sealed reader validation has incomplete, unavailable, failed or unresolved cases"
        )
    direct_rows, selected, references, assessed_rows = [], [], [], []
    for case, row in zip(cases, rows, strict=True):
        if any(row.get(key) != value for key, value in case.items()):
            raise ValueError("sealed reader measured cases differ from declared cases")
        measured = assess_measurement(
            row["r"],
            row["scalar_log_phi_nats"],
            row["matrix_log_phi_nats"],
            row["direct_reference_log_phi_nats"],
            row["refined_reference_log_phi_nats"],
        )
        direct_rows.append({**case, **measured})
        if measured["requires_independent_reference"]:
            selected.append(case)
            references.append(row.get("independent_reference"))
        assessed = finalize_measurement(
            case, measured, row.get("independent_reference")
        )
        expected = {**case, **assessed}
        if (
            assessed["status"] not in ("passed", "precision_qualified")
            or row != expected
        ):
            raise ValueError(
                "sealed reader residuals fail the declared nominal/four-ULP contract or stored accounting differs"
            )
        assessed_rows.append(expected)
    if (
        read_json(root / "data/independent-reference-cases.json") != selected
        or [
            json.loads(line)
            for line in (root / "data/independent-references.jsonl")
            .read_text()
            .splitlines()
        ]
        != references
        or [
            json.loads(line)
            for line in (root / "data/direct-rows.jsonl").read_text().splitlines()
        ]
        != direct_rows
    ):
        raise ValueError(
            "sealed reader independent reference selection or direct record differs"
        )
    metrics = summarize_measurements(assessed_rows)
    if any(result.get(key) != value for key, value in metrics.items()):
        raise ValueError(
            "sealed reader measured classes or error metrics differ from summary"
        )
    if result.get("independent_high_precision_reference") is not bool(selected):
        raise ValueError(
            "sealed reader independent reference coverage differs from summary"
        )
    summary = read_json(root / "data/summary.json")
    if any(result.get(k) != v for k, v in summary.items()):
        raise ValueError("sealed reader result and measured summary differ")


def sealed_profile_result(root, result, specification):
    """Require all declared same-profile comparisons, using their actual residuals."""
    import numpy as np

    loss, mass = (
        specification["loss_tolerance_bits"],
        specification["raw_mass_tolerance"],
    )
    if not (0 < loss <= 1e-5 and 0 < mass <= 1e-7):
        raise ValueError(
            "sealed profile gates may not relax 1e-5-bit / 1e-7 raw-mass tolerances"
        )
    expected = {c["id"]: c for c in specification["cases"]}
    cases = result["cases"]
    if (
        not expected
        or len(expected) != len(specification["cases"])
        or len(cases) != len(expected)
        or {c["id"] for c in cases} != expected.keys()
        or result.get("unavailable_cases") != []
        or not result.get("source_unchanged")
        or not result.get("store_unchanged")
        or result.get("config_sha256") != canonical_hash(specification)
    ):
        raise ValueError(
            "sealed cross-provider evidence has incomplete or changed declared cases"
        )
    for row in cases:
        case = expected[row["id"]]
        if row.get("status") != "passed" or row.get("case") != case:
            raise ValueError(
                "sealed cross-provider case failed or differs from specification"
            )
        position = list(expected).index(row["id"])
        sample = root / f"data/case-{position:03d}.npz"
        if sha256(sample) != row.get("sample_sha256"):
            raise ValueError("sealed cross-provider saved sample hash differs")
        with np.load(sample, allow_pickle=False) as saved:
            counts, target = saved["counts"], saved["target"]
            if (
                counts.shape != (case["d"],)
                or target.shape != (case["d"],)
                or int(counts.sum()) != case["n"]
                or canonical_hash(counts.tolist()) != row.get("counts_sha256")
                or canonical_hash(target.tolist()) != row.get("target_sha256")
            ):
                raise ValueError(
                    "sealed cross-provider saved counts/target differ from the declared case"
                )
        measurements = row["measurements"]
        for name in (
            "maximum_codelength_difference_bits_per_token",
            "mixture_codelength_difference_bits_per_token",
        ):
            if not _within(measurements[name], loss):
                raise ValueError(
                    "sealed cross-provider codelength residual exceeds gate"
                )
        if case["predictive"]:
            if not _within(
                measurements["maximum_augmented_codelength_difference_bits_per_token"],
                loss,
            ):
                raise ValueError(
                    "sealed cross-provider augmented codelength residual exceeds gate"
                )
            for name in (
                "maximum_predictive_kl_change_bits",
                "mixture_predictive_kl_change_bits",
            ):
                if not _within(measurements[name], loss):
                    raise ValueError(
                        "sealed cross-provider predictive residual exceeds gate"
                    )
            if not _within(measurements["maximum_raw_mass_error"], mass):
                raise ValueError(
                    "sealed cross-provider raw mass exceeds validation gate"
                )
    candidate = read_json(root / "data/candidate-engine.json")
    legacy = read_json(root / "data/legacy-engine.json")
    if (
        result.get("candidate_configuration_sha256") != canonical_hash(candidate)
        or result.get("legacy_configuration_sha256") != canonical_hash(legacy)
        or candidate["store"].get("format") != "sealed"
        or legacy["store"].get("format", "legacy") != "legacy"
        or legacy["store"].get("saddle_min_depth") != 54
    ):
        raise ValueError(
            "sealed cross-provider engine identities are incomplete or inconsistent"
        )
    for key in (
        "grid_step",
        "minimum_grid_step",
        "u_max",
        "maximum_u_max",
        "upper_window_increment",
        "scan_mode",
        "significance_gap",
        "minimum_right_gap",
    ):
        if candidate["store"].get(key) != legacy["store"].get(key):
            raise ValueError("sealed cross-provider outer integration settings differ")


def sealed_independent_kernel(item, pins, engine):
    """Check the sealed reader against the independent decimal Mellin references."""
    from .kernel_validation import _decimal_difference, expand_cases

    root = Path(item["path"])
    config = read_json(root / "data/config.json")
    declared = item["specification"]["config"]
    if {k: v for k, v in config.items() if k != "stores"} != {
        k: v for k, v in declared.items() if k != "stores"
    }:
        raise ValueError(
            "independent sealed kernel cases/settings differ from the committed grid"
        )
    stores = config["stores"]
    cases = expand_cases(declared)
    if len(stores) != 1:
        raise ValueError("independent kernel suite requires exactly one sealed store")
    expected_store = {
        "id": "sealed_candidate",
        "format": "sealed",
        "path": stores[0].get("path"),
        "files_sha256": pins,
        "depths": sorted({c["depth"] for c in cases}),
        "counts": sorted({c["r"] for c in cases}),
    }
    native_requested = (
        engine["store"].get("interpolation_backend", "python") == "native"
    )
    if native_requested:
        expected_store["native_library"] = {
            "path": stores[0].get("native_library", {}).get("path"),
            "sha256": engine["store"]["native_library_sha256"],
        }
        if not expected_store["native_library"]["path"]:
            raise ValueError(
                "independent kernel suite lacks its actual native library path"
            )
    if stores[0] != expected_store:
        raise ValueError(
            "independent kernel suite does not bind the complete sealed store/domain/backend"
        )
    if engine_options(item["record"]["engine"]) != engine_options(engine):
        raise ValueError("independent kernel suite engine differs from production")
    metadata = read_json(root / "data/store-inputs.json")
    if (
        len(metadata) != 1
        or metadata[0].get("id") != "sealed_candidate"
        or metadata[0].get("format") != "sealed"
        or metadata[0].get("files_sha256") != pins
        or metadata[0].get("unchanged_after_read") is not True
    ):
        raise ValueError(
            "independent kernel suite lacks unchanged complete store identity"
        )
    _validate_sealed_backend(
        {
            "interpolation_backend": "native" if native_requested else "python",
            "native_identity": metadata[0].get("native_identity"),
        },
        engine,
    )
    if (
        declared["reference_dps"] < 45
        or declared["refined_reference_dps"] < 60
        or declared["reference_convergence_nats"] > 1e-25
    ):
        raise ValueError("independent kernel precision settings are insufficient")
    rows = [
        json.loads(line) for line in (root / "data/rows.jsonl").read_text().splitlines()
    ]
    if len(rows) != len(cases):
        raise ValueError("independent sealed kernel rows are incomplete")
    for case, row in zip(cases, rows, strict=True):
        tolerance = 1e-11 if row["r"] <= 3 else 3e-9
        if (
            any(row.get(k) != v for k, v in case.items())
            or row["tolerance_nats"] != tolerance
        ):
            raise ValueError("independent sealed kernel case or absolute gate differs")
        reference, refined = row["reference"], row["refined_reference"]
        if (
            reference["dps"] != declared["reference_dps"]
            or refined["dps"] != declared["refined_reference_dps"]
            or not _within(
                _decimal_difference(reference["log_phi_nats"], refined["log_phi_nats"]),
                declared["reference_convergence_nats"],
            )
        ):
            raise ValueError("independent kernel references did not converge")
        value = row["stores"]["sealed_candidate"]
        if value.get("status") != "evaluated":
            raise ValueError("independent sealed kernel value is unavailable")
        for key, error_key in (
            ("log_phi_nats", "error_nats"),
            ("matrix_log_phi_nats", "matrix_error_nats"),
        ):
            error = _decimal_difference(value[key], refined["log_phi_nats"])
            if not _within(error, tolerance) or value[error_key] != error:
                raise ValueError(
                    "independent sealed kernel residual exceeds unchanged row gate"
                )
    if kernel_status(item["result"]["measurements"], root / "data", config) != "passed":
        raise ValueError("independent sealed kernel suite has failed measured checks")


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
        elif suite != "regressions" and name in EXECUTION_FILES:
            reason = "distributed_partition_not_executed_by_this_numerical_suite"
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
    if suite == "sealed_store":
        sealed_store_result(root, result, specification)
    elif suite == "sealed_profile":
        sealed_profile_result(root, result, specification)
    elif suite == "kernel":
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
    sealed = engine.get("store", {}).get("format", "legacy") == "sealed"
    candidate_path = (
        repo
        / "experiments/alt2027"
        / ("sealed-store-candidate.json" if sealed else "store-candidate.json")
    )
    if sealed and not candidate_path.is_file():
        raise ValueError(
            "sealed production requires committed sealed-store-candidate.json"
        )
    candidate = read_json(candidate_path)
    if (
        engine.get("mode") != "store"
        or engine.get("store", {}).get("files_sha256") != candidate["files_sha256"]
    ):
        raise ValueError("production requires the complete pinned store identity")
    if sealed and (
        candidate.get("format") != "sealed"
        or not {"plan.json", "manifest.json"} <= candidate["files_sha256"].keys()
    ):
        raise ValueError(
            "sealed store specification must identify format and pin plan plus manifest"
        )
    required_suites = (
        *SUITES,
        *(("sealed_store", "sealed_profile") if sealed else ()),
        "regressions",
    )
    unknown = set(evidence) - set(required_suites)
    if unknown:
        raise ValueError(f"unknown evidence keys: {sorted(unknown)}")
    for suite in required_suites:
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
            ArithmeticError,
            ValueError,
            OSError,
            KeyError,
            IndexError,
            TypeError,
            subprocess.CalledProcessError,
        ) as error:
            gates[suite] = {"status": "failed", "reason": str(error)}
    if sealed:
        if "sealed_store" not in accepted:
            gates["sealed_store_identity"] = {
                "status": "pending",
                "reason": "verified full-coverage sealed reader evidence required",
            }
        else:
            try:
                item = accepted["sealed_store"]
                plan = _sealed_plan(Path(item["path"]), candidate["files_sha256"])
                from .sealed_store_build import source_identity as builder_identity

                if (
                    item["result"]["store_files_sha256"] != candidate["files_sha256"]
                    or item["result"]["engine_sha256"]
                    != canonical_hash(item["record"]["engine"])
                    or _configuration_options(item["record"]["engine"])
                    != engine_options(engine)
                    or plan["source_sha256"] != builder_identity()
                    or engine["store"]["max_depth"] < 138
                    or engine["store"]["max_count"] + 3 > plan["support_max_count"]
                    or engine_options(engine)["store"]["maximum_u_max"] > plan["u_max"]
                ):
                    raise ValueError(
                        "sealed reader evidence does not bind the production engine/store/builder"
                    )
                _validate_sealed_backend(
                    item["result"]["reader_backend"], item["record"]["engine"]
                )
                gates["sealed_store_identity"] = {
                    "status": "passed",
                    "store_spec_sha256": sha256(candidate_path),
                }
            except (
                ArithmeticError,
                ValueError,
                OSError,
                KeyError,
                IndexError,
                TypeError,
            ) as error:
                gates["sealed_store_identity"] = {
                    "status": "failed",
                    "reason": str(error),
                }
        if "kernel" not in accepted:
            gates["sealed_independent_kernel"] = {
                "status": "pending",
                "reason": "independent sealed reader values in the kernel suite are required",
            }
        else:
            try:
                sealed_independent_kernel(
                    accepted["kernel"], candidate["files_sha256"], engine
                )
                gates["sealed_independent_kernel"] = {"status": "passed"}
            except (
                ArithmeticError,
                ValueError,
                OSError,
                KeyError,
                IndexError,
                TypeError,
            ) as error:
                gates["sealed_independent_kernel"] = {
                    "status": "failed",
                    "reason": str(error),
                }
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
            if sealed:
                if "kernel" in accepted:
                    kernel_metadata = read_json(
                        Path(accepted["kernel"]["path"]) / "data/store-inputs.json"
                    )
                    _validate_sealed_backend(
                        {
                            "interpolation_backend": depth["store"].get(
                                "interpolation_backend", "python"
                            ),
                            "native_identity": kernel_metadata[0].get(
                                "native_identity"
                            ),
                        },
                        depth,
                    )
                if (
                    "sealed_store" in accepted
                    and accepted["sealed_store"]["record"]["engine"] != depth
                ):
                    raise ValueError(
                        "sealed reader validation differs from the full chain engine identity"
                    )
                if "sealed_profile" in accepted:
                    path = Path(accepted["sealed_profile"]["path"])
                    candidate_engine = read_json(path / "data/candidate-engine.json")
                    legacy_engine = read_json(path / "data/legacy-engine.json")
                    legacy_pins = read_json(
                        repo / "experiments/alt2027/store-candidate.json"
                    )["files_sha256"]
                    if (
                        candidate_engine != depth
                        or accepted["sealed_profile"]["record"]["engine"]
                        != {"candidate": candidate_engine, "legacy": legacy_engine}
                        or legacy_engine["store"]["files_sha256"] != legacy_pins
                        or any(
                            legacy_engine.get(k) != depth.get(k)
                            for k in (
                                "runtime",
                                "implementation_sha256",
                                "vendor_provenance_sha256",
                                "vendor_source_sha256",
                                "native_kernel",
                                "depth_truncation",
                            )
                        )
                    ):
                        raise ValueError(
                            "cross-provider engines differ from admitted sealed/corrected-direct identities"
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
