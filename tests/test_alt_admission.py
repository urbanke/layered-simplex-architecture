"""Admission consumes verified evidence; these fixtures perform no production run."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lsa.alt import admission
from lsa.alt.artifacts import (
    Run,
    canonical_hash,
    read_json,
    source_identity,
    write_json,
)
from lsa.alt.depth import DepthEvaluator, _json_hash
from lsa.alt.distributed import engine_options, power_options
from lsa.alt.kernel_validation import expand_cases


def commit(repo):
    for arguments in (
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", *arguments], cwd=repo, check=True)


@pytest.fixture
def bundle(tmp_path):
    repo = tmp_path / "repo"
    specs = repo / "experiments/alt2027"
    specs.mkdir(parents=True)
    package = Path(admission.__file__).parent
    for name in ("batch_depth.py", "powers.py"):
        target = repo / "src/lsa/alt" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    protocol = {
        "status": "frozen",
        "experiments": {
            "benchmark_primary": {},
            "benchmark_powers": {},
            "architecture": {},
        },
    }
    configurations = {
        "kernel": {
            "groups": [{"id": "tiny", "depths": [2], "counts": [0], "u_values": [0.0]}],
            "special_function_cases": [{}],
            "stores": [],
            "nominal_tolerance_nats": 3e-9,
            "reference_convergence_nats": 1e-25,
        },
        "prior": {"cases": ["tiny-independent-identity"]},
        "depth": {"cases": [{"id": "tiny"}], "workers": 4},
        "power": {
            "targets": ["uniform"],
            "workers": 2,
            "default_settings": power_options({}),
        },
        "chain": {"steps": 2, "depths": [0, 1], "d": 2},
    }
    engine = {
        "mode": "store",
        "prediction_tolerance": 1e-3,
        "store": {
            "path": "/unused-fixture-store",
            "files_sha256": {"manifest.json": "a" * 64},
            "max_depth": 2,
            "max_d": 2,
            "max_n": 4,
            "max_count": 4,
        },
    }
    write_json(
        specs / "store-candidate.json",
        {"files_sha256": engine["store"]["files_sha256"]},
    )
    write_json(specs / "protocol.json", protocol)
    for suite, config in configurations.items():
        write_json(specs / admission.SPECIFICATIONS[suite], config)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    commit(repo)
    with DepthEvaluator(mode="reference", prediction_tolerance=1e-3) as evaluator:
        depth = copy.deepcopy(evaluator.configuration)
    depth.update(mode="store", store=engine_options(engine)["store"])
    results = {
        "kernel": {
            "status": "passed",
            "measurements": {
                "cases": 1,
                "source_unchanged": True,
                "reference_converged_cases": 1,
                "direct_column_nominal_passes": 1,
            },
        },
        "prior": {"status": "passed", "declared_suite_complete": True, "failed": 0},
        "depth": {
            "status": "passed",
            "assessment": {
                "status": "passed",
                "source_unchanged": True,
                "pending_case_ids": [],
                "cases": [{"id": "tiny", "status": "passed"}],
                "original_config_sha256": canonical_hash(configurations["depth"]),
            },
        },
        "power": {
            "status": "passed",
            "source_unchanged": True,
            "profiles": [{"target": "uniform", "status": "passed", "failures": []}],
        },
        "chain": {"status": "passed", **configurations["chain"]},
        "regressions": {
            "status": "passed",
            "checks": {
                name: {"exit_code": 0} for name in admission.REGRESSION_COMMANDS
            },
        },
    }
    evidence = {}
    for suite, result in results.items():
        path = tmp_path / suite
        specification = (
            {"checks": admission.REGRESSION_COMMANDS}
            if suite == "regressions"
            else {
                "suite": suite,
                "config": configurations[suite],
                "workers_override": None,
            }
        )
        with Run(
            path,
            repo=repo,
            protocol=specification,
            purpose="validation",
            experiment="production_regressions"
            if suite == "regressions"
            else f"calibration_{suite}",
            engine_identity=engine if suite in ("depth", "chain") else None,
        ) as run:
            data = run.path / "data"
            data.mkdir()
            write_json(run.path / "result.json", result)
            if suite == "chain":
                write_json(data / "engine.json", depth)
            if suite == "kernel":
                write_json(
                    data / "resolved-cases.json", expand_cases(configurations["kernel"])
                )
                write_json(
                    data / "special-functions.json", [{"meijer_error_nats": 0.0}]
                )
                (data / "rows.jsonl").write_text(
                    json.dumps(
                        {
                            "batched_direct_column_error_nats": 0.0,
                            "direct_column_refined_error_nats": 0.0,
                            "tolerance_nats": 1e-11,
                            "stores": {},
                        }
                    )
                    + "\n"
                )
        evidence[suite] = path
    return {
        "repo": repo,
        "protocol": protocol,
        "engine": engine,
        "evidence": evidence,
    }, depth


def test_complete_verified_evidence_automatically_creates_exact_certificate(bundle):
    arguments, depth = bundle
    result = admission.assemble(**arguments)
    assert result["status"] == "passed"
    assert result["required_checks_complete"] is True
    assert result["configuration_sha256"] == _json_hash(depth)
    assert set(result["covered_experiments"]) == set(
        arguments["protocol"]["experiments"]
    )
    engines = admission.experiment_engines(
        arguments["protocol"], depth, power_options({}), 20, arguments["repo"]
    )
    assert result["engine_sha256_by_experiment"] == {
        name: canonical_hash(value) for name, value in engines.items()
    }


@pytest.mark.parametrize(
    "missing", ["kernel", "prior", "depth", "power", "chain", "regressions"]
)
def test_no_missing_suite_can_create_passed_certificate(bundle, missing):
    arguments, _ = bundle
    arguments["evidence"].pop(missing)
    result = admission.assemble(**arguments)
    assert result["status"] == "pending"
    assert not result["required_checks_complete"]
    assert "engine_sha256_by_experiment" not in result


def test_failed_or_tampered_artifacts_stop_admission(bundle):
    arguments, _ = bundle
    path = arguments["evidence"]["power"] / "result.json"
    path.write_text(json.dumps({"status": "failed"}))
    result = admission.assemble(**arguments)
    assert result["status"] == "failed"
    assert result["gates"]["power"]["status"] == "failed"
    assert not result["required_checks_complete"]


def test_dirty_source_and_unfrozen_protocol_remain_pending(bundle):
    arguments, _ = bundle
    (arguments["repo"] / "README.md").write_text("documentation change\n")
    arguments["protocol"]["status"] = "implementation"
    result = admission.assemble(**arguments)
    assert result["status"] == "pending"
    assert result["gates"]["source"]["status"] == "pending"
    assert result["gates"]["protocol"]["status"] == "pending"


def test_power_changes_reuse_unaffected_suites_but_force_power_and_regressions(bundle):
    arguments, _ = bundle
    repo = arguments["repo"]
    old = source_identity(repo)
    with (repo / "src/lsa/alt/powers.py").open("a") as stream:
        stream.write("\n# changed powered numerical implementation\n")
    commit(repo)
    new = source_identity(repo)
    for suite in ("kernel", "prior", "depth", "chain"):
        reuse = admission.source_reuse(suite, old, new, repo)
        assert (
            reuse["permitted_changes"][0]["reason"]
            == "powered_model_not_executed_by_this_suite"
        )
    for suite in ("power", "regressions"):
        with pytest.raises(ValueError, match="needs rerun"):
            admission.source_reuse(suite, old, new, repo)


def test_partition_changes_preserve_numerical_evidence_but_require_regressions(bundle):
    arguments, _ = bundle
    repo = arguments["repo"]
    old = source_identity(repo)
    (repo / "src/lsa/alt/distributed.py").write_text("# revised job partition\n")
    commit(repo)
    new = source_identity(repo)
    for suite in ("kernel", "prior", "depth", "power", "chain"):
        reuse = admission.source_reuse(suite, old, new, repo)
        assert reuse["permitted_changes"][0]["reason"] == (
            "distributed_partition_not_executed_by_this_numerical_suite"
        )
    with pytest.raises(ValueError, match="needs rerun"):
        admission.source_reuse("regressions", old, new, repo)


def test_protocol_reuse_allows_only_status_freeze(bundle):
    arguments, _ = bundle
    repo = arguments["repo"]
    path = repo / "experiments/alt2027/protocol.json"
    original = read_json(path)
    original["status"] = "implementation"
    path.write_text(json.dumps(original))
    commit(repo)
    old = source_identity(repo)
    original["status"] = "frozen"
    path.write_text(json.dumps(original))
    commit(repo)
    reuse = admission.source_reuse("kernel", old, source_identity(repo), repo)
    assert reuse["permitted_changes"][0]["reason"] == "protocol_status_only_freeze"
    original["experiments"]["new_experiment"] = {}
    path.write_text(json.dumps(original))
    commit(repo)
    with pytest.raises(ValueError, match="needs rerun"):
        admission.source_reuse("kernel", old, source_identity(repo), repo)


def test_changed_executable_wrapper_requires_new_regressions(bundle):
    arguments, _ = bundle
    repo = arguments["repo"]
    before = source_identity(repo)
    wrapper = repo / "cluster/alt2027/launch.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("# changed executable launcher\n")
    commit(repo)
    after = source_identity(repo)
    assert admission.source_reuse("kernel", before, after, repo)["permitted_changes"]
    with pytest.raises(ValueError, match="needs rerun"):
        admission.source_reuse("regressions", before, after, repo)


def test_changed_engine_and_power_settings_cannot_inherit_evidence(bundle):
    arguments, _ = bundle
    arguments["engine"]["prediction_tolerance"] = 1e-2
    result = admission.assemble(**arguments)
    assert result["gates"]["engine_and_runtime"]["status"] == "failed"
    arguments["engine"]["prediction_tolerance"] = 1e-3
    result = admission.assemble(**arguments, powers={"kernel_tail_drop": 50.0})
    assert result["gates"]["engine_and_runtime"]["status"] == "failed"


def test_host_local_paths_and_worker_counts_are_the_only_normalized_fields():
    a = {"stores": [{"id": "same", "path": "/laptop", "counts": [1]}]}
    b = {"stores": [{"id": "same", "path": "/cluster", "counts": [1]}]}
    assert admission.normalized_specification(
        "kernel", a
    ) == admission.normalized_specification("kernel", b)
    b["stores"][0]["counts"] = [2]
    assert admission.normalized_specification(
        "kernel", a
    ) != admission.normalized_specification("kernel", b)
    assert admission.normalized_specification(
        "power", {"workers": 2, "tolerance": 1e-5}
    ) == admission.normalized_specification("power", {"workers": 72, "tolerance": 1e-5})


def test_sealed_engine_requires_its_own_committed_candidate_and_extra_evidence(bundle):
    arguments, _ = bundle
    arguments["engine"]["store"]["format"] = "sealed"
    with pytest.raises(ValueError, match="sealed-store-candidate"):
        admission.assemble(**arguments)
    pins = {"plan.json": "b" * 64, "manifest.json": "a" * 64}
    arguments["engine"]["store"]["files_sha256"] = pins
    write_json(
        arguments["repo"] / "experiments/alt2027/sealed-store-candidate.json",
        {"format": "sealed", "files_sha256": pins},
    )
    result = admission.assemble(**arguments)
    assert not result["required_checks_complete"]
    assert result["gates"]["sealed_store"]["status"] == "pending"
    assert result["gates"]["sealed_profile"]["status"] == "pending"
    assert "sealed_independent_kernel" in result["gates"]
    assert "engine_sha256_by_experiment" not in result


@pytest.fixture
def sealed_reader_evidence(tmp_path):
    from lsa.alt.sealed_store_validation import (
        accuracy_contract,
        assess_measurement,
        summarize_measurements,
    )

    root = tmp_path / "sealed-reader"
    (root / "data").mkdir(parents=True)
    levels = list(range(2, 139))
    plan = {
        "format": "lsa-sealed-kernels-v1",
        "levels": levels,
        "support_max_count": 1045889,
        "u_max": 80.0,
        "left_drop": 60.0,
        "grid_step": 0.02,
        "interpolation_degree": 7,
        "count_degree": 11,
        "anchors": {str(L): list(range(257)) + [1045889] for L in levels},
    }
    write_json(root / "store-plan.json", plan)
    files = {
        f"level_{L:03d}.{suffix}": "a" * 64
        for L in levels
        for suffix in ("bin", "index.json")
    }
    from lsa.alt.artifacts import sha256

    files["plan.json"] = sha256(root / "store-plan.json")
    manifest = {
        "format": "lsa-sealed-kernels-v1",
        "sealed": True,
        "levels": levels,
        "plan_sha256": files["plan.json"],
        "files": files,
    }
    write_json(root / "store-manifest.json", manifest)
    specification = {
        "cases": [{"depth": 2, "r": 0, "u": 0.007}],
        "accuracy_contract": accuracy_contract(),
    }
    cases, missing = admission._sealed_declared_cases(plan, specification)
    assert not missing
    rows = [{**cases[0], **assess_measurement(0, 0, 0, 0, 0)}]
    write_json(root / "data/config.json", specification)
    write_json(root / "data/resolved-cases.json", cases)
    write_json(root / "data/unavailable-strata.json", [])
    write_json(
        root / "data/reader-backend.json",
        {"interpolation_backend": "python", "native_identity": None},
    )
    write_json(root / "data/independent-reference-cases.json", [])
    (root / "data/independent-references.jsonl").write_text("")
    (root / "data/direct-rows.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (root / "data/rows.jsonl").write_text(json.dumps(rows[0]) + "\n")
    result = {
        "status": "passed",
        "unavailable_strata": [],
        "resolved_cases_sha256": canonical_hash(cases),
        "config_sha256": canonical_hash(specification),
        **summarize_measurements(rows),
        "accuracy_contract": accuracy_contract(),
        "independent_high_precision_reference": False,
        "reader_backend": {"interpolation_backend": "python", "native_identity": None},
        "reader_backend_unchanged": True,
        "source_unchanged": True,
        "store_unchanged": True,
        "gates_nats": {"general": 3e-9, "counts_0_to_3": 1e-11},
    }
    write_json(root / "data/summary.json", result)
    return (
        root,
        result,
        specification,
        {**files, "manifest.json": sha256(root / "store-manifest.json")},
    )


def test_complete_sealed_metadata_requires_every_manifest_pin(sealed_reader_evidence):
    root, result, specification, pins = sealed_reader_evidence
    admission._sealed_plan(root, pins)
    admission.sealed_store_result(root, result, specification)
    with pytest.raises(ValueError, match="pins"):
        admission._sealed_plan(root, {**pins, "level_138.bin": "f" * 64})
    manifest = read_json(root / "store-manifest.json")
    del manifest["files"]["level_138.index.json"]
    (root / "store-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="omits"):
        admission._sealed_plan(root)


@pytest.mark.parametrize(
    "mutation", ["unavailable", "precision", "relaxed", "missing_row", "forged_error"]
)
def test_sealed_reader_status_cannot_hide_missing_or_failed_measurements(
    sealed_reader_evidence, mutation
):
    root, result, specification, _ = sealed_reader_evidence
    if mutation == "unavailable":
        result["unavailable_strata"] = [{"depth": 138}]
    elif mutation == "precision":
        result["unresolved_reference_cases"] = 1
    elif mutation == "relaxed":
        result["gates_nats"]["counts_0_to_3"] = 3e-9
    elif mutation == "missing_row":
        (root / "data/rows.jsonl").write_text("")
    else:
        row = json.loads((root / "data/rows.jsonl").read_text())
        row["scalar_log_phi_nats"] = 1e-4
        (root / "data/rows.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        admission.sealed_store_result(root, result, specification)


@pytest.fixture
def sealed_independent_evidence(tmp_path):
    root = tmp_path / "independent"
    (root / "data").mkdir(parents=True)
    config = {
        "groups": [{"id": "tiny", "depths": [2], "counts": [0], "u_values": [0.0]}],
        "reference_dps": 45,
        "refined_reference_dps": 60,
        "reference_convergence_nats": 1e-25,
        "nominal_tolerance_nats": 3e-9,
        "special_function_cases": [{}],
        "stores": [],
    }
    pins = {"plan.json": "a" * 64, "manifest.json": "b" * 64}
    engine = {
        "mode": "store",
        "store": {
            "format": "sealed",
            "path": "/store",
            "files_sha256": pins,
            "max_depth": 138,
            "max_d": 1000000,
            "max_n": 915861,
            "max_count": 915861,
        },
    }
    resolved = {
        **config,
        "stores": [
            {
                "id": "sealed_candidate",
                "format": "sealed",
                "path": "/store",
                "files_sha256": pins,
                "depths": [2],
                "counts": [0],
            }
        ],
    }
    write_json(root / "data/config.json", resolved)
    write_json(
        root / "data/store-inputs.json",
        [
            {
                "id": "sealed_candidate",
                "format": "sealed",
                "files_sha256": pins,
                "unchanged_after_read": True,
            }
        ],
    )
    write_json(root / "data/special-functions.json", [{"meijer_error_nats": 0}])
    row = {
        **expand_cases(config)[0],
        "tolerance_nats": 1e-11,
        "reference": {"dps": 45, "log_phi_nats": "0.0"},
        "refined_reference": {"dps": 60, "log_phi_nats": "0.0"},
        "batched_direct_column_error_nats": 0,
        "direct_column_refined_error_nats": 0,
        "stores": {
            "sealed_candidate": {
                "status": "evaluated",
                "log_phi_nats": 0,
                "matrix_log_phi_nats": 0,
                "error_nats": 0,
                "matrix_error_nats": 0,
            }
        },
    }
    (root / "data/rows.jsonl").write_text(json.dumps(row) + "\n")
    item = {
        "path": str(root),
        "record": {"engine": engine},
        "specification": {"config": config},
        "result": {
            "measurements": {
                "cases": 1,
                "source_unchanged": True,
                "reference_converged_cases": 1,
                "direct_column_nominal_passes": 1,
            }
        },
    }
    return item, pins, engine


def test_independent_sealed_gate_requires_actual_decimal_reference_measurements(
    sealed_independent_evidence,
):
    item, pins, engine = sealed_independent_evidence
    admission.sealed_independent_kernel(item, pins, engine)
    path = Path(item["path"]) / "data/rows.jsonl"
    row = json.loads(path.read_text())
    row["stores"]["sealed_candidate"]["log_phi_nats"] = 2e-11
    # A declared pass and a forged zero residual cannot hide low-count failure.
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="residual"):
        admission.sealed_independent_kernel(item, pins, engine)


@pytest.mark.parametrize(
    "mutation", ["unavailable", "low_precision", "wrong_store", "legacy_only"]
)
def test_same_builder_or_incomplete_kernel_evidence_cannot_satisfy_independent_gate(
    sealed_independent_evidence, mutation
):
    item, pins, engine = sealed_independent_evidence
    root = Path(item["path"])
    row = json.loads((root / "data/rows.jsonl").read_text())
    if mutation == "unavailable":
        row["stores"]["sealed_candidate"]["status"] = "unavailable"
    elif mutation == "low_precision":
        row["refined_reference"]["dps"] = 45
    elif mutation == "wrong_store":
        pins = {**pins, "manifest.json": "c" * 64}
    else:
        config = read_json(root / "data/config.json")
        config["stores"] = []
        (root / "data/config.json").write_text(json.dumps(config))
    (root / "data/rows.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        admission.sealed_independent_kernel(item, pins, engine)


@pytest.mark.parametrize("reference_cutoff", [2, 54, None])
def test_cross_provider_profile_checks_actual_residuals_and_raw_mass(tmp_path, reference_cutoff):
    import numpy as np

    from lsa.alt.artifacts import sha256
    from lsa.alt.sealed_profile_validation import REFERENCE_PROTOCOL

    (tmp_path / "data").mkdir()
    candidate = {"store": {"format": "sealed"}}
    legacy = {"store": {"format": "legacy", "saddle_min_depth": reference_cutoff}}
    write_json(tmp_path / "data/candidate-engine.json", candidate)
    write_json(tmp_path / "data/legacy-engine.json", legacy)
    case = {"id": "predictive", "predictive": True, "d": 2, "n": 2}
    config = {"schema_version": 2, "reference": copy.deepcopy(REFERENCE_PROTOCOL),
              "cases": [case], "loss_tolerance_bits": 1e-5, "raw_mass_tolerance": 1e-7}
    measurements = {
        "maximum_codelength_difference_bits_per_token": 0,
        "mixture_codelength_difference_bits_per_token": 0,
        "maximum_predictive_kl_change_bits": 0,
        "mixture_predictive_kl_change_bits": 0,
        "maximum_raw_mass_error": 0,
        "maximum_augmented_codelength_difference_bits_per_token": 0,
    }
    sample = tmp_path / "data/case-000.npz"
    np.savez_compressed(sample, counts=np.array([2, 0]), target=np.array([0.5, 0.5]))
    result = {
        "status": "passed",
        "source_unchanged": True,
        "store_unchanged": True,
        "unavailable_cases": [],
        "config_sha256": canonical_hash(config),
        "reference": copy.deepcopy(REFERENCE_PROTOCOL),
        "candidate_configuration_sha256": canonical_hash(candidate),
        "legacy_configuration_sha256": canonical_hash(legacy),
        "cases": [
            {
                "id": "predictive",
                "case": case,
                "status": "passed",
                "measurements": measurements,
                "sample_sha256": sha256(sample),
                "counts_sha256": canonical_hash([2, 0]),
                "target_sha256": canonical_hash([0.5, 0.5]),
            }
        ],
    }
    if reference_cutoff != 2:
        with pytest.raises(ValueError, match="engine identities"):
            admission.sealed_profile_result(tmp_path, result, config)
        return
    admission.sealed_profile_result(tmp_path, result, config)
    measurements["maximum_raw_mass_error"] = 2e-7
    with pytest.raises(ValueError, match="raw mass"):
        admission.sealed_profile_result(tmp_path, result, config)
    config["raw_mass_tolerance"] = 1e-3
    with pytest.raises(ValueError, match="may not relax"):
        admission.sealed_profile_result(tmp_path, result, config)
    config["raw_mass_tolerance"] = 1e-7
    config["reference"]["minimum_direct_depth"] = 54
    with pytest.raises(ValueError, match="protocol v2"):
        admission.sealed_profile_result(tmp_path, result, config)


@pytest.fixture
def qualified_reader_evidence(sealed_reader_evidence):
    from decimal import Decimal

    from lsa.alt.sealed_store_validation import (
        assess_measurement,
        finalize_measurement,
        summarize_measurements,
    )

    root, result, specification, pins = sealed_reader_evidence
    specification["cases"] = [{"depth": 138, "r": 1000000, "u": 80.0}]
    cases, missing = admission._sealed_declared_cases(
        read_json(root / "store-plan.json"), specification
    )
    assert not missing
    case = cases[0]
    reference = {
        "case": {key: case[key] for key in ("depth", "r", "u")},
        "input_u_hex": float(case["u"]).hex(),
        "input_u_exact_decimal": str(Decimal.from_float(float(case["u"]))),
        "method": "independent_mpmath_mellin",
        "settings": {"dps": [45, 60], "tail_digits": [65, 80]},
        "status": "complete",
        "references": [
            {"dps": 45, "log_phi_nats": "1000000000.00000003"},
            {"dps": 60, "log_phi_nats": "1000000000.00000003"},
        ],
    }
    measured = assess_measurement(case["r"], 1e9, 1e9, 1e9, 1e9)
    row = {**case, **finalize_measurement(case, measured, reference)}
    result.update(
        **summarize_measurements([row]),
        independent_high_precision_reference=True,
        config_sha256=canonical_hash(specification),
        resolved_cases_sha256=canonical_hash(cases),
    )
    (root / "data/config.json").write_text(json.dumps(specification))
    (root / "data/resolved-cases.json").write_text(json.dumps(cases))
    (root / "data/independent-reference-cases.json").write_text(json.dumps(cases))
    (root / "data/independent-references.jsonl").write_text(
        json.dumps(reference) + "\n"
    )
    (root / "data/direct-rows.jsonl").write_text(
        json.dumps({**case, **measured}) + "\n"
    )
    (root / "data/rows.jsonl").write_text(json.dumps(row) + "\n")
    (root / "data/summary.json").write_text(json.dumps(result))
    return root, result, specification, pins


def test_sealed_admission_accepts_separately_counted_qualified_evidence(
    qualified_reader_evidence,
):
    root, result, specification, _ = qualified_reader_evidence
    admission.sealed_store_result(root, result, specification)
    assert result["precision_qualified_cases"] == 1
    assert result["nominal_pass_cases"] == 0
    assert not result["all_cases_meet_nominal_limits"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_reference",
        "wrong_case",
        "five_ulps",
        "forged_decimal_error",
        "summary",
        "selection",
        "relaxed_contract",
    ],
)
def test_qualified_admission_recalculates_references_and_rejects_forged_accounting(
    qualified_reader_evidence, mutation
):
    import math

    root, result, specification, _ = qualified_reader_evidence
    row = json.loads((root / "data/rows.jsonl").read_text())
    if mutation == "missing_reference":
        del row["independent_reference"]
    elif mutation == "wrong_case":
        row["independent_reference"]["case"]["r"] -= 1
    elif mutation == "five_ulps":
        row["scalar_log_phi_nats"] += 5 * math.ulp(1e9)
        row["matrix_log_phi_nats"] = row["scalar_log_phi_nats"]
    elif mutation == "forged_decimal_error":
        row["independent_assessment"]["scalar"]["true_error_decimal_nats"] = "0"
    elif mutation == "summary":
        result["nominal_pass_cases"] = 1
    elif mutation == "selection":
        (root / "data/independent-reference-cases.json").write_text("[]")
    else:
        specification["accuracy_contract"]["qualified_arithmetic_budget_ulps"] = 5
    (root / "data/rows.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        admission.sealed_store_result(root, result, specification)


@pytest.fixture
def native_independent_evidence(sealed_independent_evidence):
    from lsa.alt.artifacts import sha256
    from lsa.alt.sealed_native import ABI_VERSION, COMPILE_FLAGS, FORMAT, SOURCE_PATH

    item, pins, engine = sealed_independent_evidence
    engine["store"].update(
        interpolation_backend="native",
        native_library_path="/host/library.so",
        native_library_sha256="c" * 64,
    )
    root = Path(item["path"])
    config = read_json(root / "data/config.json")
    config["stores"][0]["native_library"] = {
        "path": "/host/library.so",
        "sha256": "c" * 64,
    }
    (root / "data/config.json").write_text(json.dumps(config))
    metadata = read_json(root / "data/store-inputs.json")
    metadata[0]["native_identity"] = {
        "format": FORMAT,
        "abi_version": ABI_VERSION,
        "flags": list(COMPILE_FLAGS),
        "binary_sha256": "c" * 64,
        "source_sha256": sha256(SOURCE_PATH),
        "wrapper_sha256": sha256(SOURCE_PATH.with_name("sealed_native.py")),
    }
    (root / "data/store-inputs.json").write_text(json.dumps(metadata))
    return item, pins, engine


def test_native_independent_kernel_gate_binds_actual_backend(
    native_independent_evidence,
):
    item, pins, engine = native_independent_evidence
    admission.sealed_independent_kernel(item, pins, engine)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_native_metadata",
        "source",
        "wrapper",
        "binary",
        "spec_binary",
        "python_disguised",
    ],
)
def test_native_independent_gate_rejects_mismatched_provider_evidence(
    native_independent_evidence, mutation
):
    item, pins, engine = native_independent_evidence
    root = Path(item["path"])
    metadata = read_json(root / "data/store-inputs.json")
    if mutation == "missing_native_metadata":
        del metadata[0]["native_identity"]
    elif mutation in ("source", "wrapper", "binary"):
        metadata[0]["native_identity"][mutation + "_sha256"] = "d" * 64
    elif mutation == "spec_binary":
        config = read_json(root / "data/config.json")
        config["stores"][0]["native_library"]["sha256"] = "d" * 64
        (root / "data/config.json").write_text(json.dumps(config))
    else:
        engine["store"].update(
            interpolation_backend="python",
            native_library_path=None,
            native_library_sha256=None,
        )
    (root / "data/store-inputs.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        admission.sealed_independent_kernel(item, pins, engine)


def test_native_reader_identity_must_match_complete_chain_identity(
    native_independent_evidence,
):
    item, _, engine = native_independent_evidence
    root = Path(item["path"])
    identity = read_json(root / "data/store-inputs.json")[0]["native_identity"]
    engine["sealed_native_identity"] = {**identity, "compiled_compiler": "one"}
    with pytest.raises(ValueError, match="identity"):
        admission._validate_sealed_backend(
            {
                "interpolation_backend": "native",
                "native_identity": {**identity, "compiled_compiler": "other"},
            },
            engine,
        )
