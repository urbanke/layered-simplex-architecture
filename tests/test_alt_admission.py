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
