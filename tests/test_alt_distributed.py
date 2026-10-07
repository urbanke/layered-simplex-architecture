"""Plan integrity, parallel endpoint smoke, immutable restart and failure records."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from lsa.alt import cli, distributed
from lsa.alt.artifacts import (
    Run,
    canonical_hash,
    read_json,
    sha256,
    source_identity,
    verify_run,
    write_json,
)
from lsa.alt.benchmark import prepare_samples

THREAD_LIMITS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@pytest.fixture(autouse=True)
def single_thread_libraries(monkeypatch):
    for name in THREAD_LIMITS:
        monkeypatch.setenv(name, "1")
    # Resource-policy tests supply their own allocation; the runner's host and
    # inherited batch-job environment must not change the test cases.
    for name in tuple(os.environ):
        if name.startswith("SLURM_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(distributed.socket, "gethostname", lambda: "test-laptop")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(distributed.socket, "gethostname", lambda: "test-laptop.local")


@pytest.fixture
def repo(tmp_path):
    """Stable provenance while the development checkout is edited elsewhere."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "source.py").write_text("# immutable fixture source\n")
    for command in (
        ["init", "-q"],
        ["add", "source.py"],
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
        subprocess.run(["git", *command], cwd=root, check=True)
    return root


def protocol(*, all_families=False, frozen=False):
    result = read_json(
        Path(__file__).resolve().parents[1] / "experiments/alt2027/smoke.json"
    )
    if not all_families:
        result["experiments"] = {
            "benchmark_primary": result["experiments"]["benchmark_primary"]
        }
    if frozen:
        result["status"] = "frozen"
    for name, config in result["experiments"].items():
        if name.startswith("benchmark_"):
            config.update(
                targets=["uniform", "dirichlet_half"],
                trials=3,
                depths=[0, 1],
                fixed_depth=1,
                powers=[0, 1],
            )
        elif name in ("spectrum", "factorial", "depth_scaling"):
            config["trials"] = 5
    return result


def plan_for(repo, **kwargs):
    return distributed.make_plan(
        kwargs.pop("protocol", protocol()),
        repo=repo,
        purpose=kwargs.pop("purpose", "smoke"),
        engine={"mode": "reference"},
        block_size=2,
        factorial_block_size=3,
        batch_size=2,
        **kwargs,
    )


def calibration_for(plan):
    return {
        "status": "passed",
        "required_checks_complete": True,
        "protocol_sha256": plan["protocol_sha256"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "covered_experiments": list(plan["protocol"]["experiments"]),
        "engine_sha256_by_experiment": {"benchmark_primary": "a" * 64},
        "configuration_sha256": "b" * 64,
    }


def work_root(path):
    path.mkdir()
    for name in ("jobs", "attempts", "claims", "specs", "workers"):
        (path / name).mkdir()
    return path


def snapshot(path):
    return {str(p.relative_to(path)): sha256(p) for p in path.rglob("*") if p.is_file()}


def test_partition_covers_each_original_trial_once_and_is_deterministic(repo):
    specification = protocol(all_families=True)
    saved = copy.deepcopy(specification)
    plan = plan_for(repo, protocol=specification)
    assert plan == plan_for(repo, protocol=specification)
    assert specification == saved
    assert distributed.validate_plan(plan) == plan
    ids = [job["id"] for job in plan["jobs"]]
    assert len(ids) == len(set(ids))
    actual, expected = Counter(), Counter()
    for name, config in specification["experiments"].items():
        if name.startswith("benchmark_"):
            for target in config["targets"]:
                for trial in range(config["trials"]):
                    for n in config["n_values"]:
                        expected[name, target, trial, n] += 1
        elif name in ("spectrum", "factorial", "depth_scaling"):
            count = len(config["alphas"])
            if name == "spectrum":
                count *= len(config["panels"])
            elif name == "factorial":
                count *= len(config["ds"]) * len(config["ns"])
            for cell_id in range(count):
                for trial in range(config["trials"]):
                    expected[name, cell_id, trial] += 1
        else:
            expected[name] += 1
    for job in plan["jobs"]:
        name, config = job["experiment"], job["config"]
        if name.startswith("benchmark_"):
            assert (
                config["sampling_n_values"]
                == specification["experiments"][name]["n_values"]
            )
            for target in config["targets"]:
                for trial in range(
                    config["trial_start"], config["trial_start"] + config["trials"]
                ):
                    for n in config["n_values"]:
                        actual[name, target, trial, n] += 1
        elif name in ("spectrum", "factorial", "depth_scaling"):
            for cell_id in config["cell_ids"]:
                for trial in range(
                    config["trial_start"], config["trial_start"] + config["trials"]
                ):
                    actual[name, cell_id, trial] += 1
        else:
            actual[name] += 1
    assert actual == expected
    assert set(actual.values()) == {1}


@pytest.mark.parametrize(
    "mutation",
    ["protocol", "missing", "duplicate", "config", "unsafe_id", "unknown_job"],
)
def test_tampered_plan_is_rejected_before_creating_output(tmp_path, repo, mutation):
    plan = plan_for(repo)
    if mutation == "protocol":
        plan["protocol"]["experiments"]["benchmark_primary"]["seed"] += 1
    elif mutation == "missing":
        plan["jobs"].pop()
    elif mutation == "duplicate":
        plan["jobs"].append(copy.deepcopy(plan["jobs"][0]))
    elif mutation == "config":
        plan["jobs"][0]["config"]["trial_start"] += 1
    elif mutation == "unsafe_id":
        plan["jobs"][0]["id"] = "../outside"
    else:
        plan["jobs"].append({"id": "unexpected", "experiment": "unknown", "config": {}})
    with pytest.raises(ValueError):
        distributed.work(plan, tmp_path / "out", repo=repo, workers=1)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("selector", ["trial_start", "cell_ids", "sampling_n_values"])
def test_parent_protocol_must_not_already_be_sharded(repo, selector):
    specification = protocol()
    specification["experiments"]["benchmark_primary"][selector] = 0
    with pytest.raises(ValueError, match="complete unsharded grid"):
        plan_for(repo, protocol=specification)


@pytest.mark.parametrize("field", ["source_tree_sha256", "source_commit"])
def test_source_mismatch_is_rejected_before_output(tmp_path, repo, field):
    plan = plan_for(repo)
    plan[field] = "f" * len(plan[field])
    with pytest.raises(ValueError, match="source differs"):
        distributed.work(plan, tmp_path / "out", repo=repo, workers=1)
    assert not (tmp_path / "out").exists()


def test_production_requires_frozen_protocol_clean_source_and_calibration(
    tmp_path, repo
):
    with pytest.raises(ValueError, match="frozen protocol and clean source"):
        plan_for(repo, purpose="production")
    frozen = protocol(frozen=True)
    plan = plan_for(repo, purpose="production", protocol=frozen)
    with pytest.raises(ValueError, match="requires parent calibration"):
        distributed.work(plan, tmp_path / "out", repo=repo, workers=1)
    (repo / "source.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="clean source"):
        plan_for(repo, purpose="production", protocol=frozen)
    # Even if a dirty tree's hash is inserted into a plan, production refuses it.
    plan["source_tree_sha256"] = source_identity(repo)["tree_sha256"]
    with pytest.raises(ValueError, match="source differs"):
        distributed.work(plan, tmp_path / "out", repo=repo, workers=1)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field", ["engine_config", "power_settings"])
def test_numerical_setting_mismatch_is_rejected_before_output(tmp_path, repo, field):
    plan = plan_for(repo)
    path = tmp_path / "settings.json"
    write_json(
        path,
        {"mode": "reference", "prediction_tolerance": 1e-5}
        if field == "engine_config"
        else {"kernel_tail_drop": 50.0},
    )
    with pytest.raises(ValueError, match="numerical settings differ"):
        distributed.work(plan, tmp_path / "out", repo=repo, workers=1, **{field: path})
    assert not (tmp_path / "out").exists()


def test_derived_calibration_binds_exact_subset_and_preserves_parent(repo):
    plan = plan_for(repo, purpose="production", protocol=protocol(frozen=True))
    parent = calibration_for(plan)
    snapshot_parent = copy.deepcopy(parent)
    job = plan["jobs"][0]
    child = distributed.derive_calibration(plan, job, parent)
    assert parent == snapshot_parent
    assert child["protocol_sha256"] == canonical_hash(
        distributed.job_protocol(plan, job)
    )
    assert child["protocol_sha256"] != plan["protocol_sha256"]
    assert child["engine_sha256_by_experiment"] == parent["engine_sha256_by_experiment"]
    assert child["configuration_sha256"] == parent["configuration_sha256"]
    assert child["derivation"] == {
        "type": "exact_deterministic_subset",
        "job_id": job["id"],
        "parent_calibration_sha256": canonical_hash(parent),
        "parent_protocol_sha256": plan["protocol_sha256"],
        "plan_sha256": canonical_hash(plan),
    }
    changed = copy.deepcopy(job)
    changed["config"]["trials"] += 1
    with pytest.raises(ValueError, match="unknown execution shard"):
        distributed.derive_calibration(plan, changed, parent)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "partial"},
        {"required_checks_complete": False},
        {"protocol_sha256": "0" * 64},
        {"source_tree_sha256": "0" * 64},
        {"covered_experiments": []},
    ],
)
def test_derived_calibration_rejects_missing_parent_coverage(repo, change):
    plan = plan_for(repo, purpose="production", protocol=protocol(frozen=True))
    parent = {**calibration_for(plan), **change}
    with pytest.raises(ValueError, match="parent calibration does not cover"):
        distributed.derive_calibration(plan, plan["jobs"][0], parent)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, 11])
def test_invalid_laptop_process_counts_are_rejected(workers):
    with pytest.raises(ValueError):
        distributed.check_resources(workers, "laptop")


def test_resource_limits_and_required_single_thread_environment(monkeypatch):
    distributed.check_resources(10, "laptop")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="OPENBLAS_NUM_THREADS=1"):
        distributed.check_resources(1, "laptop")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="allocation"):
        distributed.check_resources(1, "scitas")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "jed001")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setattr(distributed.socket, "gethostname", lambda: "jed001.epfl.ch")
    monkeypatch.setattr(
        distributed.subprocess, "check_output", lambda *args, **kwargs: "jed001\n"
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    distributed.check_resources(2, "scitas")
    with pytest.raises(ValueError, match="limit 2"):
        distributed.check_resources(3, "scitas")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "128")
    distributed.check_resources(72, "scitas")
    with pytest.raises(ValueError, match="limit 72"):
        distributed.check_resources(73, "scitas")
    with pytest.raises(ValueError, match="SCITAS profile"):
        distributed.check_resources(1, "laptop")
    monkeypatch.setattr(distributed.socket, "gethostname", lambda: "jed-login")
    with pytest.raises(ValueError, match="outside the allocated"):
        distributed.check_resources(1, "scitas")
    monkeypatch.setattr(distributed.socket, "gethostname", lambda: "jed001")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    with pytest.raises(ValueError, match="one Slurm node and task"):
        distributed.check_resources(1, "scitas")


@pytest.mark.parametrize("index,count", [(True, 2), (-1, 2), (2, 2), (0, 0), (0, True)])
def test_invalid_controller_assignment_is_rejected(tmp_path, repo, index, count):
    with pytest.raises(ValueError):
        distributed.work(
            plan_for(repo),
            tmp_path / "out",
            repo=repo,
            workers=1,
            worker_index=index,
            worker_count=count,
        )
    assert not (tmp_path / "out").exists()


def test_concurrent_metadata_creation_is_atomic_and_conflicts_do_not_overwrite(
    tmp_path,
):
    path = tmp_path / "plan.json"
    value = {"large": "x" * 200_000, "nested": {"identity": [1, 2, 3]}}
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: distributed._same_or_create(path, value), range(24)))
    assert read_json(path) == value
    original = path.read_bytes()
    with pytest.raises(ValueError, match="immutable specification differs"):
        distributed._same_or_create(path, {"different": True})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_two_process_smoke_matches_saved_samples_and_restart_is_read_only(
    tmp_path, repo
):
    specification = protocol()
    plan = plan_for(repo, protocol=specification)
    root = tmp_path / "parallel"
    result = distributed.work(plan, root, repo=repo, workers=2)
    assert result["status"] == "complete"
    assert result["processes"] == 2
    assert {row["id"] for row in result["results"]} == {
        job["id"] for job in plan["jobs"]
    }
    assert {row["status"] for row in result["results"]} == {"complete"}
    assert not list((root / "attempts").iterdir())
    config = specification["experiments"]["benchmark_primary"]
    reference_samples = tmp_path / "reference-samples"
    expected_samples = prepare_samples(config, reference_samples)
    expected_paths = {
        (entry["target_id"], entry["trial"]): entry["path"]
        for entry in expected_samples["files"]
    }
    observed = Counter()
    for job in plan["jobs"]:
        path = root / "jobs" / job["id"]
        assert verify_run(path)["status"] == "complete"
        rows = [
            json.loads(line)
            for line in (path / "data/trials.jsonl").read_text().splitlines()
        ]
        observed.update((row["target_id"], row["trial"], row["n"]) for row in rows)
        sample_manifest = read_json(path / "data/samples/manifest.json")
        for entry in sample_manifest["files"]:
            original = (
                reference_samples / expected_paths[entry["target_id"], entry["trial"]]
            )
            with (
                np.load(path / "data/samples" / entry["path"]) as actual,
                np.load(original) as expected,
            ):
                assert actual.files == expected.files
                for key in expected.files:
                    np.testing.assert_array_equal(actual[key], expected[key])
    assert observed == Counter(
        (target, trial, n)
        for target in config["targets"]
        for trial in range(config["trials"])
        for n in config["n_values"]
    )
    saved_jobs = snapshot(root / "jobs")
    saved_specs = snapshot(root / "specs")
    resumed = distributed.work(plan, root, repo=repo, workers=2)
    assert {row["status"] for row in resumed["results"]} == {"verified_existing"}
    assert snapshot(root / "jobs") == saved_jobs
    assert snapshot(root / "specs") == saved_specs
    assert not list((root / "attempts").iterdir())
    assert len(list((root / "workers").glob("*-finished.json"))) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_bytes",
        "engine_settings",
        "batch_settings",
        "source",
        "depth_runtime",
        "run_runtime",
    ],
)
def test_restart_rejects_corrupted_existing_job_without_overwriting(
    tmp_path, repo, mutation
):
    plan = plan_for(repo)
    root = work_root(tmp_path / "runs")
    job = plan["jobs"][0]
    distributed._execute_job(plan, job, root, repo, None, None, None)
    path = root / "jobs" / job["id"]
    if mutation == "raw_bytes":
        with (path / "data/trials.jsonl").open("a") as stream:
            stream.write("changed\n")
    else:
        manifest = read_json(path / "manifest.json")
        if mutation == "engine_settings":
            manifest["engine"]["depth"]["prediction_tolerance"] = 1e-5
        elif mutation == "batch_settings":
            manifest["engine"]["execution"]["batch_size"] = 999
        elif mutation == "depth_runtime":
            manifest["engine"]["depth"]["runtime"]["numpy"] = "0.0.0"
        elif mutation == "run_runtime":
            manifest["environment"]["packages"]["numpy"] = "0.0.0"
        else:
            manifest["source"]["commit"] = "f" * 40
        (path / "manifest.json").write_text(json.dumps(manifest))
    corrupted = snapshot(path)
    with pytest.raises(ValueError):
        distributed._execute_job(plan, job, root, repo, None, None, None)
    assert snapshot(path) == corrupted
    assert not list((root / "attempts").iterdir())


def test_failed_attempt_survives_successful_retry_and_lock_is_released(
    tmp_path, repo, monkeypatch
):
    plan = plan_for(repo)
    job = plan["jobs"][0]
    root = work_root(tmp_path / "runs")
    real_run_one = cli.run_one

    def fail_with_diagnostic(name, local_protocol, args, directory):
        with Run(
            directory,
            repo=args.repo,
            protocol=local_protocol,
            experiment=name,
            purpose=args.purpose,
        ) as run:
            write_json(
                run.path / "diagnostic.json", {"failure": "injected numerical check"}
            )
            raise ArithmeticError("injected numerical check")

    monkeypatch.setattr(cli, "run_one", fail_with_diagnostic)
    with pytest.raises(ArithmeticError, match="injected numerical check"):
        distributed._execute_job(plan, job, root, repo, None, None, None)
    attempts = list((root / "attempts").iterdir())
    assert len(attempts) == 1
    assert verify_run(attempts[0], require_complete=False)["status"] == "failed"
    assert not (root / "jobs" / job["id"]).exists()
    saved = snapshot(attempts[0])
    monkeypatch.setattr(cli, "run_one", real_run_one)
    result = distributed._execute_job(plan, job, root, repo, None, None, None)
    assert result == {"id": job["id"], "status": "complete"}
    assert list((root / "attempts").iterdir()) == attempts
    assert snapshot(attempts[0]) == saved
    assert verify_run(root / "jobs" / job["id"])["status"] == "complete"


def test_concurrent_claim_rejects_duplicate_execution(tmp_path, repo, monkeypatch):
    plan = plan_for(repo)
    job = plan["jobs"][0]
    root = work_root(tmp_path / "runs")

    def unexpected_run(*args, **kwargs):
        pytest.fail("a duplicate claimant must not execute a job")

    monkeypatch.setattr(cli, "run_one", unexpected_run)
    with (root / "claims" / job["id"]).open("a") as claim:
        fcntl.flock(claim.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            distributed._execute_job(plan, job, root, repo, None, None, None)
    assert not list((root / "attempts").iterdir())
    assert not list((root / "jobs").iterdir())


def admitted_fixture(plan):
    """Synthetic engine identity for admission gates, without a production run."""
    from lsa.alt.depth import _json_hash

    engine = {"depth": {"mode": "reference", "fixture": "admission-only"}}
    parent = calibration_for(plan)
    parent["engine_sha256_by_experiment"]["benchmark_primary"] = canonical_hash(engine)
    parent["configuration_sha256"] = _json_hash(engine["depth"])
    return {"engine": engine}, parent


def test_saved_admission_requires_parent_engine_and_depth_configuration(repo):
    plan = plan_for(repo, purpose="production", protocol=protocol(frozen=True))
    job = plan["jobs"][0]
    record, parent = admitted_fixture(plan)
    distributed.validate_saved_admission(plan, job, record, parent)
    with pytest.raises(ValueError, match="lacks calibration"):
        distributed.validate_saved_admission(plan, job, record, None)
    wrong_engine = copy.deepcopy(parent)
    wrong_engine["engine_sha256_by_experiment"]["benchmark_primary"] = "0" * 64
    with pytest.raises(ValueError, match="engine is not covered"):
        distributed.validate_saved_admission(plan, job, record, wrong_engine)
    wrong_depth = {
        **parent,
        "configuration_sha256": canonical_hash(record["engine"]["depth"]),
    }
    assert wrong_depth["configuration_sha256"] != parent["configuration_sha256"]
    with pytest.raises(ValueError, match="depth configuration is not covered"):
        distributed.validate_saved_admission(plan, job, record, wrong_depth)


def test_existing_production_job_cannot_skip_saved_admission(
    tmp_path, repo, monkeypatch
):
    plan = plan_for(repo, purpose="production", protocol=protocol(frozen=True))
    job = plan["jobs"][0]
    record, parent = admitted_fixture(plan)
    root = work_root(tmp_path / "runs")
    (root / "jobs" / job["id"]).mkdir()
    certificate = tmp_path / "parent.json"
    write_json(certificate, parent)
    monkeypatch.setattr(distributed, "_verify_job", lambda *args: record)
    assert distributed._execute_job(plan, job, root, repo, None, None, certificate) == {
        "id": job["id"],
        "status": "verified_existing",
    }
    parent["engine_sha256_by_experiment"]["benchmark_primary"] = "0" * 64
    certificate.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="engine is not covered"):
        distributed._execute_job(plan, job, root, repo, None, None, certificate)
    assert not list((root / "attempts").iterdir())


def test_root_admission_captures_parent_and_refuses_changed_certificate(
    tmp_path, repo, monkeypatch
):
    plan = plan_for(repo, purpose="production", protocol=protocol(frozen=True))
    _, parent = admitted_fixture(plan)
    certificate = tmp_path / "parent.json"
    write_json(certificate, parent)
    root = tmp_path / "runs"
    submitted_paths = []

    class AdmissionOnlyPool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, _function, *args):
            submitted_paths.append(args[-1])
            assert read_json(args[-1]) == parent
            # A change to the original file after capture cannot reach workers.
            certificate.write_text(json.dumps({**parent, "revision_note": "changed"}))
            future = Future()
            future.set_result({"id": args[1]["id"], "status": "verified_existing"})
            return future

    monkeypatch.setattr(distributed, "ProcessPoolExecutor", AdmissionOnlyPool)
    distributed.work(plan, root, repo=repo, workers=2, calibration=certificate)
    assert submitted_paths == [root / "parent-calibration.json"] * len(plan["jobs"])
    assert read_json(root / "parent-calibration.json") == parent
    assert read_json(root / "admission.json") == {
        "calibration_sha256": canonical_hash(parent),
        "purpose": "production",
    }
    original = snapshot(root)
    with pytest.raises(ValueError, match="immutable specification differs.*admission"):
        distributed.work(plan, root, repo=repo, workers=2, calibration=certificate)
    assert snapshot(root) == original
