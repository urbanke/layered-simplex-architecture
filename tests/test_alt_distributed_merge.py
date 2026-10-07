"""Saved sample equivalence and adversarial checks for distributed merging."""

from __future__ import annotations

import copy
import gzip
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from lsa.alt.artifacts import Run, read_json, sha256, verify_run, write_json
from lsa.alt.baselines import add_constant, dirichlet_log_evidence
from lsa.alt.benchmark import run_benchmark
from lsa.alt.benchmark_report import generate_reports, load_summary
from lsa.alt.distributed_merge import _merge_benchmark, merge_campaign
from lsa.alt.scaling import report_scaling, run_scaling


class EndpointEvaluator:
    def predict(self, counts, *, depths=None, powers=None):
        grid = depths if depths is not None else powers
        assert list(grid) == [0, 1]
        d, n = len(counts), sum(counts)
        logs = np.array([-n * np.log(d), dirichlet_log_evidence(counts, 1)])
        components = np.array([np.full(d, 1 / d), add_constant(counts, 1)])
        weights = np.exp(logs - logsumexp(logs))
        return SimpleNamespace(
            component_probabilities=components,
            mixture_probabilities=weights @ components,
            posterior=weights,
            component_log_evidence=logs,
            diagnostics={"fixture": "analytic endpoints"},
        )

    def evidence_at_depths(self, d, partition, depths):
        return SimpleNamespace(
            depths=tuple(depths),
            log_evidence=np.full(len(depths), -sum(partition) * np.log(d)),
            diagnostics={"fixture": "uniform sequence law"},
        )


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "source.py").write_text("# frozen fixture source\n")
    for command in (
        ["init", "-q"],
        ["add", "source.py"],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", *command], cwd=root, check=True)
    return root


def protocol(*, all_targets=False, all_families=False):
    path = Path(__file__).resolve().parents[1] / "experiments/alt2027/smoke.json"
    result = read_json(path)
    names = ["benchmark_primary", "factorial", "architecture"]
    if all_families:
        names += ["benchmark_powers", "spectrum", "depth_scaling"]
    result["experiments"] = {name: result["experiments"][name] for name in names}
    for name, config in result["experiments"].items():
        if name.startswith("benchmark_"):
            config.update(depths=[0, 1], fixed_depth=1, powers=[0, 1], trials=3)
            if not all_targets:
                config["targets"] = ["uniform", "dirichlet_half"]
        elif name == "factorial":
            config.update(fixed_depth=0, trials=3)
        elif name in ("spectrum", "depth_scaling"):
            config["trials"] = 3
    return result


def build_plan(protocol, repo):
    from lsa.alt.distributed import make_plan

    return make_plan(
        protocol,
        repo=repo,
        purpose="smoke",
        engine={"mode": "reference"},
        block_size=2,
        factorial_block_size=2,
        batch_size=2,
    )


def engine_record(name):
    from lsa.alt import batch_depth, depth, powers

    if name in ("architecture", "bible_secondary"):
        return {"depth": None}
    with depth.DepthEvaluator(prediction_tolerance=1e-3) as evaluator:
        result = {"depth": evaluator.configuration}
    if name.startswith("benchmark_") or name in (
        "factorial",
        "spectrum",
        "depth_scaling",
    ):
        result["execution"] = {
            "batch_size": 2,
            "batch_implementation_sha256": sha256(batch_depth.__file__),
        }
    if name == "benchmark_powers":
        result["power"] = {
            "settings": asdict(powers.PowerSettings()),
            "implementation_sha256": sha256(powers.__file__),
        }
    return result


def create_shards(tmp_path, repo, specification):
    from lsa.alt.distributed import job_protocol, runtime_identity

    plan = build_plan(specification, repo)
    root = tmp_path / "runs"
    root.mkdir()
    write_json(root / "runtime.json", runtime_identity(plan))
    write_json(
        root / "admission.json", {"purpose": "smoke", "calibration_sha256": None}
    )
    evaluator = EndpointEvaluator()
    for job in plan["jobs"]:
        name, config = job["experiment"], job["config"]
        with Run(
            root / "jobs" / job["id"],
            repo=repo,
            protocol=job_protocol(plan, job),
            experiment=name,
            purpose="smoke",
            engine_identity=engine_record(name),
        ) as run:
            if name.startswith("benchmark_"):
                run_benchmark(
                    config,
                    run.path / "data",
                    depth_evaluator=evaluator,
                    power_evaluator=evaluator,
                )
            elif name in ("factorial", "spectrum", "depth_scaling"):
                run_scaling(name, config, run.path / "data", evaluator=evaluator)
            else:
                (run.path / "data").mkdir()
                write_json(run.path / "data/architecture.json", {"fixture": True})
    return plan, root


def rows(path):
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    with opener(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def reseal(path):
    """Rehash deliberately malformed raw data to exercise semantic validation."""
    manifest = read_json(path / "manifest.json")
    for name in manifest["outputs"]:
        output = path / name
        manifest["outputs"][name] = {
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
        }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_merge_matches_unsharded_records_and_existing_reports(tmp_path, repo):
    specification = protocol(all_targets=True, all_families=True)
    plan, runs = create_shards(tmp_path, repo, specification)
    merged = tmp_path / "merged"
    result = merge_campaign(plan, runs, merged, repo=repo)
    assert set(result["experiments"]) == set(specification["experiments"])
    evaluator = EndpointEvaluator()
    for name, config in specification["experiments"].items():
        record = verify_run(merged / name)
        assert read_json(merged / name / "protocol.json") == specification
        assert record["status"] == "complete"
        assert read_json(merged / name / "source-runtime.json") == read_json(
            runs / "runtime.json"
        )
        source_jobs = read_json(merged / name / "source-jobs.json")["jobs"]
        assert len(source_jobs) == result["experiments"][name]["jobs"]
        data = merged / name / "data"
        if name.startswith("benchmark_"):
            direct = tmp_path / f"direct-{name}"
            original = run_benchmark(
                config, direct, depth_evaluator=evaluator, power_evaluator=evaluator
            )
            assert load_summary(data)["targets"] == original["targets"]
            blocks = read_json(data / "block-diagnostics.json")
            assert blocks["block_count"] == 3
            assert blocks["requested_blocks"] == 10
            assert (
                blocks["pooled_targets"]["uniform"]["4"]["methods"]
                == original["targets"]["uniform"]["4"]["methods"]
            )
            assert (
                read_json(data / "samples/manifest.json")["sampling"]
                == read_json(direct / "samples/manifest.json")["sampling"]
            )
            for entry in read_json(data / "samples/manifest.json")["files"]:
                with (
                    np.load(data / "samples" / entry["path"]) as a,
                    np.load(direct / "samples" / entry["path"]) as b,
                ):
                    assert set(a.files) == set(b.files)
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
            assert {row["trial"] for row in rows(data / "trials.jsonl")} == {0, 1, 2}
            assert all(
                row["source_job"]["id"] in source_jobs
                for row in rows(data / "trials.jsonl")
            )
        elif name in ("factorial", "spectrum", "depth_scaling"):
            direct = tmp_path / f"direct-{name}"
            original = run_scaling(name, config, direct, evaluator=evaluator)
            assert read_json(data / "summary.json") == original
            for path in data.glob("samples-*.jsonl.gz"):
                restored = [
                    {k: v for k, v in row.items() if k != "source_job"}
                    for row in rows(path)
                ]
                assert restored == rows(direct / path.name)
            report_scaling(data, tmp_path / f"report-{name}")
        else:
            assert read_json(data / "architecture.json") == {"fixture": True}
    report = generate_reports(
        merged / "benchmark_primary/data",
        merged / "benchmark_powers/data",
        tmp_path / "benchmark-report",
        config=specification["reports"],
    )
    assert set(report["roles"]) == set(specification["reports"]["roles"])
    assert not list(merged.glob("*/.merge-summary-*"))
    with pytest.raises(FileExistsError):
        merge_campaign(plan, runs, merged, repo=repo)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unexpected",
        "tampered",
        "failed",
        "source",
        "purpose",
        "engine",
        "protocol",
        "runtime",
        "runtime_missing",
    ],
)
def test_merge_rejects_invalid_shards_before_output(tmp_path, repo, mutation):
    specification = protocol()
    plan, runs = create_shards(tmp_path, repo, specification)
    job = next(job for job in plan["jobs"] if job["experiment"] == "benchmark_primary")
    path = runs / "jobs" / job["id"]
    if mutation == "missing":
        shutil.rmtree(path)
    elif mutation == "unexpected":
        shutil.copytree(path, runs / "jobs/unplanned-copy")
    elif mutation == "runtime_missing":
        (runs / "runtime.json").unlink()
    elif mutation == "tampered":
        with (path / "data/trials.jsonl").open("a") as stream:
            stream.write("{}\n")
    else:
        manifest = read_json(path / "manifest.json")
        if mutation == "failed":
            manifest["status"] = "failed"
        elif mutation == "source":
            manifest["source"]["tree_sha256"] = "0" * 64
        elif mutation == "purpose":
            manifest["purpose"] = "production"
        elif mutation == "runtime":
            manifest["environment"]["packages"]["mpmath"] = "different runtime"
        elif mutation == "engine":
            manifest["engine"]["depth"]["prediction_tolerance"] = 0.5
        else:
            changed = read_json(path / "protocol.json")
            changed["protocol_id"] = "different protocol"
            (path / "protocol.json").write_text(json.dumps(changed))
            from lsa.alt.artifacts import canonical_hash

            manifest["protocol_sha256"] = canonical_hash(changed)
        (path / "manifest.json").write_text(json.dumps(manifest))
        if mutation == "protocol":
            reseal(path)
    with pytest.raises(ValueError):
        merge_campaign(plan, runs, tmp_path / "merged", repo=repo)
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("family", ["benchmark_primary", "factorial"])
def test_resealed_duplicate_raw_trial_is_rejected(tmp_path, repo, family):
    specification = protocol()
    plan, runs = create_shards(tmp_path, repo, specification)
    job = next(
        job
        for job in plan["jobs"]
        if job["experiment"] == family and job["config"]["trials"] == 2
    )
    shard = runs / "jobs" / job["id"]
    if family == "benchmark_primary":
        path = shard / "data/trials.jsonl"
        original = rows(path)
        original[1] = copy.deepcopy(original[0])
        path.write_text("".join(json.dumps(row) + "\n" for row in original))
    else:
        path = next((shard / "data").glob("trials-*.jsonl.gz"))
        original = rows(path)
        original[1] = copy.deepcopy(original[0])
        with gzip.open(path, "wt", encoding="utf8") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in original))
    reseal(shard)
    with pytest.raises(ValueError, match="duplicate|differs"):
        merge_campaign(plan, runs, tmp_path / "merged", repo=repo)
    assert (
        read_json(tmp_path / "merged" / family / "manifest.json")["status"] == "failed"
    )


def test_n_subset_merge_rebuilds_full_saved_common_sample_files(tmp_path):
    config = protocol()["experiments"]["benchmark_primary"]
    config.update(targets=["dirichlet_half"], trials=3)
    evaluator = EndpointEvaluator()
    direct = tmp_path / "direct"
    expected = run_benchmark(config, direct, depth_evaluator=evaluator)
    shards = []
    for n in config["n_values"]:
        for start, length in ((0, 2), (2, 1)):
            settings = {
                **config,
                "n_values": [n],
                "sampling_n_values": config["n_values"],
                "trial_start": start,
                "trials": length,
            }
            job_id = f"n-{n}-t-{start}"
            path = tmp_path / job_id
            run_benchmark(settings, path / "data", depth_evaluator=evaluator)
            shards.append(
                {
                    "job": {"id": job_id, "config": settings},
                    "path": path,
                    "manifest_sha256": job_id,
                }
            )
    destination = tmp_path / "merged"
    _merge_benchmark(shards, config, destination, 2)
    assert load_summary(destination)["targets"] == expected["targets"]
    manifest = read_json(destination / "samples/manifest.json")
    for entry in manifest["files"]:
        assert len(entry["source_samples"]) == 2
        with (
            np.load(destination / "samples" / entry["path"]) as actual,
            np.load(direct / "samples" / entry["path"]) as original,
        ):
            for key in original.files:
                np.testing.assert_array_equal(actual[key], original[key])
    hashes = {
        (entry["target_id"], entry["trial"]): entry["sha256"]
        for entry in manifest["files"]
    }
    assert all(
        row["sample_sha256"] == hashes[row["target_id"], row["trial"]]
        for row in rows(destination / "trials.jsonl")
    )


def test_collection_preserves_a_different_recorded_platform_runtime(tmp_path, repo):
    specification = protocol()
    config = specification["experiments"]["benchmark_primary"]
    config.update(targets=["uniform"], trials=1)
    specification["experiments"] = {"benchmark_primary": config}
    plan, runs = create_shards(tmp_path, repo, specification)
    runtime = read_json(runs / "runtime.json")
    runtime.update(system="Linux", machine="x86_64", python="3.11.99")
    (runs / "runtime.json").write_text(json.dumps(runtime))
    for path in (runs / "jobs").iterdir():
        record = read_json(path / "manifest.json")
        record["environment"].update(
            python="3.11.99 (archived worker)",
            platform="Linux-archived-worker",
            machine="x86_64",
        )
        record["engine"]["depth"]["runtime"] = {
            key: runtime[key]
            for key in ("python", "numpy", "scipy", "system", "machine")
        }
        (path / "manifest.json").write_text(json.dumps(record))
    merged = tmp_path / "merged"
    merge_campaign(plan, runs, merged, repo=repo)
    assert read_json(merged / "benchmark_primary/source-runtime.json") == runtime
    assert (
        verify_run(merged / "benchmark_primary")["engine"]["depth"]["runtime"]["python"]
        == "3.11.99"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing_admission",
        "missing_calibration",
        "certificate_hash",
        "engine_coverage",
    ],
)
def test_production_collection_requires_and_preserves_admission(
    tmp_path, repo, mutation
):
    from lsa.alt.artifacts import canonical_hash
    from lsa.alt.distributed import job_protocol, make_plan, runtime_identity

    specification = protocol()
    specification["status"] = "frozen"
    specification["experiments"] = {
        "architecture": specification["experiments"]["architecture"]
    }
    plan = make_plan(
        specification,
        repo=repo,
        purpose="production",
        engine={"mode": "reference"},
        batch_size=2,
    )
    root = tmp_path / "production"
    root.mkdir()
    engine = {"depth": None}
    calibration = {
        "status": "passed",
        "required_checks_complete": True,
        "protocol_sha256": plan["protocol_sha256"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "covered_experiments": ["architecture"],
        "engine_sha256_by_experiment": {"architecture": canonical_hash(engine)},
    }
    if mutation == "engine_coverage":
        calibration["engine_sha256_by_experiment"]["architecture"] = "0" * 64
    admission = {
        "purpose": "production",
        "calibration_sha256": canonical_hash(calibration),
    }
    if mutation == "certificate_hash":
        admission["calibration_sha256"] = "0" * 64
    write_json(root / "runtime.json", runtime_identity(plan))
    if mutation != "missing_admission":
        write_json(root / "admission.json", admission)
    if mutation != "missing_calibration":
        write_json(root / "parent-calibration.json", calibration)
    job = plan["jobs"][0]
    with Run(
        root / "jobs" / job["id"],
        repo=repo,
        protocol=job_protocol(plan, job),
        experiment="architecture",
        purpose="production",
        engine_identity=engine,
    ) as run:
        (run.path / "data").mkdir()
        write_json(run.path / "data/fixture.json", {"complete": True})
    output = tmp_path / "merged-production"
    if mutation is not None:
        with pytest.raises(ValueError, match="admission|calibration|covered"):
            merge_campaign(plan, root, output, repo=repo)
        assert not output.exists()
        return
    merge_campaign(plan, root, output, repo=repo)
    assert read_json(output / "architecture/source-admission.json") == admission
    assert read_json(output / "architecture/source-calibration.json") == calibration
    record = verify_run(output / "architecture")
    pinned = {Path(item["path"]).name for item in record["inputs"]}
    assert {"admission.json", "parent-calibration.json"} <= pinned
