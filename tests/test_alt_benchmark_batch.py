"""Saved-sample batching integration; only analytic depth-zero/one models."""

import hashlib
import json

import numpy as np
import pytest

from lsa.alt import benchmark
from lsa.alt.depth import DepthEvaluator


@pytest.fixture
def implementation():
    return benchmark


def settings(*, powers=False, targets=benchmark.TARGET_IDS, trials=3):
    return {
        "d": 8,
        "n_values": [3, 7],
        "trials": trials,
        "seed": 7321,
        "sample_set_id": "batch-integration",
        "targets": list(targets),
        "methods": list(
            benchmark.POWER_METHODS if powers else benchmark.PRIMARY_METHODS
        ),
        "depths": [0, 1],
        "fixed_depth": 1,
        "powers": [0, 1],
        "normalize_numerical": True,
        "numerical_normalization_tolerance": 1e-3,
        "dirichlet_exponents": list(range(-24, 5)),
        "dirichlet_target_policy": "redraw_per_trial_shared_across_n",
        "sample_size_policy": "independent_multinomial_per_n",
    }


class RecordingDepth(DepthEvaluator):
    def __init__(self):
        super().__init__()
        self.batch_profiles = []
        self.single_calls = 0

    def predict(self, *args, **kwargs):
        self.single_calls += 1
        return super().predict(*args, **kwargs)

    def prediction_by_count_batch(self, d, profiles, *, depths):
        self.batch_profiles.append((d, dict(profiles), tuple(depths)))
        return super().prediction_by_count_batch(d, profiles, depths=depths)


class AnalyticPredictOnly:
    """Old predict-only fakes remain usable; power endpoints share these laws."""

    def __init__(self):
        self.engine = DepthEvaluator()
        self.calls = []

    def predict(self, counts, *, depths=None, powers=None):
        self.calls.append(hashlib.sha256(np.asarray(counts).tobytes()).hexdigest())
        return self.engine.predict(counts, depths=depths if powers is None else powers)


def records(path):
    rows = [
        json.loads(line) for line in (path / "trials.jsonl").read_text().splitlines()
    ]
    keyed = {(row["target_id"], row["trial"], row["n"]): row for row in rows}
    assert len(keyed) == len(rows)
    return keyed


def equal_numerics(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_numerics(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            equal_numerics(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=0, abs=2e-13)
    else:
        assert actual == expected


def test_all_targets_keep_samples_results_and_power_pairs_across_bounded_batches(
    implementation, tmp_path
):
    config = settings(powers=True)
    samples = tmp_path / "samples"
    implementation.prepare_samples(config, samples)
    hashes_before = {
        path.name: benchmark.sha256_file(path) for path in samples.iterdir()
    }
    single, batched = tmp_path / "single", tmp_path / "batched"
    single_engine, batch_engine = RecordingDepth(), RecordingDepth()
    power_single, power_batch = AnalyticPredictOnly(), AnalyticPredictOnly()
    summary_single = implementation.run_benchmark(
        config,
        single,
        depth_evaluator=single_engine,
        power_evaluator=power_single,
        samples_dir=samples,
        batch_size=None,
    )
    summary_batch = implementation.run_benchmark(
        config,
        batched,
        depth_evaluator=batch_engine,
        power_evaluator=power_batch,
        samples_dir=samples,
        batch_size=2,
    )
    original, grouped = records(single), records(batched)
    assert original.keys() == grouped.keys()
    assert len(original) == 11 * 3 * 2
    assert batch_engine.single_calls == 0
    assert single_engine.single_calls == len(original)
    assert sorted(power_single.calls) == sorted(power_batch.calls)
    for d, profiles, grid in batch_engine.batch_profiles:
        assert d == 8 and grid == (0, 1) and 1 <= len(profiles) <= 2
        assert len({sum(partition) for partition in profiles.values()}) == 1
    for identity, row in original.items():
        other = grouped[identity]
        for key in (
            "sample_set_id",
            "target_id",
            "trial",
            "n",
            "sample_file",
            "sample_sha256",
            "losses",
            "posteriors",
            "paired_differences",
            "derived",
        ):
            equal_numerics(other[key], row[key])
        for family in ("depth", "power"):
            for key in (
                "indices",
                "component_normalization",
                "component_log_evidence_nats",
            ):
                equal_numerics(
                    other["diagnostics"][family][key], row["diagnostics"][family][key]
                )
        assert other["diagnostics"]["depth"]["evaluator"]["batch_cache"]["hit"]
    equal_numerics(summary_batch["targets"], summary_single["targets"])
    assert (
        summary_batch["sample_manifest_sha256"]
        == summary_single["sample_manifest_sha256"]
    )
    assert {
        path.name: benchmark.sha256_file(path) for path in samples.iterdir()
    } == hashes_before
    preparation = [
        json.loads(line)
        for line in (batched / "batching.jsonl").read_text().splitlines()
    ]
    assert sum(len(row["samples"]) for row in preparation) == len(original)
    assert summary_batch["batching"]["cohorts_by_n"] == len(preparation)
    assert summary_batch["batching"][
        "preparation_records_sha256"
    ] == benchmark.sha256_file(batched / "batching.jsonl")
    assert "exclude shared" in summary_batch["batching"]["timing"]
    assert "batching" not in summary_single


def test_none_retains_predict_only_api_and_enabled_batch_requires_it(
    implementation, tmp_path
):
    config = settings(targets=["uniform"], trials=1)
    old_fake = AnalyticPredictOnly()
    implementation.run_benchmark(config, tmp_path / "single", depth_evaluator=old_fake)
    assert len(old_fake.calls) == 2
    with pytest.raises(TypeError, match="prediction_by_count_batch"):
        implementation.run_benchmark(
            config, tmp_path / "invalid", depth_evaluator=old_fake, batch_size=2
        )
    assert not (tmp_path / "invalid").exists()


def test_every_file_in_a_cohort_is_verified_before_any_preparation(
    implementation, tmp_path
):
    config = settings(targets=["uniform"], trials=2)
    samples = tmp_path / "samples"
    manifest = implementation.prepare_samples(config, samples)
    with (samples / manifest["files"][1]["path"]).open("ab") as stream:
        stream.write(b"tampered")
    engine = RecordingDepth()
    with pytest.raises(ValueError, match="checksum"):
        implementation.run_benchmark(
            config,
            tmp_path / "run",
            depth_evaluator=engine,
            samples_dir=samples,
            batch_size=2,
        )
    assert engine.batch_profiles == []
    assert not (tmp_path / "run/summary.json").exists()


def test_duplicate_manifest_identity_is_rejected_before_evaluation(
    implementation, tmp_path
):
    config = settings(targets=["uniform"], trials=2)
    samples = tmp_path / "samples"
    manifest = implementation.prepare_samples(config, samples)
    manifest["files"][1]["trial"] = 0
    (samples / "manifest.json").write_text(json.dumps(manifest))
    engine = RecordingDepth()
    with pytest.raises(ValueError, match="trial identities"):
        implementation.run_benchmark(
            config,
            tmp_path / "run",
            depth_evaluator=engine,
            samples_dir=samples,
            batch_size=2,
        )
    assert engine.batch_profiles == []


def test_batch_size_must_be_explicit_positive_integer(implementation, tmp_path):
    for value in (0, -1, True, 1.5):
        with pytest.raises((TypeError, ValueError), match="batch_size"):
            implementation.run_benchmark(
                settings(targets=["uniform"]),
                tmp_path / str(value),
                depth_evaluator=DepthEvaluator(),
                batch_size=value,
            )
        assert not (tmp_path / str(value)).exists()
