"""Sharding preserves the declared draws; failed references block acceptance."""

import json

import numpy as np
import pytest

from lsa.alt.calibration import depth_shards, kernel_status, run_suite


def test_sharding_preserves_random_samples_and_case_order():
    config = {
        "seed": 42,
        "cases": [
            {"id": str(i), "kind": "synthetic", "target": "uniform"} for i in range(7)
        ],
    }
    config["cases"][3]["seed_coordinates"] = [1, 8, 4]
    single = {c["id"]: c for c in depth_shards(config, 1)[0]["cases"]}
    parallel = {c["id"]: c for shard in depth_shards(config, 3) for c in shard["cases"]}
    assert single == parallel
    for key in single:
        a = np.random.default_rng(single[key]["seed_coordinates"]).multinomial(
            100, [0.2, 0.8]
        )
        b = np.random.default_rng(parallel[key]["seed_coordinates"]).multinomial(
            100, [0.2, 0.8]
        )
        np.testing.assert_array_equal(a, b)
    assert "seed_coordinates" not in config["cases"][0]
    config["cases"].append(config["cases"][0])
    with pytest.raises(ValueError, match="unique"):
        depth_shards(config, 3)


def test_kernel_acceptance_requires_batched_and_independent_reference_checks(tmp_path):
    summary = {
        "cases": 1,
        "source_unchanged": True,
        "reference_converged_cases": 1,
        "direct_column_nominal_passes": 1,
    }
    config = {
        "special_function_cases": [{}],
        "nominal_tolerance_nats": 3e-9,
        "reference_convergence_nats": 1e-25,
    }
    row = {
        "batched_direct_column_error_nats": 0,
        "direct_column_refined_error_nats": 0,
        "tolerance_nats": 1e-11,
        "stores": {},
    }
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    special = [{"meijer_error_nats": 1e-40, "recursion_error_nats": 1e-40}]
    (tmp_path / "special-functions.json").write_text(json.dumps(special))
    assert kernel_status(summary, tmp_path, config) == "passed"
    row["batched_direct_column_error_nats"] = 2e-11
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    assert kernel_status(summary, tmp_path, config) == "failed"
    row["batched_direct_column_error_nats"] = 0
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    special[0]["recursion_error_nats"] = 1e-12
    (tmp_path / "special-functions.json").write_text(json.dumps(special))
    assert kernel_status(summary, tmp_path, config) == "failed"


def test_missing_engine_fails_before_creating_calibration_run(tmp_path):
    with pytest.raises(ValueError, match="engine-config"):
        run_suite("chain", tmp_path / "missing.json", tmp_path / "run", repo=tmp_path)
    assert not (tmp_path / "run").exists()
