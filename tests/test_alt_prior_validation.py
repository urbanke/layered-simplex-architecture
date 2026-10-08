"""Independent finite identities, Monte Carlo uncertainty, and simplex checks."""

import json
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from lsa.alt.prior_validation import (
    finite_identity_suite,
    prior_monte_carlo,
    run_prior_validation,
    simplex_moments_d2,
)


def laplace_log_evidence(counts, depth):
    if depth == 0:
        return -sum(counts) * math.log(len(counts))
    assert depth == 1
    # Sequential rising products, separate from implementation's Gamma form.
    return sum(math.log(k) for count in counts for k in range(1, int(count) + 1)) - sum(
        math.log(len(counts) + i) for i in range(sum(counts))
    )


def test_mc_batch_combination_matches_individual_sample_variance():
    result = prior_monte_carlo(
        [2, 1, 0], 2, samples=103, batch_size=16, seed_coordinates=[1234, 7]
    )
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1234, 7])))
    log_y = np.log(rng.exponential(size=(103, 2, 3))).sum(axis=1)
    log_theta = log_y - logsumexp(log_y, axis=1, keepdims=True)
    moments = np.exp(log_theta @ np.array([2, 1, 0]))
    assert sum(batch["samples"] for batch in result["batches"]) == 103
    assert result["mean_probability"] == pytest.approx(moments.mean(), rel=1e-14)
    assert result["standard_error_probability"] == pytest.approx(
        moments.std(ddof=1) / np.sqrt(103), rel=1e-14
    )
    repeated = prior_monte_carlo(
        [2, 1, 0], 2, samples=103, batch_size=16, seed_coordinates=[1234, 7]
    )
    assert repeated["sample_moments_sha256"] == result["sample_moments_sha256"]


def test_direct_simplex_layer_one_matches_beta_moments_without_normalizing():
    cases = [[0, 0], [1, 0], [2, 1], [3, 1]]
    result = simplex_moments_d2(
        cases,
        1,
        node_orders=[128, 256, 512],
        logit_bound=30,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    )
    expected = [
        math.factorial(a) * math.factorial(b) / math.factorial(a + b + 1)
        for a, b in cases
    ]
    np.testing.assert_allclose(
        result["sequence_probabilities"], expected, atol=3e-13, rtol=0
    )
    assert result["converged"] is True
    assert result["normalization_applied"] is False


def test_simplex_depth_two_symmetry_and_sequence_partition_identity():
    result = simplex_moments_d2(
        [[2, 0], [1, 1], [0, 2]],
        2,
        node_orders=[128, 256],
        logit_bound=26,
        absolute_tolerance=2e-10,
        relative_tolerance=2e-10,
    )
    first, mixed, last = result["sequence_probabilities"]
    assert first == pytest.approx(last, abs=1e-14)
    assert first + 2 * mixed + last == pytest.approx(1, abs=3e-11)
    assert result["converged"] is True


def test_insufficient_simplex_refinement_is_explicitly_not_converged():
    result = simplex_moments_d2(
        [[3, 1]],
        3,
        node_orders=[8, 16],
        logit_bound=20,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    assert result["converged"] is False


def test_full_enumeration_checks_components_and_equal_mixture():
    result = finite_identity_suite(
        [0.2, 0.3, 0.5],
        3,
        [0, 1],
        log_evidence=laplace_log_evidence,
        maximum_sequences=1000,
    )
    assert set(result["models"]) == {"depth_0", "depth_1", "equal_prior_mixture"}
    assert result["enumerated_through_n"] == 4
    for model in result["models"].values():
        for key in (
            "max_sequence_normalization_error",
            "max_predictive_normalization_error",
            "kl_chain_residual_nats",
            "discovered_set_decomposition_residual_nats",
        ):
            assert abs(model[key]) < 5e-14
        assert model["joint_kl_nats"] >= 0


def test_identity_checker_detects_unnormalized_evidence_without_repair():
    def broken(counts, depth):
        return laplace_log_evidence(counts, depth) + 0.01 * sum(counts)

    result = finite_identity_suite(
        [0.3, 0.7], 2, [1], log_evidence=broken, maximum_sequences=100
    )
    component = result["models"]["depth_1"]
    assert component["max_sequence_normalization_error"] > 0.02
    assert component["max_predictive_normalization_error"] > 0.009
    assert result["normalization_applied"] is False


def test_small_recorded_suite_saves_cases_units_se_and_module_hashes(tmp_path):
    config = {
        "protocol_id": "unit-test-bounded-prior-check",
        "seed": 7,
        "reference": {
            "contour_steps": [0.01, 0.005],
            "contour_refinement_tolerance_nats": 2e-8,
        },
        "monte_carlo": {
            "samples_per_case": 1000,
            "batch_size": 128,
            "maximum_absolute_z_score": 5,
            "cases": [{"id": "L1-d2", "depth": 1, "counts": [2, 1]}],
        },
        "simplex": {
            "depths": [1],
            "counts": [[2, 1]],
            "node_orders": [128, 256, 512],
            "logit_bound": 30,
            "refinement_absolute_tolerance": 1e-10,
            "refinement_relative_tolerance": 1e-10,
            "comparison_absolute_tolerance": 1e-10,
        },
        "identities": {
            "probability_tolerance": 1e-12,
            "log_identity_tolerance_nats": 1e-12,
            "maximum_sequences": 100,
            "cases": [{"id": "binary", "target": [0.3, 0.7], "n": 2, "depths": [0, 1]}],
        },
    }
    summary = run_prior_validation(config, tmp_path / "run")
    assert summary["status"] == "passed"
    assert summary["production_certified"] is False
    checks = json.loads((tmp_path / "run/checks.json").read_text())
    mc = checks[0]
    assert mc["units"] == "Monte Carlo standard errors of sequence probability"
    assert mc["details"]["samples"] == 1000
    fingerprints = json.loads((tmp_path / "run/module-fingerprints.json").read_text())
    assert "prior_validation.py" in fingerprints
    with pytest.raises(FileExistsError):
        run_prior_validation(config, tmp_path / "run")
