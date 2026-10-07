"""Independent checks of E**w, not just a second call to its outer integral."""

import json
import math
from dataclasses import replace
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfcx, gammaln, logsumexp

from lsa.alt.powers import (
    PowerEvaluator,
    PowerIntegrationError,
    PowerSettings,
    _legendre,
    log_scaled_kernel,
)


def simplex_evidence_d2(a, b, w):
    # A Dirichlet(1,1) layer has U~Uniform(0,1); normalize U**w directly.
    # This reference uses neither Laplace kernels nor the outer evidence identity.
    def f(u):
        if u <= 0 or u >= 1:
            return 0.0
        z = w * (math.log(u) - math.log1p(-u))
        return math.exp(-a * np.logaddexp(0, -z) - b * np.logaddexp(0, z))

    return quad(f, 0, 1, points=[0.5], epsabs=2e-13, epsrel=2e-13)[0]


def test_analytic_endpoints_and_empty_sample():
    counts = np.array([8, 3, 0, 1, 0])
    result = PowerEvaluator().predict(counts, powers=[0, 1])
    assert np.allclose(result.component_probabilities[0], 0.2, atol=0, rtol=0)
    assert np.allclose(result.component_probabilities[1], (counts + 1) / 17)
    assert result.component_log_evidence[0] == -12 * math.log(5)
    assert result.component_log_evidence[1] == pytest.approx(
        gammaln(5) - gammaln(17) + np.sum(gammaln(counts + 1)))
    empty = PowerEvaluator().predict(np.zeros(9, dtype=int), powers=range(81))
    assert np.all(empty.component_log_evidence == 0)
    assert np.allclose(empty.component_probabilities, 1 / 9)
    assert np.allclose(empty.posterior, 1 / 81)
    one_symbol = PowerEvaluator().predict([15], powers=[0, 1, 2, 80])
    assert np.all(one_symbol.component_probabilities == 1)
    assert np.all(one_symbol.component_log_evidence == 0)


@pytest.mark.parametrize("v", [-3.0, 0.0, 2.0, 4.0])
def test_w2_zero_kernel_against_independent_closed_form(v):
    # Integral_0^infty exp(-x-t*x*x) dx has an erfcx closed form.
    got, diag = log_scaled_kernel(0, 2, v)
    a = math.exp(v)
    expected = math.log(a * math.sqrt(math.pi) * erfcx(a / 2) / 2)
    assert got == pytest.approx(expected, abs=2e-11)
    assert diag["tail_relative_estimate"] < 1e-12


@pytest.mark.parametrize("nodes", [32, 64, 128, 256, 512])
def test_legendre_rule_polynomial_moments(nodes):
    x, weights = _legendre(nodes)
    assert np.all(weights > 0)
    # These integrals are analytic; checking weight sum alone misses the
    # platform-dependent endpoint weight errors behind the calibration failure.
    for degree in (2, 20, 50):
        assert float(weights @ x**degree) == pytest.approx(
            2 / (degree + 1), rel=0, abs=5e-15)


@pytest.mark.parametrize("w,v", [
    (59, 0.02737464341427795), (80, 0.09115050933976537),
    (60, 0.031201174682110977), (77, 0.08495046975937914),
])
def test_strict_zero_kernel_against_high_precision_exponential_integral(w, v):
    # Actual SCITAS calibration failures; integrate directly in E coordinates
    # independently of the production log-coordinate Gauss-Legendre rule.
    with mp.workdps(60):
        cutoff = mp.exp(mp.mpf(v))
        expected = mp.quad(lambda x: mp.exp(-x - (x / cutoff)**w),
                           [0, cutoff / 2, cutoff * mp.mpf(".9"), cutoff,
                            cutoff * mp.mpf("1.1"), cutoff * mp.mpf("1.5"),
                            2 * cutoff])
        # For E>=2*cutoff, exp(-(E/cutoff)^w)<=exp(-2^w).
        tail_bound = mp.exp(-mp.power(2, w) - 2 * cutoff)
        assert tail_bound / expected < mp.mpf("1e-60")
        actual, diagnostics = log_scaled_kernel(
            0, w, v, settings=replace(PowerSettings(),
                                      kernel_log_tolerance=5e-13,
                                      kernel_tail_drop=48))
        assert abs(mp.mpf(actual) - mp.log(expected)) < mp.mpf("3e-14")
        assert diagnostics["refinement_log_difference"] <= 5e-13


@pytest.mark.parametrize("r", [0, 1, 2])
def test_high_power_cutoff_against_direct_exponential_integral(r):
    # This point has a small but significant cutoff tail away from the mode;
    # a single coarse log-grid can incorrectly look converged here.
    w, v = 80, 2.4766421435720125
    a = math.exp(v)

    def f(x):
        if x == 0:
            return 1.0 if r == 0 else 0.0
        log_y = w * math.log(x / a)
        if log_y > 7:
            return 0.0  # exp(-exp(7)) is already below double precision.
        return math.exp(-x + r * log_y - math.exp(log_y))

    expected, error = quad(f, 0, 80, points=[a * 0.8, a, a * 1.2],
                           epsabs=2e-15, epsrel=2e-13)
    actual, _ = log_scaled_kernel(r, w, v)
    assert error / expected < 2e-10
    assert actual == pytest.approx(math.log(expected), abs=2e-10)


@pytest.mark.parametrize("w", [2, 5, 80])
def test_power_evidence_and_predictive_against_direct_simplex_integral(w):
    counts = [3, 2]
    result = PowerEvaluator().predict(counts, powers=[w])
    evidence = simplex_evidence_d2(3, 2, w)
    independent = np.array([simplex_evidence_d2(4, 2, w),
                            simplex_evidence_d2(3, 3, w)]) / evidence
    assert result.component_log_evidence[0] == pytest.approx(math.log(evidence), abs=2e-9)
    assert np.allclose(result.component_probabilities[0], independent, atol=2e-9, rtol=2e-9)
    assert result.diagnostics["component_normalization"][0] == pytest.approx(1, abs=2e-9)
    assert result.diagnostics["normalization_applied"] is False


def test_w2_against_independent_three_symbol_simplex_integral():
    # Uniform density 2 on the 2-simplex, integrated after U2=(1-U1)*z.
    def inner(u):
        def f(z):
            q = np.array([u, (1-u)*z, (1-u)*(1-z)]) ** 2
            q /= q.sum()
            return 2*(1-u)*q[0]*q[1]
        return quad(f, 0, 1, epsabs=1e-11, epsrel=1e-11)[0]
    expected = quad(inner, 0, 1, epsabs=1e-11, epsrel=1e-11)[0]
    result = PowerEvaluator().predict([1, 1, 0], powers=[2])
    assert math.exp(result.component_log_evidence[0]) == pytest.approx(expected, abs=2e-10)


def test_evidence_ratios_chain_rule_and_mixture_posterior():
    evaluator = PowerEvaluator()
    powers = [0, 1, 2, 80]
    base = evaluator.predict([2, 1, 0], powers=powers)
    augmented = evaluator.predict([2, 1, 1], powers=powers)
    ratios = np.exp(augmented.component_log_evidence - base.component_log_evidence)
    assert np.allclose(ratios, base.component_probabilities[:, 2], atol=2e-8, rtol=2e-8)
    assert np.allclose(base.posterior,
                       np.exp(base.component_log_evidence - logsumexp(base.component_log_evidence)))
    assert np.allclose(base.mixture_probabilities, base.posterior @ base.component_probabilities)
    mixture_ratio = math.exp(augmented.diagnostics["mixture_log_evidence"]
                             - base.diagnostics["mixture_log_evidence"])
    assert mixture_ratio == pytest.approx(base.mixture_probabilities[2], rel=2e-8)


def test_high_power_at_benchmark_dimensions_with_stricter_convergence():
    # Edge-shaped profile exercises unseen-mass amplification by 9,995 and r=500.
    counts = np.zeros(10_000, dtype=int)
    counts[:5] = [500, 300, 150, 40, 10]
    standard = PowerEvaluator().predict(counts, powers=[80])
    strict = PowerEvaluator(replace(PowerSettings(), kernel_log_tolerance=5e-13,
                                    kernel_tail_drop=48, outer_relative_tolerance=3e-9,
                                    outer_tail_drop=38)).predict(counts, powers=[80])
    assert np.all(np.isfinite(standard.component_log_evidence))
    assert np.all(standard.component_probabilities > 0)
    assert abs(standard.component_log_evidence[0] - strict.component_log_evidence[0]) < 2e-7
    assert np.max(np.abs(np.log(standard.component_probabilities
                                / strict.component_probabilities))) < 2e-7
    assert abs(standard.diagnostics["mixture_normalization"] - 1) < 2e-8


def test_failed_quadrature_is_not_silently_accepted():
    cfg = replace(PowerSettings(), kernel_max_nodes=32)
    with pytest.raises(PowerIntegrationError, match="refinement failed"):
        PowerEvaluator(cfg).predict([2, 1, 0], powers=[2])


@pytest.mark.parametrize("counts,powers", [
    ([1, -1], [2]), ([1, 0.5], [2]), ([1, 0], []), ([1, 0], [2, 2]),
    ([1, 0], [81]), ([1, 0], [2.5]), ([1001, 0], [2]),
])
def test_reject_invalid_or_out_of_domain_inputs(counts, powers):
    with pytest.raises(ValueError):
        PowerEvaluator().predict(counts, powers=powers)


def test_high_dimensional_w2_against_closed_form_moment_reference():
    from lsa.alt.power_validation import w2_closed_form_reference

    counts = np.zeros(10_000, dtype=int)
    counts[:900] = 1
    counts[900:950] = 2
    result = PowerEvaluator().prediction_by_count(10_000, counts, powers=[2])
    diag = result.diagnostics["components"][0]
    reference = w2_closed_form_reference(
        10_000, counts, mode=diag["v_mode"], window=diag["v_window"])
    assert abs(reference["log_evidence"]-result.component_log_evidence[0]) < 2e-8
    assert np.max(np.abs(np.log(reference["probabilities"]
                               / result.component_probabilities[0]))) < 2e-9
    assert abs(reference["diagnostics"]["raw_normalization"]-1) < 2e-9


@pytest.mark.parametrize("r,w,v", [(967, 80, 1.0), (967, 80, 2.0), (400, 40, 1.5)])
def test_large_count_kernel_against_high_precision_direct_exponential_integral(r, w, v):
    # Independent x-domain integral with arbitrary precision. Its mode omits
    # the log-coordinate Jacobian used in the production kernel's s integral.
    with mp.workdps(80):
        a, wr = mp.exp(v), mp.mpf(w*r)
        left, right = mp.mpf(0), min(wr, a*mp.power(r, mp.mpf(1)/w))
        for _ in range(300):
            middle = (left+right)/2
            if middle+w*(middle/a)**w < wr:
                left = middle
            else:
                right = middle
        mode = (left+right)/2
        q = (mode/a)**w
        scale = 1/mp.sqrt(mode+w*w*q)

        def drop(z):
            delta = scale*z
            return wr*mp.log1p(delta)-mode*delta-q*mp.expm1(w*mp.log1p(delta))

        integral = mp.quad(lambda z: mp.exp(drop(z)), [-20, -10, -4, 0, 4, 10, 20])
        # Strict concavity in x bounds both omitted tails by tangent integrals.
        tail_bound = (mp.exp(drop(-20))/mp.diff(drop, -20)
                      - mp.exp(drop(20))/mp.diff(drop, 20))
        assert tail_bound/integral < mp.mpf("1e-60")
        expected = mp.log(mode*scale)+wr*mp.log(mode/a)-mode-q+mp.log(integral)
        actual, _ = log_scaled_kernel(r, w, v)
        assert abs(mp.mpf(actual)-expected) < mp.mpf("2e-10")


def _small_calibration_config():
    config = json.loads((Path(__file__).parents[1]
                         / "experiments/alt2027/power-validation.json").read_text())
    config.update(d=8, n=4, workers=1, targets=["uniform"], powers=[0, 1, 2],
                  independent_w2_targets=[])
    return config


def test_power_calibration_retains_profiles_components_and_hashes(tmp_path):
    from lsa.alt.artifacts import sha256
    from lsa.alt.power_validation import run_power_validation

    out = tmp_path / "pilot"
    result = run_power_validation(_small_calibration_config(), out)
    assert result["status"] == "passed"
    assert result["source_unchanged"] is True
    assert result["production_domain_certified"] is False
    records = [json.loads(line) for line in (out / "uniform/components.jsonl").read_text().splitlines()]
    assert len(records) == 6
    assert all(record["status"] == "passed" for record in records)
    sample = json.loads((out / "samples.json").read_text())["profiles"][0]
    assert sha256(out / sample["path"]) == sample["sha256"]
    summary = json.loads((out / "uniform/summary.json").read_text())
    assert summary["normalization_applied"] is False
    assert summary["default"]["max_raw_normalization_error"] < 2e-8
    assert summary["comparison"]["max_predictive_kl_change_bits"] < 1e-5
    with pytest.raises(FileExistsError):
        run_power_validation(_small_calibration_config(), out)


def test_power_calibration_records_failed_components_without_mixture(tmp_path):
    from lsa.alt.power_validation import run_power_validation

    config = _small_calibration_config()
    config["default_settings"]["kernel_max_nodes"] = 32
    out = tmp_path / "failed-pilot"
    result = run_power_validation(config, out)
    assert result["status"] == "failed"
    summary = json.loads((out / "uniform/summary.json").read_text())
    assert summary["failures"][0]["power"] == 2
    assert "default" not in summary
    records = [json.loads(line) for line in (out / "uniform/components.jsonl").read_text().splitlines()]
    assert len(records) == 6
    assert records[2]["status"] == "failed"
    assert records[5]["status"] == "passed"
