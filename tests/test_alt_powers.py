"""Independent checks of E**w, not just a second call to its outer integral."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfcx, gammaln, logsumexp

from lsa.alt.powers import (
    PowerEvaluator,
    PowerIntegrationError,
    PowerSettings,
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
