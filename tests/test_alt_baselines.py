"""Independent identities and domain boundaries for the ALT comparators."""

import math
from itertools import product

import numpy as np
import pytest

from lsa.alt.baselines import (
    DIRICHLET_EXPONENTS,
    absolute_discounting,
    add_constant,
    dirichlet_concentration_mixture,
    dirichlet_log_evidence,
    good_turing_hybrid,
    kl_bits,
    natural_oracle,
    ristad,
    validate_counts,
)


def test_dirichlet_evidence_sequence_enumeration_and_predictive_identity():
    # Counts do not carry a multinomial coefficient: summing sequences gives 1.
    for alpha in (2 ** -24, 0.5, 1.0, 16.0):
        total = 0.0
        for seq in product(range(3), repeat=3):
            counts = np.bincount(seq, minlength=3)
            total += math.exp(dirichlet_log_evidence(counts, alpha))
        assert total == pytest.approx(1, abs=2e-12)
        counts = np.array([3, 0, 1])
        ratios = [math.exp(dirichlet_log_evidence(counts + np.eye(3, dtype=int)[i], alpha)
                           - dirichlet_log_evidence(counts, alpha)) for i in range(3)]
        np.testing.assert_allclose(ratios, (counts + alpha) / (4 + 3 * alpha), rtol=2e-12)


def test_concentration_mixture_sequential_chain_rule():
    counts = np.zeros(5, dtype=int)
    log_probability = 0.0
    for symbol in [2, 2, 4, 2, 0, 0, 1]:
        result = dirichlet_concentration_mixture(counts, exponents=DIRICHLET_EXPONENTS)
        log_probability += math.log(result.probabilities[symbol])
        counts[symbol] += 1
    result = dirichlet_concentration_mixture(counts, exponents=DIRICHLET_EXPONENTS)
    assert log_probability == pytest.approx(result.diagnostics["mixture_log_evidence_nats"], abs=2e-12)
    np.testing.assert_allclose(result.probabilities.sum(), 1, atol=2e-13)
    assert len(result.diagnostics["posterior"]) == 29


def test_concentration_empty_sample_is_uniform_with_equal_prior():
    result = dirichlet_concentration_mixture([0] * 7, exponents=DIRICHLET_EXPONENTS)
    np.testing.assert_allclose(result.probabilities, np.full(7, 1 / 7))
    np.testing.assert_allclose(result.diagnostics["posterior"], np.full(29, 1 / 29))
    assert result.diagnostics["mixture_log_evidence_nats"] == pytest.approx(0)


def test_gt_empirical_switch_equality_and_zero_class():
    # For t=1, c_2=1, equality uses GT, not the empirical branch.
    np.testing.assert_allclose(good_turing_hybrid([1, 1, 2, 0, 0]), np.array([2, 2, 2, 1.5, 1.5]) / 9)
    np.testing.assert_allclose(good_turing_hybrid([0, 0, 0]), [1 / 3] * 3)
    np.testing.assert_allclose(good_turing_hybrid([3, 4]), [3 / 7, 4 / 7])


def test_ristad_boundaries_and_normalization():
    np.testing.assert_allclose(ristad([0, 0]), [0.5, 0.5])
    np.testing.assert_allclose(ristad([1, 4]), add_constant([1, 4], 1))
    for counts in product(range(4), repeat=3):
        q = ristad(counts)
        assert np.all(q > 0)
        assert q.sum() == pytest.approx(1, abs=2e-15)


def test_ad_preserves_distinct_undefined_and_infinite_cases():
    undefined = absolute_discounting([0, 3, 4])
    assert undefined.status == "undefined" and undefined.probabilities is None
    assert undefined.reason == "c1_and_c2_are_zero"
    zero_discount = absolute_discounting([0, 2, 4])
    np.testing.assert_allclose(zero_discount.probabilities, [0, 1 / 3, 2 / 3])
    assert math.isinf(kl_bits([0.2, 0.3, 0.5], zero_discount.probabilities))
    # Zero target mass at an unseen symbol does not itself cause infinite KL.
    assert math.isfinite(kl_bits([0, 0.3, 0.7], zero_discount.probabilities))
    assert absolute_discounting([1, 2]).reason == "no_unseen_symbol_for_reserved_mass"
    all_singletons = absolute_discounting([0, 1, 1])
    np.testing.assert_allclose(all_singletons.probabilities, [1, 0, 0])
    assert math.isinf(kl_bits([0.2, 0.3, 0.5], all_singletons.probabilities))


def test_ad_finite_case_allocates_the_reserved_mass():
    result = absolute_discounting([3, 2, 1, 0, 0])
    delta = 1 / 3
    expected = np.array([(3 - delta) / 6, (2 - delta) / 6,
                         (1 - delta) / 6, delta * 3 / 12, delta * 3 / 12])
    np.testing.assert_allclose(result.probabilities, expected)
    assert expected.sum() == pytest.approx(1)


def test_oracle_is_class_average_and_lower_bound():
    counts = np.array([0, 0, 1, 1, 3])
    p = np.array([0.03, 0.17, 0.25, 0.15, 0.4])
    q = natural_oracle(counts, p)
    np.testing.assert_allclose(q, [0.1, 0.1, 0.2, 0.2, 0.4])
    for alternative in (add_constant(counts, 1), add_constant(counts, 0.5),
                        good_turing_hybrid(counts), ristad(counts)):
        assert kl_bits(p, q) <= kl_bits(p, alternative) + 1e-14


@pytest.mark.parametrize("counts", [[-1, 0], [0.5, 1], [], [[1]], [float("nan")]])
def test_invalid_counts_fail(counts):
    with pytest.raises(ValueError):
        validate_counts(counts)


def test_kl_rejects_missing_normalization_and_does_not_clip_zeros():
    with pytest.raises(ValueError):
        kl_bits([0.5, 0.5], [0.3, 0.3])
    assert math.isinf(kl_bits([0.5, 0.5], [1, 0]))
    assert kl_bits([0, 1], [0, 1]) == 0
