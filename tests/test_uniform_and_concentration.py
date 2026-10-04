"""Independent sequence identities for the new mixtures and AD edge cases."""
import itertools
import math

import numpy as np
import pytest
from scipy.special import gammaln

from lsa.codelength import DepthAveragedCodelength
from lsa.concentration_baselines import (
    absolute_discounting_codelengths,
    absolute_discounting_predictive,
    dirichlet_mixture_codelength_bits,
    dirichlet_mixture_predictive,
)
from lsa.estimators import lsa_predictive_by_count


def l1_result(counts):
    d, n = len(counts), sum(counts)
    lp = (gammaln(d) - gammaln(d + n) + sum(gammaln(np.array(counts) + 1))) / math.log(2)
    return DepthAveragedCodelength(d, n, 1, (lp,), lp)


def test_uniform_sequence_distribution_and_chain_rule():
    d, n = 3, 4
    mass = 0.
    for seq in itertools.product(range(d), repeat=n):
        counts = np.bincount(seq, minlength=d)
        old = l1_result(counts)
        new = old.with_uniform()
        expected = (d**(-n) + 2**old.log2_q_avg) / 2
        assert 2**new.log2_q_avg == pytest.approx(expected)
        assert new.with_uniform() == new
        assert new.depths == (0, 1)
        assert new.bits_per_token_at_depth(0) == pytest.approx(math.log2(d))
        assert sum(new.posterior) == pytest.approx(1.)
        assert new.posterior_mode in (0, 1)
        mass += expected
        augmented = [l1_result(counts + np.eye(d,dtype=int)[j]).with_uniform() for j in range(d)]
        assert sum(2**(a.log2_q_avg-new.log2_q_avg) for a in augmented) == pytest.approx(1.)
    assert mass == pytest.approx(1.)


def test_uniform_predictive_matches_evidence_ratio(tmp_path):
    counts = np.array([2,1,0,0])
    pred = lsa_predictive_by_count(counts, d=4, l_max=1, include_zero=True, cache_dir=tmp_path)
    base = l1_result(counts).with_uniform()
    expected = [2**(l1_result(counts+np.eye(4,dtype=int)[j]).with_uniform().log2_q_avg-base.log2_q_avg) for j in range(4)]
    np.testing.assert_allclose(pred.q_hat(counts),expected,rtol=1e-10)
    np.testing.assert_allclose(pred.q_hat(counts,depth=0),np.full(4,.25))
    np.testing.assert_allclose(pred.posterior,base.posterior)


def test_dir_tau_sequential_equals_batch():
    counts = np.zeros(7,dtype=int)
    bits = 0.
    for token in [0,1,0,2,0,1,3,0]:
        q = dirichlet_mixture_predictive(counts)
        assert sum(q) == pytest.approx(1.)
        bits -= math.log2(q[token])
        counts[token] += 1
        assert bits == pytest.approx(dirichlet_mixture_codelength_bits(counts,7),abs=1e-9)


def test_ad_zero_and_undefined_are_explicit():
    result=absolute_discounting_codelengths([0,0,0,1],7,[1,2,4])
    assert result[1]['bits']==pytest.approx(math.log2(7))
    assert result[2]['status']=='infinite'
    assert result[2]['first_zero_probability_token']==2
    assert result[4]['status']=='undefined'
    assert result[4]['first_undefined_prediction_token']==4
    assert result[4]['bits'] is None


def test_ad_finite_matches_direct_vectors():
    seq=[0,1,2,3,4,5]  # delta=1, but every realized token remains unseen
    counts=np.zeros(9,dtype=int);bits=0.
    for n,t in enumerate(seq):
        if n==0:q=np.full(9,1/9)
        else:
            c1=sum(counts==1);c2=sum(counts==2);s=sum(counts>0)
            delta=c1/(c1+2*c2)
            q=np.where(counts>0,(counts-delta)/n,delta*s/(n*(9-s)))
        assert sum(q)==pytest.approx(1.)
        bits-=math.log2(q[t]);counts[t]+=1
    assert absolute_discounting_codelengths(seq,9,[6])[6]['bits']==pytest.approx(bits)


def test_batch_ad_normalization_and_boundaries():
    counts = np.array([3, 2, 1, 1, 0, 0, 0])
    q = absolute_discounting_predictive(counts)
    assert sum(q) == pytest.approx(1.)
    assert q[-1] == pytest.approx(.5 * 4 / (7 * 3))
    assert absolute_discounting_predictive(np.array([1, 0, 0]))[0] == 0
    assert np.isnan(absolute_discounting_predictive(np.array([3, 0, 0]))).all()
    np.testing.assert_allclose(absolute_discounting_predictive(np.zeros(3)), 1/3)
