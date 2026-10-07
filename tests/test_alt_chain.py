import numpy as np
import pytest

from lsa.alt.chain_validation import check_sequence
from lsa.alt.depth import DepthEvaluator


def test_batched_chain_matches_independent_profile_integrals_across_chunks():
    ids = np.array([0, 1, 0, 2, 3, 0, 1, 2])
    evaluator = DepthEvaluator()
    small = check_sequence(evaluator, 8, ids, [0, 1, 2], chunk_size=3)
    one = check_sequence(evaluator, 8, ids, [0, 1, 2], chunk_size=8)
    assert max(r["probability_error"] for r in small) < 1e-12
    assert max(r["chain_error_bits"] for r in small) < 1e-10
    np.testing.assert_allclose(
        [r["cumulative_bits"] for r in small],
        [r["cumulative_bits"] for r in one],
        atol=1e-11,
        rtol=0,
    )


def test_chain_rejects_fractional_or_out_of_alphabet_tokens():
    for ids in [np.array([0, 8]), np.array([0, 0.5])]:
        with pytest.raises(ValueError):
            check_sequence(DepthEvaluator(), 8, ids, [0, 1])
