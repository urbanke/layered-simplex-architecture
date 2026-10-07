from types import SimpleNamespace

import numpy as np

from lsa.alt.depth_validation import comparison


def test_refinement_reports_loss_changes_and_raw_mass_in_separate_units():
    a = SimpleNamespace(
        component_log_evidence=np.array([-20.0, -22.0]),
        component_probabilities=np.array([[0.1, 0.3], [0.2, 0.4]]),
        mixture_probabilities=np.array([0.15, 0.35]),
        posterior=np.array([0.4, 0.6]),
    )
    b = SimpleNamespace(
        component_log_evidence=np.array([-20.001, -22.002]),
        component_probabilities=a.component_probabilities * 1.0001,
        mixture_probabilities=a.mixture_probabilities * 1.0001,
        posterior=np.array([0.401, 0.599]),
    )
    result = comparison(a, b, n=100, class_mass=[0.4, 0.6], multiplicities=[3, 1])
    assert abs(result["maximum_evidence_difference_bits"] - 0.002 / np.log(2)) < 1e-13
    assert result["maximum_predictive_kl_change_bits"] < 1e-14
    assert result["maximum_raw_mass_error"] > 0.1
