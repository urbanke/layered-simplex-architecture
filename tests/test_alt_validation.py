import json

import pytest

from lsa.alt.depth import DepthEvaluator
from lsa.alt.validation import run_validation


def test_bounded_validation_records_actual_residuals_and_pending_claims(tmp_path):
    out = tmp_path / "validation"
    summary = run_validation({"purpose": "smoke"}, out, evaluator=DepthEvaluator())
    assert summary["failed"] == 0
    assert summary["passed"] >= 10
    assert summary["required_checks_complete"] is False
    assert summary["production_certified"] is False
    checks = {x["name"]: x for x in json.loads((out / "checks.json").read_text())}
    assert checks["discovered_set_KL_decomposition"]["status"] == "passed"
    assert checks["naming_only_lower_bound_counterexample"]["residual"] > 0
    assert (
        checks["independent_L2_simplex"]["units"]
        == "absolute sequence probability error"
    )
    assert checks["kernel_recursion_L2"]["units"] == "nats (log kernel)"
    assert checks["mellin_rows_through_138"]["status"] == "pending"
    assert (
        checks["predictive_normalization[0]"]["details"]["normalization_applied"]
        is False
    )
    with pytest.raises(FileExistsError):
        run_validation({}, out, evaluator=DepthEvaluator())


def test_unsupported_requested_domain_is_pending_not_a_pass(tmp_path):
    summary = run_validation(
        {
            "purpose": "validation",
            "exchangeability": [
                {"d": 1_000_000, "depths": [22], "tolerance_nats": 0.002}
            ],
            "predictive_normalization": [],
            "independent_l2": False,
            "kernel_recursion_l2": False,
            "support_identity": False,
        },
        tmp_path / "unsupported",
        evaluator=DepthEvaluator(),
    )
    assert summary["failed"] == 0
    assert "exchangeability[0]" in summary["pending_checks"]
    assert summary["required_checks_complete"] is False


def test_failed_numeric_check_stays_failed(tmp_path):
    class Broken:
        def evidence_at_depths(self, *args, **kwargs):
            raise ArithmeticError("deliberate integration failure")

        def evaluate_profiles_at_depths(self, *args, **kwargs):
            raise ArithmeticError("deliberate integration failure")

    result = run_validation(
        {
            "exchangeability": [],
            "predictive_normalization": [],
            "independent_l2": False,
            "kernel_recursion_l2": False,
            "support_identity": False,
        },
        tmp_path / "failed",
        evaluator=Broken(),
    )
    assert result["status"] == "failed"
    assert result["failed"] == 1
    assert result["production_certified"] is False
