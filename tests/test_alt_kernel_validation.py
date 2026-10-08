import json
from decimal import Decimal

import pytest

from lsa.alt.kernel_validation import (
    _decimal_difference,
    contour_evidence_small,
    expand_cases,
    high_precision_log_phi,
    meijer_log_phi,
    recursion_l2_log_phi,
    run_kernel_validation,
)


def test_independent_contour_meijer_and_45_digit_recursion():
    reference = high_precision_log_phi(3, 2, 2, dps=45)["log_phi_nats"]
    assert abs(_decimal_difference(meijer_log_phi(3, 2, 2, dps=45), reference)) < 1e-35
    assert (
        abs(_decimal_difference(recursion_l2_log_phi(3, 2, dps=45), reference)) < 1e-35
    )


def test_decimal_comparison_preserves_actual_binary64_value():
    # Shortest decimal printing would hide binary64 rounding here.
    assert _decimal_difference(0.1, "0.1") == float(Decimal.from_float(0.1) - Decimal("0.1"))
    assert _decimal_difference(0.1, str(Decimal.from_float(0.1))) == 0


def test_high_depth_zero_count_regression_and_real_contour_refinement():
    from lsa.alt._vendor.pmwm import _runtime
    from lsa.alt._vendor.pmwm.mellin import exact_log_phi_column, log_phi_column
    from lsa.alt._vendor.pmwm.universal_tables import _saddle_row

    reference = high_precision_log_phi(0, 138, -4, dps=40)["log_phi_nats"]
    token = _runtime._settings.set({"PMM_BUILD_EXACT": "1"})
    try:
        legacy = log_phi_column(0, 138, [-4])[0]
        corrected = _saddle_row(138, 0, [-4])[0]
        refined = exact_log_phi_column(0, 138, [-4], oversample=16)[0]
    finally:
        _runtime._settings.reset(token)
    assert abs(_decimal_difference(legacy, reference)) > 0.018
    assert abs(_decimal_difference(corrected, reference)) < 1e-11
    assert abs(_decimal_difference(refined, reference)) < 1e-11


def test_small_contour_evidence_refinement_and_known_endpoint():
    coarse = contour_evidence_small(2, (2, 1), 3, step=0.02)
    fine = contour_evidence_small(2, (2, 1), 3, step=0.01)
    assert coarse["log_evidence"] == pytest.approx(fine["log_evidence"], abs=1e-8)
    assert fine["diagnostics"]["right_gap"] > 20
    with pytest.raises(ValueError):
        contour_evidence_small(10, (2,), 3)


def test_finite_case_expansion_deduplicates_and_preserves_groups():
    config = {
        "groups": [
            {"id": "a", "depths": [54], "counts": [0], "u_values": [-4.0]},
            {
                "id": "b",
                "depths": [54],
                "counts": [0],
                "series_boundary_offsets": [-4.0],
            },
        ]
    }
    rows = expand_cases(config)
    assert len(rows) == 1
    assert rows[0]["groups"] == ["a", "b"]


def test_standalone_kernel_record_is_finite_and_does_not_certify_production(tmp_path):
    config = {
        "reference_dps": 30,
        "refined_reference_dps": 40,
        "reference_convergence_nats": 1e-20,
        "low_count_tolerance_nats": 1e-11,
        "nominal_tolerance_nats": 3e-9,
        "groups": [{"id": "small", "depths": [2], "counts": [0], "u_values": [0.0]}],
    }
    out = tmp_path / "kernel"
    summary = run_kernel_validation(config, out)
    assert summary["reference_converged_cases"] == 1
    assert summary["direct_column_nominal_passes"] == 1
    assert summary["source_unchanged"]
    assert summary["production_certified"] is False
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert abs(rows[0]["direct_column_error_nats"]) < 1e-11
    with pytest.raises(FileExistsError):
        run_kernel_validation(config, out)
