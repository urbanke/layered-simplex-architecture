"""Finite reader validation keeps integrity, strict gates and precision limits."""

import json
import math
import shutil
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from lsa.alt.artifacts import canonical_hash, read_json, sha256, verify_run
from lsa.alt.sealed_store_validation import (
    accuracy_contract,
    assess_measurement,
    finalize_measurement,
    resolve_validation_cases,
    run_sealed_store_validation,
    summarize_measurements,
    validation_config,
)


def test_nominal_gates_are_not_relaxed_for_float64_resolution():
    good = assess_measurement(0, -0.01, -0.01, -0.01, -0.01)
    assert good["status"] == "passed"
    bad = assess_measurement(0, 2e-11, 2e-11, 0, 0)
    assert bad["status"] == "failed"
    assert bad["tolerance_nats"] == 1e-11
    large = assess_measurement(1000000, 1e9, 1e9, 1e9, 1e9)
    assert large["status"] == "failed"
    assert large["unresolved_reference"]
    assert large["direct_comparison_status"] == "precision_limited"
    assert large["within_nominal_tolerance"]
    assert large["tolerance_nats"] == 3e-9
    assert large["reference_float64_ulp_nats"] > large["tolerance_nats"]
    assert (
        assess_measurement(1000000, 1e9 + 1000, 1e9 + 1000, 1e9, 1e9)["status"]
        == "failed"
    )
    disagreement = assess_measurement(4, 0, 0, 1e-8, 0)
    assert disagreement["status"] == "failed"
    assert not disagreement["reference_refinement_passed"]
    with pytest.raises(ArithmeticError, match="finite"):
        assess_measurement(0, math.nan, 0, 0, 0)


def test_declared_grid_covers_dense_boundary_gaps_joins_and_all_low_levels():
    anchors = tuple(range(257)) + (300, 500, 1000, 20000, 500000, 1100000)
    metadata = SimpleNamespace(
        coverage_depths=(2, 3, 22, 53, 54, 80, 138),
        supported_count=1045889,
        maximum_u=80.0,
        grid_step=0.02,
        plan={"left_drop": 60.0},
        count_anchors=lambda L: anchors,
        column_metadata=lambda L, r: {
            "u_min": math.floor((-60 - L * math.log(r + 1)) / 0.02) * 0.02
        },
    )
    cases, unavailable = resolve_validation_cases(metadata, validation_config())
    assert not unavailable
    assert cases == resolve_validation_cases(metadata, validation_config())[0]
    for L in metadata.coverage_depths:
        assert {
            c["r"]
            for c in cases
            if c["depth"] == L and "all-level-low-count" in c["groups"]
        } == {0, 1, 2, 3}
    for r in (255, 256, 257, 1000000, 1045889):
        assert any(c["r"] == r and c["u"] == 79.993 for c in cases)
    assert any(
        c["r"] not in anchors
        and c["r"] > 500000
        and "held-out-anchor-gap" in c["groups"]
        for c in cases
    )
    assert {"stored-left-edge", "analytic-left-join"} <= {
        g for c in cases for g in c["groups"]
    }
    with pytest.raises(ValueError, match="coverage"):
        resolve_validation_cases(metadata, {"cases": [{"depth": 2, "r": 0, "u": 80.1}]})


@pytest.fixture(scope="module")
def tiny_store(tmp_path_factory):
    # A real offline store and real reader, with actual corrected kernel values.
    # The validation points below are not the builder's .02-spaced nodes.
    from lsa.alt.sealed_store_build import (
        build_store,
        create_plan,
        seal_store,
        write_plan,
    )

    path = tmp_path_factory.mktemp("sealed-validation") / "store"
    plan = create_plan(levels=[2], support_max_count=4, anchors={2: [0, 1, 2, 3, 4]})
    write_plan(plan, path)
    build_store(path)
    seal_store(path)
    return path


@pytest.fixture
def stable_repo(tmp_path, monkeypatch):
    # Isolate source identity from unrelated concurrent edits in this checkout.
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src/example.py").write_text("# validation source snapshot fixture\n")

    def identity(path):
        files = {"src/example.py": sha256(repo / "src/example.py")}
        return {
            "commit": "test",
            "branch": "test",
            "dirty": True,
            "files": files,
            "tree_sha256": canonical_hash(files),
        }

    monkeypatch.setattr("lsa.alt.artifacts.source_identity", identity)
    return repo


def test_real_reader_offgrid_validation_is_immutable_run(
    tiny_store, stable_repo, tmp_path
):
    before = {p.name: sha256(p) for p in tiny_store.iterdir() if p.is_file()}
    config = {
        "cases": [
            {"depth": 2, "r": 0, "u": -60.007},
            {"depth": 2, "r": 0, "u": -59.993},
            {"depth": 2, "r": 0, "u": 5.213},
            {"depth": 2, "r": 3, "u": 35.013},
            {"depth": 2, "r": 3, "u": 79.993},
            {"depth": 2, "r": 3, "u": 80.0},
        ]
    }
    out = tmp_path / "validation"
    summary = run_sealed_store_validation(
        tiny_store, out, config=config, repo=stable_repo
    )
    record = verify_run(out)
    assert record["experiment"] == "calibration_sealed_store"
    assert record["purpose"] == "validation"
    assert summary["passed_cases"] == summary["cases"] == 6
    assert summary["store_unchanged"] and summary["source_unchanged"]
    assert summary["maximum_low_count_error_nats"] < 1e-11
    assert summary["production_certified"] is False
    assert summary["independent_high_precision_reference"] is False
    rows = [
        json.loads(line) for line in (out / "data/rows.jsonl").read_text().splitlines()
    ]
    assert all(row["scalar_matrix_bitwise_equal"] for row in rows)
    assert (out / "source-capsule/src/example.py").read_bytes() == (
        stable_repo / "src/example.py"
    ).read_bytes()
    assert before == {p.name: sha256(p) for p in tiny_store.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        run_sealed_store_validation(tiny_store, out, config=config, repo=stable_repo)


def test_failed_comparison_remains_inspectable_and_cannot_verify_complete(
    tiny_store, stable_repo, tmp_path, monkeypatch
):
    from lsa.alt import sealed_store_validation as validation

    original = validation._direct_reference

    def biased_reference(*args, **kwargs):
        return original(*args, **kwargs) + 1e-4

    monkeypatch.setattr(validation, "_direct_reference", biased_reference)
    monkeypatch.setattr(
        validation,
        "_independent_reference",
        lambda case: {
            "status": "failed",
            "error": "deliberately unavailable test reference",
        },
    )
    out = tmp_path / "failed"
    with pytest.raises(ArithmeticError, match="failed or unresolved"):
        run_sealed_store_validation(
            tiny_store,
            out,
            repo=stable_repo,
            config={"cases": [{"depth": 2, "r": 0, "u": 5.213}]},
        )
    assert read_json(out / "result.json")["failed_cases"] == 1
    assert verify_run(out, require_complete=False)["status"] == "failed"
    with pytest.raises(ValueError, match="failed"):
        verify_run(out)


def test_missing_or_tampered_seal_rejected_before_run(
    tiny_store, stable_repo, tmp_path
):
    path = tmp_path / "tampered"
    shutil.copytree(tiny_store, path)
    data = path / "level_002.bin"
    with data.open("r+b") as stream:
        stream.write(np.float64(123).tobytes())
    out = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="hash mismatch"):
        run_sealed_store_validation(
            path,
            out,
            repo=stable_repo,
            config={"cases": [{"depth": 2, "r": 0, "u": 0}]},
        )
    assert not out.exists()
    (path / "manifest.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        run_sealed_store_validation(path, out, repo=stable_repo)


def test_engine_config_requires_complete_matching_hashes(
    tiny_store, stable_repo, tmp_path
):
    from lsa.alt.sealed_tables import SealedKernelTables

    with SealedKernelTables(tiny_store) as table:
        files = dict(table.files_identity)
    engine = {
        "mode": "store",
        "store": {
            "path": str(tiny_store),
            "format": "sealed",
            "files_sha256": files,
            "max_depth": 2,
            "max_d": 4,
            "max_n": 1,
            "max_count": 1,
        },
    }
    summary = run_sealed_store_validation(
        tiny_store,
        tmp_path / "engine",
        repo=stable_repo,
        engine_config=engine,
        config={"cases": [{"depth": 2, "r": 0, "u": 0.007}]},
    )
    manifest = verify_run(tmp_path / "engine")
    assert summary["engine_sha256"] == canonical_hash(manifest["engine"])
    assert manifest["engine"]["store"]["format"] == "sealed"
    del engine["store"]["files_sha256"]["plan.json"]
    with pytest.raises(ValueError, match="complete"):
        run_sealed_store_validation(
            tiny_store, tmp_path / "incomplete", repo=stable_repo, engine_config=engine
        )


def decimal_reference(case, value, *, coarse=None):
    """Synthetic reference for contract accounting tests, not numerical evidence."""
    return {
        "case": {key: case[key] for key in ("depth", "r", "u")},
        "input_u_hex": float(case["u"]).hex(),
        "input_u_exact_decimal": str(Decimal.from_float(float(case["u"]))),
        "method": "independent_mpmath_mellin",
        "settings": {"dps": [45, 60], "tail_digits": [65, 80]},
        "status": "complete",
        "references": [
            {"dps": 45, "log_phi_nats": str(value if coarse is None else coarse)},
            {"dps": 60, "log_phi_nats": str(value)},
        ],
    }


@pytest.mark.parametrize(
    "steps, expected",
    [(0, "precision_qualified"), (4, "precision_qualified"), (5, "failed")],
)
def test_independent_large_values_have_bounded_separate_arithmetic_budget(
    steps, expected
):
    case = {"depth": 138, "r": 1000000, "u": 80.0}
    reference = decimal_reference(case, "1000000000.00000003")
    candidate = 1e9 + steps * math.ulp(1e9)
    direct = assess_measurement(case["r"], candidate, candidate, 1e9, 1e9)
    row = finalize_measurement(case, direct, reference)
    assert row["status"] == expected
    assert row["independent_assessment"]["scalar"]["arithmetic_error_ulps"] == steps
    assert abs(row["independent_assessment"]["rounding_error_ulps"]) < 0.5
    assert not row["nominal_pass"]
    if expected == "precision_qualified":
        assert row["precision_qualified_pass"]
        assert abs(
            row["independent_assessment"]["scalar"]["true_error_nats"]
        ) <= 4.5 * math.ulp(1e9)
    summary = summarize_measurements([{**case, **row}])
    assert not summary["all_cases_meet_nominal_limits"]
    assert summary["nominal_pass_cases"] == 0
    assert summary["precision_qualified_cases"] == (expected == "precision_qualified")


@pytest.mark.parametrize(
    "r, reference, candidate",
    [
        (3, "1000000000.00000003", 1e9),
        (4, "2700000", 2.7e6 + 11 * math.ulp(2.7e6)),
    ],
)
def test_low_counts_and_sub_budget_magnitude_failures_never_qualify(
    r, reference, candidate
):
    case = {"depth": 138, "r": r, "u": 80.0}
    row = finalize_measurement(
        case,
        assess_measurement(r, candidate, candidate, 0, 0),
        decimal_reference(case, reference),
    )
    assert row["status"] == "failed"
    assert not row["independent_assessment"]["qualification_eligible"]
    assert not row["precision_qualified_pass"]


def test_independent_reference_can_resolve_direct_bias_but_cannot_hide_path_disagreement():
    case = {"depth": 2, "r": 0, "u": 0.007}
    row = finalize_measurement(
        case, assess_measurement(0, 1.0, 1.0, 1.01, 1.02), decimal_reference(case, "1")
    )
    assert row["status"] == "passed"
    assert row["direct_comparison_status"] == "failed"
    assert row["nominal_pass"]
    assert row["independent_assessment"]["scalar"]["true_error_nats"] == 0
    case = {"depth": 138, "r": 1000000, "u": 80.0}
    row = finalize_measurement(
        case,
        assess_measurement(case["r"], 1e9, math.nextafter(1e9, math.inf), 1e9, 1e9),
        decimal_reference(case, "1000000000.00000003"),
    )
    assert row["status"] == "failed"
    assert not row["scalar_matrix_passed"]
    assert (
        row["scalar_accuracy_class"]
        == row["matrix_accuracy_class"]
        == "precision_qualified"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "wrong_case",
        "wrong_input",
        "unconverged",
        "unstable_rounding",
        "nonfinite",
    ],
)
def test_qualified_pass_requires_exact_case_converged_independent_references(mutation):
    case = {"depth": 138, "r": 1000000, "u": 80.0}
    ref = decimal_reference(case, "1000000000.00000003")
    if mutation == "missing":
        ref = None
    elif mutation == "wrong_case":
        ref["case"]["r"] += 1
    elif mutation == "wrong_input":
        ref["input_u_hex"] = math.nextafter(case["u"], 0).hex()
    elif mutation == "unconverged":
        ref["references"][0]["log_phi_nats"] = "1000000000.000000030000000001"
    elif mutation == "nonfinite":
        ref["references"][0]["log_phi_nats"] = "NaN"
    else:
        with localcontext() as context:
            context.prec = 100
            halfway = Decimal.from_float(1e9) + Decimal.from_float(math.ulp(1e9)) / 2
            ref = decimal_reference(
                case, halfway + Decimal("1e-30"), coarse=halfway - Decimal("1e-30")
            )
    row = finalize_measurement(
        case, assess_measurement(case["r"], 1e9, 1e9, 1e9, 1e9), ref
    )
    assert row["status"] == "failed"
    assert row["unresolved_reference"]


def test_independent_reference_parallel_run_recovers_biased_direct_screen_in_fixed_order(
    tiny_store, stable_repo, tmp_path, monkeypatch
):
    from lsa.alt import sealed_store_validation as validation

    original = validation._direct_reference
    monkeypatch.setattr(
        validation,
        "_direct_reference",
        lambda *args, **kwargs: original(*args, **kwargs) + 1e-4,
    )
    out = tmp_path / "parallel-resolved"
    config = {
        "cases": [{"depth": 2, "r": 0, "u": 5.213}, {"depth": 2, "r": 3, "u": 0.007}]
    }
    summary = run_sealed_store_validation(
        tiny_store, out, repo=stable_repo, config=config, workers=2
    )
    verify_run(out)
    assert summary["passed_cases"] == summary["nominal_pass_cases"] == 2
    assert summary["independent_reference_converged_cases"] == 2
    assert summary["independent_reference_workers"] == 2
    assert summary["maximum_low_count_error_nats"] < 1e-11
    cases = read_json(out / "data/independent-reference-cases.json")
    refs = [
        json.loads(line)
        for line in (out / "data/independent-references.jsonl").read_text().splitlines()
    ]
    assert [ref["case"] for ref in refs] == [
        {key: case[key] for key in ("depth", "r", "u")} for case in cases
    ]
    assert all(
        ref["settings"] == {"dps": [45, 60], "tail_digits": [65, 80]} for ref in refs
    )


@pytest.mark.parametrize("workers", [0, 73, True, 1.5])
def test_workers_are_bounded_before_opening_run(tiny_store, tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        run_sealed_store_validation(tiny_store, tmp_path / "invalid", workers=workers)
    assert not (tmp_path / "invalid").exists()


def test_committed_accuracy_contract_is_exact_and_cannot_be_relaxed(
    tiny_store, tmp_path
):
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/alt2027/sealed-store-validation.json"
    )
    assert read_json(path)["accuracy_contract"] == accuracy_contract()
    config = {
        "cases": [{"depth": 2, "r": 0, "u": 0.007}],
        "accuracy_contract": accuracy_contract(),
    }
    config["accuracy_contract"]["qualified_arithmetic_budget_ulps"] = 5
    with pytest.raises(ValueError, match="accuracy contract"):
        run_sealed_store_validation(
            tiny_store, tmp_path / "invalid-contract", config=config
        )


@pytest.mark.parametrize(
    "base, steps, expected, eligible",
    [
        (math.nextafter(2.0**22, 0.0), 7, "failed", False),
        (2.0**22, 4, "precision_qualified", True),
        (2.0**22, 5, "failed", True),
        (1e7, 2, "precision_qualified", True),
        (1e7, 5, "failed", True),
    ],
)
def test_four_ulp_budget_eligibility_boundary_is_explicit(
    base, steps, expected, eligible
):
    case = {"depth": 54, "r": 1000000, "u": -0.2573}
    candidate = base + steps * math.ulp(base)
    measured = assess_measurement(case["r"], candidate, candidate, base, base)
    row = finalize_measurement(
        case, measured, decimal_reference(case, str(Decimal.from_float(base)))
    )
    assert row["status"] == expected
    assert row["independent_assessment"]["qualification_eligible"] == eligible
    assert row["independent_assessment"]["arithmetic_budget_nats"] == 4 * math.ulp(base)
    if eligible:
        assert math.ulp(base) < 3e-9 < 4 * math.ulp(base)


def test_arithmetic_budget_regime_is_independently_checked_despite_matching_doubles():
    measured = assess_measurement(1000000, 1e7, 1e7, 1e7, 1e7)
    assert measured["within_nominal_tolerance"]
    assert not measured["precision_floor_exceeds_gate"]
    assert measured["direct_arithmetic_budget_exceeds_nominal"]
    assert measured["requires_independent_reference"]
    assert measured["status"] == "failed"  # No reference, no pass.
    case = {"depth": 54, "r": 1000000, "u": -0.2573}
    row = finalize_measurement(case, measured, decimal_reference(case, "10000000"))
    assert row["status"] == "passed"
    assert row["nominal_pass"]
    assert not row["precision_qualified_pass"]


def test_v1_contract_cannot_be_silently_reinterpreted(tiny_store, tmp_path):
    contract = accuracy_contract()
    contract["id"] = "absolute-plus-four-ulp-v1"
    config = {
        "cases": [{"depth": 2, "r": 0, "u": 0.007}],
        "accuracy_contract": contract,
    }
    with pytest.raises(ValueError, match="accuracy contract"):
        run_sealed_store_validation(
            tiny_store, tmp_path / "old-contract", config=config
        )


def test_selected_native_engine_is_actually_executed_and_bound(
    tiny_store, stable_repo, tmp_path, monkeypatch
):
    from lsa.alt.sealed_native import NativeInterpolator, build_native
    from lsa.alt.sealed_tables import SealedKernelTables

    if shutil.which("cc") is None:
        pytest.skip("native engine validation requires an explicit C build")
    library = tmp_path / "sealed.so"
    metadata = build_native(library)
    calls = []
    original = NativeInterpolator.interpolate

    def measured_interpolation(self, *args, **kwargs):
        calls.append(self.identity["binary_sha256"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NativeInterpolator, "interpolate", measured_interpolation)
    with SealedKernelTables(tiny_store) as table:
        files = dict(table.files_identity)
    engine = {
        "mode": "store",
        "store": {
            "path": str(tiny_store),
            "format": "sealed",
            "files_sha256": files,
            "max_depth": 2,
            "max_d": 4,
            "max_n": 1,
            "max_count": 1,
            "interpolation_backend": "native",
            "native_library_path": str(library),
            "native_library_sha256": metadata["binary_sha256"],
        },
    }
    out = tmp_path / "native-run"
    summary = run_sealed_store_validation(
        tiny_store,
        out,
        repo=stable_repo,
        engine_config=engine,
        config={
            "cases": [
                {"depth": 2, "r": 0, "u": 0.007},
                {"depth": 2, "r": 3, "u": 5.213},
            ]
        },
    )
    record = verify_run(out)
    backend = read_json(out / "data/reader-backend.json")
    assert len(calls) >= 4 and set(calls) == {metadata["binary_sha256"]}
    assert summary["reader_backend"] == backend
    assert summary["reader_backend_unchanged"]
    assert backend["native_identity"] == record["engine"]["sealed_native_identity"]
    assert backend["native_identity"]["binary_sha256"] == metadata["binary_sha256"]
    assert summary["passed_cases"] == 2


@pytest.mark.parametrize(
    "depth,r,u,candidate,coarse,refined",
    [
        (
            2,
            1000000,
            -0.2573,
            13070541.758086044,
            "13070541.7580860478943335240743272566791614341",
            "13070541.7580860478943335240743272566791614340625852032644456",
        ),
        (
            138,
            990031,
            -4.013,
            16650792.474726189,
            "16650792.4747261921817225660447600783799141289",
            "16650792.474726192181722566044760078379914128944588864852301",
        ),
    ],
)
def test_recorded_nominal_misses_remain_explicitly_qualified_under_v2(
    depth, r, u, candidate, coarse, refined
):
    # References copied from count-remaining-reference-001, SHA256
    # b5cf8204a5957d0ab01319d64479a92e3599ffcad2349839fcaeaa3c37cdd265.
    case = {"depth": depth, "r": r, "u": u}
    row = finalize_measurement(
        case,
        assess_measurement(r, candidate, candidate, float(refined), float(refined)),
        decimal_reference(case, refined, coarse=coarse),
    )
    assert row["status"] == "precision_qualified"
    assert not row["nominal_pass"]
    assessment = row["independent_assessment"]
    assert abs(assessment["scalar"]["true_error_nats"]) > 3e-9
    assert assessment["scalar"]["arithmetic_error_ulps"] == -2
    assert (
        assessment["reference_ulp_nats"] < 3e-9 < assessment["arithmetic_budget_nats"]
    )
