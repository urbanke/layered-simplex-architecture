"""Finite, held-out checks of the sealed reader against corrected direct kernels.

Nominal gates remain absolute. A separately reported four-ULP arithmetic budget
is available only when it exceeds the nominal absolute target, after an
independent reference has been computed for that exact case. This is finite-case validation,
not a uniform error bound or a production admission certificate.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import platform
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, localcontext
from itertools import pairwise
from pathlib import Path

import mpmath
import numpy as np
import scipy

from .artifacts import Run, canonical_hash, read_json, sha256, verify_run, write_json

NOMINAL_TOLERANCE_NATS = 3e-9
LOW_COUNT_TOLERANCE_NATS = 1e-11
PILOT_DEPTHS = (2, 22, 53, 54, 80, 138)
PILOT_COUNTS = (0, 1, 2, 3, 255, 256, 257, 1000, 20000, 1000000, 1045889)


def accuracy_contract():
    """The predeclared contract; its arithmetic allowance is not storage error."""
    return {
        "id": "absolute-plus-four-ulp-v2",
        "supersedes": "absolute-plus-four-ulp-v1",
        "nominal_absolute_error_nats": 3e-9,
        "low_count_max_r": 3,
        "low_count_absolute_error_nats": 1e-11,
        "qualified_arithmetic_budget_ulps": 4,
        "qualification_requires_arithmetic_budget_above_nominal": True,
        "qualification_requires_independent_reference": True,
        "reference_dps": [45, 60],
        "reference_tail_digits": [65, 80],
        "reference_convergence_nats": 1e-25,
        "reference_arguments": "exact_binary64",
        "require_matching_nearest_binary64_references": True,
        "ulp_definition": "larger adjacent binary64 spacing at nearest(reference)",
        "reference_selection": "nominal direct check fails, reference ULP exceeds its nominal gate, or r>3 four-ULP reference budget exceeds general nominal gate",
        "scalar_matrix_agreement": "unchanged nominal absolute gate",
    }


def _resolved_config(config):
    result = dict(config)
    result.setdefault("accuracy_contract", accuracy_contract())
    if result["accuracy_contract"] != accuracy_contract():
        raise ValueError("unsupported or changed sealed-reader accuracy contract")
    return result


def validation_config():
    """Return the fixed pilot selection; resolution is saved before evaluation."""
    return {
        "accuracy_contract": accuracy_contract(),
        "depths": list(PILOT_DEPTHS),
        "counts": list(PILOT_COUNTS),
        "u_values": [
            -4.013,
            -0.2573,
            5.213,
            10.007,
            35.0,
            35.013,
            59.987,
            79.993,
            80.0,
        ],
        "shifted_u_offsets": [-4.013, 0.007, 6.013],
        "all_low_count_depths": True,
        "anchor_gap_cases": True,
        "join_cases": True,
    }


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def resolve_validation_cases(table, config):
    """Resolve a deterministic case grid from coverage, never from result values.

    Explicit ``cases`` may instead replay an already declared finite grid.
    Uncovered pilot strata are reported, not silently counted as successful.
    """
    cases, unavailable = {}, []
    levels = tuple(table.coverage_depths)
    support = table.supported_count

    def add(L, r, u, group):
        L, r = _integer(L, "depth", 1), _integer(r, "count")
        u = float(u)
        if (
            L not in levels
            or r > support
            or not math.isfinite(u)
            or u > table.maximum_u
        ):
            raise ValueError(f"validation case outside sealed coverage: {(L, r, u)}")
        key = L, r, u
        item = cases.setdefault(key, {"depth": L, "r": r, "u": u, "groups": []})
        if group not in item["groups"]:
            item["groups"].append(group)

    if "cases" in config:
        if not config["cases"]:
            raise ValueError("validation requires nonempty cases")
        for case in config["cases"]:
            add(case["depth"], case["r"], case["u"], "explicit-declared-grid")
    else:
        for L in config["depths"]:
            _integer(L, "depth", 1)
            if L not in levels:
                unavailable.append({"depth": L, "reason": "pilot level outside store"})
                continue
            counts = set()
            for r in config["counts"]:
                _integer(r, "count")
                if r <= support:
                    counts.add(r)
                else:
                    unavailable.append(
                        {"depth": L, "r": r, "reason": "count outside store"}
                    )
            anchors = tuple(int(r) for r in table.count_anchors(L))
            if config.get("anchor_gap_cases", True):
                gaps = [
                    (a, b) for a, b in pairwise(anchors) if b - a > 1 and a < support
                ]
                if gaps:
                    chosen = {
                        gaps[0],
                        gaps[len(gaps) // 2],
                        gaps[-1],
                        max(gaps, key=lambda ab: ab[1] - ab[0]),
                    }
                    for a, b in sorted(chosen):
                        for r in (a + 1, int(math.sqrt((a + 1) * (b + 1)) - 1), b - 1):
                            if a < r < b and r <= support:
                                counts.add(r)
            for r in sorted(counts):
                label = (
                    "declared-count" if r in config["counts"] else "held-out-anchor-gap"
                )
                for u in config["u_values"]:
                    if u <= table.maximum_u:
                        add(L, r, u, label)
                    else:
                        unavailable.append(
                            {"depth": L, "r": r, "u": u, "reason": "u outside store"}
                        )
                for offset in config["shifted_u_offsets"]:
                    add(L, r, -L * math.log(r + 1) + offset, "shifted-off-grid")
                if config.get("join_cases", True):
                    join = -float(table.plan["left_drop"]) - L * math.log(r + 1)
                    for step in (-0.37, 0.0, 0.37, 1.13):
                        add(L, r, join + step * table.grid_step, "analytic-left-join")
                    # Test physical column endpoints independently of the tail join.
                    # At a held-out count, include both neighboring anchor edges.
                    j = int(np.searchsorted(anchors, r))
                    near = {anchors[min(j, len(anchors) - 1)], anchors[max(0, j - 1)]}
                    for a in sorted(near):
                        edge = float(table.column_metadata(L, a)["u_min"])
                        for step in (-0.37, 0.0, 0.37):
                            add(L, r, edge + step * table.grid_step, "stored-left-edge")
        if config.get("all_low_count_depths", True):
            # Every covered level appears, with each low count, at one shifted
            # point and one stratified right-window point. No random draws.
            right_points = (35.013, 59.987, 79.993)
            for i, L in enumerate(levels):
                for r in range(min(3, support) + 1):
                    add(L, r, -L * math.log(r + 1) - 4.013, "all-level-low-count")
                    u = right_points[(i + r) % len(right_points)]
                    if u <= table.maximum_u:
                        add(L, r, u, "all-level-low-count-right-window")
    if not cases:
        raise ValueError("no validation cases fall inside the sealed store")
    ordered = sorted(cases.values(), key=lambda x: (x["depth"], x["r"], x["u"]))
    return [
        dict(c, case_id=f"sealed-{i:05d}") for i, c in enumerate(ordered)
    ], unavailable


def _float64_ulp(value):
    value = float(value)
    return max(
        value - math.nextafter(value, -math.inf),
        math.nextafter(value, math.inf) - value,
    )


def assess_measurement(r, scalar, matrix, reference, refined):
    """Screen direct comparisons and predeclare which rows need independent refs."""
    values = [float(x) for x in (scalar, matrix, reference, refined)]
    if not all(math.isfinite(x) for x in values):
        raise ArithmeticError("reader and reference values must all be finite")
    scalar, matrix, reference, refined = values
    tolerance = LOW_COUNT_TOLERANCE_NATS if r <= 3 else NOMINAL_TOLERANCE_NATS
    scalar_error, matrix_error = scalar - refined, matrix - refined
    refinement_delta, path_delta = reference - refined, scalar - matrix
    ulp = max(_float64_ulp(v) for v in (reference, refined))
    error_passed = max(abs(scalar_error), abs(matrix_error)) <= tolerance
    reference_converged = abs(refinement_delta) <= tolerance
    path_passed = abs(path_delta) <= tolerance
    precision_limited = ulp > tolerance
    passed = (
        error_passed and reference_converged and path_passed and not precision_limited
    )
    measured_failure = not (error_passed and reference_converged and path_passed)
    direct_status = "passed"
    if not passed:
        direct_status = "failed" if measured_failure else "precision_limited"
    arithmetic_regime = r > 3 and 4 * ulp > NOMINAL_TOLERANCE_NATS
    requires_reference = not passed or arithmetic_regime
    return {
        "status": "failed" if requires_reference else "passed",
        "direct_comparison_status": direct_status,
        "requires_independent_reference": requires_reference,
        "nominal_pass": not requires_reference,
        "precision_qualified_pass": False,
        "unresolved_reference": requires_reference,
        "scalar_accuracy_class": "unresolved" if requires_reference else "nominal",
        "matrix_accuracy_class": "unresolved" if requires_reference else "nominal",
        "scalar_log_phi_nats": scalar,
        "matrix_log_phi_nats": matrix,
        "direct_reference_log_phi_nats": reference,
        "refined_reference_log_phi_nats": refined,
        "scalar_error_nats": scalar_error,
        "matrix_error_nats": matrix_error,
        "scalar_matrix_delta_nats": path_delta,
        "scalar_matrix_bitwise_equal": bool(
            np.float64(scalar).tobytes() == np.float64(matrix).tobytes()
        ),
        "reference_refinement_delta_nats": refinement_delta,
        "reference_float64_ulp_nats": ulp,
        "direct_arithmetic_budget_nats": 4 * ulp,
        "direct_arithmetic_budget_exceeds_nominal": arithmetic_regime,
        "precision_floor_exceeds_gate": precision_limited,
        "tolerance_nats": tolerance,
        "within_nominal_tolerance": error_passed,
        "reference_refinement_passed": reference_converged,
        "scalar_matrix_passed": path_passed,
    }


def _case_identity(case):
    return {key: case[key] for key in ("depth", "r", "u")}


def _independent_reference(case):
    """Picklable worker: two independent Mellin references at the exact input."""
    from .kernel_validation import high_precision_log_phi

    exact_u = str(Decimal.from_float(float(case["u"])))
    result = {
        "case": _case_identity(case),
        "input_u_hex": float(case["u"]).hex(),
        "input_u_exact_decimal": exact_u,
        "method": "independent_mpmath_mellin",
        "settings": {"dps": [45, 60], "tail_digits": [65, 80]},
    }
    try:
        result["references"] = [
            high_precision_log_phi(
                case["r"], case["depth"], exact_u, dps=dps, tail_digits=tail
            )
            for dps, tail in ((45, 65), (60, 80))
        ]
        result["status"] = "complete"
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
        result.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
    return result


def assess_independent_reference(case, scalar, matrix, reference):
    """Recalculate exact-float/decimal errors; also used by immutable admission."""
    if (
        reference.get("status") != "complete"
        or reference.get("case") != _case_identity(case)
        or reference.get("input_u_hex") != float(case["u"]).hex()
        or reference.get("input_u_exact_decimal")
        != str(Decimal.from_float(float(case["u"])))
        or reference.get("method") != "independent_mpmath_mellin"
        or reference.get("settings") != {"dps": [45, 60], "tail_digits": [65, 80]}
    ):
        raise ValueError(
            "independent reference is missing or does not bind this exact case"
        )
    refs = reference["references"]
    if len(refs) != 2 or [v["dps"] for v in refs] != [45, 60]:
        raise ValueError("independent reference requires both 45 and 60 digits")
    if not all(math.isfinite(float(v)) for v in (scalar, matrix)):
        raise ValueError("candidate values must be finite")
    with localcontext() as context:
        context.prec = 110
        coarse, refined = [Decimal(v["log_phi_nats"]) for v in refs]
        if not coarse.is_finite() or not refined.is_finite():
            raise ValueError("independent decimal references must be finite")
        q_coarse, nearest = float(coarse), float(refined)
        if not math.isfinite(nearest) or not math.isfinite(q_coarse):
            raise ValueError("independent reference is outside finite binary64")
        ulp = _float64_ulp(nearest)
        if not math.isfinite(ulp) or ulp <= 0:
            raise ValueError("independent reference spacing is not finite and positive")
        spacing = Decimal.from_float(ulp)
        nearest_exact = Decimal.from_float(nearest)
        delta = coarse - refined
        stable = q_coarse.hex() == nearest.hex()
        rounding = nearest_exact - refined
        converged = (
            abs(delta) <= Decimal("1e-25") and stable and abs(rounding) <= spacing / 2
        )
        tolerance = Decimal("1e-11") if case["r"] <= 3 else Decimal("3e-9")
        eligible = case["r"] > 3 and 4 * spacing > Decimal("3e-9")
        result = {
            "reference_converged": converged,
            "reference_precision_delta_nats": float(delta),
            "reference_precision_delta_decimal_nats": str(delta),
            "nearest_binary64_stable": stable,
            "nearest_binary64_reference": nearest,
            "nearest_binary64_hex": nearest.hex(),
            "reference_ulp_nats": ulp,
            "rounding_error_nats": float(rounding),
            "rounding_error_decimal_nats": str(rounding),
            "rounding_error_ulps": float(rounding / spacing),
            "qualification_eligible": eligible,
            "arithmetic_budget_ulps": 4,
            "arithmetic_budget_nats": float(4 * spacing),
            "qualified_total_error_bound_nats": float(Decimal("4.5") * spacing),
        }
        for name, value in (("scalar", scalar), ("matrix", matrix)):
            exact = Decimal.from_float(float(value))
            error, arithmetic = exact - refined, exact - nearest_exact
            status = "unresolved"
            if converged:
                status = (
                    "nominal"
                    if abs(error) <= tolerance
                    else (
                        "precision_qualified"
                        if eligible and abs(arithmetic) <= 4 * spacing
                        else "failed"
                    )
                )
            result[name] = {
                "accuracy_class": status,
                "true_error_nats": float(error),
                "true_error_decimal_nats": str(error),
                "arithmetic_error_nats": float(arithmetic),
                "arithmetic_error_decimal_nats": str(arithmetic),
                "arithmetic_error_ulps": float(arithmetic / spacing),
            }
    return result


def finalize_measurement(case, measured, reference=None):
    """Combine independent path assessments while retaining all direct residuals."""
    result = dict(measured)
    if not measured["requires_independent_reference"]:
        return result
    if reference is None:
        return result  # Missing independent evidence always remains failed.
    result["independent_reference"] = reference
    try:
        independent = assess_independent_reference(
            case,
            measured["scalar_log_phi_nats"],
            measured["matrix_log_phi_nats"],
            reference,
        )
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        result["independent_reference_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return result
    result["independent_assessment"] = independent
    classes = [independent[name]["accuracy_class"] for name in ("scalar", "matrix")]
    accepted = (
        independent["reference_converged"]
        and all(v in ("nominal", "precision_qualified") for v in classes)
        and measured["scalar_matrix_passed"]
    )
    qualified = accepted and "precision_qualified" in classes
    result.update(
        status="precision_qualified"
        if qualified
        else "passed"
        if accepted
        else "failed",
        nominal_pass=accepted and not qualified,
        precision_qualified_pass=qualified,
        unresolved_reference=not independent["reference_converged"],
        scalar_accuracy_class=classes[0],
        matrix_accuracy_class=classes[1],
    )
    return result


def summarize_measurements(rows):
    """Summarize measured classes, without presenting qualified rows as nominal."""
    measured = [row for row in rows if "scalar_error_nats" in row]
    independent = [
        row["independent_assessment"] for row in rows if "independent_assessment" in row
    ]

    def error(row):
        if "independent_assessment" in row:
            return max(
                abs(row["independent_assessment"][path]["true_error_nats"])
                for path in ("scalar", "matrix")
            )
        return max(abs(row["scalar_error_nats"]), abs(row["matrix_error_nats"]))

    counts = {
        "cases": len(rows),
        "passed_cases": sum(
            row["status"] in ("passed", "precision_qualified") for row in rows
        ),
        "nominal_pass_cases": sum(row.get("nominal_pass") is True for row in rows),
        "precision_qualified_cases": sum(
            row.get("precision_qualified_pass") is True for row in rows
        ),
        "failed_cases": sum(row["status"] == "failed" for row in rows),
        "unresolved_reference_cases": sum(
            row.get("unresolved_reference") is True for row in rows
        ),
        "independent_reference_required_cases": sum(
            row.get("requires_independent_reference") is True for row in rows
        ),
        "independent_reference_attempted_cases": sum(
            "independent_reference" in row for row in rows
        ),
        "independent_reference_complete_cases": sum(
            row.get("independent_reference", {}).get("status") == "complete"
            for row in rows
        ),
        "independent_reference_converged_cases": sum(
            item["reference_converged"] for item in independent
        ),
        "direct_nominal_pass_cases": sum(
            row.get("direct_comparison_status") == "passed" for row in rows
        ),
        "direct_failed_cases": sum(
            row.get("direct_comparison_status") == "failed" for row in rows
        ),
        "direct_precision_limited_cases": sum(
            row.get("direct_comparison_status") == "precision_limited" for row in rows
        ),
    }
    return {
        **counts,
        "all_cases_meet_nominal_limits": counts["nominal_pass_cases"] == len(rows),
        "maximum_absolute_error_nats": max(
            (error(row) for row in measured), default=None
        ),
        "maximum_low_count_error_nats": max(
            (error(row) for row in measured if row["r"] <= 3), default=None
        ),
        "maximum_direct_comparison_error_nats": max(
            (
                max(abs(row["scalar_error_nats"]), abs(row["matrix_error_nats"]))
                for row in measured
            ),
            default=None,
        ),
        "maximum_independent_true_error_nats": max(
            (
                abs(item[path]["true_error_nats"])
                for item in independent
                for path in ("scalar", "matrix")
            ),
            default=None,
        ),
        "maximum_qualified_arithmetic_error_ulps": max(
            (
                abs(item[path]["arithmetic_error_ulps"])
                for item in independent
                for path in ("scalar", "matrix")
                if item[path]["accuracy_class"] == "precision_qualified"
            ),
            default=None,
        ),
        "maximum_independent_rounding_error_ulps": max(
            (abs(item["rounding_error_ulps"]) for item in independent), default=None
        ),
        "maximum_scalar_matrix_delta_nats": max(
            (abs(row["scalar_matrix_delta_nats"]) for row in measured), default=None
        ),
    }


def _implementation_identity():
    base = Path(__file__).parent
    sources = [
        Path(__file__),
        base / "sealed_tables.py",
        base / "sealed_store_build.py",
        base / "kernel_validation.py",
        base / "sealed_native.py",
        base / "sealed_interp.c",
        *sorted((base / "_vendor/pmwm").glob("*.py")),
    ]
    return {
        "source_sha256": {str(p): sha256(p) for p in sources if p.is_file()},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mpmath": mpmath.__version__,
        },
    }


def _store_files(path, manifest):
    # The reader verifies its sealed manifest before this helper is reached.
    return {
        name: sha256(path / name)
        for name in sorted({"plan.json", "manifest.json", *manifest["files"]})
    }


def _direct_reference(r, L, u, *, refined):
    from ._vendor.pmwm import _runtime
    from ._vendor.pmwm.mellin import exact_log_phi_column

    token = _runtime._settings.set({"PMM_BUILD_EXACT": "1"})
    try:
        return exact_log_phi_column(
            r,
            L,
            u,
            oversample=16.0 if refined else 8.0,
            series_tolerance=1e-14 if refined else 1e-13,
            contour_tail_nats=50.0 if refined else 40.0,
        )
    finally:
        _runtime._settings.reset(token)


def _execute_validation(table, outdir, config, *, workers=1, native=None):
    path, root = table.path, Path(outdir)
    try:
        cases, unavailable = resolve_validation_cases(table, config)
        initial_store = dict(table.files_identity)
        identity = _implementation_identity()
        reader_backend = {
            "interpolation_backend": "native" if native is not None else "python",
            "native_identity": native.identity if native is not None else None,
        }
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "config.json", config)
        write_json(root / "resolved-cases.json", cases)
        write_json(root / "unavailable-strata.json", unavailable)
        write_json(root / "implementation.json", identity)
        write_json(root / "reader-backend.json", reader_backend)
        write_json(
            root / "store-inputs.json", {"path": str(path), "sha256": initial_store}
        )
        levels = defaultdict(list)
        for case in cases:
            levels[case["depth"]].append(case)
        rows = []
        with (root / "direct-rows.jsonl").open("x") as stream:
            for L, group in levels.items():
                started = time.monotonic()
                rs, us = (
                    sorted({c["r"] for c in group}),
                    sorted({c["u"] for c in group}),
                )
                r_index, u_index = (
                    {r: i for i, r in enumerate(rs)},
                    {u: i for i, u in enumerate(us)},
                )
                matrix_error = None
                try:
                    matrix = np.asarray(table.log_phi_matrix(L, rs, us))
                    if matrix.shape != (len(rs), len(us)):
                        raise ValueError("reader returned incorrect matrix shape")
                except (ArithmeticError, RuntimeError, ValueError) as exc:
                    matrix_error = f"{type(exc).__name__}: {exc}"
                by_count = defaultdict(list)
                for case in group:
                    by_count[case["r"]].append(case)
                for r, count_cases in by_count.items():
                    query = np.array([c["u"] for c in count_cases])
                    failure = None
                    try:
                        direct = np.asarray(
                            _direct_reference(r, L, query, refined=False)
                        )
                        refined = np.asarray(
                            _direct_reference(r, L, query, refined=True)
                        )
                        if direct.shape != query.shape or refined.shape != query.shape:
                            raise ValueError(
                                "reference returned incorrect column shape"
                            )
                    except (ArithmeticError, RuntimeError, ValueError) as exc:
                        failure = f"{type(exc).__name__}: {exc}"
                    for i, case in enumerate(count_cases):
                        row = dict(case)
                        try:
                            if matrix_error or failure:
                                raise ArithmeticError(matrix_error or failure)
                            # A one-point scalar query exercises a genuinely
                            # different call shape from the shared matrix grid.
                            scalar = np.asarray(table.log_phi(L, r, [case["u"]]))
                            if scalar.shape != (1,):
                                raise ValueError(
                                    "reader returned incorrect scalar shape"
                                )
                            row.update(
                                assess_measurement(
                                    r,
                                    scalar[0],
                                    matrix[r_index[r], u_index[case["u"]]],
                                    direct[i],
                                    refined[i],
                                )
                            )
                        except (ArithmeticError, RuntimeError, ValueError) as exc:
                            row.update(
                                status="failed",
                                error={"type": type(exc).__name__, "message": str(exc)},
                            )
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        rows.append(row)
                stream.flush()
                print(
                    f"sealed reader L={L}: {len(group)} cases, {time.monotonic() - started:.2f}s",
                    flush=True,
                )
        selected = [row for row in rows if row.get("requires_independent_reference")]
        reference_cases = [
            case
            for case, row in zip(cases, rows, strict=True)
            if row.get("requires_independent_reference")
        ]
        write_json(root / "independent-reference-cases.json", reference_cases)
        print(
            f"sealed reader: {len(selected)} independent 45/60-digit cases, {workers} workers",
            flush=True,
        )
        references = {}
        executor = None
        try:
            if workers == 1:
                results = map(_independent_reference, reference_cases)
            else:
                executor = ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                )
                results = executor.map(
                    _independent_reference, reference_cases, chunksize=1
                )
            with (root / "independent-references.jsonl").open("x") as stream:
                for case, reference in zip(reference_cases, results, strict=True):
                    references[case["case_id"]] = reference
                    stream.write(json.dumps(reference, allow_nan=False) + "\n")
                    stream.flush()
                    print(
                        f"independent reference {case['case_id']}: {reference['status']}",
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        rows = [
            {**row, **finalize_measurement(case, row, references.get(case["case_id"]))}
            if "requires_independent_reference" in row
            else row
            for case, row in zip(cases, rows, strict=True)
        ]
        with (root / "rows.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        final_store = _store_files(path, table.manifest)
        unchanged_store = initial_store == final_store
        unchanged_source = identity == _implementation_identity()
        unchanged_backend = True
        if native is not None:
            try:
                unchanged_backend = native.identity == reader_backend["native_identity"]
            except ValueError:
                unchanged_backend = False
        metrics = summarize_measurements(rows)
        summary = {
            "status": "passed"
            if metrics["passed_cases"] == len(rows)
            and unchanged_store
            and unchanged_source
            and unchanged_backend
            else "failed",
            "scope": "declared finite reader/interpolation cases, with selected independent references; not a uniform kernel bound",
            "production_certified": False,
            "independent_high_precision_reference": bool(references),
            "config_sha256": canonical_hash(config),
            "resolved_cases_sha256": canonical_hash(cases),
            **metrics,
            "levels": sorted(levels),
            "unavailable_strata": unavailable,
            "independent_reference_workers": workers,
            "reader_backend": reader_backend,
            "reader_backend_unchanged": unchanged_backend,
            "store_unchanged": unchanged_store,
            "source_unchanged": unchanged_source,
            "accuracy_contract": accuracy_contract(),
            "gates_nats": {
                "general": NOMINAL_TOLERANCE_NATS,
                "counts_0_to_3": LOW_COUNT_TOLERANCE_NATS,
            },
            "reference": {
                "method": "corrected exact_log_phi_column; selected cases independently checked by mpmath Mellin integration",
                "relationship_to_builder": "direct screening uses the builder's numerical implementation at held-out points and refined settings; selected 45/60-digit references use a separate implementation",
                "oversample": [8, 16],
                "series_tolerance": [1e-13, 1e-14],
                "contour_tail_nats": [40, 50],
                "precision_policy": "nominal true-error gates are unchanged; r>3 cases whose four-ULP budget exceeds the general nominal gate may separately qualify with converged independent references and at most four ULP of computational error relative to the nearest reference plus half an ULP of reference rounding",
                "reported_error_basis": "independent decimal reference where computed; refined direct binary64 screening otherwise",
            },
            "required_followup": "independent full kernel suite; sealed versus direct profile/augmentation/mixture comparisons; outer refinement and raw normalization; existing full-domain depth and chain suites; source/runtime/store-bound admission",
        }
        write_json(root / "summary.json", summary)
        outputs = {p.name: sha256(p) for p in root.iterdir() if p.is_file()}
        write_json(root / "output-manifest.json", outputs)
        return summary
    finally:
        table.close()


def run_sealed_store_validation(
    store,
    outdir,
    *,
    config=None,
    repo=None,
    engine_config=None,
    config_path=None,
    engine_path=None,
    workers=1,
):
    """Save a normal immutable validation Run; a failed gate raises after saving.

    A store-only run binds the complete sealed file inventory. With an engine
    configuration, additionally bind the actual depth evaluator configuration,
    including settings, source and runtime, without evaluating any profiles.
    """
    from .sealed_tables import SealedKernelTables

    config = _resolved_config(validation_config() if config is None else config)
    _integer(workers, "workers", 1)
    if workers > 72:
        raise ValueError("workers must not exceed 72")
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
    path, root = Path(store).resolve(strict=True), Path(outdir).resolve()
    if root == path or path in root.parents:
        raise ValueError("validation output must be outside the immutable store")
    native = None
    if engine_config is not None:
        from .depth import StoreConfig

        if (
            engine_config.get("mode") != "store"
            or engine_config["store"].get("format") != "sealed"
        ):
            raise ValueError("engine configuration must select a sealed store")
        settings = StoreConfig(**engine_config["store"])
        if settings.interpolation_backend == "native":
            from .sealed_native import NativeInterpolator

            native = NativeInterpolator(
                settings.native_library_path, settings.native_library_sha256
            )
    table = SealedKernelTables(path, native_interpolator=native)
    try:
        # Resolve before opening the Run so malformed declarations cannot create
        # an apparently useful partial calibration record.
        resolve_validation_cases(table, config)
        if engine_config is not None:
            from .depth import DepthEvaluator

            if Path(engine_config["store"]["path"]).resolve() != path:
                raise ValueError("engine configuration points to a different store")
            if engine_config["store"].get("files_sha256") != dict(table.files_identity):
                raise ValueError(
                    "engine must bind the complete sealed-store hash mapping"
                )
            with DepthEvaluator(
                mode="store",
                store=settings,
                prediction_tolerance=engine_config.get("prediction_tolerance", 1e-3),
            ) as evaluator:
                engine_identity = evaluator.configuration
        else:
            engine_identity = {
                "provider": "sealed-kernel-reader",
                "store": {
                    "format": "sealed",
                    "files_sha256": dict(table.files_identity),
                    "plan_sha256": table.plan_sha256,
                },
                "implementation": _implementation_identity(),
            }
        inputs = [path / "plan.json", path / "manifest.json"]
        if native is not None:
            if engine_identity.get("sealed_native_identity") != native.identity:
                raise ValueError(
                    "executed native reader identity differs from the depth engine"
                )
            inputs += [native.path, native.manifest_path]
        inputs += [Path(p) for p in (config_path, engine_path) if p is not None]
        with Run(
            root,
            repo=repo,
            protocol={"suite": "sealed_store", "config": config, "workers": workers},
            experiment="calibration_sealed_store",
            purpose="validation",
            engine_identity=engine_identity,
            inputs=inputs,
        ) as run:
            capsule = run.path / "source-capsule"
            for name in run.record["source"]["files"]:
                if name.startswith(("src/", "scripts/")) or name in (
                    "pyproject.toml",
                    "requirements-alt.lock",
                ):
                    target = capsule / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(repo / name, target)
            write_json(run.path / "store-plan.json", read_json(path / "plan.json"))
            write_json(
                run.path / "store-manifest.json", read_json(path / "manifest.json")
            )
            if engine_config is not None:
                write_json(run.path / "engine-options.json", engine_config)
            result = _execute_validation(
                table, run.path / "data", config, workers=workers, native=native
            )
            result["engine_sha256"] = canonical_hash(engine_identity)
            result["store_files_sha256"] = dict(table.files_identity)
            write_json(run.path / "result.json", result)
            if result["status"] != "passed":
                raise ArithmeticError(
                    "sealed-store calibration has failed or unresolved declared checks"
                )
        verify_run(root)
        return result
    finally:
        table.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", help="existing complete sealed kernel store")
    parser.add_argument(
        "--engine-config", help="optional complete sealed depth-engine configuration"
    )
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument(
        "--out", required=True, help="new validation directory outside the store"
    )
    parser.add_argument(
        "--config",
        help="explicit finite-case JSON; default is the declared stratified pilot",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent-reference processes, 1..72 (default: 1)",
    )
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text()) if args.config else None
    engine = read_json(args.engine_config) if args.engine_config else None
    store = args.store or (engine["store"]["path"] if engine else None)
    if store is None:
        parser.error("--store or --engine-config is required")
    summary = run_sealed_store_validation(
        store,
        args.out,
        config=config,
        repo=args.repo,
        engine_config=engine,
        config_path=args.config,
        engine_path=args.engine_config,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
