"""Independent high-precision kernel references and declared calibration cases.

The mpmath contour implementation has its own saddle solve and quadrature.
It never calls the candidate's contour, series, or saddle approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import mpmath as mp
import numpy as np


def _mp_exact(value):
    """Lift binary64 inputs exactly; retain explicit decimal reference strings."""
    if isinstance(value, (float, np.floating)):
        numerator, denominator = float(value).as_integer_ratio()
        return mp.mpf(numerator) / denominator
    return mp.mpf(str(value))


def high_precision_log_phi(r, depth, u, *, dps=50, tail_digits=None):
    """Mellin integral on its own high-precision contour, with scaled panels.

    Returns decimal strings, avoiding a premature float rounding of the
    reference. Tail/precision convergence must be checked by the caller;
    the returned tail amplitude is a diagnostic, not a rigorous error bound.
    """
    if r < 0 or depth < 1 or dps < 25:
        raise ValueError("r>=0, depth>=1, dps>=25 required")
    with mp.workdps(dps):
        r, u = _mp_exact(r), _mp_exact(u)
        tiny = mp.power(10, -dps)
        lo, hi = tiny, r + 1 - tiny
        for _ in range(4 * dps + 20):
            z = (lo + hi) / 2
            derivative = mp.digamma(z) - depth * mp.digamma(r + 1 - z) - u
            if derivative > 0:
                hi = z
            else:
                lo = z
        c = (lo + hi) / 2
        f0 = mp.loggamma(c) - c * u + depth * mp.loggamma(r + 1 - c)
        f2 = mp.polygamma(1, c) + depth * mp.polygamma(1, r + 1 - c)
        sigma = 1 / mp.sqrt(f2)

        def scaled_exponent(y):
            z = c + mp.j * y
            return mp.loggamma(z) - z * u + depth * mp.loggamma(r + 1 - z) - f0

        cutoff = 16 * sigma
        target = -(tail_digits or dps + 12) * mp.log(10)
        for _ in range(40):
            if mp.re(scaled_exponent(cutoff)) < target:
                break
            cutoff *= mp.mpf("1.5")
        else:
            raise ArithmeticError("contour tail did not decay within bounded expansion")
        knots = [mp.mpf(0)]
        scale = mp.mpf("0.25")
        while scale * sigma < cutoff:
            knots.append(scale * sigma)
            scale *= 2
        knots.append(cutoff)
        integral = 2 * mp.quad(lambda y: mp.re(mp.exp(scaled_exponent(y))), knots)
        if integral <= 0:
            raise ArithmeticError("high-precision contour integral is not positive")
        value = f0 + mp.log(integral / (2 * mp.pi))
        return {
            "log_phi_nats": mp.nstr(value, dps),
            "dps": dps,
            "saddle": mp.nstr(c, dps),
            "sigma": mp.nstr(sigma, dps),
            "tail_endpoint": mp.nstr(cutoff, dps),
            "tail_scaled_log_amplitude": mp.nstr(mp.re(scaled_exponent(cutoff)), dps),
            "panels": len(knots) - 1,
        }


def meijer_log_phi(r, depth, u, *, dps=50):
    """Independent special-function representation (small depths only)."""
    if depth > 3:
        raise ValueError("generic Meijer-G is deliberately bounded to depth<=3")
    with mp.workdps(dps):
        r, u = _mp_exact(r), _mp_exact(u)
        value = mp.meijerg([[-r] * depth, []], [[0], []], mp.exp(u))
        if value <= 0 or not mp.isfinite(value):
            raise ArithmeticError("Meijer-G evaluation is not positive and finite")
        return mp.nstr(mp.log(value), dps)


def recursion_l2_log_phi(r, u, *, dps=45):
    """Positive, high-precision layer recursion, independent of Mellin inversion."""
    with mp.workdps(dps):
        r, u = _mp_exact(r), _mp_exact(u)
        t = mp.exp(u)
        xp = 2 * (r + 1) / (1 + mp.sqrt(1 + 4 * t * (r + 1)))
        yp = mp.log(xp)

        def log_integrand(y):
            return (
                mp.loggamma(r + 1)
                + (r + 1) * y
                - mp.exp(y)
                - (r + 1) * mp.log1p(t * mp.exp(y))
            )

        peak = log_integrand(yp)
        left = min(yp - (dps + 20) * mp.log(10), -u - (dps + 20) * mp.log(10))
        right = mp.log(r + 1) + mp.log((dps + 20) * mp.log(10))
        knots = [left, yp - 10, yp - 2, yp, yp + 2, right]
        knots = sorted({x for x in knots if left <= x <= right})
        value = mp.quad(lambda y: mp.exp(log_integrand(y) - peak), knots)
        return mp.nstr(peak + mp.log(value), dps)


@lru_cache(maxsize=64)
def _small_contour_column(r, depth, step):
    from ._vendor.pmwm import _runtime
    from ._vendor.pmwm.mellin import exact_log_phi_column

    grid = np.linspace(-80.0, 35.0, math.ceil(115.0 / step) + 1)
    token = _runtime._settings.set({"PMM_BUILD_EXACT": "1"})
    try:
        values = exact_log_phi_column(r, depth, grid)
    finally:
        _runtime._settings.reset(token)
    return grid, values


def contour_evidence_small(d, partition, depth, step=0.01):
    """Small-domain independent Mellin/Gamma route for prior-integration checks.

    No table store is used. Compare successive steps; this function reports
    diagnostics and refuses unresolved peaks. Natural-log evidence is returned.
    """
    from ._vendor.pmwm import _runtime
    from ._vendor.pmwm.layered import ProductMomentTables, log_q_lambda_scan

    parts = tuple(sorted((int(x) for x in partition), reverse=True))
    if not (
        1 <= d <= 8
        and 1 <= depth <= 3
        and 0 < sum(parts) <= 12
        and len(parts) <= d
        and all(x > 0 for x in parts)
        and 0 < step <= 0.05
    ):
        raise ValueError("small contour route requires d<=8, 1<=N<=12, depths1..3")
    rs = {0, 1, 2}
    for r in parts:
        rs.update((r, r + 1, r + 2))
    rs = tuple(sorted(rs))
    grid = _small_contour_column(0, depth, step)[0]
    matrix = np.stack([_small_contour_column(r, depth, step)[1] for r in rs])
    tables = ProductMomentTables.from_matrix(
        max_L=depth, L=depth, r_values=rs, u_grid=grid, matrix=matrix
    )
    token = _runtime._settings.set(
        {"PMM_SCAN": "full", "PMM_SCAN_LEGACY": "0", "PMM_BUILD_EXACT": "1"}
    )
    try:
        result = log_q_lambda_scan(d=d, L=depth, partition=parts, tables=tables)
    finally:
        _runtime._settings.reset(token)
    if not result.converged or "NARROW" in result.message:
        raise ArithmeticError(f"unresolved contour evidence: {result.message}")
    return {
        "log_evidence": result.log_q,
        "diagnostics": {
            "method": "direct-contour-columns-gamma-integral",
            "step": float(grid[1] - grid[0]),
            "u_min": -80.0,
            "u_max": 35.0,
            "right_gap": result.right_gap,
            "message": result.message,
            "peaks": result.peaks,
        },
    }


def _decimal_difference(a, b):
    with mp.workdps(90):
        return float(_mp_exact(a) - _mp_exact(b))


def expand_cases(config):
    """Resolve every configured r/u case into a reproducible finite grid."""
    cases = {}
    for group in config["groups"]:
        for L in group["depths"]:
            for r in group["counts"]:
                us = list(group.get("u_values", []))
                for offset in group.get("tilted_log_mean_offsets", []):
                    with mp.workdps(40):
                        us.append(float(-L * mp.digamma(r + 1) + offset))
                for offset in group.get("series_boundary_offsets", []):
                    us.append(float(-L * math.log(r + 1) + offset))
                for u in us:
                    key = (int(L), int(r), float(u))
                    if key not in cases:
                        cases[key] = {
                            "depth": int(L),
                            "r": int(r),
                            "u": float(u),
                            "groups": [],
                        }
                    cases[key]["groups"].append(group["id"])
    return [
        dict(case, case_id=f"kernel-{index:04d}")
        for index, case in enumerate(cases.values())
    ]


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _implementation_identity():
    import platform

    import scipy

    base = Path(__file__).parent
    sources = [
        Path(__file__),
        base / "sealed_tables.py",
        *sorted((base / "_vendor/pmwm").glob("*.py")),
        base / "_vendor/pmwm/provenance.json",
    ]
    return {
        "source_sha256": {str(p): _sha256(p) for p in sources},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mpmath": mp.__version__,
        },
    }


@contextmanager
def _read_only_store(path, *, ladder):
    from ._vendor.pmwm import universal_tables as ut

    path = Path(path).resolve(strict=True)
    if not (path / "manifest.json").is_file():
        raise ValueError(
            "store must already have a manifest; validation never initializes one"
        )

    class ReadOnly(ut.UniversalTables):
        def _save_manifest(self):
            raise RuntimeError("calibration cannot write a store")

        def _append_column(self, *args, **kwargs):
            raise RuntimeError("calibration cannot append a column")

        def _build_column(self, *args, **kwargs):
            raise RuntimeError("calibration cannot build a column")

    settings = {
        "_SADDLE_MIN_L": 0,
        "_LADDER_F": 0.0,
        "_LADDER_EVERY": 1 if ladder else 0,
        "_LADDER_DEGREE": 11,
        "_PHI_BIAS": 0.0,
        "_PHI_WAVE": 0.0,
        "SERIES_TAIL_NATS": math.inf,
    }
    old = {key: getattr(ut, key) for key in settings}
    table = ReadOnly(path, read_only=True)
    try:
        for key, value in settings.items():
            setattr(ut, key, value)
        yield table
    finally:
        table.close()
        for key, value in old.items():
            setattr(ut, key, value)


def _store_samples(spec, cases):
    """Read only selected levels and hash every referenced store file."""
    if spec.get("format") == "sealed":
        return _sealed_store_samples(spec, cases)
    root = Path(spec["path"])
    selected = [
        case
        for case in cases
        if case["depth"] in spec["depths"] and case["r"] in spec["counts"]
    ]
    levels = {case["depth"] for case in selected}
    files = [root / "manifest.json"]
    if spec.get("ladder", False):
        files.append(root / "anchors.json")
    for L in levels:
        for suffix in ("bin", "index.json"):
            file = root / f"level_{L:02d}.{suffix}"
            if file.is_file():
                files.append(file)
    before = {
        str(p): {
            "sha256": _sha256(p),
            "bytes": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in files
    }
    values = {}
    with _read_only_store(root, ladder=spec.get("ladder", False)) as table:
        for case in selected:
            L, r = case["depth"], case["r"]
            index_path = root / f"level_{L:02d}.index.json"
            if not index_path.exists():
                values[case["case_id"]] = {
                    "status": "unavailable",
                    "reason": "level absent",
                }
                continue
            if not spec.get("ladder", False) and str(r) not in json.loads(
                index_path.read_text()
            ):
                values[case["case_id"]] = {
                    "status": "unavailable",
                    "reason": "count column absent",
                }
                continue
            try:
                value = float(table.log_phi(L, r, [case["u"]])[0])
                values[case["case_id"]] = {"status": "evaluated", "log_phi_nats": value}
            except (RuntimeError, ValueError) as exc:
                values[case["case_id"]] = {"status": "failed", "reason": str(exc)}
    after = {
        str(p): {
            "sha256": _sha256(p),
            "bytes": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in files
    }
    if before != after:
        raise RuntimeError("store changed during read-only calibration")
    return {
        "id": spec["id"],
        "path": str(root.resolve()),
        "ladder": spec.get("ladder", False),
        "files": before,
        "unchanged_after_read": True,
    }, values


def _sealed_store_samples(spec, cases):
    """Sample the actual sealed reader, binding every declared store byte."""
    from .sealed_tables import SealedKernelTables

    root = Path(spec["path"]).resolve(strict=True)
    values = {}
    # The reader verifies all bytes itself, before we compare its identity with
    # the engine pin. Passing the pin as already-verified here would skip this.
    with SealedKernelTables(root) as table:
        identity = dict(table.files_identity)
        if identity != spec["files_sha256"]:
            raise ValueError("sealed kernel sampling differs from the engine store pin")
        before = {str(root / name): {"sha256": digest,
                  "bytes": (root / name).stat().st_size,
                  "mtime_ns": (root / name).stat().st_mtime_ns}
                  for name, digest in identity.items()}
        for case in cases:
            if case["depth"] not in spec["depths"] or case["r"] not in spec["counts"]:
                continue
            scalar = float(table.log_phi(case["depth"], case["r"], [case["u"]])[0])
            matrix = float(table.log_phi_matrix(
                case["depth"], [case["r"]], [case["u"]])[0, 0])
            values[case["case_id"]] = {
                "status": "evaluated", "log_phi_nats": scalar,
                "matrix_log_phi_nats": matrix,
            }
        after = {str(root / name): {"sha256": _sha256(root / name),
                 "bytes": (root / name).stat().st_size,
                 "mtime_ns": (root / name).stat().st_mtime_ns}
                 for name in identity}
        if before != after:
            raise RuntimeError("sealed store changed during kernel calibration")
    return {"id": spec["id"], "path": str(root), "format": "sealed",
            "files": before, "files_sha256": identity,
            "unchanged_after_read": True}, values


def run_kernel_validation(config, outdir):
    """Run a declared finite grid, recording residuals and precision limitations."""
    from ._vendor.pmwm import _runtime
    from ._vendor.pmwm.mellin import (
        exact_log_phi_column,
        log_phi_column,
        log_phi_contour,
    )

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=False)
    cases = expand_cases(config)
    identity = _implementation_identity()
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "resolved-cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (root / "implementation.json").write_text(json.dumps(identity, indent=2) + "\n")
    store_values, store_metadata = {}, []
    for spec in config.get("stores", []):
        metadata, values = _store_samples(spec, cases)
        store_metadata.append(metadata)
        store_values[spec["id"]] = values
    (root / "store-inputs.json").write_text(json.dumps(store_metadata, indent=2) + "\n")
    rows = []
    token = _runtime._settings.set({"PMM_BUILD_EXACT": "1"})
    try:
        # Exercise block-shared contour rows as used by the real provider,
        # in addition to pointwise calculations and contour-step refinement.
        grouped = {}
        for case in cases:
            grouped.setdefault((case["depth"], case["r"]), []).append(case)
        batched_values = {}
        for (L, r), group in grouped.items():
            group = sorted(group, key=lambda x: x["u"])
            values = exact_log_phi_column(r, L, [x["u"] for x in group], oversample=8.0)
            for case, value in zip(group, values, strict=True):
                batched_values[case["case_id"]] = float(value)
        with (root / "rows.jsonl").open("x") as stream:
            for index, case in enumerate(cases):
                started = time.monotonic()
                r, L, u = case["r"], case["depth"], case["u"]
                ref = high_precision_log_phi(r, L, u, dps=config["reference_dps"])
                refined = high_precision_log_phi(
                    r,
                    L,
                    u,
                    dps=config["refined_reference_dps"],
                    tail_digits=config["refined_reference_dps"] + 20,
                )
                exact = float(exact_log_phi_column(r, L, [u], oversample=8.0)[0])
                exact_fine = float(
                    exact_log_phi_column(
                        r, L, [u], oversample=16.0, contour_tail_nats=50.0
                    )[0]
                )
                scalar = log_phi_contour(r, L, u, dispatch=False, oversample=8.0)
                scalar_fine = log_phi_contour(
                    r, L, u, dispatch=False, oversample=16.0, tail_sigmas=20.0
                )
                legacy = float(log_phi_column(r, L, [u])[0])
                tolerance = (
                    config["low_count_tolerance_nats"]
                    if r <= 3
                    else config["nominal_tolerance_nats"]
                )
                vector_error = _decimal_difference(exact, refined["log_phi_nats"])
                precision_delta = _decimal_difference(
                    ref["log_phi_nats"], refined["log_phi_nats"]
                )
                floating_scale = (
                    32
                    * np.finfo(float).eps
                    * max(1.0, abs(float(refined["log_phi_nats"])))
                )
                stores = {}
                for name, values in store_values.items():
                    if case["case_id"] in values:
                        item = dict(values[case["case_id"]])
                        if item["status"] == "evaluated":
                            item["error_nats"] = _decimal_difference(
                                item["log_phi_nats"], refined["log_phi_nats"]
                            )
                            item["within_nominal_tolerance"] = (
                                abs(item["error_nats"]) <= tolerance
                            )
                            if "matrix_log_phi_nats" in item:
                                item["matrix_error_nats"] = _decimal_difference(
                                    item["matrix_log_phi_nats"], refined["log_phi_nats"]
                                )
                        stores[name] = item
                row = {
                    **case,
                    "reference": ref,
                    "refined_reference": refined,
                    "reference_precision_delta_nats": precision_delta,
                    "direct_column_log_phi_nats": exact,
                    "direct_column_error_nats": vector_error,
                    "batched_direct_column_error_nats": _decimal_difference(
                        batched_values[case["case_id"]], refined["log_phi_nats"]
                    ),
                    "direct_column_refined_error_nats": _decimal_difference(
                        exact_fine, refined["log_phi_nats"]
                    ),
                    "direct_column_refinement_delta_nats": exact_fine - exact,
                    "scalar_contour_error_nats": _decimal_difference(
                        scalar, refined["log_phi_nats"]
                    ),
                    "scalar_contour_refined_error_nats": _decimal_difference(
                        scalar_fine, refined["log_phi_nats"]
                    ),
                    "scalar_contour_refinement_delta_nats": scalar_fine - scalar,
                    "legacy_saddle_series_error_nats": _decimal_difference(
                        legacy, refined["log_phi_nats"]
                    ),
                    "tolerance_nats": tolerance,
                    "floating_scale_diagnostic_nats": floating_scale,
                    "within_nominal_tolerance": abs(vector_error) <= tolerance,
                    "reference_converged": abs(precision_delta)
                    <= config["reference_convergence_nats"],
                    "stores": stores,
                    "seconds": time.monotonic() - started,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if (index + 1) % 20 == 0:
                    print(
                        f"kernel cases {index + 1}/{len(cases)}; latest L={L} r={r} error={vector_error:.3g} nats",
                        flush=True,
                    )
    finally:
        _runtime._settings.reset(token)
    special = []
    for case in config.get("special_function_cases", []):
        r, L, u = case["r"], case["depth"], case["u"]
        ref = high_precision_log_phi(r, L, u, dps=45)["log_phi_nats"]
        meijer = meijer_log_phi(r, L, u, dps=45)
        row = {
            **case,
            "dps": 45,
            "contour_log_phi_nats": ref,
            "meijer_log_phi_nats": meijer,
            "meijer_error_nats": _decimal_difference(meijer, ref),
        }
        if L == 2:
            recursion = recursion_l2_log_phi(r, u, dps=45)
            difference = _decimal_difference(recursion, ref)
            row.update(
                recursion_log_phi_nats=recursion,
                recursion_error_nats=difference,
                recursion_relative_kernel_error=math.expm1(difference),
            )
        special.append(row)
    (root / "special-functions.json").write_text(json.dumps(special, indent=2) + "\n")
    final_identity = _implementation_identity()
    summary = {
        "cases": len(rows),
        "depths": sorted({x["depth"] for x in rows}),
        "reference_converged_cases": sum(x["reference_converged"] for x in rows),
        "direct_column_nominal_passes": sum(
            x["within_nominal_tolerance"] for x in rows
        ),
        "max_direct_column_error_nats": max(
            abs(x["direct_column_error_nats"]) for x in rows
        ),
        "max_direct_low_count_error_nats": max(
            abs(x["direct_column_error_nats"]) for x in rows if x["r"] <= 3
        ),
        "max_legacy_saddle_series_error_nats": max(
            abs(x["legacy_saddle_series_error_nats"]) for x in rows
        ),
        "max_reference_precision_delta_nats": max(
            abs(x["reference_precision_delta_nats"]) for x in rows
        ),
        "special_function_cases": len(special),
        "source_unchanged": identity == final_identity,
        "production_certified": False,
        "scope": "Only the explicitly resolved finite r/u/depth grid. Large-log floating-scale diagnostics do not relax the nominal gate or certify evidence/predictions. Generic Meijer-G checks are bounded to depths1..3; independent high-precision Mellin checks cover the full declared depth grid.",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def run_pilot(cases, outdir, *, dps=45):
    """Standalone records; no global working-tree snapshot while agents edit."""
    from ._vendor.pmwm.mellin import log_phi_column, log_phi_contour

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).parent / "_vendor/pmwm/mellin.py"]
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (root / "config.json").write_text(
        json.dumps({"cases": cases, "dps": dps, "source_sha256": provenance}, indent=2)
        + "\n"
    )
    rows = []
    with (root / "rows.jsonl").open("x") as stream:
        for case in cases:
            started = time.monotonic()
            r, L, u = case["r"], case["depth"], case["u"]
            reference = high_precision_log_phi(r, L, u, dps=dps)
            candidate = float(log_phi_column(r, L, [u])[0])
            direct = log_phi_contour(r, L, u, dispatch=False, oversample=16.0)
            row = {
                **case,
                "reference": reference,
                "candidate_log_phi_nats": candidate,
                "candidate_error_nats": _decimal_difference(
                    candidate, reference["log_phi_nats"]
                ),
                "float_contour_error_nats": _decimal_difference(
                    direct, reference["log_phi_nats"]
                ),
                "seconds": time.monotonic() - started,
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.config:
        run_kernel_validation(json.loads(Path(args.config).read_text()), args.outdir)
    else:
        cases = [
            {"r": 0, "depth": L, "u": u}
            for L, u in [(54, -4.0), (54, 10.0), (138, -4.0)]
        ]
        run_pilot(cases, args.outdir)


if __name__ == "__main__":
    main()
