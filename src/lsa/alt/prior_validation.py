"""Independent small-alphabet prior and finite-identity validation for ALT.

Three distinct routes are compared: direct draws from the exponential-layer
prior; direct integration over uniform simplex layers (d=2); and the Gamma
evidence integral using positive L2 quadrature or Mellin kernels at L3.
No production store is used. Error quantities and Monte Carlo uncertainty have
separate units and acceptance criteria. Imports perform no numerical campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp, roots_legendre


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(_jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(values):
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or np.any(~np.isfinite(array)):
        raise ValueError("counts must be a finite nonempty vector")
    if np.any(array < 0) or np.any(array != np.floor(array)):
        raise ValueError("counts must be nonnegative integers")
    return array.astype(np.int64)


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def prior_monte_carlo(
    counts: Sequence[int],
    depth: int,
    *,
    samples: int,
    batch_size: int,
    seed_coordinates: Sequence[int],
) -> dict:
    """Estimate E[prod theta_i**m_i], retaining batch sufficient statistics.

    Each prior draw uses independent unit exponentials for every layer and
    label. Log products make normalization stable. Standard error uses the
    unbiased sample variance of individual draws, not variability of means
    with an incorrect batch-size denominator.
    """
    m = _counts(counts)
    depth = _positive_integer(depth, "depth")
    samples = _positive_integer(samples, "samples", 2)
    batch_size = _positive_integer(batch_size, "batch_size")
    coordinates = [_positive_integer(x, "seed coordinate", 0) for x in seed_coordinates]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(coordinates)))
    initial_state = rng.bit_generator.state
    records = []
    count, mean, m2 = 0, 0.0, 0.0
    digest = hashlib.sha256()
    while count < samples:
        size = min(batch_size, samples - count)
        exponential = rng.exponential(size=(size, depth, len(m)))
        log_y = np.log(exponential).sum(axis=1)
        log_theta = log_y - logsumexp(log_y, axis=1, keepdims=True)
        values = np.exp(log_theta @ m)
        digest.update(np.asarray(values, dtype="<f8").tobytes())
        batch_mean = float(np.mean(values))
        batch_m2 = float(np.sum((values - batch_mean) ** 2))
        delta = batch_mean - mean
        total = count + size
        m2 += batch_m2 + delta * delta * count * size / total
        mean += delta * size / total
        records.append(
            {
                "batch": len(records),
                "offset": count,
                "samples": size,
                "mean_probability": batch_mean,
                "sum_squared_deviations": batch_m2,
            }
        )
        count = total
    variance = m2 / (count - 1)
    return {
        "counts": m.tolist(),
        "d": len(m),
        "depth": depth,
        "samples": samples,
        "batch_size": batch_size,
        "seed_coordinates": coordinates,
        "rng": "NumPy PCG64 / SeedSequence",
        "rng_initial_state": initial_state,
        "rng_final_state": rng.bit_generator.state,
        "mean_probability": mean,
        "individual_sample_variance": variance,
        "standard_error_probability": math.sqrt(variance / count),
        "sample_moments_sha256": digest.hexdigest(),
        "batches": records,
    }


def _simplex_tensor_moments_d2(counts, depth, order, bound):
    """Direct d=2 simplex integral after u=logistic(x) in each layer.

    The density du/dx is logistic(x)logistic(-x). This representation integrates
    the original independent simplex coordinates, without a moment kernel,
    Mellin contour, evidence identity, or posterior normalization correction.
    Memory is O(order**2), including for three layers.
    """
    nodes, weights = roots_legendre(order)
    x = bound * nodes
    w = bound * weights * expit(x) * expit(-x)
    exponents = np.asarray(counts, dtype=int)
    values = np.zeros(len(exponents))

    def moments(sums, joint_weights):
        first, second = expit(sums), expit(-sums)
        return np.asarray(
            [float(np.sum(joint_weights * first**a * second**b)) for a, b in exponents]
        )

    if depth == 1:
        values = moments(x, w)
    elif depth == 2:
        values = moments(x[:, None] + x[None, :], w[:, None] * w[None, :])
    elif depth == 3:
        pair_sums = x[:, None] + x[None, :]
        pair_weights = w[:, None] * w[None, :]
        for shift, weight in zip(x, w, strict=True):
            values += weight * moments(shift + pair_sums, pair_weights)
    else:
        raise ValueError("direct simplex integration supports depths 1, 2, 3")
    return values, float(w.sum() ** depth)


def simplex_moments_d2(
    counts: Sequence[Sequence[int]],
    depth: int,
    *,
    node_orders: Sequence[int],
    logit_bound: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict:
    """Refine a direct positive simplex quadrature and bound truncated mass.

    The refinement difference is an empirical convergence diagnostic. The
    omitted prior mass has the separate rigorous union bound 2L/(1+exp(A));
    since each monomial is in [0,1], this also bounds its omitted integral.
    """
    clean = [_counts(m).tolist() for m in counts]
    if not clean or any(len(m) != 2 for m in clean):
        raise ValueError("simplex cases must be nonempty, with d=2")
    _positive_integer(depth, "depth")
    if depth > 3:
        raise ValueError("simplex depth exceeds bounded implementation")
    orders = [_positive_integer(n, "quadrature order", 2) for n in node_orders]
    if len(orders) < 2 or orders != sorted(set(orders)):
        raise ValueError(
            "quadrature orders must be distinct, increasing and include a refinement"
        )
    if not math.isfinite(logit_bound) or logit_bound <= 0:
        raise ValueError("logit bound must be finite and positive")
    if absolute_tolerance <= 0 or relative_tolerance < 0:
        raise ValueError("quadrature tolerances must be positive/nonnegative")
    tail_bound = float(2 * depth * expit(-logit_bound))
    history, previous = [], None
    converged = False
    for order in orders:
        values, mass = _simplex_tensor_moments_d2(clean, depth, order, logit_bound)
        differences = None if previous is None else np.abs(values - previous)
        history.append(
            {
                "nodes_per_layer": order,
                "sequence_probabilities": values.tolist(),
                "integrated_prior_mass": mass,
                "refinement_absolute_differences": None
                if differences is None
                else differences.tolist(),
            }
        )
        if differences is not None:
            limits = absolute_tolerance + relative_tolerance * np.abs(values)
            converged = bool(
                np.all(differences + tail_bound <= limits)
                and abs(mass - 1) <= absolute_tolerance + tail_bound
            )
            if converged:
                break
        previous = values
    return {
        "d": 2,
        "depth": depth,
        "counts": clean,
        "logit_bound": logit_bound,
        "tail_probability_bound": tail_bound,
        "history": history,
        "sequence_probabilities": values.tolist(),
        "converged": converged,
        "maximum_refinement_absolute_difference": float(np.max(differences)),
        "normalization_applied": False,
        "method": "tensor Gauss-Legendre over independent uniform simplex layers, logit change of variables",
    }


class EvidenceReference:
    """Independent positive L2 and contour L3 routes, with shared profile cache."""

    def __init__(self, reference_config, *, evaluator=None, contour_evaluator=None):
        if evaluator is None:
            from .depth import DepthEvaluator

            evaluator = DepthEvaluator(mode="reference")
        self.evaluator = evaluator
        self.contour_evaluator = contour_evaluator
        self.config = reference_config
        self.cache = {}
        self.records = []

    def log_evidence(self, counts, depth):
        m = _counts(counts)
        parts = tuple(sorted((int(x) for x in m if x), reverse=True))
        key = len(m), parts, depth
        if key in self.cache:
            return self.cache[key]
        if not parts:
            value = 0.0
            diagnostics = {"method": "empty sequence exact identity"}
        elif depth <= 2:
            result = self.evaluator.evidence_at_depths(len(m), parts, [depth])
            value = float(result.log_evidence[0])
            diagnostics = result.diagnostics
        elif depth == 3:
            if self.contour_evaluator is None:
                from .kernel_validation import contour_evidence_small

                self.contour_evaluator = contour_evidence_small
            refinements = []
            for step in self.config["contour_steps"]:
                result = self.contour_evaluator(len(m), parts, depth, step=step)
                # Helper returns a mapping, exposing the full numerical route.
                if isinstance(result, Mapping):
                    log_q = float(result["log_evidence"])
                    diagnostic = result.get("diagnostics", {})
                else:
                    log_q, diagnostic = result
                    log_q = float(log_q)
                refinements.append(
                    {
                        "step": step,
                        "log_evidence_nats": log_q,
                        "diagnostics": diagnostic,
                    }
                )
            if len(refinements) < 2:
                raise ValueError("L3 contour evidence needs at least two refinements")
            change = abs(
                refinements[-1]["log_evidence_nats"]
                - refinements[-2]["log_evidence_nats"]
            )
            if change > self.config["contour_refinement_tolerance_nats"]:
                raise ArithmeticError(
                    f"L3 contour refinement changed log evidence by {change} nats"
                )
            value = refinements[-1]["log_evidence_nats"]
            diagnostics = {
                "method": "independent Mellin/Gamma evidence",
                "refinements": refinements,
                "refinement_difference_nats": change,
            }
        else:
            raise ValueError("reference scope is depths zero through three")
        if not math.isfinite(value) or value > 1e-12:
            raise ArithmeticError("invalid reference log evidence")
        self.cache[key] = value
        self.records.append(
            {
                "d": len(m),
                "partition": parts,
                "depth": depth,
                "log_evidence_nats": value,
                "diagnostics": diagnostics,
            }
        )
        return value


def finite_identity_suite(
    target: Sequence[float],
    n: int,
    depths: Sequence[int],
    *,
    log_evidence: Callable[[Sequence[int], int], float],
    maximum_sequences: int,
) -> dict:
    """Enumerate every sequence and history; verify the two KL decompositions.

    Each component and the equal-prior mixture use raw evidence ratios. No
    probability vector or sequence measure is renormalized by this checker.
    """
    p = np.asarray(target, dtype=float)
    if (
        p.ndim != 1
        or len(p) < 2
        or np.any(p <= 0)
        or not np.isclose(p.sum(), 1, rtol=0, atol=1e-14)
    ):
        raise ValueError(
            "identity target must be a strictly positive probability vector"
        )
    n = _positive_integer(n, "n")
    d = len(p)
    if d ** (n + 1) > maximum_sequences:
        raise ValueError("enumeration exceeds declared sequence limit")
    depths = tuple(_positive_integer(L, "depth", 0) for L in depths)
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("depth grid must be nonempty and unique")
    logp = np.log(p)
    cached = {}

    def model_logs(counts):
        key = tuple(counts)
        if key not in cached:
            components = np.asarray([log_evidence(counts, L) for L in depths])
            cached[key] = np.r_[
                components, logsumexp(components) - math.log(len(depths))
            ]
        return cached[key]

    model_ids = [f"depth_{L}" for L in depths] + ["equal_prior_mixture"]
    sequence_masses = np.zeros((n + 2, len(model_ids)))
    max_predictive_error = np.zeros(len(model_ids))
    cumulative_chain = np.zeros(len(model_ids))
    joint_kl = np.zeros(len(model_ids))
    ps, qs = defaultdict(float), defaultdict(lambda: np.zeros(len(model_ids)))
    pk, qk = defaultdict(float), defaultdict(lambda: np.zeros(len(model_ids)))
    final_rows = []
    for t in range(n + 2):
        for sequence in itertools.product(range(d), repeat=t):
            counts = np.bincount(sequence, minlength=d)
            logs = model_logs(counts)
            sequence_masses[t] += np.exp(logs)
            lp = float(sum(logp[i] for i in sequence))
            probability = math.exp(lp)
            if t <= n:
                child_logs = np.stack(
                    [model_logs(counts + np.eye(d, dtype=int)[i]) for i in range(d)]
                )
                predictions = np.exp(child_logs - logs)
                max_predictive_error = np.maximum(
                    max_predictive_error, np.abs(predictions.sum(axis=0) - 1)
                )
                if t < n:
                    cumulative_chain += probability * np.sum(
                        p[:, None] * (logp[:, None] - (child_logs - logs)), axis=0
                    )
            if t == n:
                joint_kl += probability * (lp - logs)
                support = tuple(sorted(set(sequence)))
                k = len(support)
                ps[support] += probability
                qs[support] += np.exp(logs)
                pk[k] += probability
                qk[k] += np.exp(logs)
                final_rows.append((support, probability, lp, logs))
    kl_cardinality = sum(
        probability * (math.log(probability) - np.log(qk[k]))
        for k, probability in pk.items()
    )
    expected_naming = sum(
        probability * math.log(math.comb(d, k)) for k, probability in pk.items()
    )
    support_entropy = -sum(
        probability * math.log(probability / pk[len(support)])
        for support, probability in ps.items()
    )
    conditional_sequence_kl = sum(
        probability * (lp - math.log(ps[support]) - logs + np.log(qs[support]))
        for support, probability, lp, logs in final_rows
    )
    decomposition = (
        joint_kl
        - kl_cardinality
        - expected_naming
        + support_entropy
        - conditional_sequence_kl
    )
    models = {}
    for i, model in enumerate(model_ids):
        models[model] = {
            "sequence_masses_by_n": sequence_masses[:, i].tolist(),
            "max_sequence_normalization_error": float(
                np.max(np.abs(sequence_masses[:, i] - 1))
            ),
            "max_predictive_normalization_error": float(max_predictive_error[i]),
            "joint_kl_nats": float(joint_kl[i]),
            "sum_expected_predictive_kl_nats": float(cumulative_chain[i]),
            "kl_chain_residual_nats": float(joint_kl[i] - cumulative_chain[i]),
            "cardinality_kl_nats": float(kl_cardinality[i]),
            "expected_naming_nats": float(expected_naming),
            "conditional_support_entropy_nats": float(support_entropy),
            "conditional_sequence_kl_nats": float(conditional_sequence_kl[i]),
            "discovered_set_decomposition_residual_nats": float(decomposition[i]),
        }
    return {
        "d": d,
        "target": p.tolist(),
        "n": n,
        "depths": list(depths),
        "enumerated_through_n": n + 1,
        "sequences_at_last_level": d ** (n + 1),
        "normalization_applied": False,
        "models": models,
    }


def run_prior_validation(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    evaluator=None,
    contour_evaluator=None,
) -> dict:
    """Execute the declared suite, preserving case/module identities and failures."""
    if any(not config[group]["cases"] for group in ("monte_carlo", "identities")):
        raise ValueError("each declared validation family requires nonempty cases")
    if not config["simplex"]["depths"] or not config["simplex"]["counts"]:
        raise ValueError(
            "direct simplex validation requires explicit depths and moments"
        )
    if config["simplex"].get("alphabet", 2) != 2:
        raise ValueError("direct simplex implementation supports exactly d=2")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "prior-validation-config.json", config)
    code_root = Path(__file__).parent
    dependencies = [
        Path(__file__),
        code_root / "depth.py",
        code_root / "kernel_validation.py",
        *[
            code_root / "_vendor" / "pmwm" / name
            for name in (
                "mellin.py",
                "layered.py",
                "fast_tables.py",
                "_runtime.py",
                "kernel.py",
                "provenance.json",
            )
        ],
    ]
    fingerprints = {
        str(p.relative_to(code_root)): _hash_file(p)
        for p in dependencies
        if p.is_file()
    }
    _write_json(root / "module-fingerprints.json", fingerprints)
    for name in fingerprints:
        snapshot = root / "sources" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as stream:
            stream.write((code_root / name).read_bytes())
    _write_json(
        root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name) for name in ("numpy", "scipy")
            },
            "command": sys.argv,
        },
    )
    reference = EvidenceReference(
        config["reference"], evaluator=evaluator, contour_evaluator=contour_evaluator
    )
    checks = []
    started = time.perf_counter()

    def record(name, result, residual, units, tolerance):
        finite = math.isfinite(float(residual))
        checks.append(
            {
                "name": name,
                "status": "passed"
                if finite and abs(residual) <= tolerance
                else "failed",
                "residual": float(residual) if finite else None,
                "units": units,
                "tolerance": tolerance,
                "details": result,
            }
        )

    def attempt(name, operation):
        start = time.perf_counter()
        try:
            operation()
        except Exception as error:  # noqa: BLE001 -- preserve failed numerical checks
            checks.append(
                {
                    "name": name,
                    "status": "failed",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        checks[-1]["seconds"] = time.perf_counter() - start

    mc = config["monte_carlo"]
    for index, case in enumerate(mc["cases"]):

        def mc_check(index=index, case=case):
            result = prior_monte_carlo(
                case["counts"],
                case["depth"],
                samples=mc["samples_per_case"],
                batch_size=mc["batch_size"],
                seed_coordinates=[config["seed"], index],
            )
            # Preserve draws' sufficient statistics even if the independent
            # evidence route fails later in this case.
            _write_json(root / f"monte-carlo-{case['id']}.json", result)
            q = math.exp(reference.log_evidence(case["counts"], case["depth"]))
            result["reference_probability"] = q
            difference = result["mean_probability"] - q
            se = result["standard_error_probability"]
            if se <= 0:
                raise ArithmeticError(
                    "zero Monte Carlo standard error in a nonconstant case"
                )
            result["difference_probability"] = difference
            result["z_score"] = difference / se
            record(
                f"prior_monte_carlo.{case['id']}",
                result,
                difference / se,
                "Monte Carlo standard errors of sequence probability",
                mc["maximum_absolute_z_score"],
            )

        attempt(f"prior_monte_carlo.{case['id']}", mc_check)

    simplex = config["simplex"]
    for depth in simplex["depths"]:

        def simplex_check(depth=depth):
            result = simplex_moments_d2(
                simplex["counts"],
                depth,
                node_orders=simplex["node_orders"],
                logit_bound=simplex["logit_bound"],
                absolute_tolerance=simplex["refinement_absolute_tolerance"],
                relative_tolerance=simplex["refinement_relative_tolerance"],
            )
            _write_json(root / f"simplex-L{depth}.json", result)
            if not result["converged"]:
                raise ArithmeticError(
                    "direct simplex quadrature did not meet declared refinement/tail gate"
                )
            comparisons = []
            for m, q in zip(
                simplex["counts"], result["sequence_probabilities"], strict=True
            ):
                reference_q = math.exp(reference.log_evidence(m, depth))
                comparisons.append(
                    {
                        "counts": m,
                        "simplex_probability": q,
                        "reference_probability": reference_q,
                        "absolute_probability_difference": abs(q - reference_q),
                    }
                )
            record(
                f"direct_simplex.L{depth}",
                {**result, "comparisons": comparisons},
                max(row["absolute_probability_difference"] for row in comparisons),
                "absolute sequence probability error",
                simplex["comparison_absolute_tolerance"],
            )

        attempt(f"direct_simplex.L{depth}", simplex_check)

    identities = config["identities"]
    for case in identities["cases"]:

        def identities_check(case=case):
            result = finite_identity_suite(
                case["target"],
                case["n"],
                case["depths"],
                log_evidence=reference.log_evidence,
                maximum_sequences=identities["maximum_sequences"],
            )
            _write_json(root / f"identities-{case['id']}.json", result)
            for model, row in result["models"].items():
                for metric in (
                    "max_sequence_normalization_error",
                    "max_predictive_normalization_error",
                    "kl_chain_residual_nats",
                    "discovered_set_decomposition_residual_nats",
                ):
                    units = (
                        "nats"
                        if metric.endswith("nats")
                        else "absolute probability mass error"
                    )
                    tolerance = (
                        identities["log_identity_tolerance_nats"]
                        if units == "nats"
                        else identities["probability_tolerance"]
                    )
                    record(
                        f"identities.{case['id']}.{model}.{metric}",
                        row,
                        row[metric],
                        units,
                        tolerance,
                    )

        attempt(f"identities.{case['id']}", identities_check)

    _write_json(root / "reference-evidence.json", reference.records)
    changed = [
        name
        for name, digest in fingerprints.items()
        if _hash_file(code_root / name) != digest
    ]
    if changed:
        checks.append(
            {"name": "module_identity", "status": "failed", "changed_modules": changed}
        )
    _write_json(root / "checks.json", checks)
    failures = sum(check["status"] == "failed" for check in checks)
    summary = {
        "protocol_id": config["protocol_id"],
        "status": "failed" if failures else "passed",
        "declared_suite_complete": failures == 0,
        "production_certified": False,
        "passed": len(checks) - failures,
        "failed": failures,
        "seconds": time.perf_counter() - started,
        "scope": "Only the explicitly saved small-alphabet validation cases; no full-domain calibration",
        "config_sha256": _hash_file(root / "prior-validation-config.json"),
        "modules_sha256": _hash_file(root / "module-fingerprints.json"),
        "artifacts": {
            str(p.relative_to(root)): {
                "sha256": _hash_file(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
    }
    _write_json(root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_prior_validation(json.loads(args.config.read_text()), args.out)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
