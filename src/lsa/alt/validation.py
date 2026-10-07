"""Measured ALT numerical checks, with unverified appendix claims kept pending.

This driver records bounded evidence. It never turns smoke checks into a
production calibration or treats an unavailable numerical domain as a pass.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

from .artifacts import write_json
from .depth import UnsupportedDomain

PENDING_CHECKS = (
    (
        "mellin_rows_through_138",
        "Step-halving and independent Meijer-G comparisons on a declared r/t grid through L=138.",
    ),
    (
        "recursion_mellin_45_digit",
        "The reported 45-digit contour comparison and 2.2e-16 relative error require their precise grid and high-precision implementation.",
    ),
    (
        "prior_monte_carlo",
        "The reported agreement within 2.6 Monte Carlo standard errors requires a frozen sample count, seeds, and case list.",
    ),
    (
        "appendix_simplex_quadrature_cases",
        "The reported 1.1e-9 simplex-quadrature row needs its full original case list; the small independent L2 check here has explicitly narrower scope.",
    ),
    (
        "appendix_independent_reference_cases",
        "The reported 5e-13 independent-reference row needs its exact cases; the finite identities below are separately identified.",
    ),
    (
        "bible_chain_rule_thousands_of_steps",
        "The reported 1e-12 ratios and 3e-4 total bits need specified corpus prefixes and a new sequential comparison.",
    ),
    (
        "full_alt_domain_calibration",
        "Validate interpolation, saddle substitution, grid/tail refinement, powered kernels, and the actual ALT domain through depth138 before production.",
    ),
)


def _depth_logs(evaluator, d, partition, depths):
    return evaluator.evidence_at_depths(d, tuple(partition), tuple(depths))


def _direct_l2_simplex():
    # d=2 uniform simplex layers can be parameterized by u,v in [0,1].
    # This integral never uses a moment kernel or the evidence integral.
    def inner(u):
        def f(v):
            a = u * v
            return (a / (a + (1 - u) * (1 - v))) ** 2

        return quad(f, 0, 1, epsabs=1e-11, epsrel=1e-11, limit=100)[0]

    return quad(inner, 0, 1, epsabs=1e-10, epsrel=1e-10, limit=100)


def _support_enumeration(evaluator, probabilities, n, depth=1):
    """Enumerate labeled sequences and all KL decomposition terms in nats."""
    p = np.asarray(probabilities, dtype=float)
    d = len(p)
    if d**n > 100_000:
        raise UnsupportedDomain("finite identity check is bounded to 100,000 sequences")
    if np.any(p <= 0) or not math.isclose(float(p.sum()), 1, abs_tol=1e-14):
        raise ValueError("enumeration target must be strictly positive and normalized")
    sequences = list(itertools.product(range(d), repeat=n))
    profiles = {tuple(sorted(Counter(seq).values(), reverse=True)) for seq in sequences}
    evidence = evaluator.evaluate_profiles_at_depths(
        d, {profile: profile for profile in profiles}, (depth,)
    )
    rows, ps, qs = [], defaultdict(float), defaultdict(float)
    pk, qk = defaultdict(float), defaultdict(float)
    for seq in sequences:
        profile = tuple(sorted(Counter(seq).values(), reverse=True))
        log_p = float(sum(math.log(p[i]) for i in seq))
        log_q = float(evidence[profile].log_evidence[0])
        prob_p, prob_q = math.exp(log_p), math.exp(log_q)
        support = tuple(sorted(set(seq)))
        k = len(support)
        ps[support] += prob_p
        qs[support] += prob_q
        pk[k] += prob_p
        qk[k] += prob_q
        rows.append((support, prob_p, log_p, log_q))
    kl = sum(prob * (lp - lq) for _, prob, lp, lq in rows)
    kl_k = sum(value * math.log(value / qk[k]) for k, value in pk.items())
    naming = sum(value * math.log(math.comb(d, k)) for k, value in pk.items())
    support_entropy = -sum(
        value * math.log(value / pk[len(s)]) for s, value in ps.items()
    )
    conditional_kl = sum(
        prob * (lp - math.log(ps[s]) - lq + math.log(qs[s])) for s, prob, lp, lq in rows
    )
    return {
        "d": d,
        "n": n,
        "depth": depth,
        "target": p.tolist(),
        "sequences": len(sequences),
        "kl_sequence_nats": kl,
        "kl_cardinality_nats": kl_k,
        "expected_naming_nats": naming,
        "conditional_support_entropy_nats": support_entropy,
        "support_information_nats": naming - support_entropy,
        "conditional_sequence_kl_nats": conditional_kl,
        "decomposition_residual_nats": kl
        - kl_k
        - (naming - support_entropy)
        - conditional_kl,
        "p_normalization": sum(pk.values()),
        "q_normalization": sum(qk.values()),
        "naming_only_violation_margin_nats": naming - kl,
    }


def run_validation(config, outdir, *, evaluator):
    """Run declared checks and write immutable ``checks.json``/``summary.json``.

    Configuration keys: purpose (smoke/validation), exchangeability (list of
    d/depths/tolerance_nats), predictive_normalization (list of d/n/alpha/seed/
    depths/tolerance, or explicit counts), l1_cases (d/partition),
    independent_l2, kernel_recursion_l2, support_identity. Optional checks
    omitted or unavailable are recorded as pending, never silently passed.
    Default small cases are suitable for ``DepthEvaluator(mode='reference')``.
    """
    purpose = config.get("purpose", "smoke")
    if purpose not in ("smoke", "validation"):
        raise ValueError("validation driver purpose must be smoke or validation")
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("validation-config.json", "checks.json", "summary.json"):
        if (root / name).exists():
            raise FileExistsError(root / name)
    write_json(root / "validation-config.json", config)
    checks = []

    def record(
        name,
        residual,
        units,
        tolerance,
        *,
        details=None,
        criterion="absolute_residual_le",
    ):
        finite = math.isfinite(float(residual))
        passed = finite and abs(residual) <= tolerance
        checks.append(
            {
                "name": name,
                "status": "passed" if passed else "failed",
                "residual": float(residual) if finite else None,
                "units": units,
                "tolerance": tolerance,
                "criterion": criterion,
                "details": details or {},
            }
        )

    def pending(name, reason, units=None, tolerance=None):
        checks.append(
            {
                "name": name,
                "status": "pending",
                "residual": None,
                "units": units,
                "tolerance": tolerance,
                "reason": reason,
            }
        )

    def attempt(name, operation):
        try:
            operation()
        except UnsupportedDomain as exc:
            pending(
                name, f"Current evaluator does not support this requested domain: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 -- preserve every failed check in the record
            checks.append(
                {
                    "name": name,
                    "status": "failed",
                    "residual": None,
                    "units": None,
                    "tolerance": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )

    l1_cases = config.get("l1_cases", [{"d": 7, "partition": [3, 2, 1]}])
    if not l1_cases:
        pending("analytic_endpoints", "No L0/L1 cases configured.")
    for index, case in enumerate(l1_cases):

        def endpoint(case=case, index=index):
            d, parts = case["d"], tuple(case["partition"])
            n = sum(parts)
            got = _depth_logs(evaluator, d, parts, (0, 1))
            # Sequential add-one denominator avoids Gamma(d+N)-Gamma(d)
            # cancellation, providing a distinct closed-form calculation.
            expected1 = sum(math.lgamma(r + 1) for r in parts) - sum(
                math.log(d + i) for i in range(n)
            )
            expected0 = -n * math.log(d)
            tolerance = case.get("tolerance_nats", 2e-9)
            record(
                f"L0_uniform[{index}]",
                got.log_evidence[0] - expected0,
                "nats",
                tolerance,
                details={
                    "d": d,
                    "partition": parts,
                    "observed": float(got.log_evidence[0]),
                    "expected": expected0,
                },
            )
            record(
                f"L1_add_one[{index}]",
                got.log_evidence[1] - expected1,
                "nats",
                tolerance,
                details={
                    "d": d,
                    "partition": parts,
                    "observed": float(got.log_evidence[1]),
                    "expected": expected1,
                },
            )
            if expected1:
                record(
                    f"L1_relative_log_evidence[{index}]",
                    (got.log_evidence[1] - expected1) / expected1,
                    "relative error of log evidence (not probability)",
                    case.get("relative_tolerance", 2e-9),
                )

        attempt(f"analytic_endpoints[{index}]", endpoint)

    exchangeability = config.get(
        "exchangeability", [{"d": 3, "depths": [0, 1, 2], "tolerance_nats": 1e-7}]
    )
    if not exchangeability:
        pending("exchangeability", "No alphabet/depth cases configured.")
    for index, case in enumerate(exchangeability):

        def exchange(case=case, index=index):
            d, depths = case["d"], case["depths"]
            if d < 2:
                raise ValueError("exchangeability pair check needs d>=2")
            result = evaluator.evaluate_profiles_at_depths(
                d, {"one": (1,), "two": (2,), "distinct": (1, 1)}, depths
            )
            for j, L in enumerate(depths):
                r1 = math.log(d) + result["one"].log_evidence[j]
                r2 = logsumexp(
                    [
                        math.log(d) + result["two"].log_evidence[j],
                        math.log(d)
                        + math.log(d - 1)
                        + result["distinct"].log_evidence[j],
                    ]
                )
                record(
                    f"exchangeability[{index}].L{L}",
                    max(abs(r1), abs(r2)),
                    "nats",
                    case["tolerance_nats"],
                    details={
                        "d": d,
                        "depth": L,
                        "log_singleton_mass": float(r1),
                        "log_pair_mass": float(r2),
                        "evidence_diagnostics": {
                            key: value.diagnostics for key, value in result.items()
                        },
                    },
                )

        attempt(f"exchangeability[{index}]", exchange)

    normalization = config.get(
        "predictive_normalization",
        [
            {
                "d": 8,
                "n": 4,
                "alpha": 1.5,
                "seed": 11,
                "depths": [0, 1, 2],
                "tolerance": 1e-6,
            }
        ],
    )
    if not normalization:
        pending("predictive_normalization", "No sampled/explicit profile configured.")
    for index, case in enumerate(normalization):

        def normalize(case=case, index=index):
            d, depths = case["d"], case["depths"]
            target = None
            if "counts" in case:
                counts = np.asarray(case["counts"])
                if counts.shape != (d,):
                    raise ValueError("counts must have exactly d entries")
            else:
                target = np.arange(1, d + 1, dtype=float) ** (-case["alpha"])
                target /= target.sum()
                counts = np.random.default_rng(case["seed"]).multinomial(
                    case["n"], target
                )
            if np.any(counts < 0) or np.any(counts != np.floor(counts)):
                raise ValueError("counts must be nonnegative integers")
            parts = tuple(sorted((int(x) for x in counts if x), reverse=True))
            multiplicities = Counter(parts)
            if len(parts) < d:
                multiplicities[0] = d - len(parts)
            profiles = {"base": parts}
            for c in multiplicities:
                aug = list(parts)
                if c:
                    aug.remove(c)
                aug.append(c + 1)
                profiles[c] = tuple(aug)
            values = evaluator.evaluate_profiles_at_depths(d, profiles, depths)
            base = values["base"].log_evidence
            probabilities = {
                c: np.exp(values[c].log_evidence - base) for c in multiplicities
            }
            masses = sum(multiplicities[c] * probabilities[c] for c in multiplicities)
            mixed_mass = sum(
                multiplicities[c]
                * math.exp(logsumexp(values[c].log_evidence) - logsumexp(base))
                for c in multiplicities
            )
            details = {
                "d": d,
                "counts": counts.astype(int).tolist(),
                "depths": list(depths),
                "target_draw_policy": "fixed Zipf target"
                if target is not None
                else "explicit counts",
                "alpha": case.get("alpha"),
                "seed": case.get("seed"),
                "component_masses": masses.tolist(),
                "mixture_mass": mixed_mass,
                "normalization_applied": False,
                "evidence_diagnostics": {
                    str(key): value.diagnostics for key, value in values.items()
                },
            }
            record(
                f"predictive_normalization[{index}]",
                max(float(np.max(np.abs(masses - 1))), abs(mixed_mass - 1)),
                "absolute probability mass error",
                case["tolerance"],
                details=details,
            )

        attempt(f"predictive_normalization[{index}]", normalize)

    if config.get("independent_l2", True):

        def independent_l2():
            result = _depth_logs(evaluator, 2, (2,), (2,))
            reference, estimated_error = _direct_l2_simplex()
            observed = math.exp(result.log_evidence[0])
            record(
                "independent_L2_simplex",
                observed - reference,
                "absolute sequence probability error",
                2e-9,
                details={
                    "d": 2,
                    "partition": [2],
                    "depth": 2,
                    "observed": observed,
                    "reference": reference,
                    "reference_outer_quad_error_estimate": estimated_error,
                    "evidence_diagnostics": result.diagnostics,
                },
            )

        attempt("independent_L2_simplex", independent_l2)
    else:
        pending("independent_L2_simplex", "Disabled by this validation configuration.")

    if config.get("kernel_recursion_l2", True):

        def kernel_recursion():
            from ._vendor.pmwm.mellin import log_phi_contour

            cases = []
            for r, t in itertools.product((0, 3, 11), (0.5, 5.0, 200.0)):

                def f(y, r=r, t=t):
                    return (
                        math.lgamma(r + 1)
                        + (r + 1) * y
                        - math.exp(y)
                        - (r + 1) * float(np.logaddexp(0, math.log(t) + y))
                    )

                xp = 2 * (r + 1) / (1 + math.sqrt(1 + 4 * t * (r + 1)))
                peak = f(math.log(xp))
                value, error = quad(
                    lambda y, f=f, peak=peak: math.exp(f(y) - peak),
                    -60,
                    math.log(r + 1) + 6,
                    points=[math.log(xp)],
                    epsabs=1e-11,
                    epsrel=1e-11,
                    limit=150,
                )
                reference = peak + math.log(value)
                got = log_phi_contour(r, 2, math.log(t), dispatch=False)
                cases.append(
                    {
                        "r": r,
                        "t": t,
                        "residual_nats": got - reference,
                        "reference_quad_relative_error_estimate": error / value,
                    }
                )
            record(
                "kernel_recursion_L2",
                max(abs(case["residual_nats"]) for case in cases),
                "nats (log kernel)",
                2e-6,
                details={
                    "reference": "independent positive adaptive quadrature",
                    "candidate": "pinned PMWM contour with dispatch=False",
                    "cases": cases,
                },
            )

        attempt("kernel_recursion_L2", kernel_recursion)
    else:
        pending("kernel_recursion_L2", "Disabled by this validation configuration.")

    if config.get("support_identity", True):

        def support_identity():
            terms = _support_enumeration(evaluator, (0.6, 0.3, 0.1), 5)
            record(
                "discovered_set_KL_decomposition",
                terms["decomposition_residual_nats"],
                "nats",
                5e-13,
                details=terms,
            )
            record(
                "finite_sequence_normalization",
                terms["q_normalization"] - 1,
                "absolute probability mass error",
                5e-13,
                details={"d": 3, "n": 5, "depth": 1},
            )
            counterexample = _support_enumeration(evaluator, (1 / 3, 1 / 3, 1 / 3), 5)
            margin = counterexample["naming_only_violation_margin_nats"]
            checks.append(
                {
                    "name": "naming_only_lower_bound_counterexample",
                    "status": "passed" if margin > 1e-10 else "failed",
                    "residual": margin,
                    "units": "nats",
                    "tolerance": 1e-10,
                    "criterion": "naming_cost_minus_actual_KL_gt_tolerance",
                    "details": counterexample,
                }
            )

        attempt("discovered_set_KL_decomposition", support_identity)
    else:
        pending(
            "discovered_set_KL_decomposition",
            "Disabled by this validation configuration.",
        )

    for name, reason in PENDING_CHECKS:
        pending(name, reason)
    summary = {
        "schema_version": 1,
        "purpose": purpose,
        "status": "failed"
        if any(x["status"] == "failed" for x in checks)
        else "partial",
        "configuration_sha256": getattr(evaluator, "configuration_sha256", None),
        "evaluator_configuration": getattr(evaluator, "configuration", None),
        "required_checks_complete": False,
        "production_certified": False,
        "passed": sum(x["status"] == "passed" for x in checks),
        "failed": sum(x["status"] == "failed" for x in checks),
        "pending": sum(x["status"] == "pending" for x in checks),
        "pending_checks": [x["name"] for x in checks if x["status"] == "pending"],
        "note": "Passing the completed checks supports only their recorded cases and tolerances. Pending appendix/full-domain checks block production certification.",
    }
    write_json(root / "checks.json", checks)
    write_json(root / "summary.json", summary)
    return summary
