"""Positive-quadrature reference for the ALT single-layer power family.

The weights are ``Y_i = E_i**w`` with independent unit exponentials.  This
is not the unit-power model at depth ``w``. Evidence is a sequence probability
(no multinomial coefficient) and all logarithms/diagnostic errors are in nats.

With ``v = -log(t)/w``, define ``psi_r(v) = t**r phi_r(t)``. Its positive
integral in ``s = log(E)`` has log-integrand
``s + w*r*(s-v) - exp(s) - exp(w*(s-v))``. Integrating the product of these
scaled kernels over v avoids subtracting large ``N log(t)`` terms. Predictive
ratios share the same outer integral. Kernels and the outer integral are
refined independently; failed checks raise :class:`PowerIntegrationError`.

This is a reference implementation, with empirical error diagnostics rather
than interval-arithmetic certificates. See ``docs/alt-powers.md`` for the
checked domain and the distinction between validation and production timing.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, logsumexp, roots_legendre


class PowerIntegrationError(RuntimeError):
    """A powered evidence/prediction did not meet its numerical checks."""


@dataclass(frozen=True)
class PowerSettings:
    kernel_log_tolerance: float = 2e-12
    kernel_tail_drop: float = 42.0
    kernel_max_nodes: int = 512
    outer_relative_tolerance: float = 2e-8
    outer_tail_drop: float = 32.0
    normalization_tolerance: float = 2e-7
    max_alphabet: int = 10_000
    max_sample_size: int = 1_000


@dataclass(frozen=True)
class PowerPredictionResult:
    powers: tuple[int, ...]
    component_probabilities: np.ndarray
    mixture_probabilities: np.ndarray
    posterior: np.ndarray
    component_log_evidence: np.ndarray
    diagnostics: dict

    @property
    def log_evidence(self):
        return self.component_log_evidence


@dataclass(frozen=True)
class PowerProfileResult:
    powers: tuple[int, ...]
    count_values: tuple[int, ...]
    component_probabilities: np.ndarray
    mixture_probabilities: np.ndarray
    posterior: np.ndarray
    component_log_evidence: np.ndarray
    diagnostics: dict

    @property
    def log_evidence(self):
        return self.component_log_evidence


@lru_cache(maxsize=8)
def _legendre(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    return roots_legendre(nodes)


def _expm1_minus_x(x):
    """Stable curvature remainder near the kernel mode."""
    x = np.asarray(x)
    with np.errstate(over="ignore"):
        direct = np.expm1(x) - x
    series = x * x * (0.5 + x * (1 / 6 + x * (1 / 24 + x / 120)))
    return np.where(np.abs(x) < 1e-3, series, direct)


def log_scaled_kernel(
    r: int, w: int, v: float, *, settings: PowerSettings | None = None
) -> tuple[float, dict]:
    """Return ``log(t**r E[E**(wr) exp(-t E**w)])``, t=exp(-wv).

    ``w >= 2``; the complete evaluator handles the two analytic endpoints.
    Tail estimates use tangents to the strictly concave log-integrand.
    """
    cfg = settings or PowerSettings()
    if r < 0 or int(r) != r or not 2 <= w <= 80 or int(w) != w:
        raise ValueError("kernel requires integer r>=0 and integer 2<=w<=80")
    if not math.isfinite(v):
        raise ValueError("v must be finite")
    a = w * r + 1.0
    log_a = math.log(a)
    hi = min(log_a, v + (log_a - math.log(w)) / w)
    mode = brentq(
        lambda s: np.logaddexp(s, math.log(w) + w * (s - v)) - log_a,
        hi - math.log(2.0) - 1e-12, hi + 1e-12,
        xtol=1e-14, rtol=1e-14,
    )
    p = math.exp(mode)
    log_q = w * (mode - v)
    q = math.exp(log_q)

    def drop(delta):
        return -p * _expm1_minus_x(delta) - q * _expm1_minus_x(w * delta)

    scale = 1.0 / math.sqrt(p + w * w * q)
    bounds = []
    for direction in (-1, 1):
        far = direction * scale
        for _ in range(60):
            if float(drop(far)) < -cfg.kernel_tail_drop:
                break
            far *= 2
        else:
            raise PowerIntegrationError("could not bracket kernel tails")
        lo, hi = sorted((0.0, far))
        bounds.append(brentq(lambda z: float(drop(z)) + cfg.kernel_tail_drop,
                             lo, hi, xtol=1e-13))
    left, right = bounds
    # For r=0 and high w the exponential prior's mode and the sharp cutoff
    # near E=exp(v) can be far apart. Resolve both rather than sampling one
    # long interval whose apparently stable nodes miss the small cutoff tail.
    splits = [left, 0.0, right]
    if r == 0:
        splits += [v - mode + offset / w for offset in (-4, 0, 4)
                   if left < v - mode + offset / w < right]
    splits = np.array(sorted(set(splits)))
    mids = (splits[1:] + splits[:-1]) / 2
    halves = np.diff(splits) / 2
    previous = None
    nodes = 32
    delta_log = math.inf
    while nodes <= cfg.kernel_max_nodes:
        x, weights = _legendre(nodes)
        z = mids[:, None] + halves[:, None] * x
        value = float(np.sum((np.exp(drop(z)) @ weights) * halves))
        if not (value > 0 and math.isfinite(value)):
            raise PowerIntegrationError("nonpositive/nonfinite kernel quadrature")
        log_integral = math.log(value)
        if previous is not None:
            delta_log = abs(log_integral - previous)
            if delta_log <= cfg.kernel_log_tolerance:
                break
        previous = log_integral
        nodes *= 2
    else:
        raise PowerIntegrationError(
            f"kernel refinement failed (r={r}, w={w}, v={v}, "
            f"last log difference={delta_log})"
        )

    def slope(z):
        return -p * math.expm1(z) - w * q * math.expm1(w * z)

    tail_relative = math.exp(-cfg.kernel_tail_drop) * (
        1 / slope(left) - 1 / slope(right)
    ) / value
    if tail_relative > cfg.kernel_log_tolerance:
        raise PowerIntegrationError("kernel tail estimate exceeds tolerance")
    log_value = mode + r * log_q - p - q + log_integral
    return log_value, {
        "nodes": nodes, "refinement_log_difference": delta_log,
        "tail_relative_estimate": tail_relative,
    }


class PowerEvaluator:
    """Checked reference evaluator for the declared integer power grid.

    No posterior component is skipped, and predictions are returned *before*
    any optional experiment-level normalization. Empty samples and d=1 are
    analytic. The configured d/N limits prevent accidental use outside the
    implementation's intended benchmark scope.
    """

    def __init__(self, settings: PowerSettings | None = None):
        self.settings = settings or PowerSettings()

    def _component(self, d: int, partition: tuple[int, ...], w: int,
                   values: tuple[int, ...]) -> tuple[float, np.ndarray, dict]:
        n = sum(partition)
        if n == 0 or d == 1 or w == 0:
            return -n * math.log(d), np.full(len(values), 1 / d), {"analytic": True}
        if w == 1:
            log_evidence = float(gammaln(d) - gammaln(d + n)
                                 + sum(gammaln(r + 1) for r in partition))
            return log_evidence, (np.asarray(values) + 1.0) / (n + d), {"analytic": True}

        cfg = self.settings
        freq = Counter(partition)
        if len(partition) < d:
            freq[0] = d - len(partition)
        base_r = np.array(sorted(freq), dtype=int)
        orders = tuple(sorted(set(values) | {r + 1 for r in values}))
        stats = {"kernel_calls": 0, "kernel_max_nodes": 0,
                 "kernel_max_refinement_log_difference": 0.0,
                 "kernel_max_tail_relative_estimate": 0.0}

        @lru_cache(maxsize=4096)
        def rows(v):
            row = {}
            for r in orders:
                row[r], diag = log_scaled_kernel(r, w, v, settings=cfg)
                stats["kernel_calls"] += 1
                stats["kernel_max_nodes"] = max(stats["kernel_max_nodes"], diag["nodes"])
                stats["kernel_max_refinement_log_difference"] = max(
                    stats["kernel_max_refinement_log_difference"],
                    diag["refinement_log_difference"])
                stats["kernel_max_tail_relative_estimate"] = max(
                    stats["kernel_max_tail_relative_estimate"], diag["tail_relative_estimate"])
            return row

        def log_integrands(v):
            row = rows(float(v))
            base = float(sum(freq[r] * row[r] for r in base_r))
            return np.array([base] + [base + row[r + 1] - row[r] - math.log(n)
                                      for r in values])

        # Positive moment scale is only an initial bracket, never a tail cutoff.
        center = float((math.log(d / n) + gammaln(w + 1)) / w)
        lower, upper = center - 4, center + 4
        for _ in range(8):
            mode = minimize_scalar(lambda v: -log_integrands(v)[0],
                                   bounds=(lower, upper), method="bounded",
                                   options={"xatol": 2e-8})
            if not mode.success:
                raise PowerIntegrationError("outer mode search failed")
            if lower + 0.1 < mode.x < upper - 0.1:
                break
            lower -= 4
            upper += 4
        else:
            raise PowerIntegrationError("outer mode remained at search boundary")
        peak = float(mode.x)
        shifts = log_integrands(peak)
        bounds = []
        for direction in (-1, 1):
            edge = peak
            for _ in range(80):
                edge += direction * 0.5
                if np.max(log_integrands(edge) - shifts) < -cfg.outer_tail_drop:
                    break
            else:
                raise PowerIntegrationError("outer tail window did not close")
            bounds.append(edge)
        left, right = bounds

        def integrand(v):
            return np.exp(log_integrands(v) - shifts)

        def integrate(a, b, tolerance):
            result, err, info = quad_vec(integrand, a, b, epsabs=tolerance,
                                        epsrel=tolerance, norm="max", limit=400,
                                        points=[peak], full_output=True)
            if not info.success or not np.all(np.isfinite(result)) or np.any(result <= 0):
                raise PowerIntegrationError(f"outer quadrature failed: {info.message}")
            return result, float(err), int(info.neval)

        coarse, _, _ = integrate(left, right, cfg.outer_relative_tolerance * 4)
        # A wider interval checks tail truncation independently of adaptive mesh.
        fine, err, neval = integrate(left - 0.5, right + 0.5,
                                     cfg.outer_relative_tolerance / 4)
        log_difference = float(np.max(np.abs(np.log(fine / coarse))))
        relative_error_estimate = err / float(np.min(fine))
        if max(log_difference, relative_error_estimate) > cfg.outer_relative_tolerance:
            raise PowerIntegrationError(
                f"outer refinement failed w={w}: change={log_difference}, "
                f"quadrature estimate={relative_error_estimate}"
            )
        log_integrals = np.log(fine) + shifts
        log_evidence = float(log_integrals[0] + math.log(w) - gammaln(n))
        probabilities = np.exp(log_integrals[1:] - log_integrals[0])
        mass = float(sum(freq[r] * probabilities[j] for j, r in enumerate(values)))
        if not math.isfinite(log_evidence) or abs(mass - 1) > cfg.normalization_tolerance:
            raise PowerIntegrationError(f"predictive normalization failed w={w}: mass={mass}")
        return log_evidence, probabilities, {
            **stats, "analytic": False, "v_mode": peak,
            "v_window": [left - 0.5, right + 0.5],
            "outer_refinement_log_difference": log_difference,
            "outer_relative_error_estimate": relative_error_estimate,
            "outer_evaluations": neval, "raw_normalization": mass,
            "kernel_log_error_budget_estimate": d * (
                stats["kernel_max_refinement_log_difference"]
                + stats["kernel_max_tail_relative_estimate"]),
        }

    def prediction_by_count(self, d: int, partition: Sequence[int], *,
                            powers: Sequence[int]) -> PowerProfileResult:
        if int(d) != d or not 1 <= d <= self.settings.max_alphabet:
            raise ValueError("d outside configured powered-evaluator domain")
        raw = np.asarray(partition)
        if raw.ndim != 1 or np.any(~np.isfinite(raw)) or np.any(raw < 0) or np.any(raw != np.floor(raw)):
            raise ValueError("partition must contain nonnegative integer counts")
        part = tuple(sorted((int(r) for r in raw if r > 0), reverse=True))
        if len(part) > d or sum(part) > self.settings.max_sample_size:
            raise ValueError("count profile outside configured powered-evaluator domain")
        powers = tuple(powers)
        if (not powers or len(set(powers)) != len(powers)
                or any(not isinstance(w, (int, np.integer)) or not 0 <= w <= 80 for w in powers)):
            raise ValueError("powers must be distinct integers in [0,80]")
        values = tuple(sorted(set(part) | ({0} if len(part) < d else set())))
        logs, probabilities, diagnostics = [], [], []
        for w in powers:
            log_q, q, diag = self._component(d, part, int(w), values)
            logs.append(log_q)
            probabilities.append(q)
            diagnostics.append(diag)
        logs = np.asarray(logs)
        components = np.asarray(probabilities)
        posterior = np.exp(logs - logsumexp(logs))  # declared equal component prior
        mixture = posterior @ components
        freq = Counter(part)
        freq[0] = d - len(part)
        weights = np.array([freq[r] for r in values])
        return PowerProfileResult(
            tuple(int(w) for w in powers), values, components, mixture, posterior, logs,
            {"family": "single_layer_exponential_power", "log_units": "nats",
             "settings": asdict(self.settings), "components": diagnostics,
             "component_normalization": (components @ weights).tolist(),
             "mixture_normalization": float(mixture @ weights),
             "mixture_log_evidence": float(logsumexp(logs) - math.log(len(powers))),
             "normalization_applied": False},
        )

    def predict(self, counts: Sequence[int], *, powers: Sequence[int]) -> PowerPredictionResult:
        counts = np.asarray(counts)
        if counts.ndim != 1 or len(counts) == 0:
            raise ValueError("counts must be a nonempty one-dimensional vector")
        profile = self.prediction_by_count(len(counts), counts, powers=powers)
        indices = np.searchsorted(profile.count_values, counts)
        return PowerPredictionResult(
            profile.powers, profile.component_probabilities[:, indices],
            profile.mixture_probabilities[indices], profile.posterior,
            profile.log_evidence, profile.diagnostics,
        )


def predict(counts: Sequence[int], *, powers: Sequence[int],
            settings: PowerSettings | None = None) -> PowerPredictionResult:
    """Convenience entry point; grids are explicit and receive equal prior mass."""
    return PowerEvaluator(settings).predict(counts, powers=powers)
