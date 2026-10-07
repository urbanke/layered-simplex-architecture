"""The baseline definitions in the ALT manuscript, without numerical-engine imports.

Probabilities are per labelled symbol. Evidence is for an ordered sequence,
not a multinomial count event, and is returned in natural logarithms.
Undefined absolute-discounting cases are explicit, not replaced by another rule.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp

DIRICHLET_EXPONENTS = tuple(range(-24, 5))
BASELINE_METHODS = (
    "add_one", "kt", "ristad", "good_turing_hybrid",
    "dirichlet_concentration_mixture", "absolute_discounting", "oracle",
)


@dataclass(frozen=True)
class BaselinePrediction:
    probabilities: np.ndarray | None
    status: str = "defined"
    reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def validate_counts(counts: Sequence[int]) -> np.ndarray:
    raw = np.asarray(counts)
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("counts must be a nonempty one-dimensional numeric vector")
    if np.any(~np.isfinite(raw)) or np.any(raw < 0) or np.any(raw != np.floor(raw)):
        raise ValueError("counts must be finite nonnegative integers")
    if np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("counts exceed int64 range")
    return raw.astype(np.int64, copy=False)


def validate_probabilities(p: Sequence[float], *, size: int | None = None) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not p.size or (size is not None and p.size != size):
        raise ValueError("probability vector has wrong shape")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValueError(f"probability vector sums to {p.sum():.17g}, not one")
    return p


def add_constant(counts: Sequence[int], concentration: float) -> np.ndarray:
    m = validate_counts(counts)
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("per-coordinate concentration must be positive and finite")
    return (m + concentration) / (float(m.sum()) + m.size * concentration)


def good_turing_hybrid(counts: Sequence[int]) -> np.ndarray:
    """The stated GT/empirical hybrid, including the zero-count class.

    The common 1/n cancels in normalization. At n=0, symmetry gives 1/d.
    """
    m = validate_counts(counts)
    if m.sum() == 0:
        return np.full(m.size, 1.0 / m.size)
    prevalence = Counter(map(int, m))
    weights = {
        t: float(t) if t > prevalence.get(t + 1, 0)
        else (prevalence.get(t + 1, 0) + 1.0) * (t + 1) / c_t
        for t, c_t in prevalence.items()
    }
    q = np.asarray([weights[int(t)] for t in m])
    return q / q.sum()


def ristad(counts: Sequence[int]) -> np.ndarray:
    m = validate_counts(counts)
    n, d, s = int(m.sum()), m.size, int(np.count_nonzero(m))
    if n == 0:
        return np.full(d, 1.0 / d)
    if s == d:
        return add_constant(m, 1.0)
    denominator = n * n + n + 2 * s
    q = (m + 1.0) * (n + 1.0 - s) / denominator
    q[m == 0] = s * (s + 1.0) / ((d - s) * denominator)
    return q


def absolute_discounting(counts: Sequence[int]) -> BaselinePrediction:
    m = validate_counts(counts)
    n, d, s = int(m.sum()), m.size, int(np.count_nonzero(m))
    c1, c2 = int(np.count_nonzero(m == 1)), int(np.count_nonzero(m == 2))
    diagnostics = {"n": n, "distinct": s, "c1": c1, "c2": c2}
    if c1 + 2 * c2 == 0:
        return BaselinePrediction(None, "undefined", "c1_and_c2_are_zero", diagnostics)
    delta = c1 / (c1 + 2 * c2)
    diagnostics["discount"] = delta
    if s == d and delta > 0:
        # The manuscript defines allocation to unseen labels only. Do not
        # invent a fallback when this class is empty (outside d > n tests).
        return BaselinePrediction(
            None, "undefined", "no_unseen_symbol_for_reserved_mass", diagnostics,
        )
    q = np.zeros(d, dtype=float)
    q[m > 0] = (m[m > 0] - delta) / n
    if s < d:
        q[m == 0] = delta * s / (n * (d - s))
    return BaselinePrediction(q, diagnostics=diagnostics)


def natural_oracle(counts: Sequence[int], target: Sequence[float]) -> np.ndarray:
    m = validate_counts(counts)
    p = validate_probabilities(target, size=m.size)
    q = np.empty_like(p)
    for count in np.unique(m):
        mask = m == count
        q[mask] = p[mask].sum() / np.count_nonzero(mask)
    return q


def dirichlet_log_evidence(counts: Sequence[int], concentration: float) -> float:
    """Exact Dirichlet formula, evaluated in float64 (no count multiplicity)."""
    m = validate_counts(counts)
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("per-coordinate concentration must be positive and finite")
    n = int(m.sum())
    if n == 0:
        return 0.0
    positive, multiplicities = np.unique(m[m > 0], return_counts=True)
    tau = m.size * concentration
    return float(
        gammaln(tau) - gammaln(tau + n)
        + np.dot(multiplicities, gammaln(positive + concentration) - gammaln(concentration))
    )


def dirichlet_concentration_mixture(
    counts: Sequence[int], *, exponents: Sequence[int],
) -> BaselinePrediction:
    m = validate_counts(counts)
    exponents = tuple(exponents)
    if not exponents or len(set(exponents)) != len(exponents):
        raise ValueError("concentration exponent grid must be nonempty and unique")
    if any(isinstance(j, bool) or not isinstance(j, (int, np.integer)) for j in exponents):
        raise ValueError("concentration exponents must be integers")
    concentrations = np.exp2(np.asarray(exponents, dtype=float))
    if np.any(~np.isfinite(concentrations)) or np.any(concentrations <= 0):
        raise ValueError("concentration grid is outside float64 range")
    logs = np.asarray([dirichlet_log_evidence(m, a) for a in concentrations])
    log_normalizer = logsumexp(logs)
    posterior = np.exp(logs - log_normalizer)
    # This expression avoids a 29-by-d temporary array.
    denominators = m.sum() + m.size * concentrations
    q = m * np.sum(posterior / denominators) + np.sum(posterior * concentrations / denominators)
    return BaselinePrediction(q, diagnostics={
        "exponents": list(map(int, exponents)), "posterior": posterior.tolist(),
        "component_log_evidence_nats": logs.tolist(),
        "mixture_log_evidence_nats": float(log_normalizer - np.log(len(logs))),
    })


def evaluate_baseline(
    method: str, counts: Sequence[int], *, target: Sequence[float],
    dirichlet_exponents: Sequence[int],
) -> BaselinePrediction:
    if method == "add_one":
        return BaselinePrediction(add_constant(counts, 1.0))
    if method == "kt":
        return BaselinePrediction(add_constant(counts, 0.5))
    if method == "ristad":
        return BaselinePrediction(ristad(counts))
    if method == "good_turing_hybrid":
        return BaselinePrediction(good_turing_hybrid(counts))
    if method == "absolute_discounting":
        return absolute_discounting(counts)
    if method == "oracle":
        return BaselinePrediction(natural_oracle(counts, target))
    if method == "dirichlet_concentration_mixture":
        return dirichlet_concentration_mixture(counts, exponents=dirichlet_exponents)
    raise ValueError(f"unknown baseline {method!r}")


def kl_bits(target: Sequence[float], probabilities: Sequence[float]) -> float:
    """KL in bits; genuine zeros retain infinite loss, never epsilon-clipped."""
    p = validate_probabilities(target)
    q = validate_probabilities(probabilities, size=p.size)
    active = p > 0
    if np.any(q[active] == 0):
        return float("inf")
    loss = float(np.dot(p[active], (np.log(p[active]) - np.log(q[active]))) / np.log(2))
    if loss < -1e-10:
        raise ArithmeticError(f"negative KL loss {loss}; check normalization")
    return max(0.0, loss)  # Only roundoff below zero is clipped.
