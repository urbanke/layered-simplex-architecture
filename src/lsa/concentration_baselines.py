"""Concentration-averaged Dirichlet and unregularized absolute discounting.

Dir-tau uses per-coordinate beta=2**j, j=-24,...,4, with equal prior weights.
Mixtures of symmetric Dirichlets over concentration precede this implementation
(e.g. Nemenman, Shafee & Bialek, NIPS 2001); this grid is an experimental choice,
not the NSB entropy estimator. AD uses the Ney--Essen--Kneser discount estimate
c1/(c1+2*c2), with freed mass uniform over unseen symbols (Chen & Goodman 1999,
Sec. 2.6). No clipping or fallback is silently applied to that rule.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.special import gammaln, logsumexp

BETA_EXPONENTS = tuple(range(-24, 5))


def dirichlet_log_evidence(counts: np.ndarray, d: int) -> np.ndarray:
    """Natural log sequence evidence of each component (no count multiplicity)."""
    counts = np.asarray(counts, dtype=np.int64)
    if d < len(counts) or np.any(counts < 0):
        raise ValueError('invalid alphabet size or counts')
    beta = np.exp2(BETA_EXPONENTS)
    r, c = np.unique(counts[counts > 0], return_counts=True)
    return (gammaln(d * beta) - gammaln(counts.sum() + d * beta)
            + ((gammaln(r[:, None] + beta) - gammaln(beta)) * c[:, None]).sum(0))


def dirichlet_mixture_codelength_bits(counts: np.ndarray, d: int) -> float:
    logq = dirichlet_log_evidence(counts, d)
    return float(-(logsumexp(logq) - math.log(len(logq))) / math.log(2))


def dirichlet_mixture_predictive(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    d, n = len(counts), int(counts.sum())
    beta = np.exp2(BETA_EXPONENTS)
    logq = dirichlet_log_evidence(counts, d)
    w = np.exp(logq - logsumexp(logq))
    return counts * np.sum(w / (n + d * beta)) + np.sum(w * beta / (n + d * beta))


def absolute_discounting_codelengths(
    ids: Sequence[int], d: int, checkpoints: Sequence[int],
) -> dict[int, dict]:
    """O(N+d) prequential evaluation with explicit nonfinite diagnostics.

    Empty history predicts uniformly. Delta=1 may give a repeated singleton
    probability zero; c1=c2=0 makes the specified rule undefined. When the
    alphabet is exhausted, discounting has nowhere to put freed mass, so the
    rule is also marked undefined rather than silently renormalized.
    """
    ids = np.asarray(ids)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or d < 1 or np.any(ids < 0) or np.any(ids >= d)):
        raise ValueError('invalid token IDs or alphabet size')
    checkpoints = set(checkpoints)
    if not checkpoints or min(checkpoints) < 1 or max(checkpoints) > len(ids):
        raise ValueError('checkpoints must lie in 1..len(ids)')
    counts = np.zeros(d, dtype=np.int64)
    seen = c1 = c2 = 0
    first_zero = first_undefined = None
    bits = 0.0
    out = {}
    for n, token in enumerate(ids[:max(checkpoints)]):
        count = int(counts[token])
        if n == 0:
            q = 1.0 / d
        elif c1 + 2 * c2 == 0 or seen == d:
            q = None
        else:
            delta = c1 / (c1 + 2 * c2)
            q = ((count - delta) / n if count else
                 delta * seen / (n * (d - seen)))
        if q is None:
            if first_undefined is None:
                first_undefined = n + 1
        elif q == 0:
            if first_zero is None:
                first_zero = n + 1
        else:
            bits -= math.log2(q)
        if count == 0:
            seen += 1
            c1 += 1
        elif count == 1:
            c1 -= 1
            c2 += 1
        elif count == 2:
            c2 -= 1
        counts[token] += 1
        if n + 1 in checkpoints:
            status = ('undefined' if first_undefined is not None else
                      'infinite' if first_zero is not None else 'finite')
            out[n + 1] = dict(
                bits=bits if status == 'finite' else None, status=status,
                first_zero_probability_token=first_zero,
                first_undefined_prediction_token=first_undefined,
            )
    return out


def absolute_discounting_predictive(counts: np.ndarray) -> np.ndarray:
    """Batch AD; undefined formula returns NaNs, never an invented fallback."""
    counts = np.asarray(counts, dtype=np.int64)
    d, n = len(counts), int(counts.sum())
    if n == 0:
        return np.full(d, 1 / d)
    c1, c2 = np.sum(counts == 1), np.sum(counts == 2)
    seen = counts > 0
    if c1 + 2 * c2 == 0 or seen.all():
        return np.full(d, np.nan)
    delta = c1 / (c1 + 2 * c2)
    return np.where(seen, (counts - delta) / n,
                    delta * seen.sum() / (n * (~seen).sum()))
