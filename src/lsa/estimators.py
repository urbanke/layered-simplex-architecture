"""The estimators of the competitive comparison (paper, Section 5).

Every batch estimator maps a count vector ``m`` (length ``d``, summing to
``n``) to a probability assignment ``q_hat`` over the ``d``-symbol alphabet.
The sequential coders accumulate the codelength of a token sequence one
step at a time (predict, suffer the log-loss, update).

Implemented, exactly as specified in Section 5.1 of the paper:

- add-constant rules: add-one (Laplace), add-half (Krichevsky--Trofimov),
  and the Braess--Sauer rule;
- the Good--Turing + empirical hybrid of Orlitsky--Suresh;
- Ristad's natural law of succession;
- the natural oracle (knows the target, but must give symbols with equal
  counts the same probability) -- a lower bound for every count-based rule;
- the predictive distribution of the LSA prior at fixed depth L and of the
  depth-averaged predictor, both computed exactly from the count profile by
  the methods of Appendix B (see :func:`lsa_predictive_by_count`).

All returned assignments sum to one (validated in tests/test_estimators.py).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.special import gammaln

from lsa.codelength import (
    default_l_max,
    depth_averaged_codelength_families,
    profile_of,
)

LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# add-constant rules
# ---------------------------------------------------------------------------

def add_constant(counts: np.ndarray, beta: float) -> np.ndarray:
    """The rule q(i) = (m_i + beta) / (n + beta d)."""

    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    return (counts + beta) / (n + beta * counts.size)


def add_one(counts: np.ndarray) -> np.ndarray:
    """Laplace's rule; by Proposition 1 this is the LSA prior at L = 1."""

    return add_constant(counts, 1.0)


def add_half(counts: np.ndarray) -> np.ndarray:
    """The Krichevsky--Trofimov rule."""

    return add_constant(counts, 0.5)


def braess_sauer(counts: np.ndarray) -> np.ndarray:
    """Braess--Sauer: add 1/2 to unseen, 1 to once-seen, 3/4 to
    multiply-seen symbols, then normalize."""

    counts = np.asarray(counts, dtype=float)
    w = counts + 0.75
    w[counts == 0] = 0.5
    w[counts == 1] = 2.0
    return w / w.sum()


# ---------------------------------------------------------------------------
# Good--Turing + empirical hybrid (Orlitsky--Suresh)
# ---------------------------------------------------------------------------

def good_turing_hybrid(counts: np.ndarray) -> np.ndarray:
    """The hybrid of [Orlitsky--Suresh 2015], as stated in Section 5.1.

    With ``c_t`` the number of symbols appearing ``t`` times, a symbol seen
    ``t`` times receives (before normalization) the empirical mass ``t/n``
    if ``t > c_{t+1}``, and the Good--Turing mass
    ``(c_{t+1} + 1)(t + 1) / (n c_t)`` otherwise; unseen symbols share the
    ``t = 0`` assignment.  The result is normalized at the end.
    """

    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    if n == 0:
        return np.full(counts.size, 1.0 / counts.size)
    prevalence = Counter(int(t) for t in counts)  # includes t = 0 class
    w_of_t: dict[int, float] = {}
    for t, c_t in prevalence.items():
        c_next = prevalence.get(t + 1, 0)
        if t > 0 and t > c_next:
            w_of_t[t] = t / n
        else:
            w_of_t[t] = (c_next + 1) * (t + 1) / (n * c_t)
    w = np.array([w_of_t[int(t)] for t in counts])
    return w / w.sum()


# ---------------------------------------------------------------------------
# Ristad's natural law of succession
# ---------------------------------------------------------------------------

def ristad_natural_law(counts: np.ndarray) -> np.ndarray:
    """Ristad (1995): a closed-form rule from a two-level hierarchical
    uniform prior (support size, then identity, then uniform Dirichlet).

    With ``s <= d`` distinct symbols observed in ``n`` samples: if
    ``s = d`` the rule is add-one; otherwise

        q(i) = (m_i + 1)(n + 1 - s) / (n^2 + n + 2 s)          if m_i > 0,
        q(i) = s (s + 1) / ((d - s)(n^2 + n + 2 s))            if m_i = 0.

    The assignment is exactly normalized for every ``n``, ``s``, ``d``.
    """

    counts = np.asarray(counts, dtype=np.int64)
    d = counts.size
    n = int(counts.sum())
    s = int((counts > 0).sum())
    if n == 0:
        return np.full(d, 1.0 / d)
    if s == d:
        return add_one(counts)
    denominator = n * n + n + 2 * s
    q = (counts + 1.0) * (n + 1.0 - s) / denominator
    q[counts == 0] = s * (s + 1.0) / ((d - s) * denominator)
    return q


# ---------------------------------------------------------------------------
# the natural oracle
# ---------------------------------------------------------------------------

def natural_oracle(counts: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The genie of [Orlitsky--Suresh 2015]: it knows the target ``p`` but
    must assign the same probability to all symbols with the same count,

        q(i) = S_{m_i} / c_{m_i},

    with ``S_t`` the true total probability of the symbols appearing ``t``
    times and ``c_t`` how many there are.  It lower-bounds every natural
    (count-based) estimator, including all of the above.
    """

    counts = np.asarray(counts, dtype=np.int64)
    p = np.asarray(p, dtype=float)
    q = np.empty_like(p)
    for t in np.unique(counts):
        mask = counts == t
        q[mask] = p[mask].sum() / mask.sum()
    return q


# ---------------------------------------------------------------------------
# LSA predictive distributions (fixed depth and depth-averaged)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LsaPredictive:
    """Predictive probabilities of the LSA mixtures given a count profile.

    ``by_count[L]`` maps each count value ``c`` occurring in the sample
    (plus ``c = 0`` for unseen symbols) to the predictive probability of
    one particular symbol with that count under the depth-``L`` mixture;
    ``avg_by_count`` is the same for the depth-averaged predictor
    (posterior-weighted over ``L = 1 .. l_max``); ``posterior[L-1]`` is the
    posterior weight of depth ``L`` given the sample. With ``include_zero``
    enabled, the posterior starts at depth zero instead, and ``by_count[0]``
    is the fixed uniform predictor.
    """

    d: int
    l_max: int
    by_count: Mapping[int, Mapping[int, float]]
    avg_by_count: Mapping[int, float]
    posterior: tuple[float, ...]
    include_zero: bool = False  # posterior starts at L=0 when enabled

    def q_hat(self, counts: np.ndarray, depth: int | None = None) -> np.ndarray:
        """The full assignment over the alphabet behind ``counts``.

        ``depth=None`` gives the depth-averaged predictor; an integer gives
        that fixed depth.
        """

        table = self.avg_by_count if depth is None else self.by_count[depth]
        counts = np.asarray(counts, dtype=np.int64)
        q = np.empty(counts.size, dtype=float)
        for c in np.unique(counts):
            q[counts == c] = table[int(c)]
        if not q.sum() > 0.5:
            raise ValueError(
                f"predictives at depth {depth} were truncated as "
                "negligible; run with LSA_NO_TRUNCATE=1 for exact "
                "evaluation of every depth"
            )
        return q


def lsa_predictive_by_count(
    counts_or_profile: np.ndarray | Sequence[int],
    *,
    d: int,
    l_max: int | None = None,
    cache_dir: str | Path | None = None,
    jobs: int = 1,
    include_unseen: bool = True,
    include_zero: bool = False,
) -> LsaPredictive:
    """Exact LSA predictives from a count profile (Appendix B).

    The predictive probability of one symbol currently at count ``c`` is
    the ratio ``q_{lambda + c} / q_lambda`` of two mixture weights whose
    profiles differ by moving that symbol from ``c`` to ``c + 1``
    (Section 3.1); the depth-averaged predictive weights each depth by its
    posterior given the sample.  Everything is evaluated exactly by the
    machinery of Appendix B; no sequential simulation is involved.
    ``include_zero`` adds the fixed uniform component with equal prior weight
    and sequence evidence d**(-n), without changing positive-depth kernels.
    """

    if l_max is None:
        l_max = default_l_max(d)
    arr = np.asarray(counts_or_profile, dtype=np.int64)
    base = profile_of(int(c) for c in arr[arr > 0])
    cs = sorted(set(base))
    if include_unseen and len(base) < d:
        cs = [0, *cs]
    result = depth_averaged_codelength_families(
        {"family": (base, cs)},
        d=d,
        l_max=l_max,
        cache_dir=cache_dir,
        jobs=jobs,
    )
    base_result, augmented = result["family"]
    base_log2 = np.asarray(base_result.log2_q_by_depth)
    posterior = np.asarray(base_result.posterior)
    by_count: dict[int, dict[int, float]] = {}
    avg_by_count: dict[int, float] = {}
    for c in cs:
        aug_log2 = np.asarray(augmented[c].log2_q_by_depth)
        # Depths whose likelihood was truncated as negligible (see the
        # level window in lsa.codelength) carry -inf; their posterior
        # weight is zero, so they contribute nothing to the average.
        # Their FIXED-depth predictives are unavailable; q_hat() below
        # raises if such a depth is requested.  Set LSA_NO_TRUNCATE=1 to
        # evaluate every depth exactly.
        with np.errstate(invalid="ignore"):
            ratios = np.exp2(aug_log2 - base_log2)
        ratios = np.where(np.isfinite(ratios), ratios, 0.0)
        for L in range(1, l_max + 1):
            by_count.setdefault(L, {})[c] = float(ratios[L - 1])
        avg_by_count[c] = float(np.dot(posterior, ratios))
    if include_zero:
        posterior = np.asarray(base_result.with_uniform().posterior)
        w0 = posterior[0]
        for c in cs:
            by_count.setdefault(0, {})[c] = 1.0 / d
            ratios = np.array([by_count[L][c] for L in range(1, l_max + 1)])
            avg_by_count[c] = float(w0 / d + np.dot(posterior[1:], ratios))
    return LsaPredictive(
        d=d,
        l_max=l_max,
        by_count=by_count,
        avg_by_count=avg_by_count,
        posterior=tuple(float(w) for w in posterior),
        include_zero=include_zero,
    )


# ---------------------------------------------------------------------------
# batch codelengths of the exchangeable add-constant mixtures
# ---------------------------------------------------------------------------

def add_constant_codelength_bits(counts: np.ndarray, beta: float,
                                 d: int | None = None) -> float:
    """Exact batch codelength -log2 q(x^n) of the Dirichlet(beta) mixture.

    For beta = 1 this is the LSA prior at L = 1 (Proposition 1); for
    beta = 1/2 it is the Krichevsky--Trofimov code.  The batch codelength
    equals the accumulated sequential log-loss of the corresponding
    add-constant rule (chain rule; Appendix C).
    """

    counts = np.asarray(counts, dtype=np.int64)
    counts = counts[counts > 0]
    n = int(counts.sum())
    if d is None:
        d = counts.size
    log_q = (
        gammaln(d * beta)
        - gammaln(n + d * beta)
        + np.sum(gammaln(counts + beta))
        - counts.size * gammaln(beta)
    )
    return float(-log_q / LOG2)


# ---------------------------------------------------------------------------
# sequential coders (order-dependent rules; Section 5.3)
# ---------------------------------------------------------------------------

def sequential_codelength_bits(ids: Sequence[int], d: int, method: str,
                               checkpoints: Sequence[int] | None = None,
                               ) -> dict[int, float]:
    """Code a token sequence one symbol at a time with a classical rule.

    ``ids`` are integer token ids in ``0 .. d-1``; ``method`` is one of
    ``add_one``, ``add_half``, ``braess_sauer``, ``good_turing``,
    ``ristad``.  Returns cumulative bits at each requested checkpoint
    (default: only at ``len(ids)``).  This is an honest sequential code:
    predict the next token from the counts so far, suffer ``-log2 q``,
    update the counts.

    The add-constant rules are included for cross-checks only -- their
    accumulated log-loss equals the closed form
    :func:`add_constant_codelength_bits` by the chain rule.
    """

    ids = np.asarray(ids, dtype=np.int64)
    n_total = len(ids)
    checkpoints = sorted(set(checkpoints or [n_total]))
    if any(c < 1 or c > n_total for c in checkpoints):
        raise ValueError("checkpoints must lie in 1 .. len(ids)")
    checkpoint_set = set(checkpoints)

    counts = np.zeros(d, dtype=np.int64)
    prevalence: Counter[int] = Counter({0: d})  # c_t: symbols with count t
    s = 0        # distinct symbols seen
    n = 0        # tokens seen
    bits = 0.0
    out: dict[int, float] = {}

    # --- Good--Turing normalizer, maintained incrementally.
    # Both branches of the hybrid weight scale as 1/n, so track the
    # n-free class totals T(t):
    #   empirical branch (t > 0 and t > c_{t+1}):  T(t) = t c_t
    #   Good--Turing branch:                       T(t) = (c_{t+1}+1)(t+1)
    #     (the per-symbol mass (c_{t+1}+1)(t+1)/(n c_t) times c_t symbols)
    # and the normalizer is Z = (sum of T over occupied classes) / n.
    # One update moves a symbol from count t to t + 1, changing c_t and
    # c_{t+1} only; the class totals that can change are T(t-1), T(t),
    # T(t+1) (T(t') involves c_{t'} and c_{t'+1}).  Each step therefore
    # touches O(1) classes.  The running sum is refreshed from scratch
    # every 100,000 steps to keep floating-point drift negligible.
    def class_total(t: int) -> float:
        c_t = prevalence.get(t, 0)
        if c_t == 0:
            return 0.0
        c_next = prevalence.get(t + 1, 0)
        if t > 0 and t > c_next:
            return float(t) * c_t
        return float(c_next + 1) * (t + 1)

    def symbol_weight_nfree(t: int) -> float:
        c_next = prevalence.get(t + 1, 0)
        if t > 0 and t > c_next:
            return float(t)
        return float(c_next + 1) * (t + 1) / prevalence[t]

    track_gt = method == "good_turing"
    gt_zn = sum(class_total(t) for t in prevalence) if track_gt else 0.0

    for step, x in enumerate(ids, start=1):
        t = int(counts[x])
        if method == "add_one":
            q = (t + 1.0) / (n + d)
        elif method == "add_half":
            q = (t + 0.5) / (n + 0.5 * d)
        elif method == "braess_sauer":
            # closed-form normalizer:
            #   sum w = 0.5 c_0 + 2 c_1 + sum_{t>=2} (t + 3/4) c_t
            #         = 0.5 c_0 + 2 c_1 + (n - c_1) + 0.75 (s - c_1)
            c0 = prevalence.get(0, 0)
            c1 = prevalence.get(1, 0)
            z = 0.5 * c0 + 2.0 * c1 + (n - c1) + 0.75 * (s - c1)
            w = 0.5 if t == 0 else (2.0 if t == 1 else t + 0.75)
            q = w / z
        elif method == "good_turing":
            if n == 0:
                q = 1.0 / d
            else:
                q = symbol_weight_nfree(t) / gt_zn
        elif method == "ristad":
            if n == 0:
                q = 1.0 / d
            elif s == d:
                q = (t + 1.0) / (n + d)
            else:
                denominator = n * n + n + 2 * s
                if t > 0:
                    q = (t + 1.0) * (n + 1.0 - s) / denominator
                else:
                    q = s * (s + 1.0) / ((d - s) * denominator)
        else:
            raise ValueError(f"unknown method {method!r}")

        bits -= math.log2(q)

        # update the counts and, for Good--Turing, the class totals
        if track_gt:
            touched = (t - 1, t, t + 1)
            before = sum(class_total(u) for u in touched)
        counts[x] = t + 1
        prevalence[t] -= 1
        if prevalence[t] == 0:
            del prevalence[t]
        prevalence[t + 1] += 1
        if t == 0:
            s += 1
        n += 1
        if track_gt:
            gt_zn += sum(class_total(u) for u in touched) - before
            if step % 100_000 == 0:
                gt_zn = sum(class_total(u) for u in prevalence)

        if step in checkpoint_set:
            out[step] = bits
    return out
