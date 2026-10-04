#!/usr/bin/env python3
"""Competitive comparison on the Orlitsky--Suresh benchmark (Section 5).

Reproduces the data behind Figure 6 and Tables 5-6 of the paper.  For each
of eleven targets on a support of size d (default 10^4) and each sample
size n, a sample of size n is drawn; every estimator is given the
resulting counts (and d) and is scored by the divergence D(p || q_hat)
between the true distribution and its estimate, averaged over independent
trials with common samples across estimators.

Estimators: add-one (Laplace), add-half (Krichevsky--Trofimov),
Braess--Sauer, the Good--Turing + empirical hybrid, Ristad's natural law,
the natural oracle, the LSA prior at fixed depths (default 5 and 22), and
the depth-averaged LSA predictor (default L_max = 80).  The LSA
predictives are computed exactly from the count profile by the methods of
Appendix B; the only statistical error is the trial average.

Full run (the paper's setting; writes results.json):

    python scripts/benchmark_experiment.py --out output/benchmark \
        --jobs 8

Smoke run (a few minutes; reduced grid, same code paths):

    python scripts/benchmark_experiment.py --out output/benchmark_smoke \
        --d 300 --n-values 100,300,1000 --trials 3 --l-max 40

Tables 5 and 6 and Figure 6 are then rendered from results.json by
scripts/benchmark_report.py.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

# The comparison quotes fixed-depth rows and posterior weights, so every
# depth's likelihood is evaluated exactly rather than truncated as
# negligible (the truncation is a speed device for corpus runs where only
# the depth average is quoted).
os.environ.setdefault("LSA_NO_TRUNCATE", "1")

from lsa.codelength import default_l_max
from lsa.concentration_baselines import (
    absolute_discounting_predictive,
    dirichlet_mixture_predictive,
)
from lsa.estimators import (
    add_half,
    add_one,
    braess_sauer,
    good_turing_hybrid,
    lsa_predictive_by_count,
    natural_oracle,
    ristad_natural_law,
)

C_STAR = 1.0 / (1.0 - float(np.euler_gamma))

CLASSICAL = {
    "add_one": add_one,
    "add_half": add_half,
    "braess_sauer": braess_sauer,
    "good_turing": good_turing_hybrid,
    "ristad": ristad_natural_law,
}


# ---------------------------------------------------------------------------
# the eleven targets of Section 5.1
# ---------------------------------------------------------------------------

def zipf(d: int, alpha: float) -> np.ndarray:
    w = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return w / w.sum()


def make_targets(d: int) -> dict:
    """Fixed targets return an array; per-trial targets return a callable
    taking the trial's random generator."""

    return {
        "uniform": np.full(d, 1.0 / d),
        "step": np.concatenate([
            np.full(d // 2, 1.0 / (2 * d)),
            np.full(d - d // 2, 3.0 / (2 * d)),
        ]),
        "zipf_1": zipf(d, 1.0),
        "zipf_1.5": zipf(d, 1.5),
        "zipf_2": zipf(d, 2.0),
        "zipf_3": zipf(d, 3.0),
        "zipf_4": zipf(d, 4.0),
        "zipf_5": zipf(d, 5.0),
        "geometric": (lambda w: w / w.sum())(
            0.998 ** np.arange(1, d + 1, dtype=float)
        ),
        "dirichlet_1": lambda rng: rng.dirichlet(np.full(d, 1.0)),
        "dirichlet_half": lambda rng: rng.dirichlet(np.full(d, 0.5)),
    }


def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def finite_mean(values):
    return float(np.mean(values)) if np.all(np.isfinite(values)) else None


def finite_stderr(values):
    if not np.all(np.isfinite(values)):
        return None
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=10_000)
    parser.add_argument("--n-values",
                        default="1000,2000,3000,5000,7000,10000,14000,20000")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--l-max", type=int, default=80,
                        help="depth ceiling of the depth-averaged predictor "
                             "(paper: 80, comfortably above 2 c* ln d)")
    parser.add_argument("--fixed-depths", default="auto",
                        help="comma-separated fixed depths to report; "
                             "'auto' = 5 and round(c* ln d)")
    parser.add_argument("--targets", default="all",
                        help="comma-separated subset of targets (default all)")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--transient-cache", action="store_true",
                        help="delete disposable moment tables after each sample")
    parser.add_argument("--checkpoint", action="store_true",
                        help="save each completed cell and resume matching runs")
    parser.add_argument("--include-zero", action="store_true",
                        help="include fixed uniform L=0 with equal depth prior weight")
    parser.add_argument("--extra-baselines", action="store_true",
                        help="add AD and Dir-tau to the comparison")
    args = parser.parse_args()
    classical = dict(CLASSICAL)
    if args.extra_baselines:
        classical.update(absolute_discounting=absolute_discounting_predictive,
                         dir_tau=dirichlet_mixture_predictive)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    d = args.d
    n_values = [int(x) for x in args.n_values.split(",")]
    l_max = args.l_max or default_l_max(d)
    if args.fixed_depths == "auto":
        fixed_depths = sorted({5, int(round(C_STAR * np.log(d)))})
    else:
        fixed_depths = sorted({int(x) for x in args.fixed_depths.split(",")})
    if any(L > l_max for L in fixed_depths):
        raise SystemExit("every fixed depth must be <= --l-max")

    targets = make_targets(d)
    if args.targets != "all":
        keep = args.targets.split(",")
        unknown = [k for k in keep if k not in targets]
        if unknown:
            raise SystemExit(f"unknown targets {unknown}; "
                             f"known: {list(targets)}")
        targets = {k: targets[k] for k in keep}

    estimator_names = (
        list(classical)
        + ["oracle"]
        + [f"lsa_L{L}" for L in fixed_depths]
        + ["lsa_avg"]
    )
    checkpoint_dir = out_dir / "cells"
    if args.checkpoint:
        checkpoint_dir.mkdir(exist_ok=True)
        config = {"d": d, "n_values": n_values, "trials": args.trials,
                  "l_max": l_max, "fixed_depths": fixed_depths,
                  "targets": list(targets), "seed": args.seed,
                  "estimators": estimator_names, "include_zero": args.include_zero,
                  "no_truncate": os.environ.get("LSA_NO_TRUNCATE")}
        config_file = checkpoint_dir / "config.json"
        if config_file.exists() and json.loads(config_file.read_text()) != config:
            raise SystemExit("checkpoint configuration mismatch; use a new --out")
        config_file.write_text(json.dumps(config, indent=2))
    if args.transient_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results: dict = {}
    total_cells = len(targets) * len(n_values) * args.trials
    done = 0
    for t_index, (target_name, target) in enumerate(targets.items()):
        per_n: dict = {n: {name: [] for name in estimator_names}
                       for n in n_values}
        posteriors: dict = {n: [] for n in n_values}
        for trial in range(args.trials):
            rng = np.random.default_rng(
                [args.seed, t_index, trial]
            )
            p = target(rng) if callable(target) else target
            for n in n_values:
                counts = rng.multinomial(n, p)
                checkpoint_file = checkpoint_dir / f"{target_name}_{trial}_{n}.npz"
                if args.checkpoint and checkpoint_file.exists():
                    with np.load(checkpoint_file, allow_pickle=False) as saved:
                        for name, loss in zip(estimator_names, saved["losses"], strict=True):
                            per_n[n][name].append(float(loss))
                        posteriors[n].append(saved["posterior"].copy())
                    done += 1
                    continue
                for name, rule in classical.items():
                    per_n[n][name].append(kl_bits(p, rule(counts)))
                per_n[n]["oracle"].append(
                    kl_bits(p, natural_oracle(counts, p))
                )
                cache_context = (tempfile.TemporaryDirectory(dir=cache_dir,
                                 prefix="cell-") if args.transient_cache
                                 else nullcontext(cache_dir))
                with cache_context as cell_cache:
                    pred = lsa_predictive_by_count(
                        counts, d=d, l_max=l_max,
                        cache_dir=cell_cache, jobs=args.jobs,
                        include_zero=args.include_zero,
                    )
                for L in fixed_depths:
                    per_n[n][f"lsa_L{L}"].append(
                        kl_bits(p, pred.q_hat(counts, depth=L))
                    )
                per_n[n]["lsa_avg"].append(kl_bits(p, pred.q_hat(counts)))
                posteriors[n].append(pred.posterior)
                if args.checkpoint:
                    temporary = checkpoint_file.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez(stream, losses=[per_n[n][name][-1]
                                 for name in estimator_names],
                                 posterior=pred.posterior)
                    temporary.replace(checkpoint_file)
                done += 1
            print(
                f"[{time.time()-t0:7.0f}s] {target_name:>14} "
                f"trial {trial+1:>2}/{args.trials} done "
                f"({done}/{total_cells} cells)",
                flush=True,
            )
        results[target_name] = {
            "n_values": n_values,
            "mean": {
                name: [finite_mean(per_n[n][name]) for n in n_values]
                for name in estimator_names
            },
            "stderr": {
                name: [
                    finite_stderr(per_n[n][name])
                    for n in n_values
                ]
                for name in estimator_names
            },
            "nonfinite_trials": {
                name: [{"infinite": int(np.isinf(per_n[n][name]).sum()),
                        "undefined": int(np.isnan(per_n[n][name]).sum())}
                       for n in n_values] for name in estimator_names
            },
            "mean_posterior": {
                str(n): list(np.mean(np.asarray(posteriors[n]), axis=0))
                for n in n_values
            },
        }

    payload = {
        "d": d,
        "n_values": n_values,
        "trials": args.trials,
        "l_max": l_max,
        "include_zero": args.include_zero,
        "depths": list(range(0 if args.include_zero else 1, l_max + 1)),
        "fixed_depths": fixed_depths,
        "c_star": C_STAR,
        "seed": args.seed,
        "estimators": estimator_names,
        "loss": "KL(p || q_hat) in bits, mean over trials, "
                "common samples across estimators",
        "seconds": time.time() - t0,
        "targets": results,
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"written: {out_file} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
