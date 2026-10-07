"""Dictionary-conditioned Bible code lengths, controls, and chain checks."""

from __future__ import annotations

import gzip
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .artifacts import read_json, sha256, write_json
from .baselines import dirichlet_log_evidence
from .benchmark import jsonable

LOG2 = math.log(2)


def load_corpus(config, repo):
    path = Path(repo) / config["corpus"]
    expected = read_json(Path(repo) / config["manifest"])
    if sha256(path) != expected["compressed_sha256"]:
        raise ValueError("compressed corpus checksum mismatch")
    data = gzip.decompress(path.read_bytes())
    if hashlib.sha256(data).hexdigest() != expected["decompressed_sha256"]:
        raise ValueError("uncompressed corpus checksum mismatch")
    tokens = data.decode("utf8").split()
    if (
        len(tokens) != expected["tokens"]
        or len(set(tokens)) != expected["distinct_types"]
    ):
        raise ValueError("corpus token/type counts mismatch")
    return tokens, expected


def sequential_classical(ids, d, methods, exponents):
    """O(n times grid-size) online evaluation, before updating each token.

    The GT normalizer uses exact integer class totals; a transition changes
    only classes t-1,t,t+1. Per-token probabilities remain available for checks.
    """
    allowed = {
        "add_one",
        "kt",
        "ristad",
        "good_turing_hybrid",
        "dirichlet_concentration_mixture",
    }
    if set(methods) - allowed:
        raise ValueError("unknown sequential baseline")
    counts = np.zeros(d, dtype=np.int64)
    prevalence = Counter({0: d})
    log_losses = {m: np.empty(len(ids)) for m in methods}
    concentrations = np.exp2(np.asarray(exponents, dtype=float))
    evidence = np.zeros(len(concentrations))
    s = 0

    def total(t):
        ct, cn = prevalence.get(t, 0), prevalence.get(t + 1, 0)
        return 0 if ct == 0 else (t * ct if t > cn else (cn + 1) * (t + 1))

    normalizer = sum(total(t) for t in prevalence)
    for n, symbol in enumerate(ids):
        if symbol < 0 or symbol >= d:
            raise ValueError("token ID outside alphabet")
        t = int(counts[symbol])
        q = {"add_one": (t + 1) / (n + d), "kt": (t + 0.5) / (n + 0.5 * d)}
        if n == 0:
            q["ristad"] = q["good_turing_hybrid"] = 1 / d
        else:
            denominator = n * n + n + 2 * s
            q["ristad"] = (
                (t + 1) / (n + d)
                if s == d
                else (
                    (t + 1) * (n + 1 - s) / denominator
                    if t > 0
                    else s * (s + 1) / ((d - s) * denominator)
                )
            )
            nxt = prevalence.get(t + 1, 0)
            q["good_turing_hybrid"] = (
                t if t > nxt else (nxt + 1) * (t + 1) / prevalence[t]
            ) / normalizer
        component_q = (t + concentrations) / (n + d * concentrations)
        q["dirichlet_concentration_mixture"] = float(
            np.dot(np.exp(evidence - logsumexp(evidence)), component_q)
        )
        for method in methods:
            if not math.isfinite(q[method]) or not 0 < q[method] <= 1:
                raise ArithmeticError(f"invalid next-token probability for {method}")
            log_losses[method][n] = -math.log2(q[method])
        evidence += np.log(component_q)
        touched = (t - 1, t, t + 1)
        previous = sum(total(u) for u in touched)
        counts[symbol] += 1
        prevalence[t] -= 1
        if prevalence[t] == 0:
            del prevalence[t]
        prevalence[t + 1] += 1
        normalizer += sum(total(u) for u in touched) - previous
        if t == 0:
            s += 1
    return log_losses


def run_bible(config, output_dir, *, evaluator, repo):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    tokens, corpus = load_corpus(config, repo)
    checkpoints = sorted(set(config["prefixes"] + config["short_prefixes"]))
    longest = max(max(checkpoints), config["chain_rule_steps"])
    if longest > len(tokens):
        raise ValueError("prefix exceeds corpus length")
    vocabulary = {}
    ids = np.asarray(
        [vocabulary.setdefault(t, len(vocabulary)) for t in tokens[:longest]],
        dtype=np.int64,
    )
    if len(vocabulary) > config["d"]:
        raise ValueError("prefix vocabulary exceeds declared alphabet")
    write_json(root / "config.json", config)
    write_json(root / "corpus.json", corpus)
    write_json(root / "dictionary.json", vocabulary)
    np.savez_compressed(root / "token-ids.npz", ids=ids)
    methods = [m for m in config["methods"] if m != "lsa_depth_mixture"]
    losses = sequential_classical(
        ids, config["d"], methods, config["dirichlet_exponents"]
    )
    np.savez_compressed(root / "sequential-loss-bits.npz", **losses)
    cumulative = {m: np.cumsum(x) for m, x in losses.items()}
    rows = []
    checks = []
    for n in checkpoints:
        counts = np.bincount(ids[:n], minlength=config["d"])
        profile = tuple(sorted(map(int, counts[counts > 0]), reverse=True))
        np.savez_compressed(root / f"counts-{n}.npz", counts=counts)
        empirical = counts[counts > 0] / n
        entropy = float(-np.dot(empirical, np.log2(empirical)))
        totals = {method: float(value[n - 1]) for method, value in cumulative.items()}
        for method, a in [("add_one", 1.0), ("kt", 0.5)]:
            if method in totals:
                exact = -dirichlet_log_evidence(counts, a) / LOG2
                checks.append(
                    {
                        "check": "sequential_batch",
                        "method": method,
                        "n": n,
                        "error_bits": abs(totals[method] - exact),
                    }
                )
        if "dirichlet_concentration_mixture" in totals:
            logs = [
                dirichlet_log_evidence(counts, 2.0**j)
                for j in config["dirichlet_exponents"]
            ]
            exact = -(logsumexp(logs) - math.log(len(logs))) / LOG2
            checks.append(
                {
                    "check": "sequential_batch",
                    "method": "dirichlet_concentration_mixture",
                    "n": n,
                    "error_bits": abs(
                        totals["dirichlet_concentration_mixture"] - exact
                    ),
                }
            )
        posterior = None
        diagnostics = None
        if "lsa_depth_mixture" in config["methods"]:
            result = evaluator.evidence_at_depths(
                config["d"], profile, config["depths"]
            )
            logs = np.asarray(result.log_evidence)
            totals["lsa_depth_mixture"] = float(
                -(logsumexp(logs) - math.log(len(logs))) / LOG2
            )
            posterior = np.exp(logs - logsumexp(logs))
            diagnostics = result.diagnostics
        rows.append(
            {
                "n": n,
                "empirical_entropy_bits": entropy,
                "distinct": len(profile),
                "total_bits": totals,
                "redundancy_bits": {m: x / n - entropy for m, x in totals.items()},
                "posterior": posterior,
                "depths": config["depths"],
                "diagnostics": diagnostics,
            }
        )
    if any(c["error_bits"] > 3e-4 for c in checks):
        raise ArithmeticError("classical sequential/batch chain check failed")
    if config["chain_rule_steps"]:
        from .chain_validation import check_sequence

        chain_rows = check_sequence(
            evaluator,
            config["d"],
            ids[: config["chain_rule_steps"]],
            config["depths"],
            chunk_size=config.get("chain_chunk_size", 100),
        )
        write_json(root / "chain-rule.json", chain_rows)
        if (
            max(r["probability_error"] for r in chain_rows) > 1e-12
            or max(r["chain_error_bits"] for r in chain_rows) > 3e-4
        ):
            raise ArithmeticError("LSA sequential/batch chain check failed")
    summary = jsonable(
        {"config": config, "rows": rows, "classical_chain_checks": checks}
    )
    write_json(root / "summary.json", summary)
    return summary


def report_bible(source, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    summary = read_json(Path(source) / "summary.json")
    config = summary["config"]
    methods = config["methods"]
    labels = {
        "add_one": "Add-one",
        "kt": "KT",
        "ristad": "Ristad",
        "good_turing_hybrid": "GT",
        "dirichlet_concentration_mixture": r"Dir-$\tau$",
        "lsa_depth_mixture": "LSA",
    }
    lines = [
        r"\begin{tabular}{rrr"
        + "r" * (len(methods) + ("lsa_depth_mixture" in methods))
        + "}",
        r"$n$ & distinct & $\hat H_n$ & "
        + " & ".join(labels[m] for m in methods)
        + (r" & mode $L$" if "lsa_depth_mixture" in methods else "")
        + r" \\ \hline",
    ]
    controls = []
    for row in summary["rows"]:
        if row["n"] in config["prefixes"]:
            line = (
                str(row["n"])
                + f" & {row['distinct']} & {row['empirical_entropy_bits']:.2f} & "
                + " & ".join(
                    (r"\textbf{" + f"{row['redundancy_bits'][m]:.3f}" + "}")
                    if round(row["redundancy_bits"][m], 3)
                    == min(round(v, 3) for v in row["redundancy_bits"].values())
                    else f"{row['redundancy_bits'][m]:.3f}"
                    for m in methods
                )
            )
            if "lsa_depth_mixture" in methods:
                line += " & " + str(row["depths"][int(np.argmax(row["posterior"]))])
            lines.append(line + r" \\")
        else:
            controls.append(row)
    lines.append(r"\end{tabular}")
    (root / "bible-table.tex").write_text("\n".join(lines) + "\n")
    write_json(root / "bible-table.json", summary)
    write_json(root / "short-prefix-controls.json", controls)
    return summary
