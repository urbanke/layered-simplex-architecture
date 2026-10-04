#!/usr/bin/env python3
"""Render Figure 6 and Tables 5-6 from benchmark_experiment.py results.

    python scripts/benchmark_report.py --results output/benchmark/results.json \
        --out output/benchmark

Writes fig6_benchmark.pdf, table5.tsv, table6.tsv and prints both tables.
Table 5 reports each estimator at the largest n; Table 6 reports the mean
posterior weights over depths of the depth-averaged predictor (entries
below 0.05 omitted, as in the paper).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRETTY = {
    "add_one": "add-1 (Laplace)",
    "add_half": "add-1/2 (KT)",
    "braess_sauer": "Braess-Sauer",
    "good_turing": "Good-Turing + emp.",
    "ristad": "Ristad natural law",
    "oracle": "natural oracle",
    "lsa_avg": "depth-averaged LSA",
}

STYLE = {
    "add_one": dict(color="0.6", marker="+"),
    "add_half": dict(color="0.45", marker="x"),
    "braess_sauer": dict(color="tab:orange", marker="v"),
    "good_turing": dict(color="tab:green", marker="s"),
    "ristad": dict(color="tab:purple", marker="^"),
    "oracle": dict(color="black", linestyle=":", marker=""),
    "lsa_avg": dict(color="tab:red", marker="D"),
}
FIXED_STYLE = [dict(color="tab:blue", marker="o"),
               dict(color="tab:brown", marker="P"),
               dict(color="tab:pink", marker="*")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text())
    out_dir = Path(args.out) if args.out else Path(args.results).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed = [f"lsa_L{L}" for L in results["fixed_depths"]]
    order = ["add_one", "add_half", "braess_sauer", "ristad",
             "good_turing", "oracle", *fixed, "lsa_avg"]
    order += [key for key in ["absolute_discounting", "dir_tau"] if key in results["estimators"]]
    pretty = dict(PRETTY)
    pretty.update(absolute_discounting="AD", dir_tau="Dir-tau")
    style = dict(STYLE)
    style.update(absolute_discounting=dict(color="black", marker="x"),
                 dir_tau=dict(color="tab:green", marker="+"))
    for i, name in enumerate(fixed):
        pretty[name] = f"LSA L={name[5:]}"
        style[name] = FIXED_STYLE[i % len(FIXED_STYLE)]

    targets = list(results["targets"])
    n_values = results["n_values"]

    # ----- Figure 6 -----
    columns = 4
    rows = (len(targets) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns,
                             figsize=(3.1 * columns, 2.6 * rows))
    axes = np.atleast_2d(axes)
    for k, target in enumerate(targets):
        ax = axes[k // columns][k % columns]
        block = results["targets"][target]
        for name in order:
            mean = np.asarray(block["mean"][name], dtype=float)
            err = np.asarray(block["stderr"][name], dtype=float)
            positive = mean > 0
            ax.errorbar(np.asarray(n_values)[positive], mean[positive],
                        yerr=err[positive], label=pretty[name],
                        markersize=3.5, linewidth=1.0, **style[name])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(target.replace("_", " "), fontsize=9)
        ax.tick_params(labelsize=7)
        if k // columns == rows - 1:
            ax.set_xlabel("sample size n", fontsize=8)
        if k % columns == 0:
            ax.set_ylabel(r"$E\,D(p\,\|\,\hat q)$ [bits]", fontsize=8)
    for k in range(len(targets), rows * columns):
        axes[k // columns][k % columns].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=7, ncol=2,
               frameon=False)
    fig.tight_layout()
    fig_path = out_dir / "fig6_benchmark.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"figure: {fig_path}")

    # ----- Table 5 (largest n) -----
    n_last = n_values[-1]
    header = ["target", *[pretty[name] for name in order]]
    comparisons = [key for key in ("good_turing", "dir_tau")
                   if "dir_tau" in results["estimators"]]
    header += [f"LSA change vs {pretty[key]} (%)" for key in comparisons]
    lines = ["\t".join(header)]
    print(f"\nTable 5 at n = {n_last:,} (mean KL in bits):")
    print(f"{'target':>16} " + " ".join(f"{pretty[o][:10]:>11}" for o in order))
    max_err = 0.0
    for target in targets:
        block = results["targets"][target]
        vals = [block["mean"][name][-1] for name in order]
        max_err = max([max_err] + [block["stderr"][name][-1] for name in order
                                   if block["stderr"][name][-1] is not None])
        def format_value(name, value, block=block):
            if value is not None:
                return f"{value:.6g}"
            status = block["nonfinite_trials"][name][-1]
            return "undefined" if status["undefined"] else "infinite"
        formatted = [format_value(name, value) for name, value in zip(order, vals)]
        changes = []
        for comparator in comparisons:
            baseline = block["mean"][comparator][-1]
            lsa = block["mean"]["lsa_avg"][-1]
            change = (100 * (lsa / baseline - 1)
                      if lsa is not None and baseline is not None and baseline > 0 else None)
            changes.append(f"{change:+.1f}%" if change is not None else "undefined")
        lines.append("\t".join([target, *formatted, *changes]))
        print(f"{target:>16} " + " ".join(f"{v:>11}" for v in formatted))
    (out_dir / "table5.tsv").write_text("\n".join(lines) + "\n")
    print(f"largest standard error: {max_err:.4g} bits")
    print(f"table: {out_dir / 'table5.tsv'}")

    # ----- Table 6 (posterior over depths) -----
    show_n = [n_values[0], n_last] if len(n_values) > 1 else [n_last]
    lines = ["\t".join(["target",
                        *[f"posterior at n={n:,}" for n in show_n]])]
    print("\nTable 6 (mean posterior weights over depths; entries < 0.05 "
          "omitted):")
    for target in targets:
        block = results["targets"][target]
        cells = []
        for n in show_n:
            post = np.asarray(block["mean_posterior"][str(n)], dtype=float)
            top = [(L, w) for L, w in zip(
                results.get("depths", range(1, len(post) + 1)), post) if w >= 0.05]
            top.sort(key=lambda t: -t[1])
            cells.append(", ".join(f"L={L}: {w:.2f}" for L, w in top))
        lines.append("\t".join([target, *cells]))
        print(f"{target:>16}  " + "   |   ".join(cells))
    (out_dir / "table6.tsv").write_text("\n".join(lines) + "\n")
    print(f"table: {out_dir / 'table6.tsv'}")


if __name__ == "__main__":
    main()
