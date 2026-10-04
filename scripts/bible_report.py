#!/usr/bin/env python3
"""Assemble Table 7 and Figure 7 from the Bible runs.

Inputs: the results.json of scripts/unigram_experiment.py (LSA columns and
per-depth codelengths) and of scripts/bible_baselines_experiment.py (the
classical columns).

    python scripts/bible_report.py \
        --unigram output/table7_bible/results.json \
        --baselines output/bible_baselines/results.json \
        --out output/bible_report

Writes table7.tsv, fig7_bible.pdf and prints the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLUMNS = [
    ("add_one", "add-1"),
    ("add_half", "KT"),
    ("braess_sauer", "BS"),
    ("ristad", "Ristad"),
    ("good_turing", "GT"),
]

STYLE = {
    "add_one": dict(color="0.6", marker="s"),
    "add_half": dict(color="tab:olive", marker="v"),
    "braess_sauer": dict(color="tab:cyan", marker="^"),
    "ristad": dict(color="tab:pink", marker="D"),
    "good_turing": dict(color="tab:blue", marker="o"),
    "absolute_discounting": dict(color="black", marker="x"),
    "dir_tau": dict(color="tab:green", marker="+"),
    "lsa_avg": dict(color="tab:red", marker="*", linewidth=1.6),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unigram", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    unigram = json.loads(Path(args.unigram).read_text())
    baselines = json.loads(Path(args.baselines).read_text())
    out_dir = Path(args.out) if args.out else Path(args.baselines).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    lsa_rows = {row["n"]: row for row in unigram["rows"]}
    base_rows = {int(k): v for k, v in baselines["rows"].items()}
    checkpoints = sorted(set(lsa_rows) & set(base_rows))
    if not checkpoints:
        raise SystemExit("no common checkpoints between the two runs")

    if unigram["d"] != baselines["d"]:
        raise SystemExit("alphabet sizes do not match")
    for n in checkpoints:
        if abs(lsa_rows[n]["empirical_entropy_bits"] - base_rows[n]["empirical_entropy_bits"]) > 1e-9:
            raise SystemExit("prefix entropies do not match")
    columns = list(COLUMNS)
    for key, label in [("absolute_discounting", "AD"), ("dir_tau", "Dir-tau")]:
        if key in baselines["methods"]:
            columns.append((key, label))

    def cell(base, key):
        value = base[key + "_redundancy"]
        if value is None:
            return base[key + "_diagnostics"]["status"]
        return f"{value:.3f}"

    # ----- Table 7 -----
    header = ["n", "distinct", "H_n", *[label for _, label in columns],
              "LSA avg", "posterior mode"]
    lines = ["\t".join(header)]
    print(" ".join(f"{h:>10}" for h in header))
    for n in checkpoints:
        lsa = lsa_rows[n]
        base = base_rows[n]
        cells = [
            f"{n:,}", f"{lsa['distinct_types']:,}",
            f"{lsa['empirical_entropy_bits']:.2f}",
            *[cell(base, key) for key, _ in columns],
            f"{lsa['redundancy_bits_per_token']:.3f}",
            f"L = {lsa['posterior_mode_depth']}",
        ]
        lines.append("\t".join(cells))
        print(" ".join(f"{c:>10}" for c in cells))
    (out_dir / "table7.tsv").write_text("\n".join(lines) + "\n")
    print(f"table: {out_dir / 'table7.tsv'}")
    if "absolute_discounting" in baselines["methods"]:
        (out_dir / "baseline_notes.txt").write_text(
            "AD uses c1/(c1+2*c2), uniform empty history, and uniform unseen mass. "
            "Infinite/undefined entries are not clipped or replaced. "
            "See absolute_discounting_diagnostics in baseline results.json.\n"
            "Dir-tau averages per-coordinate beta=2**j, j=-24,...,4, with equal prior weights.\n"
        )

    # ----- Figure 7 -----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.4, 3.4))

    ns = np.asarray(checkpoints, dtype=float)
    for key, label in columns:
        red = [base_rows[n][key + "_redundancy"] for n in checkpoints]
        if not any(x is not None and np.isfinite(x) for x in red):
            continue  # Undefined/infinite AD remains explicit in the table.
        ax_a.plot(ns, red, label=label, markersize=4, **STYLE[key])
    ax_a.plot(ns, [lsa_rows[n]["redundancy_bits_per_token"]
                   for n in checkpoints],
              label="depth-averaged LSA", markersize=6, **STYLE["lsa_avg"])
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("tokens n")
    ax_a.set_ylabel("regret [bits/token]")
    ax_a.set_title("(a) redundancy above empirical entropy", fontsize=9)
    ax_a.legend(fontsize=7, frameon=False)

    l_max = unigram["l_max"]
    depths = np.asarray(unigram.get("depths", list(range(1, l_max + 1))))
    for n in checkpoints:
        lsa = lsa_rows[n]
        by_depth = np.asarray(lsa["bits_per_token_by_depth"], dtype=float)
        redundancy = by_depth - lsa["empirical_entropy_bits"]
        finite = np.isfinite(redundancy)
        line, = ax_b.plot(depths[finite], redundancy[finite],
                          label=f"n = {n:,}", linewidth=1.2)
        best = int(np.nanargmin(np.where(finite, redundancy, np.inf)))
        ax_b.plot(depths[best], redundancy[best], "o", markersize=5,
                  color=line.get_color())
    ax_b.set_yscale("log")
    ax_b.set_xlabel("depth L")
    ax_b.set_ylabel(r"regret of $Q^{(L)}$ [bits/token]")
    ax_b.set_title("(b) regret vs. depth; dot = best depth", fontsize=9)
    ax_b.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig_path = out_dir / "fig7_bible.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
