"""Generate current ALT table/plot roles from saved benchmark trial records.

Figure and table data are aggregated from the same records. No historical
transcribed numbers or private finite-only averages enter these reports.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import validate_probabilities
from .benchmark import (
    POWER_METHODS,
    PRIMARY_METHODS,
    TARGET_IDS,
    aggregate_records,
    jsonable,
    sha256_file,
)

METHOD_LABELS = {
    "add_one": "add-1", "kt": "KT", "ristad": "Ristad",
    "good_turing_hybrid": "GT", "dirichlet_concentration_mixture": "Dir-$\\tau$",
    "absolute_discounting": "AD", "lsa_fixed": "LSA fixed",
    "lsa_depth_mixture": "LSA depth avg", "power_mixture": "power avg", "oracle": "natural oracle",
}
TARGET_LABELS = {
    "uniform": "uniform", "step": "step", "geometric": "geometric",
    "dirichlet_1": "Dirichlet-1", "dirichlet_half": r"Dirichlet-$\frac{1}{2}$",
    **{f"zipf_{a}": f"Zipf $\\alpha={a}$" for a in ("1", "1.5", "2", "3", "4", "5")},
}


def load_summary(run_dir: str | Path) -> dict:
    """Reaggregate complete trial files and verify the recorded file checksum."""
    run_dir = Path(run_dir)
    stored = json.loads((run_dir / "summary.json").read_text())
    config = json.loads((run_dir / "benchmark-config.json").read_text())
    if stored["config"] != config:
        raise ValueError("summary/config mismatch")
    if sha256_file(run_dir / "trials.jsonl") != stored["trial_records_sha256"]:
        raise ValueError("trial records changed after run completion")
    records = []
    with (run_dir / "trials.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                record.pop("diagnostics", None)
                records.append(record)
    result = aggregate_records(records, config)
    if result["targets"] != stored["targets"]:
        raise ValueError("stored summary disagrees with saved trial records")
    result["source"] = {
        "run_directory": str(run_dir.resolve()),
        "trial_records_sha256": stored["trial_records_sha256"],
        "sample_manifest_sha256": stored["sample_manifest_sha256"],
    }
    return result


def posterior_summary(indices: Sequence[int], weights: Sequence[float], *,
                      mass: float, broad_min_depths: int,
                      individual_min_weight: float) -> dict:
    """Shortest contiguous interval; ties prefer greater enclosed mass.

    Contiguous means adjacent integer model indices, not adjacent locations in
    a sparse plotting grid. The actual mass is retained without rounding.
    """
    indices = list(indices)
    p = validate_probabilities(weights, size=len(indices))
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("posterior summary requires a contiguous integer grid")
    if not (0 < mass <= 1) or broad_min_depths < 1 or not (0 <= individual_min_weight <= 1):
        raise ValueError("invalid posterior reporting rule")
    cumulative = np.r_[0.0, np.cumsum(p)]
    candidates = []
    for left in range(len(p)):
        for right in range(left, len(p)):
            enclosed = float(cumulative[right + 1] - cumulative[left])
            if enclosed >= mass - 1e-14:
                candidates.append((right - left + 1, -enclosed, left, right))
                break
    if not candidates:
        raise ArithmeticError("posterior has insufficient total mass")
    width, neg_mass, left, right = min(candidates)
    result = {"interval": [indices[left], indices[right]], "enclosed_mass": -neg_mass,
              "number_of_depths": width, "requested_mass": mass}
    if width >= broad_min_depths:
        result.update(kind="interval", text=f"broad: {-neg_mass:.1%} on L in [{indices[left]},{indices[right]}]")
    else:
        kept = [(i, float(w)) for i, w in zip(indices, p) if w >= individual_min_weight]
        result.update(kind="individual", weights=kept,
                      text=", ".join(f"L={i}: {w:.2f}" for i, w in kept))
    return result


def _ensure_cells(summary: Mapping[str, Any], targets: Sequence[str], n_values: Sequence[int],
                  methods: Sequence[str]) -> None:
    for target in targets:
        if target not in summary["targets"]:
            raise ValueError(f"missing report target {target}")
        for n in n_values:
            if str(n) not in summary["targets"][target]:
                raise ValueError(f"missing report cell {target}, n={n}")
            cell = summary["targets"][target][str(n)]
            missing = set(methods) - cell["methods"].keys()
            if missing:
                raise ValueError(f"missing enabled report methods {sorted(missing)}")


def _display(cell: Mapping[str, Any], *, tex: bool = False) -> str:
    if cell["status"] == "undefined":
        return r"n.a.$^{*}$" if tex else "undefined"
    if cell["status"] == "infinite":
        return r"$\infty^{*}$" if tex else "+infinity"
    return f"{cell['mean_bits']:.4f}"


def _write_tsv(path: Path, rows: Sequence[Sequence[Any]]) -> None:
    with path.open("x", encoding="utf8", newline="") as stream:
        csv.writer(stream, delimiter="\t").writerows(rows)


def write_loss_table(summary: Mapping[str, Any], output_stem: str | Path, *,
                     n: int, targets: Sequence[str], methods: Sequence[str],
                     label: str) -> dict:
    """Full precision TSV plus manuscript precision TeX from identical cells."""
    _ensure_cells(summary, targets, [n], methods)
    stem = Path(output_stem)
    rows = [["target", "n", "method", "status", "mean_bits", "se_bits", "trials",
             "finite_trials", "infinite_trials", "undefined_trials"]]
    tex = [f"% label: {label}; sample_set={summary['config']['sample_set_id']}; n={n}",
           r"\begin{tabular}{l" + "r" * len(methods) + "}", r"\toprule",
           "target & " + " & ".join(METHOD_LABELS[m] for m in methods) + r" \\", r"\midrule"]
    finite_ses = []
    for target in targets:
        cells = summary["targets"][target][str(n)]["methods"]
        candidates = [round(cells[m]["mean_bits"], 4) for m in methods
                      if m != "oracle" and cells[m]["status"] == "finite"]
        best = min(candidates) if candidates else None
        values = []
        for method in methods:
            cell = cells[method]
            rows.append([target, n, method, cell["status"], cell["mean_bits"], cell["se_bits"],
                         cell["trials"], *[cell["counts"][s] for s in ("finite", "infinite", "undefined")]])
            if cell["se_bits"] is not None:
                finite_ses.append(cell["se_bits"])
            value = _display(cell, tex=True)
            if method != "oracle" and cell["status"] == "finite" and round(cell["mean_bits"], 4) == best:
                value = r"\textbf{" + value + "}"
            values.append(value)
        tex.append(TARGET_LABELS[target] + " & " + " & ".join(values) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    _write_tsv(stem.with_suffix(".tsv"), rows)
    with stem.with_suffix(".tex").open("x", encoding="utf8") as stream:
        stream.write("\n".join(tex) + "\n")
    return {"label": label, "n": n, "methods": list(methods),
            "max_finite_standard_error_bits": max(finite_ses) if finite_ses else None,
            "rounding": "four decimal places; bold ties at that precision, oracle excluded",
            "source": summary.get("source")}


def write_posterior_table(summary: Mapping[str, Any], output_stem: str | Path, *,
                          targets: Sequence[str], n_values: Sequence[int],
                          mass: float, broad_min_depths: int,
                          individual_min_weight: float) -> dict:
    _ensure_cells(summary, targets, n_values, ["lsa_depth_mixture"])
    stem = Path(output_stem)
    rows = [["target", *[f"n={n}" for n in n_values]]]
    details = {}
    tex = [r"\begin{tabular}{l" + "l" * len(n_values) + "}", r"\toprule",
           "target & " + " & ".join(f"$n={n}$" for n in n_values) + r" \\", r"\midrule"]
    for target in targets:
        cells = []
        details[target] = {}
        for n in n_values:
            post = summary["targets"][target][str(n)]["mean_posteriors"]["depth"]
            detail = posterior_summary(post["indices"], post["weights"], mass=mass,
                                       broad_min_depths=broad_min_depths,
                                       individual_min_weight=individual_min_weight)
            cells.append(detail["text"])
            details[target][str(n)] = detail
        rows.append([target, *cells])
        tex.append(TARGET_LABELS[target] + " & " + " & ".join(s.replace("%", r"\%") for s in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    _write_tsv(stem.with_suffix(".tsv"), rows)
    with stem.with_suffix(".tex").open("x", encoding="utf8") as stream:
        stream.write("\n".join(tex) + "\n")
    return {"label": "tab:posteriors", "source": summary.get("source"), "cells": details}


def plot_learning_curves(summary: Mapping[str, Any], output_path: str | Path, *,
                          targets: Sequence[str], methods: Sequence[str],
                          uniform_floor: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter

    n_values = summary["config"]["n_values"]
    _ensure_cells(summary, targets, n_values, methods)
    if uniform_floor <= 0 or not np.isfinite(uniform_floor):
        raise ValueError("uniform plot floor must be positive")
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    styles = {
        "add_one": ("0.6", "+", "-"), "kt": ("0.4", "x", "-"),
        "ristad": ("tab:purple", "^", "-"), "good_turing_hybrid": ("tab:green", "s", "-"),
        "dirichlet_concentration_mixture": ("tab:cyan", "P", "-"),
        "absolute_discounting": ("tab:orange", "v", "-"),
        "lsa_fixed": ("tab:blue", "o", "-"), "lsa_depth_mixture": ("tab:red", "D", "-"),
        "oracle": ("black", "", ":"),
    }
    if set(methods) - styles.keys():
        raise ValueError("learning curve requested a method absent from the current figure")
    columns = 4
    rows = (len(targets) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.65 * rows), squeeze=False)
    details = {"nonfinite_points": [], "uniform_floor_points": [], "zero_points_outside_uniform": []}
    handles = []
    for method in methods:
        color, marker, line = styles[method]
        label = METHOD_LABELS[method]
        if method == "lsa_fixed":
            label = f"LSA L={summary['config']['fixed_depth']}"
        elif method == "lsa_depth_mixture":
            depths = summary["config"]["depths"]
            label = f"LSA avg, L={depths[0]}–{depths[-1]}"
        handles.append(Line2D([], [], color=color, marker=marker, linestyle=line, label=label))
    for panel, target in enumerate(targets):
        ax = axes.flat[panel]
        for method in methods:
            color, marker, line = styles[method]
            x, means, ses = [], [], []
            for n in n_values:
                cell = summary["targets"][target][str(n)]["methods"][method]
                if cell["status"] != "finite":
                    details["nonfinite_points"].append({"target": target, "n": n, "method": method,
                                                       "status": cell["status"], "counts": cell["counts"]})
                    # NaN breaks a line; do not bridge omitted AD points.
                    x.append(n); means.append(np.nan); ses.append(np.nan)
                    continue
                mean, se = cell["mean_bits"], cell["se_bits"]
                if se is None:
                    raise ValueError("SE figure requires at least two trials per cell")
                if target == "uniform" and mean <= uniform_floor:
                    ax.scatter([n], [uniform_floor], marker="v", color=color, s=24, zorder=5)
                    details["uniform_floor_points"].append({"method": method, "n": n, "mean_bits": mean})
                    x.append(n); means.append(np.nan); ses.append(np.nan)
                elif mean <= 0:
                    details["zero_points_outside_uniform"].append({"target": target, "method": method, "n": n})
                    x.append(n); means.append(np.nan); ses.append(np.nan)
                else:
                    x.append(n); means.append(mean); ses.append(se)
            ax.errorbar(x, means, yerr=ses, color=color, marker=marker, linestyle=line,
                        markersize=3, linewidth=0.9, capsize=1.5)
        omitted = [item for item in details["nonfinite_points"] if item["target"] == target]
        if omitted:
            names = list(dict.fromkeys(METHOD_LABELS[item["method"]] for item in omitted))
            ax.text(.03, .03, ", ".join(names) + ": some points ∞ / n.a.",
                    transform=ax.transAxes, fontsize=6)
        ax.set(xscale="log", yscale="log", title=TARGET_LABELS[target])
        tick_indices = np.unique(np.linspace(0, len(n_values) - 1, min(4, len(n_values))).round().astype(int))
        tick_values = [n_values[i] for i in tick_indices]
        ax.set_xticks(tick_values, labels=[f"{n:,}" for n in tick_values])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=7)
        if panel // columns == rows - 1:
            ax.set_xlabel("sample size n", fontsize=8)
        if panel % columns == 0:
            ax.set_ylabel("mean next-symbol KL [bits]", fontsize=8)
    for ax in axes.flat[len(targets):]:
        ax.axis("off")
    # The twelfth panel holds the same nine-method legend as the manuscript.
    if len(targets) < rows * columns:
        axes.flat[-1].legend(handles=handles, loc="center", fontsize=8, frameon=False)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.text(.5, .006,
             f"Bars: one standard error; triangles: uniform loss ≤ {uniform_floor:g} bits; "
             "nonfinite AD cases marked in panels.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {"label": "fig:orlitsky", "methods": list(methods), "source": summary.get("source"),
            "standard_errors": "sample SD / sqrt(trials); not confidence intervals",
            "uniform_floor_bits": uniform_floor,
            "nonfinite_policy": "nonfinite means not plotted; line broken and locations recorded",
            **details}


def write_paired_differences(summary: Mapping[str, Any], output_path: str | Path, *,
                             targets: Sequence[str], n: int) -> dict:
    rows = [["target", "n", "comparison", "status", "mean_difference_bits", "paired_se_bits", "trials"]]
    for target in targets:
        for name, cell in summary["targets"][target][str(n)]["paired_differences"].items():
            rows.append([target, n, name, cell["status"], cell["mean_bits"], cell["se_bits"], cell["trials"]])
    _write_tsv(Path(output_path), rows)
    return {"source": summary.get("source"), "n": n,
            "uncertainty": "standard error of per-trial paired differences"}


def write_uniform_ablation(summary: Mapping[str, Any], output_path: str | Path, *, n: int) -> dict:
    rows = [["target", "n", "positive_depth_mean_bits", "including_zero_mean_bits",
             "difference_bits", "paired_se_bits", "trials"]]
    for target in ("uniform", "step"):
        cell = summary["targets"][target][str(n)]
        positive = cell["derived"]["positive_depth_mixture"]
        mixture = cell["methods"]["lsa_depth_mixture"]
        delta = cell["paired_differences"]["lsa_depth_mixture_minus_positive_depth_mixture"]
        if any(c["status"] != "finite" for c in (positive, mixture, delta)):
            raise ValueError("ablation contains a nonfinite loss")
        rows.append([target, n, positive["mean_bits"], mixture["mean_bits"],
                     delta["mean_bits"], delta["se_bits"], delta["trials"]])
    _write_tsv(Path(output_path), rows)
    return {"role": "uniform_component_ablation", "source": summary.get("source"), "n": n,
            "samples": "same saved primary samples, same positive component evidences"}


def generate_reports(primary_dir: str | Path, power_dir: str | Path | None,
                     output_dir: str | Path, *, config: Mapping[str, Any]) -> dict:
    """Generate selected current manuscript roles; missing sources fail loudly.

    config declares roles, table_n, posterior_n_values, posterior_mass,
    posterior_broad_min_depths, posterior_individual_min_weight, uniform_floor.
    Paper role grids use all eleven targets and exactly the manuscript methods.
    """
    required = {"roles", "table_n", "posterior_n_values", "posterior_mass",
                "posterior_broad_min_depths", "posterior_individual_min_weight", "uniform_floor"}
    if required - config.keys():
        raise ValueError(f"missing report settings {sorted(required - config.keys())}")
    allowed = {"tab:bench", "fig:orlitsky", "tab:posteriors", "tab:powers-bench",
               "paired_differences", "uniform_component_ablation"}
    roles = list(config["roles"])
    if not roles or len(set(roles)) != len(roles) or set(roles) - allowed:
        raise ValueError("report roles are empty, duplicated, or outside current manuscript scope")
    primary = load_summary(primary_dir)
    power = None
    if set(roles) & {"tab:powers-bench", "paired_differences"}:
        if power_dir is None:
            raise ValueError("power-replication report requires its saved run")
        power = load_summary(power_dir)
        if primary["config"]["seed"] == power["config"]["seed"]:
            raise ValueError("independent sample sets must have distinct seeds")
        if primary["config"]["sample_set_id"] == power["config"]["sample_set_id"]:
            raise ValueError("independent sample sets must have distinct IDs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "report_config": dict(config), "roles": {}}
    for role in roles:
        if role == "tab:bench":
            detail = write_loss_table(primary, output_dir / "tab-bench", n=config["table_n"],
                                      targets=TARGET_IDS, methods=PRIMARY_METHODS, label=role)
        elif role == "tab:powers-bench":
            detail = write_loss_table(power, output_dir / "tab-powers-bench", n=config["table_n"],
                                      targets=TARGET_IDS, methods=POWER_METHODS, label=role)
        elif role == "fig:orlitsky":
            detail = plot_learning_curves(primary, output_dir / "fig-orlitsky.pdf", targets=TARGET_IDS,
                                          methods=PRIMARY_METHODS, uniform_floor=config["uniform_floor"])
        elif role == "tab:posteriors":
            detail = write_posterior_table(
                primary, output_dir / "tab-posteriors", targets=TARGET_IDS,
                n_values=config["posterior_n_values"], mass=config["posterior_mass"],
                broad_min_depths=config["posterior_broad_min_depths"],
                individual_min_weight=config["posterior_individual_min_weight"],
            )
        elif role == "paired_differences":
            detail = write_paired_differences(power, output_dir / "paired-differences.tsv",
                                              targets=TARGET_IDS, n=config["table_n"])
        else:
            detail = write_uniform_ablation(primary, output_dir / "uniform-component-ablation.tsv",
                                             n=config["table_n"])
        metadata["roles"][role] = detail
    metadata["artifacts"] = [{"path": p.name, "sha256": sha256_file(p)}
                             for p in sorted(output_dir.iterdir()) if p.is_file()]
    with (output_dir / "benchmark-report.json").open("x", encoding="utf8") as stream:
        json.dump(jsonable(metadata), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return metadata
