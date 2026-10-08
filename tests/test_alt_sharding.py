"""Disjoint global-ID shards reproduce saved samples and analytic results."""

import gzip
import json

import numpy as np
import pytest

from lsa.alt.benchmark import (
    POWER_METHODS,
    PRIMARY_METHODS,
    aggregate_records,
    run_benchmark,
    validate_config,
)
from lsa.alt.benchmark_report import load_summary
from lsa.alt.depth import DepthEvaluator
from lsa.alt.scaling import (
    cells,
    report_scaling,
    run_scaling,
    selected_cells,
    summarize_scaling,
)


class EndpointPower:
    def predict(self, counts, *, powers):
        return DepthEvaluator().predict(counts, depths=powers)


def benchmark_config(power):
    return {
        "d": 8,
        "n_values": [3, 5, 8],
        "trials": 4,
        "seed": 8421,
        "sample_set_id": "powers" if power else "primary",
        "targets": ["uniform", "dirichlet_half"],
        "methods": list(POWER_METHODS if power else PRIMARY_METHODS),
        "depths": [0, 1],
        "fixed_depth": 1,
        "powers": [0, 1] if power else [],
        "normalize_numerical": True,
        "numerical_normalization_tolerance": 1e-3,
        "dirichlet_exponents": list(range(-24, 5)),
        "dirichlet_target_policy": "redraw_per_trial_shared_across_n",
        "sample_size_policy": "independent_multinomial_per_n",
    }


def benchmark_rows(path):
    return [
        json.loads(line) for line in (path / "trials.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("power", [False, True])
def test_benchmark_trial_target_and_n_shards_recombine_exactly(tmp_path, power):
    config = benchmark_config(power)
    engine = DepthEvaluator()
    power_engine = EndpointPower() if power else None
    whole = tmp_path / "whole"
    full = run_benchmark(
        config, whole, depth_evaluator=engine, power_evaluator=power_engine
    )
    original = {(r["target_id"], r["trial"], r["n"]): r for r in benchmark_rows(whole)}
    combined = []
    for target in reversed(config["targets"]):
        for start, count in ((0, 1), (1, 2), (3, 1)):
            for ns in ([3, 8], [5]):
                shard = tmp_path / f"{target}-{start}-{ns[0]}"
                local = {
                    **config,
                    "targets": [target],
                    "trial_start": start,
                    "trials": count,
                    "n_values": ns,
                    "sampling_n_values": config["n_values"],
                }
                result = run_benchmark(
                    local,
                    shard,
                    depth_evaluator=engine,
                    power_evaluator=power_engine,
                    batch_size=2,
                )
                assert load_summary(shard)["targets"] == result["targets"]
                manifest = json.loads((shard / "samples/manifest.json").read_text())
                assert [entry["trial"] for entry in manifest["files"]] == list(
                    range(start, start + count)
                )
                assert manifest["sampling"]["sampling_n_values"] == config["n_values"]
                for entry in manifest["files"]:
                    with (
                        np.load(shard / "samples" / entry["path"]) as got,
                        np.load(whole / "samples" / entry["path"]) as expected,
                    ):
                        for key in got.files:
                            np.testing.assert_array_equal(got[key], expected[key])
                for row in benchmark_rows(shard):
                    expected = original[row["target_id"], row["trial"], row["n"]]
                    for key in (
                        "sample_set_id",
                        "target_id",
                        "trial",
                        "n",
                        "sample_file",
                        "losses",
                        "posteriors",
                        "paired_differences",
                        "derived",
                    ):
                        assert row[key] == expected[key]
                    assert (
                        row["diagnostics"]["depth"]["component_log_evidence_nats"]
                        == expected["diagnostics"]["depth"][
                            "component_log_evidence_nats"
                        ]
                    )
                    if count == 1:
                        assert (
                            result["targets"][target][str(row["n"])]["methods"][
                                "lsa_depth_mixture"
                            ]["se_bits"]
                            is None
                        )
                    combined.append(row)
    assert (
        aggregate_records(list(reversed(combined)), config)["targets"]
        == full["targets"]
    )
    for altered in (combined[:-1], [*combined, combined[0]]):
        with pytest.raises(ValueError, match="global benchmark trial identities"):
            aggregate_records(altered, config)
    extra = [{**row, "trial": row["trial"] + 1} for row in combined]
    with pytest.raises(ValueError, match="global benchmark trial identities"):
        aggregate_records(extra, config)


def scaling_config(kind):
    base = {"seed": 927, "trials": 5, "alphas": [0, 2]}
    if kind == "spectrum":
        return {
            **base,
            "panels": [
                {"d": 6, "n": 4, "max_depth": 1, "show_depths": [0, 1]},
                {"d": 8, "n": 5, "max_depth": 1, "show_depths": [0, 1]},
            ],
        }
    if kind == "factorial":
        return {
            **base,
            "alphas": [2],
            "ds": [4, 8],
            "ns": [3, 5],
            "table_ds": [4, 8],
            "table_ns": [3, 5],
            "depth_rule": "fixed",
            "fixed_depth": 1,
        }
    return {
        **base,
        "d": 8,
        "n": 6,
        "c_values": [0.25, 0.5],
        "heuristic_offset": "alpha_log2_e_minus_2",
    }


def scaling_rows(path, prefix):
    rows = []
    for file in sorted(path.glob(f"{prefix}-*.jsonl.gz")):
        with gzip.open(file, "rt") as stream:
            rows.extend(json.loads(line) for line in stream)
    return {(row["cell_id"], row["trial"]): row for row in rows}


def write_scaling_rows(path, cell_id, rows):
    with gzip.open(path / f"trials-{cell_id:04d}.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


@pytest.mark.parametrize("kind", ["spectrum", "factorial", "depth_scaling"])
def test_scaling_cell_and_global_trial_shards_recombine_exactly(tmp_path, kind):
    config = scaling_config(kind)
    engine = DepthEvaluator()
    whole = tmp_path / "whole"
    full = run_scaling(kind, config, whole, evaluator=engine)
    original_samples, original_results = (
        scaling_rows(whole, "samples"),
        scaling_rows(whole, "trials"),
    )
    combined_samples, combined_results = {}, {}
    total_cells = len(list(cells(kind, config)))
    selectors = [
        list(reversed(range(0, total_cells, 2))),
        list(range(1, total_cells, 2)),
    ]
    for group, selector in enumerate(selectors):
        for start, count in ((0, 2), (2, 2), (4, 1)):
            shard = tmp_path / f"shard-{group}-{start}"
            local = {
                **config,
                "cell_ids": selector,
                "trial_start": start,
                "trials": count,
            }
            assert [i for i, _ in selected_cells(kind, local)] == sorted(selector)
            result = run_scaling(kind, local, shard, evaluator=engine, batch_size=2)
            assert [cell["cell_id"] for cell in result["cells"]] == sorted(selector)
            assert result["config"]["trial_start"] == start
            for cell in result["cells"]:
                assert cell["trials"] == count
                if count == 1:
                    assert cell["mixture_se_bits"] is None
                    assert cell["regret_se_bits"] == [None] * len(cell["depths"])
            samples, results = (
                scaling_rows(shard, "samples"),
                scaling_rows(shard, "trials"),
            )
            assert not (samples.keys() & combined_samples.keys())
            combined_samples.update(samples)
            combined_results.update(results)
            with pytest.raises(ValueError, match="complete scaling cell grid"):
                report_scaling(shard, tmp_path / f"rejected-report-{group}-{start}")
    assert combined_samples == original_samples
    assert combined_results.keys() == original_results.keys()
    for identity, row in combined_results.items():
        assert {key: value for key, value in row.items() if key != "diagnostics"} == {
            key: value
            for key, value in original_results[identity].items()
            if key != "diagnostics"
        }
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text(json.dumps({"kind": kind, **config}))
    for cell_id in range(total_cells):
        rows = [
            row
            for (cell, _), row in sorted(combined_results.items(), reverse=True)
            if cell == cell_id
        ]
        write_scaling_rows(merged, cell_id, rows)
    assert summarize_scaling(merged) == full
    path = merged / "trials-0000.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        first_rows = [json.loads(line) for line in stream]
    for altered in (first_rows[:-1], [first_rows[0], *first_rows[:-1]]):
        write_scaling_rows(merged, 0, altered)
        with pytest.raises(ValueError, match="trial group"):
            summarize_scaling(merged)


@pytest.mark.parametrize(
    "change",
    [
        {"trial_start": -1},
        {"trial_start": True},
        {"sampling_n_values": [5, 3, 8]},
        {"sampling_n_values": [3, 5]},
    ],
)
def test_invalid_benchmark_shard_config_rejected(change):
    with pytest.raises((ValueError, TypeError)):
        validate_config({**benchmark_config(False), **change})


@pytest.mark.parametrize(
    "change",
    [
        {"trial_start": -1},
        {"trial_start": True},
        {"cell_ids": []},
        {"cell_ids": [0, 0]},
        {"cell_ids": [4]},
        {"cell_ids": [False]},
    ],
)
def test_invalid_scaling_shard_config_rejected_before_artifacts(tmp_path, change):
    with pytest.raises((ValueError, TypeError)):
        run_scaling(
            "factorial",
            {**scaling_config("factorial"), **change},
            tmp_path / "invalid",
            evaluator=DepthEvaluator(),
        )
    assert not (tmp_path / "invalid").exists()
