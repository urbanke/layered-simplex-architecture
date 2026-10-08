"""Traceability and grid-completeness checks using an analytic uniform fixture."""

import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lsa.alt.scaling import report_scaling, run_scaling, summarize_scaling


class UniformFixture:
    def evidence_at_depths(self, d, partition, depths):
        return SimpleNamespace(depths=tuple(depths),
                               log_evidence=np.full(len(depths), -sum(partition) * np.log(d)),
                               diagnostics={"test_fixture": "uniform sequence law at every index"})


def factorial_config():
    return {"ds": [8, 16], "ns": [4, 8], "alphas": [2], "trials": 2,
            "seed": 90, "depth_rule": "fixed", "fixed_depth": 0,
            "table_ds": [8, 16], "table_ns": [4, 8]}


def test_failed_evaluator_leaves_all_common_profiles_saved(tmp_path):
    class Failure:
        def evidence_at_depths(self, d, partition, depths):
            raise ArithmeticError("intentional numerical failure")
    with pytest.raises(ArithmeticError, match="intentional"):
        run_scaling("factorial", factorial_config(), tmp_path / "run", evaluator=Failure())
    with gzip.open(tmp_path / "run/samples-0000.jsonl.gz", "rt") as stream:
        draws = [json.loads(line) for line in stream]
    assert [r["trial"] for r in draws] == [0, 1]
    assert all(sum(r["counts"]) == 4 for r in draws)
    assert not (tmp_path / "run/summary.json").exists()


def test_full_factorial_uniform_fixture_has_zero_data_scaling_exponent(tmp_path):
    result = run_scaling("factorial", factorial_config(), tmp_path / "run", evaluator=UniformFixture())
    assert len(result["cells"]) == 4
    report_scaling(tmp_path / "run", tmp_path / "report")
    table = json.loads((tmp_path / "report/scaling-tables.json").read_text())
    assert all(r["exponent"] == pytest.approx(0, abs=1e-14) for r in table["data"])
    # A missing whole cell cannot quietly change either fitted table.
    (tmp_path / "run/trials-0003.jsonl.gz").unlink()
    with pytest.raises(ValueError, match="cell files"):
        summarize_scaling(tmp_path / "run")


def test_batched_scaling_preserves_draws_and_losses(tmp_path):
    from lsa.alt.depth import DepthEvaluator

    config = factorial_config()
    config['trials'] = 5
    config['fixed_depth'] = 1
    with DepthEvaluator() as evaluator:
        single = run_scaling('factorial', config, tmp_path / 'single', evaluator=evaluator)
        batched = run_scaling('factorial', config, tmp_path / 'batch', evaluator=evaluator, batch_size=3)
    assert single == batched
    for index in range(4):
        with (
            gzip.open(tmp_path / f'single/samples-{index:04d}.jsonl.gz', 'rt') as a,
            gzip.open(tmp_path / f'batch/samples-{index:04d}.jsonl.gz', 'rt') as b,
        ):
            assert a.read() == b.read()


@pytest.mark.parametrize("mutation", ["wrong_alpha", "duplicate_trial", "wrong_n"])
def test_trial_metadata_must_match_the_complete_declared_grid(tmp_path, mutation):
    run_scaling("factorial", factorial_config(), tmp_path / "run", evaluator=UniformFixture())
    path = tmp_path / "run/trials-0000.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if mutation == "wrong_alpha":
        rows[0]["alpha"] = 3
    elif mutation == "duplicate_trial":
        rows[1]["trial"] = 0
    else:
        rows[0]["n"] = 5
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="trial|metadata"):
        summarize_scaling(tmp_path / "run")


def test_undeclared_heuristic_is_not_silently_substituted(tmp_path):
    settings = {"d": 8, "n": 4, "alphas": [2], "trials": 2, "seed": 90,
                "c_values": [.25, .5], "heuristic_offset": "unspecified"}
    with pytest.raises(ValueError, match="heuristic"):
        run_scaling("depth_scaling", settings, tmp_path / "run", evaluator=UniformFixture())
    assert not (tmp_path / "run").exists()
