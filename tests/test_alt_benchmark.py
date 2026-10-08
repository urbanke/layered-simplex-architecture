"""Small saved-sample integration tests; no production numerical campaign."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from lsa.alt.baselines import DIRICHLET_EXPONENTS, add_constant, dirichlet_log_evidence
from lsa.alt.benchmark import (
    POWER_METHODS,
    PRIMARY_METHODS,
    TARGET_IDS,
    loss_record,
    make_target,
    prepare_samples,
    run_benchmark,
    summarize_losses,
    validate_config,
)
from lsa.alt.benchmark_report import generate_reports, load_summary, posterior_summary


class EndpointEvaluator:
    """Analytic zero/one endpoints, shared by the depth and power families."""

    def predict(self, counts, *, depths=None, powers=None):
        grid = depths if depths is not None else powers
        assert list(grid) == [0, 1]
        d, n = len(counts), sum(counts)
        logs = np.array([-n * np.log(d), dirichlet_log_evidence(counts, 1)])
        components = np.array([np.full(d, 1 / d), add_constant(counts, 1)])
        weights = np.exp(logs - logsumexp(logs))
        return SimpleNamespace(component_probabilities=components,
                               mixture_probabilities=weights @ components,
                               posterior=weights, component_log_evidence=logs,
                               diagnostics={"test_fixture": "analytic_uniform_and_Laplace"})


def config(*, seed=140, sample_set_id="primary", methods=PRIMARY_METHODS, targets=TARGET_IDS):
    return {"d": 8, "n_values": [3, 7], "trials": 2, "seed": seed, "sample_set_id": sample_set_id,
                "targets": list(targets), "methods": list(methods), "depths": [0, 1], "fixed_depth": 1,
                "powers": [0, 1], "normalize_numerical": True, "numerical_normalization_tolerance": 1e-3,
                "dirichlet_exponents": list(DIRICHLET_EXPONENTS),
                "dirichlet_target_policy": "redraw_per_trial_shared_across_n",
                "sample_size_policy": "independent_multinomial_per_n"}


def test_all_eleven_targets_and_exact_step():
    rng = np.random.default_rng(3)
    for target in TARGET_IDS:
        p = make_target(target, 8, rng)
        assert p.sum() == pytest.approx(1)
        assert np.all(p > 0)
    np.testing.assert_allclose(make_target("step", 8, rng), [1 / 16] * 4 + [3 / 16] * 4)
    with pytest.raises(ValueError):
        make_target("step", 7, rng)


def test_common_saved_samples_stable_under_target_selection(tmp_path):
    full = config()
    subset = config(targets=["dirichlet_half", "step"])
    first = prepare_samples(full, tmp_path / "full")
    second = prepare_samples(subset, tmp_path / "subset")
    assert first["seed_scheme"] == second["seed_scheme"]
    for entry in second["files"]:
        with np.load(tmp_path / "full" / entry["path"]) as a, np.load(tmp_path / "subset" / entry["path"]) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            assert a["counts_3"].sum() == 3 and a["counts_7"].sum() == 7
    with pytest.raises(FileExistsError):
        prepare_samples(subset, tmp_path / "subset")


def test_run_saves_every_loss_posterior_and_paired_difference(tmp_path):
    settings = config(methods=POWER_METHODS, targets=["uniform", "step"])
    evaluator = EndpointEvaluator()
    result = run_benchmark(settings, tmp_path / "run", depth_evaluator=evaluator, power_evaluator=evaluator)
    records = [json.loads(line) for line in (tmp_path / "run/trials.jsonl").read_text().splitlines()]
    assert len(records) == 8
    for record in records:
        assert set(record["losses"]) == set(POWER_METHODS)
        assert record["posteriors"]["depth"]["indices"] == [0, 1]
        assert record["diagnostics"]["lsa_depth_mixture"]["raw_mass"] == pytest.approx(1)
        assert record["paired_differences"]["power_mixture_minus_lsa_depth_mixture"]["bits"] == 0
        # Excluding the atom leaves exactly the analytic add-one component.
        assert record["derived"]["positive_depth_mixture"]["loss"]["bits"] == pytest.approx(record["losses"]["add_one"]["bits"])
    assert load_summary(tmp_path / "run")["targets"] == result["targets"]
    with pytest.raises(FileExistsError):
        run_benchmark(settings, tmp_path / "run", depth_evaluator=evaluator, power_evaluator=evaluator)


def test_nonfinite_aggregate_never_drops_trials():
    result = summarize_losses([loss_record(1), loss_record(float("inf"))])
    assert result["status"] == "infinite" and result["mean_bits"] is None
    result = summarize_losses([loss_record(1), loss_record(None, reason="undefined discount")])
    assert result["status"] == "undefined" and result["counts"]["undefined"] == 1
    single = summarize_losses([loss_record(1)])
    assert single["se_bits"] is None
    paired = summarize_losses([loss_record(1), loss_record(3)])
    assert paired["mean_bits"] == 2 and paired["se_bits"] == pytest.approx(1)


def test_missing_enabled_evaluators_fail_before_artifacts(tmp_path):
    with pytest.raises(ValueError, match="no depth evaluator"):
        run_benchmark(config(), tmp_path / "bad", depth_evaluator=None)
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="no power evaluator"):
        run_benchmark(config(methods=POWER_METHODS), tmp_path / "bad", depth_evaluator=EndpointEvaluator())


def test_numerical_gate_fails_loudly(tmp_path):
    class Broken(EndpointEvaluator):
        def predict(self, *args, **kwargs):
            result = super().predict(*args, **kwargs)
            result.component_probabilities *= 1.01
            result.mixture_probabilities *= 1.01
            return result
    with pytest.raises(ArithmeticError, match="normalization error"):
        run_benchmark(config(targets=["uniform"]), tmp_path / "bad", depth_evaluator=Broken())
    assert not (tmp_path / "bad/summary.json").exists()


def test_saved_sample_tampering_is_detected(tmp_path):
    settings = config(targets=["uniform"])
    manifest = prepare_samples(settings, tmp_path / "samples")
    path = tmp_path / "samples" / manifest["files"][0]["path"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_benchmark(settings, tmp_path / "run", depth_evaluator=EndpointEvaluator(), samples_dir=tmp_path / "samples")


def test_posterior_interval_rule_and_tie_break():
    result = posterior_summary(range(7), [.01, .04, .25, .25, .25, .19, .01],
                               mass=.95, broad_min_depths=5, individual_min_weight=.05)
    assert result["interval"] == [1, 5]
    assert result["kind"] == "interval" and result["enclosed_mass"] == pytest.approx(.98)
    # Both width-2 windows exceed .7; choose the one enclosing .8, not .75.
    result = posterior_summary(range(3), [.2, .55, .25], mass=.7,
                               broad_min_depths=5, individual_min_weight=.05)
    assert result["interval"] == [1, 2]
    assert result["kind"] == "individual"
    with pytest.raises(ValueError, match="contiguous"):
        posterior_summary([0, 2], [.5, .5], mass=.95,
                          broad_min_depths=5, individual_min_weight=.05)


def test_reports_use_current_roles_and_shared_trial_records(tmp_path):
    evaluator = EndpointEvaluator()
    run_benchmark(config(), tmp_path / "primary", depth_evaluator=evaluator)
    run_benchmark(config(seed=141, sample_set_id="powers", methods=POWER_METHODS),
                  tmp_path / "powers", depth_evaluator=evaluator, power_evaluator=evaluator)
    report_config = {"roles": ["tab:bench", "fig:orlitsky", "tab:posteriors",
                               "tab:powers-bench", "paired_differences", "uniform_component_ablation"],
                         "table_n": 3, "posterior_n_values": [3, 7], "posterior_mass": .95,
                         "posterior_broad_min_depths": 5, "posterior_individual_min_weight": .05,
                         "uniform_floor": 1e-14}
    result = generate_reports(tmp_path / "primary", tmp_path / "powers", tmp_path / "reports",
                              config=report_config)
    assert set(result["roles"]) == set(report_config["roles"])
    assert result["roles"]["tab:bench"]["n"] == 3
    assert result["roles"]["fig:orlitsky"]["methods"] == list(PRIMARY_METHODS)
    assert (tmp_path / "reports/fig-orlitsky.pdf").stat().st_size > 1000
    assert "power_mixture_minus_lsa_depth_mixture" in (tmp_path / "reports/paired-differences.tsv").read_text()
    assert "braess" not in (tmp_path / "reports/tab-bench.tsv").read_text().lower()


def test_undeclared_config_does_not_get_silent_defaults():
    settings = config()
    del settings["sample_size_policy"]
    with pytest.raises(ValueError, match="missing explicit"):
        validate_config(settings)
