import gzip
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from lsa.alt import depth_validation_assessment as assessment_module
from lsa.alt.artifacts import sha256
from lsa.alt.depth import DepthEvaluator, EvidenceResult, StoreConfig
from lsa.alt.depth_validation import comparison
from lsa.alt.depth_validation_assessment import (
    _merge,
    assess_depth_run,
    gate_measurements,
    grid_assessment,
)


def test_mixture_loss_gate_catches_posterior_shift_with_unchanged_components():
    components = np.array([[0.9, 0.1], [0.1, 0.9]])

    def result(logs):
        logs = np.asarray(logs)
        weights = np.exp(logs - logsumexp(logs))
        return SimpleNamespace(
            component_log_evidence=logs,
            component_probabilities=components,
            mixture_probabilities=weights @ components,
            posterior=weights,
        )

    values = comparison(
        result([-1000, -1000]),
        result([-999.994, -1000.006]),
        n=1000,
        class_mass=[1, 0],
        multiplicities=[1, 1],
    )
    assert values["maximum_predictive_kl_change_bits"] == 0
    assert values["maximum_codelength_difference_bits_per_token"] < 1e-5
    assert values["mixture_predictive_kl_change_bits"] > 0.006
    checked = gate_measurements(
        values,
        {"loss_tolerance_bits": 1e-5, "raw_mass_tolerance": 1e-7},
        predictive=True,
    )
    assert checked == {
        "status": "failed",
        "failed_checks": ["mixture_predictive_kl_change_bits"],
    }


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1e-10, None])
def test_invalid_loss_cannot_pass(bad):
    values = {
        "maximum_codelength_difference_bits_per_token": 0.0,
        "mixture_codelength_difference_bits_per_token": bad,
    }
    assert (
        gate_measurements(values, {"loss_tolerance_bits": 1e-5}, predictive=False)[
            "status"
        ]
        == "failed"
    )


def _evidence(steps):
    return {
        "depths": [0, 2, 3],
        "log_evidence": [-10.0, -9.0, -11.0],
        "diagnostics": {
            "components": [
                {"depth": 0},
                *[
                    {"depth": depth, "outer_grid_step": step}
                    for depth, step in zip([2, 3], steps, strict=True)
                ],
            ]
        },
    }


def test_actual_spacing_controls_refinement_coverage():
    rows = grid_assessment(_evidence([0.01, 0.02]), _evidence([0.01, 0.01]))
    assert [r["status"] for r in rows] == ["analytic", "pending", "halved"]
    assert rows[1]["supplement_step"] == 0.005


def test_supplement_restores_original_full_grid_mixture():
    base = _evidence([0.01, 0.02])
    base["component_log_evidence"] = base.pop("log_evidence")
    base.update(
        counts=[0, 1],
        multiplicities=[1, 1],
        component_probabilities=[[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]],
    )
    extra = {
        "depths": [2],
        "component_log_evidence": [-10.0],
        "counts": [0, 1],
        "multiplicities": [1, 1],
        "component_probabilities": [[0.8, 0.2]],
        "diagnostics": {"components": [{"depth": 2, "outer_grid_step": 0.005}]},
    }
    merged = _merge(base, extra, predictive=True)
    logs = np.array([-10.0, -10.0, -11.0])
    weights = np.exp(logs - logsumexp(logs))
    np.testing.assert_allclose(merged["posterior"], weights)
    np.testing.assert_allclose(
        merged["mixture_probabilities"],
        weights @ np.array([[0.5, 0.5], [0.8, 0.2], [0.1, 0.9]]),
    )
    assert merged["depths"] == [0, 2, 3]
    assert merged["diagnostics"]["components"][1]["outer_grid_step"] == 0.005


def test_unfinished_run_stays_pending_and_assessment_is_immutable(tmp_path):
    run = tmp_path / "running"
    run.mkdir()
    config = {
        "seed": 7,
        "cases": [{"id": "uncomputed", "kind": "synthetic", "target": "uniform"}],
    }
    (run / "config.json").write_text(json.dumps(config))
    output = tmp_path / "assessment"
    result = assess_depth_run(run, output, engine_config={"store": {}})
    assert result["status"] == "pending"
    assert result["completed_cases_read"] == 0
    assert result["pending_case_ids"] == ["uncomputed"]
    assert result["source_unchanged"]
    with pytest.raises(FileExistsError):
        assess_depth_run(run, output, engine_config={"store": {}})


def test_saved_case_supplements_only_missing_depth_and_keeps_pending_cases(
    tmp_path, monkeypatch
):
    run = tmp_path / "running"
    shard = run / "shard-0"
    shard.mkdir(parents=True)
    case = {
        "id": "ready",
        "kind": "synthetic",
        "target": "uniform",
        "d": 2,
        "n": 2,
        "predictive": False,
        "depths": [0, 2],
        "seed_coordinates": [7, 0],
    }
    pending = dict(case, id="pending", seed_coordinates=[7, 1])
    config = {
        "seed": 7,
        "cases": [case, pending],
        "loss_tolerance_bits": 1e-5,
        "raw_mass_tolerance": 1e-7,
    }
    (run / "config.json").write_text(json.dumps(config))
    (shard / "config.json").write_text(json.dumps(dict(config, cases=[case])))
    module_dir = Path(assessment_module.__file__).parent
    (shard / "implementation.json").write_text(
        json.dumps(
            {
                name: sha256(module_dir / name)
                for name in ("depth.py", "depth_validation.py")
            }
        )
    )
    store = StoreConfig(
        path=tmp_path, files_sha256={}, max_depth=2, max_d=2, max_n=3, max_count=3
    )
    settings = asdict(store)
    settings.pop("path")
    identity = dict(DepthEvaluator().configuration, mode="store", store=settings)
    fine_identity = dict(identity, store=dict(settings, grid_step=0.01))
    for name, value in [("default", identity), ("refined", fine_identity)]:
        (shard / f"{name}-engine.json").write_text(json.dumps(value))
    np.savez_compressed(
        shard / "case-000.npz", counts=np.array([2, 0]), target=np.array([0.5, 0.5])
    )
    result = {
        "id": "ready",
        "case": case,
        "status": "passed",
        "sample_sha256": sha256(shard / "case-000.npz"),
    }
    (shard / "result-000.json").write_text(json.dumps(result))
    evaluation = {
        "depths": [0, 2],
        "log_evidence": [-2 * math.log(2), -1.0],
        "diagnostics": {
            "components": [{"depth": 0}, {"depth": 2, "outer_grid_step": 0.01}]
        },
    }
    with gzip.open(shard / "evaluation-000.json.gz", "wt") as stream:
        json.dump({"default": evaluation, "refined": evaluation}, stream)
    calls = []

    class MockEvaluator:
        def __init__(self, *, mode, store, prediction_tolerance):
            self.store = store
            self.configuration = fine_identity

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def evidence_at_depths(self, d, profile, depths):
            calls.append((d, profile, depths, self.store.grid_step))
            return EvidenceResult(
                tuple(depths),
                np.array([-1.0 + 1e-8]),
                {"components": [{"depth": 2, "outer_grid_step": self.store.grid_step}]},
            )

    monkeypatch.setattr(assessment_module, "DepthEvaluator", MockEvaluator)
    before = sha256(shard / "evaluation-000.json.gz")
    summary = assess_depth_run(
        run,
        tmp_path / "assessed",
        engine_config={"store": asdict(store)},
        run_supplemental=True,
    )
    assert calls == [(2, (2,), [2], 0.005)]
    assert summary["cases"][0]["status"] == "passed"
    assert summary["status"] == "pending"
    assert summary["pending_case_ids"] == ["pending"]
    assert sha256(shard / "evaluation-000.json.gz") == before
