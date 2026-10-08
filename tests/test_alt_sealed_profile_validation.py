"""Paired-provider numerical gates and immutable high-depth diagnostic runs."""

import json
import subprocess
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lsa.alt.artifacts import read_json, sha256, verify_run
from lsa.alt.depth import StoreConfig
from lsa.alt.sealed_profile_validation import (
    REFERENCE_PROTOCOL,
    assess_measurements,
    compare_profiles,
    run_sealed_profile_validation,
    validate_config,
)
from lsa.alt.sealed_store_build import build_store, create_plan, seal_store, write_plan


def _config():
    return {"schema_version": 2, "reference": deepcopy(REFERENCE_PROTOCOL),
            "purpose": "validation", "loss_tolerance_bits": 1e-5,
            "raw_mass_tolerance": 1e-7, "cases": [{
                "id": "tiny-low-and-high-depth", "kind": "explicit", "d": 10000,
                "n": 3, "profile": [2, 1], "target": "uniform",
                "depths": [0, 1, 2, 9, 54, 138], "predictive": True,
            }]}


def _prediction(scale=1., evidence_shift=0.):
    return SimpleNamespace(
        depths=(0, 1), counts=(0, 1), multiplicities=(3, 1),
        component_log_evidence=np.array([-20., -22.]) + evidence_shift,
        component_probabilities=np.array([[.1, .7], [.2, .4]]) * scale,
        mixture_probabilities=np.array([.15, .55]) * scale,
        posterior=np.array([.5, .5]),
    )


def test_loss_normalization_cannot_hide_bad_raw_mass():
    result = compare_profiles(_prediction(), _prediction(scale=1.00001),
                              n=100, class_mass=[.8, .2], multiplicities=[3, 1])
    assert result["maximum_predictive_kl_change_bits"] < 1e-14
    assert result["maximum_raw_mass_error"] == pytest.approx(1e-5)
    gates = assess_measurements(result, predictive=True, config=_config())
    assert not gates["maximum_raw_mass_error"]
    assert gates["maximum_predictive_kl_change_bits"]
    assert result["candidate_component_raw_mass"] == pytest.approx([1.00001, 1.00001])


def test_total_bits_are_reported_separately_from_per_token_gate():
    result = compare_profiles(_prediction(), _prediction(evidence_shift=1e-4),
                              n=100, class_mass=[.8, .2], multiplicities=[3, 1])
    assert result["maximum_evidence_difference_bits"] > 1e-5
    assert result["maximum_codelength_difference_bits_per_token"] < 1e-5
    assert all(assess_measurements(result, predictive=True, config=_config()).values())
    result["mixture_predictive_kl_change_bits"] = 1e-3
    assert not assess_measurements(result, predictive=True, config=_config())["mixture_predictive_kl_change_bits"]
    result["maximum_codelength_difference_bits_per_token"] = float("nan")
    assert not assess_measurements(result, predictive=True, config=_config())["maximum_codelength_difference_bits_per_token"]


@pytest.mark.parametrize("key,value", [("loss_tolerance_bits", 2e-5), ("raw_mass_tolerance", 1e-3)])
def test_nominal_validation_tolerances_cannot_be_relaxed(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError, match="no greater"):
        validate_config(config)


@pytest.fixture(scope="module")
def paired_engines(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-sealed-profiles")
    sealed = root / "sealed"
    levels = [2, 9, 54, 138]
    write_plan(create_plan(levels=levels, support_max_count=8,
                           anchors={L: list(range(9)) for L in levels}), sealed)
    build_store(sealed)
    seal_store(sealed)
    hashes = {p.name: sha256(p) for p in sealed.iterdir()
              if p.name in ("plan.json", "manifest.json") or p.name.startswith("level_")}
    candidate = StoreConfig(path=str(sealed), files_sha256=hashes, format="sealed",
                            max_depth=138, max_d=10000, max_n=4, max_count=4)
    legacy = root / "legacy"
    legacy.mkdir()
    (legacy / "manifest.json").write_text(json.dumps({"version": "v2", "H": .02, "U_MAX": 35.}))
    direct = replace(candidate, path=str(legacy), format="legacy", ladder_every=0,
                     saddle_min_depth=2, files_sha256={"manifest.json": sha256(legacy / "manifest.json")})
    return ({"mode": "store", "prediction_tolerance": 1e-3, "store": asdict(candidate)},
            {"mode": "store", "prediction_tolerance": 1e-3, "store": asdict(direct)})


def _test_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "fixed-input.txt").write_text("isolated Run source identity fixture\n")
    subprocess.run(["git", "add", "fixed-input.txt"], cwd=path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=path, check=True)
    return path


def test_actual_low_and_high_depth_pair_saved_as_immutable_normal_run(tmp_path, paired_engines):
    candidate, legacy = paired_engines
    out, repo = tmp_path / "run", _test_repo(tmp_path / "repo")
    result = run_sealed_profile_validation(_config(), out, candidate_engine=candidate,
                                           legacy_engine=legacy, repo=repo)
    assert result["status"] == "passed"
    record = verify_run(out)
    assert record["experiment"] == "calibration_sealed_profile"
    assert record["purpose"] == "validation"
    assert read_json(out / "result.json") == result
    assert result["reference"] == REFERENCE_PROTOCOL
    case = result["cases"][0]
    assert case["augmented_profiles"] == {"0": [2, 1, 1], "1": [2, 2], "2": [3, 1]}
    assert case["count_classes"] == [0, 1, 2]
    assert all(case["gates"].values())
    assert case["measurements"]["maximum_raw_mass_error"] < 1e-7
    assert read_json(out / "data/candidate-engine.json")["store"]["format"] == "sealed"
    assert read_json(out / "data/legacy-engine.json")["store"]["saddle_min_depth"] == 2
    assert (out / "data/evaluation-000.json.gz").is_file()
    with np.load(out / "data/case-000.npz") as sample:
        assert sample["counts"].sum() == 3
        assert sample["counts"][:3].tolist() == [2, 1, 0]
    with pytest.raises(FileExistsError):
        run_sealed_profile_validation(_config(), out, candidate_engine=candidate,
                                      legacy_engine=legacy, repo=repo)


def test_different_outer_settings_are_rejected_before_run(tmp_path, paired_engines):
    candidate, legacy = deepcopy(paired_engines)
    candidate["store"]["grid_step"] = .01
    with pytest.raises(ValueError, match="identical outer"):
        run_sealed_profile_validation(_config(), tmp_path / "run", candidate_engine=candidate,
                                      legacy_engine=legacy, repo=tmp_path)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("cutoff", [54, None])
def test_historical_hybrid_reference_is_rejected_before_run(tmp_path, paired_engines, cutoff):
    candidate, legacy = deepcopy(paired_engines)
    legacy["store"]["saddle_min_depth"] = cutoff
    legacy["store"]["max_depth"] = 54
    with pytest.raises(ValueError, match="direct kernels from depth2"):
        run_sealed_profile_validation(_config(), tmp_path / "run", candidate_engine=candidate,
                                      legacy_engine=legacy, repo=tmp_path)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("change", ["version", "missing", "hybrid"])
def test_reference_protocol_must_be_explicit_and_all_direct(change):
    config = _config()
    if change == "version":
        config["schema_version"] = 1
    elif change == "missing":
        config.pop("reference")
    else:
        config["reference"]["minimum_direct_depth"] = 54
    with pytest.raises(ValueError, match="protocol v2"):
        validate_config(config)


def test_failed_comparison_keeps_measurements_in_failed_run(tmp_path, paired_engines, monkeypatch):
    from lsa.alt import sealed_profile_validation as validation

    real_compare = validation.compare_profiles

    def failing(*args, **kwargs):
        values = real_compare(*args, **kwargs)
        values["maximum_raw_mass_error"] = .001
        return values

    monkeypatch.setattr(validation, "compare_profiles", failing)
    candidate, legacy = paired_engines
    out, repo = tmp_path / "failed-run", _test_repo(tmp_path / "repo")
    with pytest.raises(ArithmeticError, match="validation failed"):
        run_sealed_profile_validation(_config(), out, candidate_engine=candidate,
                                      legacy_engine=legacy, repo=repo)
    assert verify_run(out, require_complete=False)["status"] == "failed"
    assert read_json(out / "result.json")["status"] == "failed"
    assert read_json(out / "data/result-000.json")["measurements"]["maximum_raw_mass_error"] == .001
    with pytest.raises(ValueError, match="run is failed"):
        verify_run(out)


def test_committed_plan_has_complete_spectrum_depths_and_original_draws():
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / "experiments/alt2027/sealed-profile-validation.json")
    original = read_json(root / "experiments/alt2027/depth-validation.json")
    validate_config(config)
    assert len(config["cases"]) == len(original["cases"]) + 2
    for prior, new in zip(original["cases"], config["cases"]):
        assert prior["id"] == new["id"]
        if "seed_coordinates" in prior:
            assert new["seed_coordinates"] == prior["seed_coordinates"]
        assert new["d"] == prior["d"] and new["n"] == prior["n"]
        if new["id"].startswith("spectrum-"):
            assert new["depths"] == list(range(139))
