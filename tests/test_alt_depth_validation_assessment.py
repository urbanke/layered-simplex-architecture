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


@pytest.fixture(scope="module")
def real_native_depth_case(tmp_path_factory):
    """Use actual stored kernels, a compiled reader and a saved depth comparison."""
    import shutil

    from lsa.alt.depth_validation import run_depth_validation
    from lsa.alt.sealed_native import build_native
    from lsa.alt.sealed_store_build import (
        build_store,
        create_plan,
        seal_store,
        write_plan,
    )

    if shutil.which("cc") is None:
        pytest.skip("native depth assessment requires an explicit C build")
    root = tmp_path_factory.mktemp("native-depth-assessment")
    store_path = root / "store"
    write_plan(
        create_plan(levels=[2], support_max_count=8, anchors={2: list(range(9))}),
        store_path,
    )
    build_store(store_path)
    seal_store(store_path)
    library = root / "sealed.so"
    built = build_native(library)
    files = {p.name: sha256(p) for p in store_path.iterdir() if p.is_file()}
    store = StoreConfig(
        path=str(store_path),
        files_sha256=files,
        max_depth=2,
        max_d=3,
        max_n=4,
        max_count=4,
        format="sealed",
        interpolation_backend="native",
        native_library_path=str(library),
        native_library_sha256=built["binary_sha256"],
    )
    config = {
        "seed": 7,
        "loss_tolerance_bits": 1e-5,
        "raw_mass_tolerance": 1e-7,
        # The same actual spacing deliberately leaves one required halving.
        "refinement": {"grid_step": store.grid_step},
        "cases": [
            {
                "id": "native",
                "kind": "synthetic",
                "target": "uniform",
                "d": 3,
                "n": 3,
                "predictive": False,
                "depths": [0, 1, 2],
                "seed_coordinates": [7, 0, 0],
            }
        ],
    }
    engine = {"mode": "store", "store": asdict(store)}
    run = root / "saved-depth"
    result = run_depth_validation(
        config, run, engine_config=engine, repo=Path(__file__).resolve().parents[1]
    )
    assert result["status"] == "passed"
    return run, engine


def test_native_depth_assessment_restores_relocated_real_path_and_halves_grid(
    real_native_depth_case,
    tmp_path,
    monkeypatch,
):
    import copy
    import shutil

    from lsa.alt.sealed_native import NativeInterpolator, manifest_path

    run, original_engine = real_native_depth_case
    engine = copy.deepcopy(original_engine)
    original_library = Path(engine["store"]["native_library_path"])
    library = tmp_path / "relocated.so"
    shutil.copyfile(original_library, library)
    shutil.copyfile(manifest_path(original_library), manifest_path(library))
    engine["store"]["native_library_path"] = str(library)
    before = {str(p): sha256(p) for p in run.iterdir() if p.is_file()}
    paths = []
    initialize = NativeInterpolator.__init__

    def recorded_init(self, path, sha256):
        paths.append(str(path))
        initialize(self, path, sha256)

    monkeypatch.setattr(NativeInterpolator, "__init__", recorded_init)
    out = tmp_path / "assessment"
    summary = assess_depth_run(run, out, engine_config=engine, run_supplemental=True)
    assert summary["status"] == "passed"
    row = summary["cases"][0]
    assert row["original_grid_checks"][2]["status"] == "pending"
    assert row["final_grid_checks"][2]["status"] == "halved"
    assert row["supplements"][0]["status"] == "evaluated"
    assert len(paths) >= 2 and set(paths) == {str(library)}
    supplemental = json.loads((out / "supplement-native-00-engine.json").read_text())
    saved = json.loads((run / "refined-engine.json").read_text())
    assert supplemental["store"]["native_library_path"] == "/host-local-native"
    assert supplemental["sealed_native_identity"] == saved["sealed_native_identity"]
    assert (out / "native-inputs/relocated.so").read_bytes() == library.read_bytes()
    assert before == {str(p): sha256(p) for p in run.iterdir() if p.is_file()}


@pytest.mark.parametrize(
    "which,field",
    [
        ("default", "binary_sha256"),
        ("refined", "wrapper_sha256"),
        ("refined", "build_manifest_sha256"),
        ("refined", "source_sha256"),
    ],
)
def test_native_depth_assessment_rejects_mismatched_saved_identity(
    real_native_depth_case,
    tmp_path,
    which,
    field,
):
    import shutil

    original, engine = real_native_depth_case
    run = tmp_path / "altered-record"
    shutil.copytree(original, run)
    path = run / f"{which}-engine.json"
    saved = json.loads(path.read_text())
    saved["sealed_native_identity"][field] = "0" * 64
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="native interpolation identity"):
        assess_depth_run(
            run, tmp_path / "assessment", engine_config=engine, run_supplemental=True
        )


def test_native_supplement_rejects_changed_actual_identity(
    real_native_depth_case,
    tmp_path,
    monkeypatch,
):
    run, engine = real_native_depth_case

    class ChangedEvaluator(DepthEvaluator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.configuration["sealed_native_identity"]["compiled_compiler"] = (
                "changed"
            )

    monkeypatch.setattr(assessment_module, "DepthEvaluator", ChangedEvaluator)
    summary = assess_depth_run(
        run, tmp_path / "assessment", engine_config=engine, run_supplemental=True
    )
    assert summary["status"] == "failed"
    supplement = summary["cases"][0]["supplements"][0]
    assert supplement["status"] == "failed"
    assert "sealed_native_identity" in supplement["error"]
