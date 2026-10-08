"""Actual table-serving integration, independent reference and admission guards."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from lsa.alt.depth import DepthEvaluator, NumericalError, StoreConfig, UnsupportedDomain
from lsa.alt.sealed_store_build import build_store, create_plan, seal_store, write_plan


@pytest.fixture(scope="module")
def sealed_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("alt-sealed-depth")
    plan = create_plan(levels=[2, 54, 80, 138], support_max_count=8,
                       anchors={L: list(range(9)) for L in [2, 54, 80, 138]})
    write_plan(plan, path)
    build_store(path)
    seal_store(path)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()
             if p.name in ("manifest.json", "plan.json")
             or p.name.endswith((".bin", ".index.json", ".complete.json"))}
    return StoreConfig(path=path, files_sha256=files, max_depth=138,
                       max_d=10000, max_n=4, max_count=4, format="sealed")


def test_sealed_depth_two_matches_independent_simplex_and_never_integrates_kernels(
    sealed_config, monkeypatch,
):
    from lsa.alt._vendor.pmwm import mellin, universal_tables

    reference = DepthEvaluator().predict([2, 1, 0], depths=(0, 1, 2))

    def forbidden(*args, **kwargs):
        raise AssertionError("online kernel integration was called")

    monkeypatch.setattr(mellin, "exact_log_phi_column", forbidden)
    monkeypatch.setattr(mellin, "log_phi_contour", forbidden)
    monkeypatch.setattr(universal_tables, "_saddle_row", forbidden)
    with DepthEvaluator(mode="store", store=sealed_config) as engine:
        got = engine.predict([2, 1, 0], depths=(0, 1, 2))
        np.testing.assert_allclose(got.component_log_evidence,
                                   reference.component_log_evidence, atol=2e-9, rtol=0)
        np.testing.assert_allclose(got.component_probabilities,
                                   reference.component_probabilities, atol=2e-10, rtol=0)
        assert got.diagnostics["maximum_normalization_error"] < 1e-8
        assert got.diagnostics["base"]["components"][2]["kernel_branch"] == "sealed-interpolation"


def test_high_depths_share_family_grids_and_preserve_raw_mass(sealed_config, monkeypatch):
    from lsa.alt._vendor.pmwm import mellin, universal_tables

    def forbidden(*args, **kwargs):
        raise AssertionError("online high-depth fallback was called")

    monkeypatch.setattr(mellin, "exact_log_phi_column", forbidden)
    monkeypatch.setattr(universal_tables, "_saddle_row", forbidden)
    with DepthEvaluator(mode="store", store=sealed_config) as engine:
        got = engine.prediction_by_count(10000, (2, 1), depths=(0, 1, 54, 80, 138))
        assert got.diagnostics["maximum_normalization_error"] < 1e-7
        assert got.diagnostics["normalization_applied"] is False
        for j, base in enumerate(got.diagnostics["base"]["components"]):
            if base["depth"] < 2:
                continue
            assert base["kernel_branch"] == "sealed-interpolation"
            for child in got.diagnostics["augmented"].values():
                assert child["components"][j]["outer_grid_step"] == base["outer_grid_step"]
                assert child["components"][j]["actual_u_max"] == base["actual_u_max"]


def test_sealed_domain_and_substitution_are_explicit(sealed_config):
    with pytest.raises(ValueError, match="saddle_min_depth"):
        replace(sealed_config, saddle_min_depth=54)
    with pytest.raises(UnsupportedDomain, match="maximum_u_max"):
        DepthEvaluator(mode="store", store=replace(sealed_config, maximum_u_max=81))
    with pytest.raises(UnsupportedDomain, match="augmented-count"):
        DepthEvaluator(mode="store", store=replace(sealed_config, max_count=8))
    with (
        DepthEvaluator(mode="store", store=sealed_config) as engine,
        pytest.raises(UnsupportedDomain, match="absent"),
    ):
        engine.evidence_at_depths(4, (2, 1), (22,))


def test_verified_file_cache_rechecks_bytes_after_change(sealed_config, monkeypatch):
    from lsa.alt import depth

    with DepthEvaluator(mode="store", store=sealed_config):
        pass
    original_hash = depth._hash_file
    calls = []

    def counted(path):
        calls.append(str(path))
        return original_hash(path)

    monkeypatch.setattr(depth, "_hash_file", counted)
    with DepthEvaluator(mode="store", store=sealed_config):
        pass
    assert not any(p.endswith(".bin") for p in calls)
    data = sealed_config.path / "level_054.bin"
    original = data.read_bytes()
    try:
        altered = bytearray(original)
        altered[0] ^= 1
        data.write_bytes(altered)
        with pytest.raises(NumericalError, match="hash mismatch"):
            DepthEvaluator(mode="store", store=sealed_config)
    finally:
        data.write_bytes(original)


def test_independent_kernel_sampling_uses_actual_sealed_reader(sealed_config):
    from lsa.alt.kernel_validation import (
        _decimal_difference,
        _store_samples,
        high_precision_log_phi,
    )

    spec = {"id": "sealed_candidate", "format": "sealed",
            "path": str(sealed_config.path), "files_sha256": dict(sealed_config.files_sha256),
            "depths": [138], "counts": [0]}
    cases = [{"case_id": "off-grid", "depth": 138, "r": 0, "u": -4.013}]
    metadata, values = _store_samples(spec, cases)
    ref = high_precision_log_phi(0, 138, -4.013, dps=35)["log_phi_nats"]
    assert metadata["files_sha256"] == spec["files_sha256"]
    assert metadata["unchanged_after_read"]
    assert abs(_decimal_difference(values["off-grid"]["log_phi_nats"], ref)) < 1e-11
    spec["files_sha256"] = dict(spec["files_sha256"], **{"plan.json": "0" * 64})
    with pytest.raises(ValueError, match="engine store pin"):
        _store_samples(spec, cases)
