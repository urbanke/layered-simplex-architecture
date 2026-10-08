"""Independent identities, read-only store behavior, and explicit ALT priors."""

import hashlib
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from lsa.alt.depth import DepthEvaluator, NumericalError, StoreConfig, UnsupportedDomain


def test_uniform_add_one_mixture_and_sparse_depth_selection():
    evaluator = DepthEvaluator()
    result = evaluator.predict([2, 1, 0, 0], depths=(0, 1))
    np.testing.assert_allclose(result.component_probabilities[0], np.full(4, 0.25))
    np.testing.assert_allclose(
        result.component_probabilities[1], [3 / 7, 2 / 7, 1 / 7, 1 / 7]
    )
    q0, q1 = 4**-3, math.factorial(3) * 2 / math.factorial(6)
    np.testing.assert_allclose(result.posterior, np.array([q0, q1]) / (q0 + q1))
    np.testing.assert_allclose(
        result.mixture_probabilities, result.posterior @ result.component_probabilities
    )
    evidence = evaluator.evidence_at_depths(4, (2, 1), (1, 0))
    assert evidence.depths == (1, 0)
    np.testing.assert_allclose(evidence.log_evidence, np.log([q1, q0]))
    assert evidence.mixture_log_evidence == pytest.approx(math.log((q1 + q0) / 2))
    assert result.diagnostics["normalization_applied"] is False


def test_independent_l2_matches_direct_simplex_prior_and_normalizes():
    evaluator = DepthEvaluator()
    evidence = evaluator.evidence_at_depths(2, (2,), (2,))

    def inner(u):
        def integrand(v):
            numerator = u * v
            theta = numerator / (numerator + (1 - u) * (1 - v))
            return theta**2

        return quad(integrand, 0, 1, epsabs=1e-10, epsrel=1e-10)[0]

    direct = quad(inner, 0, 1, epsabs=1e-9, epsrel=1e-9)[0]
    assert math.exp(evidence.log_evidence[0]) == pytest.approx(direct, abs=2e-9)
    other = evaluator.evidence_at_depths(2, (1, 1), (2,))
    assert 2 * (
        math.exp(evidence.log_evidence[0]) + math.exp(other.log_evidence[0])
    ) == pytest.approx(1, abs=2e-9)
    assert evidence.diagnostics["components"][0]["tail_relative_bound"] < 1e-10


def test_batch_family_prediction_chain_rule_and_count_classes():
    evaluator = DepthEvaluator()
    profiles = {"a": (2, 1), "b": (1, 1)}
    batch = evaluator.evaluate_profiles(3, profiles, 2)
    families = evaluator.prediction_by_count_batch(3, profiles, depths=(0, 1, 2))
    for key, parts in profiles.items():
        classes = families[key]
        assert classes.diagnostics["maximum_normalization_error"] < 1e-8
        for c, probability in classes.by_count.items():
            aug = list(parts)
            if c:
                aug.remove(c)
            aug.append(c + 1)
            augmented = evaluator.evidence(3, aug, 2)
            assert math.log(probability) == pytest.approx(
                augmented.mixture_log_evidence - batch[key].mixture_log_evidence,
                abs=1e-10,
            )
    full = evaluator.predict([2, 0, 1], depths=(0, 1, 2))
    np.testing.assert_allclose(
        full.mixture_probabilities, [families["a"].by_count[x] for x in [2, 0, 1]]
    )
    # Excluding depth zero uses evidence, even if some posterior weights vanish.
    weights = np.exp(
        full.component_log_evidence[1:] - logsumexp(full.component_log_evidence[1:])
    )
    positive_only = evaluator.predict([2, 0, 1], depths=(1, 2))
    np.testing.assert_allclose(
        weights @ full.component_probabilities[1:], positive_only.mixture_probabilities
    )


def test_empty_profile_initial_prediction_and_label_symmetry():
    evaluator = DepthEvaluator()
    np.testing.assert_array_equal(evaluator.evidence(4, (), 2).log_evidence, 0)
    np.testing.assert_allclose(
        evaluator.predict([0, 0, 0], depths=(0, 1, 2)).mixture_probabilities,
        np.ones(3) / 3,
    )
    a = evaluator.predict([2, 1, 0], depths=(0, 1))
    b = evaluator.predict([0, 2, 1], depths=(0, 1))
    np.testing.assert_allclose(
        b.mixture_probabilities, a.mixture_probabilities[[2, 0, 1]]
    )


@pytest.mark.parametrize(
    "d,parts,depths", [(33, (2,), (2,)), (2, (21,), (2,)), (3, (2,), (3,))]
)
def test_reference_refuses_unvalidated_domain(d, parts, depths):
    with pytest.raises(UnsupportedDomain):
        DepthEvaluator().evidence_at_depths(d, parts, depths)


@pytest.mark.parametrize("counts", [[-1, 2], [1.5, 0], [True, 0], []])
def test_invalid_counts_fail(counts):
    with pytest.raises(ValueError):
        DepthEvaluator().predict(counts, depths=(0, 1))


def test_production_requires_configuration_specific_calibration():
    with pytest.raises(NumericalError, match="calibration"):
        DepthEvaluator(purpose="production")
    with pytest.raises(ValueError):
        DepthEvaluator(mode="store")


@pytest.fixture(scope="module")
def tiny_store(tmp_path_factory):
    # Build only a small, disposable fixture. Never access a sibling store.
    from lsa.alt._vendor.pmwm import _runtime
    from lsa.alt._vendor.pmwm.universal_tables import UniversalTables

    path = tmp_path_factory.mktemp("alt-exact-store")
    token = _runtime._settings.set({"PMM_BUILD_EXACT": "1"})
    try:
        table = UniversalTables(path)
        table.ensure_columns(2, range(7))
        table.close()
    finally:
        _runtime._settings.reset(token)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.iterdir()
        if p.name == "manifest.json"
        or p.suffix == ".bin"
        or p.name.endswith("index.json")
    }
    return StoreConfig(
        path,
        files,
        max_depth=2,
        max_d=32,
        max_n=20,
        max_count=4,
        ladder_every=0,
        grid_step=0.02,
    )


def test_pinned_store_matches_independent_reference_and_never_writes(
    tiny_store, monkeypatch
):
    # Ambient experiment settings must have no effect on the private engine.
    monkeypatch.setenv("PMM_PHI_BIAS", "10")
    monkeypatch.setenv("PMM_SCAN_LEGACY", "1")
    monkeypatch.setenv("PMM_PHI_SADDLE_MIN_L", "2")
    path = tiny_store.path
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in path.iterdir()}
    reference = DepthEvaluator().predict([2, 1, 0], depths=(0, 1, 2))
    with DepthEvaluator(mode="store", store=tiny_store) as engine:
        result = engine.predict([2, 1, 0], depths=(0, 1, 2))
        batched = engine.evaluate_profiles(3, {"a": (2, 1), "b": (1, 1)}, 2)
        np.testing.assert_allclose(
            result.component_log_evidence,
            reference.component_log_evidence,
            atol=2e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            result.component_probabilities,
            reference.component_probabilities,
            atol=2e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            batched["a"].log_evidence, result.component_log_evidence, atol=1e-10, rtol=0
        )
        assert result.diagnostics["base"]["components"][2]["kernel_branch"] == "stored"
        calibration = {
            "status": "passed",
            "configuration_sha256": engine.configuration_sha256,
        }
    with DepthEvaluator(
        mode="store", store=tiny_store, purpose="production", calibration=calibration
    ) as calibrated:
        assert calibrated.production_ready
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in path.iterdir()}
    assert before == after


def test_store_hash_and_domain_guards(tiny_store):
    from dataclasses import replace

    hashes = dict(tiny_store.files_sha256)
    hashes["manifest.json"] = "0" * 64
    with pytest.raises(NumericalError, match="hash mismatch"):
        DepthEvaluator(mode="store", store=replace(tiny_store, files_sha256=hashes))
    hashes = dict(tiny_store.files_sha256)
    hashes.pop("level_02.bin")
    with pytest.raises(NumericalError, match="missing store content"):
        DepthEvaluator(mode="store", store=replace(tiny_store, files_sha256=hashes))
    with DepthEvaluator(mode="store", store=tiny_store) as evaluator:
        with pytest.raises(UnsupportedDomain):
            evaluator.evidence_at_depths(3, (2,), (3,))
        with pytest.raises(UnsupportedDomain):
            evaluator.predict([4, 0, 0], depths=(2,))  # augmentation exceeds max_count


def test_narrow_peak_is_rejected(tiny_store, monkeypatch):
    from lsa.alt._vendor.pmwm import layered

    original = layered.log_q_lambda_scan

    def narrow(**kwargs):
        from dataclasses import replace

        return replace(original(**kwargs), message="NARROW unresolved peak")

    monkeypatch.setattr(layered, "log_q_lambda_scan", narrow)
    with (
        DepthEvaluator(mode="store", store=tiny_store) as evaluator,
        pytest.raises(NumericalError, match="unresolved"),
    ):
        evaluator.evidence_at_depths(3, (2, 1), (2,))


def test_outer_refinement_retries_entire_family_on_one_grid(tiny_store, monkeypatch):
    from dataclasses import replace

    from lsa.alt._vendor.pmwm import layered

    original = layered.log_q_lambda_scan_family
    seen_steps = []

    def narrow_on_coarse_grid(**kwargs):
        grid = kwargs["tables"].u_grid
        step = float(grid[1] - grid[0])
        seen_steps.append(step)
        parent, children = original(**kwargs)
        if step > 0.006:
            parent = replace(parent, message="NARROW synthetic coarse-grid failure")
        return parent, children

    monkeypatch.setattr(layered, "log_q_lambda_scan_family", narrow_on_coarse_grid)
    with DepthEvaluator(mode="store", store=tiny_store) as evaluator:
        parent, child = evaluator.transition_log_evidence(
            3, {"x": ((2, 1), 0)}, depths=(2,)
        )["x"]
        assert len(seen_steps) == 3
        assert parent.diagnostics["components"][0]["grid_refinements"] == 2
        assert child.diagnostics["components"][0]["outer_grid_step"] == seen_steps[-1]
        direct = DepthEvaluator().evidence_at_depths(3, (2, 1, 1), (2,))
        np.testing.assert_allclose(
            child.log_evidence, direct.log_evidence, atol=3e-7, rtol=0
        )


def test_observed_transition_equals_full_predictor():
    evaluator = DepthEvaluator()
    prediction = evaluator.predict([2, 1, 0, 0], depths=(0, 1, 2))
    for count, symbol in [(2, 0), (1, 1), (0, 2)]:
        parent, child = evaluator.transition_log_evidence(
            4, {"x": ((2, 1), count)}, depths=(0, 1, 2)
        )["x"]
        np.testing.assert_allclose(
            np.exp(child.log_evidence - parent.log_evidence),
            prediction.component_probabilities[:, symbol],
            atol=1e-11,
            rtol=0,
        )


def _mock_window_engine(tiny_store, monkeypatch, *, direct=True, maximum=80, increment=25):
    """Real adapter/store guards; only numerical tables/scans are mocked."""
    from dataclasses import replace
    from types import SimpleNamespace

    config = replace(tiny_store, max_depth=80 if direct else 2,
                     saddle_min_depth=54 if direct else None,
                     maximum_u_max=maximum, upper_window_increment=increment)
    engine = DepthEvaluator(mode="store", store=config)
    monkeypatch.setattr(engine._store, "level_tables",
                        lambda L, rs, grid: SimpleNamespace(u_grid=grid))
    return engine


def _mock_scan_result(kwargs, partition, **changes):
    from lsa.alt._vendor.pmwm.layered import QLambdaResult

    # A common window-dependent offset makes stale parents/earlier families
    # detectable: their evidence must come from the final complete attempt.
    d, n = kwargs["d"], sum(partition)
    upper = float(kwargs["tables"].u_grid[-1])
    log_q = math.lgamma(d)-math.lgamma(d+n)
    log_q += sum(math.lgamma(r+1) for r in partition) + upper/100
    fields = {"log_q": log_q, "method": "mock-window", "d": d, "L": kwargs["L"],
              "N": n, "partition": tuple(partition), "converged": True,
              "right_gap": 100., "message": "resolved peak"}
    fields.update(changes)
    return QLambdaResult(**fields)


def test_window_retry_recomputes_all_families_and_preserves_step_refinement(
    tiny_store, monkeypatch
):
    from dataclasses import replace

    from lsa.alt._vendor.pmwm import layered

    calls = []

    def scan_family(**kwargs):
        grid, part = kwargs["tables"].u_grid, kwargs["base_partition"]
        upper, step = float(grid[-1]), float(grid[1]-grid[0])
        calls.append((part, upper, step))
        parent = _mock_scan_result(kwargs, part)
        children = {}
        for c in kwargs["cs"]:
            augmented = list(part)
            if c:
                augmented.remove(c)
            augmented.append(c+1)
            children[c] = _mock_scan_result(kwargs, augmented)
        # The final child of the second family fails, after other records have
        # already been collected. All records must be discarded and recomputed.
        if part == (3,):
            if step > 0.006:
                children[3] = replace(children[3], message="NARROW synthetic peak")
            elif upper < 80:
                children[3] = replace(children[3], converged=False, right_gap=0.,
                                      message="left tail only")
        return parent, children

    monkeypatch.setattr(layered, "log_q_lambda_scan_family", scan_family)
    with _mock_window_engine(tiny_store, monkeypatch) as engine:
        results = engine.prediction_by_count_batch(4, {"a": (2, 1), "b": (3,)}, depths=[80])
    assert [call[1] for call in calls[::2]] == [35, 35, 35, 60, 80]
    assert [call[0] for call in calls] == [(2, 1), (3,)]*5
    assert all(call[2] < 0.006 for call in calls[4:])
    for name, part in {"a": (2, 1), "b": (3,)}.items():
        result = results[name]
        expected = math.lgamma(4)-math.lgamma(7)+sum(math.lgamma(r+1) for r in part)+.8
        assert result.component_log_evidence[0] == pytest.approx(expected, abs=1e-14)
        assert result.diagnostics["maximum_normalization_error"] < 1e-14
        diagnostics = [result.diagnostics["base"], *result.diagnostics["augmented"].values()]
        for record in diagnostics:
            diag = record["components"][0]
            assert diag["initial_u_max"] == 35
            assert diag["actual_u_max"] == 80
            assert diag["upper_window_history"] == [35, 60, 80]
            assert diag["window_expansions"] == 2
            assert diag["grid_refinements"] == 2
            assert diag["requested_grid_step"] == .005


@pytest.mark.parametrize("direct,maximum,expected", [
    (True, 80, [35, 60, 80]), (True, 35, [35]), (False, 80, [35]),
])
def test_window_expansion_is_capped_and_disabled_for_stored_levels(
    tiny_store, monkeypatch, direct, maximum, expected
):
    from lsa.alt._vendor.pmwm import layered

    calls = []

    def failed(**kwargs):
        calls.append(float(kwargs["tables"].u_grid[-1]))
        return _mock_scan_result(kwargs, kwargs["partition"], converged=False,
                                 right_gap=0., message="left tail only")

    monkeypatch.setattr(layered, "log_q_lambda_scan", failed)
    with (
        _mock_window_engine(tiny_store, monkeypatch, direct=direct, maximum=maximum) as engine,
        pytest.raises(NumericalError, match="right_gap=0") as caught,
    ):
        engine.evidence_at_depths(4, (2, 1), [80 if direct else 2])
    assert calls == expected
    assert ("upper-window limit" in str(caught.value)) == direct


@pytest.mark.parametrize("message,gap,converged", [
    ("unresolved curvature", 0., False), ("unresolved kernel", 100., False),
    ("invalid kernel", None, False), ("invalid gap", -math.inf, False),
])
def test_window_does_not_retry_unrelated_scan_failures(tiny_store, monkeypatch, message, gap, converged):
    from lsa.alt._vendor.pmwm import layered

    calls = []

    def failed(**kwargs):
        calls.append(float(kwargs["tables"].u_grid[-1]))
        return _mock_scan_result(kwargs, kwargs["partition"], converged=converged,
                                 right_gap=gap, message=message)

    monkeypatch.setattr(layered, "log_q_lambda_scan", failed)
    with (
        _mock_window_engine(tiny_store, monkeypatch) as engine,
        pytest.raises(NumericalError),
    ):
        engine.evidence_at_depths(4, (2, 1), [80])
    assert calls == [35]


def test_window_retry_rejects_increment_too_small_to_advance(tiny_store, monkeypatch):
    from lsa.alt._vendor.pmwm import layered

    calls = []

    def failed(**kwargs):
        calls.append(float(kwargs["tables"].u_grid[-1]))
        return _mock_scan_result(kwargs, kwargs["partition"], converged=False,
                                 right_gap=0., message="left tail only")

    monkeypatch.setattr(layered, "log_q_lambda_scan", failed)
    with (
        _mock_window_engine(tiny_store, monkeypatch, increment=1e-20) as engine,
        pytest.raises(NumericalError, match="no representable progress"),
    ):
        engine.evidence_at_depths(4, (2, 1), [80])
    assert calls == [35]


def test_resolved_scan_keeps_initial_window(tiny_store, monkeypatch):
    from lsa.alt._vendor.pmwm import layered

    calls = []

    def resolved(**kwargs):
        calls.append(float(kwargs["tables"].u_grid[-1]))
        return _mock_scan_result(kwargs, kwargs["partition"])

    monkeypatch.setattr(layered, "log_q_lambda_scan", resolved)
    with _mock_window_engine(tiny_store, monkeypatch) as engine:
        result = engine.evidence_at_depths(4, (2, 1), [80])
    assert calls == [35]
    diag = result.diagnostics["components"][0]
    assert diag["window_expansions"] == 0
    assert diag["initial_u_max"] == diag["actual_u_max"] == 35


@pytest.mark.parametrize("values", [
    {"maximum_u_max": 34}, {"maximum_u_max": math.inf},
    {"upper_window_increment": 0}, {"upper_window_increment": math.nan},
])
def test_window_policy_requires_finite_consistent_bounds(tiny_store, values):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(tiny_store, **values)


def test_vendor_hashes_match_recorded_transformations():
    from pathlib import Path

    from lsa.alt._vendor import pmwm

    directory = Path(pmwm.__file__).parent
    manifest = json.loads((directory / "provenance.json").read_text())
    assert manifest["commit"] == "240406d16be0e7c5dcd7d2ee0e14d5ee4f28c915"
    for name, hashes in manifest["files"].items():
        assert (
            hashlib.sha256((directory / name).read_bytes()).hexdigest()
            == hashes["vendored_sha256"]
        )
