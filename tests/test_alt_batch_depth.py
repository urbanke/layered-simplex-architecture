"""Batch preparation preserves identities, labels, grids and every record."""

import hashlib

import numpy as np
import pytest

from lsa.alt.batch_depth import BatchedDepthEvaluator, iter_evidence_batches
from lsa.alt.depth import DepthEvaluator, StoreConfig


def test_cached_endpoints_match_individual_and_remap_permuted_labels():
    engine = DepthEvaluator()
    batch = BatchedDepthEvaluator(engine, chunk_size=2, max_cached_profiles=3)
    inputs = [[2, 1, 0, 0], [0, 2, 0, 1], [1, 1, 1, 0]]
    info = batch.prepare_predictions(inputs, depths=[0, 1])
    assert info["samples"] == 3 and info["unique_profiles"] == 2
    assert info["kernel_batches"] == 1
    for counts in inputs:
        actual = batch.predict(counts, depths=[0, 1])
        expected = engine.predict(counts, depths=[0, 1])
        np.testing.assert_array_equal(
            actual.component_probabilities, expected.component_probabilities
        )
        np.testing.assert_array_equal(
            actual.mixture_probabilities, expected.mixture_probabilities
        )
        np.testing.assert_array_equal(actual.posterior, expected.posterior)
        assert actual.diagnostics["batch_cache"]["hit"] is True
        assert actual.diagnostics["normalization_applied"] is False
    assert batch.prepare(inputs, depths=[0, 1])["newly_evaluated_profiles"] == 0
    with pytest.raises(KeyError):
        batch.predict(inputs[0], depths=[1, 0])  # ordered prior grid is explicit


def test_reference_l2_batch_agrees_with_individual_and_preserves_diagnostics():
    engine = DepthEvaluator()
    batch = BatchedDepthEvaluator(engine, chunk_size=2, max_cached_profiles=4)
    inputs = [[2, 1, 0], [3, 0, 0], [1, 1, 1]]
    batch.prepare(inputs, depths=[0, 1, 2])
    for counts in inputs:
        actual, expected = (
            batch.predict(counts, depths=[0, 1, 2]),
            engine.predict(counts, depths=[0, 1, 2]),
        )
        np.testing.assert_allclose(
            actual.component_probabilities,
            expected.component_probabilities,
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            actual.component_log_evidence,
            expected.component_log_evidence,
            atol=1e-12,
            rtol=0,
        )
        assert actual.diagnostics["base"] == expected.diagnostics["base"]
        assert actual.diagnostics["augmented"] == expected.diagnostics["augmented"]


def test_cohort_capacity_misses_and_evictions_are_explicit():
    batch = BatchedDepthEvaluator(DepthEvaluator(), chunk_size=1, max_cached_profiles=2)
    with pytest.raises(ValueError, match="cohort"):
        batch.prepare([[3, 0, 0], [2, 1, 0], [1, 1, 1]], depths=[0, 1])
    assert batch.cache_info()["cached_profiles"] == 0
    batch.prepare([[3, 0, 0], [2, 1, 0]], depths=[0, 1])
    batch.prepare([[1, 1, 1]], depths=[0, 1])
    assert batch.cache_info()["cached_profiles"] == 2
    assert batch.cache_info()["evictions"] == 1
    with pytest.raises(KeyError):
        batch.predict([3, 0, 0], depths=[0, 1])
    batch.predict([2, 1, 0], depths=[0, 1])
    fallback = BatchedDepthEvaluator(
        DepthEvaluator(), chunk_size=1, max_cached_profiles=1, on_miss="evaluate"
    )
    assert (
        fallback.predict([1, 0], depths=[0, 1]).diagnostics["batch_cache"]["hit"]
        is False
    )


def test_evidence_iterator_is_lazy_bounded_and_never_drops_duplicate_profiles():
    class CountingEngine(DepthEvaluator):
        def __init__(self):
            super().__init__()
            self.sizes = []

        def evaluate_profiles_at_depths(self, d, profiles, depths):
            self.sizes.append(len(profiles))
            return super().evaluate_profiles_at_depths(d, profiles, depths)

    engine = CountingEngine()
    consumed = []

    def source():
        for i, parts in enumerate([(2, 1), (1, 2), (3,), (1, 1, 1), (2, 1)]):
            consumed.append(i)
            yield f"trial-{i}", parts

    outputs = iter_evidence_batches(engine, source(), d=3, depths=[0, 1], chunk_size=2)
    first = next(outputs)
    assert consumed == [0, 1]
    rows = [first, *outputs]
    assert [key for key, _ in rows] == [f"trial-{i}" for i in range(5)]
    assert engine.sizes == [1, 2, 1]
    np.testing.assert_array_equal(rows[0][1].log_evidence, rows[1][1].log_evidence)
    assert rows[0][1].diagnostics["batch_preparation"]["input_profiles"] == 2


def test_small_pinned_store_batch_matches_individual_and_independent_l2(tmp_path):
    from lsa.alt._vendor.pmwm import _runtime
    from lsa.alt._vendor.pmwm.universal_tables import UniversalTables

    path = tmp_path / "disposable-store"
    token = _runtime._settings.set({"PMM_BUILD_EXACT": "1"})
    try:
        tables = UniversalTables(path)
        tables.ensure_columns(2, range(7))
        tables.close()
    finally:
        _runtime._settings.reset(token)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.iterdir()
        if p.name == "manifest.json"
        or p.suffix == ".bin"
        or p.name.endswith("index.json")
    }
    store = StoreConfig(
        path,
        hashes,
        max_depth=2,
        max_d=8,
        max_n=12,
        max_count=4,
        ladder_every=0,
        grid_step=0.02,
    )
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in path.iterdir()}
    inputs = [[2, 1, 0], [3, 0, 0]]
    with DepthEvaluator(mode="store", store=store) as engine:
        batch = BatchedDepthEvaluator(engine, chunk_size=2, max_cached_profiles=2)
        batch.prepare(inputs, depths=[0, 1, 2])
        for counts in inputs:
            got = batch.predict(counts, depths=[0, 1, 2])
            single = engine.predict(counts, depths=[0, 1, 2])
            independent = DepthEvaluator().predict(counts, depths=[0, 1, 2])
            np.testing.assert_allclose(
                got.component_probabilities,
                single.component_probabilities,
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                got.component_log_evidence,
                single.component_log_evidence,
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                got.component_probabilities,
                independent.component_probabilities,
                rtol=0,
                atol=3e-7,
            )
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in path.iterdir()}
    assert before == after
