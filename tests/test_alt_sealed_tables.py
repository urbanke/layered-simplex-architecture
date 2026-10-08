"""Sealed kernels interpolate declared data and never integrate online."""

import hashlib
import json
import math

import numpy as np
import pytest
from scipy.special import loggamma

from lsa.alt.sealed_tables import (
    FORMAT,
    LEFT_LIMIT_LOG_ERROR_BOUND,
    SealedKernelTables,
    SealedTableError,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _reseal(path):
    plan = json.loads((path / "plan.json").read_text())
    manifest = {
        "schema_version": 1,
        "format": FORMAT,
        "sealed": True,
        "plan_sha256": _hash(path / "plan.json"),
        "levels": plan["levels"],
        "files": {p.name: _hash(p) for p in path.iterdir()
                  if p.name == "plan.json" or p.suffix == ".bin"
                  or p.name.endswith(".index.json")},
    }
    _save(path / "manifest.json", manifest)


def _surface(L, r, u):
    # Independent tensor polynomial: degree 3 in log(r+1), degree 2 in u.
    x = math.log(r + 1.)
    return L * float(loggamma(r + 1.)) - .01 * (u + 2.) ** 2 - .03 * x**3


@pytest.fixture
def store_path(tmp_path):
    anchors = [r for r in range(16) if r != 7]
    plan = {
        "schema_version": 1, "format": FORMAT, "grid_step": .25,
        "u_max": 1., "left_drop": 60., "interpolation_degree": 7,
        "count_degree": 11, "levels": [2], "anchors": {"2": anchors},
        "support_max_count": 14, "builder_settings": {}, "source_sha256": {},
    }
    _save(tmp_path / "plan.json", plan)
    columns = {}
    chunks = []
    offset = 0
    for r in anchors:
        u_min = math.floor((-60. - 2 * math.log(r + 1.)) / .25) * .25
        length = round((1. - u_min) / .25) + 1
        u = u_min + .25 * np.arange(length)
        values = np.asarray(_surface(2, r, u), dtype="<f8")
        raw = values.tobytes()
        columns[str(r)] = {"offset": offset, "length": length, "u_min": u_min,
                           "sha256": hashlib.sha256(raw).hexdigest()}
        chunks.append(raw)
        offset += length
    (tmp_path / "level_002.bin").write_bytes(b"".join(chunks))
    _save(tmp_path / "level_002.index.json", {
        "schema_version": 1, "format": FORMAT, "level": 2,
        "grid_step": .25, "u_max": 1., "columns": columns,
    })
    _reseal(tmp_path)
    return tmp_path


def test_polynomial_reproduction_scalar_matrix_and_scan_tables(store_path):
    # The gamma-residual polynomial is reproduced in the aligned-coordinate
    # branch. Exact-node/right-boundary behavior has its own test below.
    u = np.array([-14.37, -13.125, -10., -8., -7.])
    rs = [7, 0, 14, 3, 7]
    with SealedKernelTables(store_path) as table:
        assert table.coverage_depths == (2,)
        assert table.maximum_u == 1.
        assert table.supported_count == 14
        matrix = table.log_phi_matrix(2, rs, u)
        expected = np.array([_surface(2, r, u) for r in rs])
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=2e-13)
        for i, r in enumerate(rs):
            np.testing.assert_array_equal(table.log_phi(2, r, u), matrix[i])
        scan = table.level_tables(2, rs, u)
        assert scan.r_values == (0, 3, 7, 14)
        for r in scan.r_values:
            np.testing.assert_array_equal(scan.log_phi[(2, r)], table.log_phi(2, r, u))


def test_exact_anchor_nodes_boundaries_and_bounded_tail(store_path):
    with SealedKernelTables(store_path) as table:
        for r in (0, 3, 14):
            row = table.column_metadata(2, r)
            u = row["u_min"] + table.grid_step * np.arange(row["length"])
            np.testing.assert_array_equal(table.log_phi(2, r, u), _surface(2, r, u))
            below = np.array([row["u_min"] - .001, -1000.])
            np.testing.assert_array_equal(table.log_phi(2, r, below),
                                          np.full(2, 2 * float(loggamma(r + 1.))))
        assert LEFT_LIMIT_LOG_ERROR_BOUND == pytest.approx(math.exp(-60.), rel=1e-15)
        assert table.log_phi(2, 0, 1.).shape == (1,)
        assert table.log_phi_matrix(2, [], []).shape == (0, 0)
        with pytest.raises(SealedTableError, match="upper bound"):
            table.log_phi(2, 0, np.nextafter(1., math.inf))
        columns, data = table._load_level(2)
        with pytest.raises(SealedTableError, match="shifted anchor"):
            table._anchor_values(2, 0, np.array([1.001]), columns, data)


def test_no_online_kernel_calls_or_writes_and_metadata_is_immutable(store_path, monkeypatch):
    from lsa.alt._vendor.pmwm import mellin

    def forbidden(*args, **kwargs):
        raise AssertionError("online moment evaluation attempted")

    for name in ("log_phi_contour", "exact_log_phi_column", "log_phi_column", "series_column"):
        monkeypatch.setattr(mellin, name, forbidden)
    before = {p.name: (p.stat().st_mtime_ns, _hash(p)) for p in store_path.iterdir()}
    table = SealedKernelTables(store_path)
    table.ensure_columns(2, [0, 7, 14])
    table.log_phi_matrix(2, [7, 14], [-100., -1., 1.])
    with pytest.raises(TypeError):
        table.plan["u_max"] = 9
    with pytest.raises(TypeError):
        table.plan["anchors"]["2"] = ()
    with pytest.raises(TypeError):
        table.column_metadata(2, 0)["offset"] = 1
    _, mmap = table._load_level(2)
    assert not mmap.flags.writeable
    with pytest.raises(ValueError):
        mmap[0] = 0
    table.close()
    table.close()
    with pytest.raises(SealedTableError, match="closed"):
        table.log_phi(2, 0, 0.)
    assert before == {p.name: (p.stat().st_mtime_ns, _hash(p)) for p in store_path.iterdir()}


@pytest.mark.parametrize("L,r,u", [
    (3, 0, [0.]), (2, 15, [0.]), (2, -1, [0.]), (True, 0, [0.]),
    (2, True, [0.]), (2, 1.5, [0.]), (2, 0, [math.nan]),
    (2, 0, [math.inf]), (2, 0, [[0.]]),
])
def test_uncovered_or_invalid_queries_fail(store_path, L, r, u):
    with SealedKernelTables(store_path) as table, pytest.raises(SealedTableError):
        table.log_phi(L, r, u)


def test_pinned_hashes_must_match_seal_and_bind_metadata(store_path):
    hashes = {p.name: _hash(p) for p in store_path.iterdir()}
    with SealedKernelTables(store_path, files_sha256=hashes) as table:
        table.ensure_columns(2, [7])
    for name in ("plan.json", "manifest.json", "level_002.bin"):
        bad = dict(hashes, **{name: "0" * 64})
        with pytest.raises(SealedTableError, match="hash mismatch"):
            SealedKernelTables(store_path, files_sha256=bad)


@pytest.mark.parametrize("change", ["unsealed", "plan", "missing_hash", "data", "symlink"])
def test_seal_or_file_tampering_is_rejected_at_open(store_path, change):
    manifest = json.loads((store_path / "manifest.json").read_text())
    if change == "unsealed":
        manifest["sealed"] = False
    elif change == "plan":
        manifest["plan_sha256"] = "0" * 64
    elif change == "missing_hash":
        del manifest["files"]["level_002.bin"]
    elif change == "data":
        with (store_path / "level_002.bin").open("r+b") as stream:
            stream.write(b"corrupt!")
    else:
        file = store_path / "level_002.bin"
        file.rename(store_path / "other.bin")
        file.symlink_to(store_path / "other.bin")
    _save(store_path / "manifest.json", manifest)
    with pytest.raises(SealedTableError):
        SealedKernelTables(store_path)


@pytest.mark.parametrize("change,match", [
    ("overlap", "overlapping"), ("short", "length"),
    ("endpoint", "endpoint"), ("left_gap", "left gap"),
    ("missing_anchor", "planned anchors"), ("bad_grid", "identity"),
    ("nonfinite", "nonfinite"), ("column_hash", "column hash"),
])
def test_malformed_columns_rejected_even_under_self_consistent_seal(store_path, change, match):
    file = store_path / "level_002.index.json"
    index = json.loads(file.read_text())
    row = index["columns"]["0"]
    if change == "overlap":
        index["columns"]["1"]["offset"] = 0
    elif change == "short":
        data = store_path / "level_002.bin"
        data.write_bytes(data.read_bytes()[:-8])
    elif change == "endpoint":
        row["length"] += 1
    elif change == "left_gap":
        row["u_min"] += .25
        row["length"] -= 1
    elif change == "missing_anchor":
        del index["columns"]["1"]
    elif change == "bad_grid":
        index["grid_step"] = .125
    elif change == "column_hash":
        row["sha256"] = "0" * 64
    else:
        with (store_path / "level_002.bin").open("r+b") as stream:
            stream.write(np.array([math.nan], dtype="<f8").tobytes())
    _save(file, index)
    _reseal(store_path)
    with SealedKernelTables(store_path) as table, pytest.raises(SealedTableError, match=match):
        table.ensure_columns(2, [0])


def test_changes_after_open_are_rejected(store_path):
    with SealedKernelTables(store_path) as table:
        table.log_phi(2, 0, [0.])
        with (store_path / "level_002.bin").open("r+b") as stream:
            stream.write(b"modified")
        with pytest.raises(SealedTableError, match="changed after opening"):
            table.log_phi(2, 0, [0.])


def test_duplicate_json_keys_and_path_escape_rejected(store_path):
    file = store_path / "manifest.json"
    manifest = json.loads(file.read_text())
    manifest["files"]["../outside.bin"] = "0" * 64
    _save(file, manifest)
    with pytest.raises(SealedTableError, match="basenames"):
        SealedKernelTables(store_path)
    _reseal(store_path)
    file.write_text(file.read_text().replace('"sealed": true', '"sealed": true, "sealed": true'))
    with pytest.raises(SealedTableError, match="duplicate JSON key"):
        SealedKernelTables(store_path)


@pytest.fixture(scope="module")
def accurate_transition_store(tmp_path_factory):
    from lsa.alt.sealed_store_build import (
        build_store,
        create_plan,
        seal_store,
        write_plan,
    )

    # Independent count targets266 and1000 are absent. These are their balanced
    # source-grid neighborhoods, including padding, rather than clustered252..256.
    anchors = [0, 165, 178, 193, 209, 226, 245, 267, 289, 313, 339, 367, 397, 430,
               642, 695, 753, 815, 883, 956, 1036, 1122, 1215, 1316, 1425, 1543]
    path = tmp_path_factory.mktemp("accurate-transition-store")
    plan = create_plan(levels=[54, 138], support_max_count=1000,
                       anchors={54: anchors, 138: anchors})
    write_plan(plan, path)
    build_store(path)
    seal_store(path)
    return path


@pytest.mark.parametrize("L,r", [(54, 266), (54, 1000), (138, 266), (138, 1000)])
def test_aligned_transition_and_handover_match_refined_contour(
    accurate_transition_store, L, r, monkeypatch,
):
    from lsa.alt._vendor.pmwm import _runtime, mellin

    shifted = np.array([-61., -59.913, -30.137, -4.013, .007, 5.213, 20.11, 60.157, 100.019,
                        2 * L - .137, 2 * L, 2 * L + .137])
    u = np.sort(np.r_[shifted - L * math.log(r + 1.), -.2573, 10.007, 79.993, 80.])
    token = _runtime._settings.set({"PMM_BUILD_EXACT": "1"})
    try:
        reference = mellin.exact_log_phi_column(
            r, L, u, oversample=16, series_tolerance=1e-14, contour_tail_nats=50,
        )
    finally:
        _runtime._settings.reset(token)

    def forbidden(*args, **kwargs):
        raise AssertionError("a stored query called the numerical kernel")

    monkeypatch.setattr(mellin, "exact_log_phi_column", forbidden)
    with SealedKernelTables(accurate_transition_store) as table:
        stencil = table.ladder_anchors_for(L, r)
        assert len(stencil) == 12
        assert r not in stencil
        if r == 266:
            assert 100 < min(stencil) < 200
        actual = table.log_phi(L, r, u)
        np.testing.assert_allclose(actual, reference, rtol=0, atol=3e-9)
        assert actual[0] == L * float(loggamma(r + 1.))
        np.testing.assert_array_equal(table.log_phi_matrix(L, [r, r], u),
                                      np.stack([actual, actual]))


def test_large_count_centered_coordinates_match_independent_reference(tmp_path):
    from lsa.alt.sealed_store_build import (
        build_store,
        create_plan,
        seal_store,
        write_plan,
    )

    # One count above an anchor exposed loss from subtracting two rounded logs
    # near14. The reference below is a converged independent45/60-digit contour
    # value for the exact binary64 input u=10.007, not a reader-generated oracle.
    anchors = [0, 664299, 719483, 779252, 843985, 914095, 990030, 1072273,
               1161348, 1257822, 1362311, 1475479, 1598049]
    plan = create_plan(levels=[138], support_max_count=990031, anchors={138: anchors})
    write_plan(plan, tmp_path)
    build_store(tmp_path)
    seal_store(tmp_path)
    reference = 2770565.46281214279900707620803920722849264388947107654188841
    with SealedKernelTables(tmp_path) as table:
        actual = table.log_phi(138, 990031, [10.007])[0]
    assert abs(actual - reference) <= 3e-9
