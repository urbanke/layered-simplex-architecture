"""Pinned native serving reproduces the physical-node Python stencil exactly."""

import hashlib
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from lsa.alt.sealed_native import (
    COMPILE_FLAGS,
    SOURCE_PATH,
    NativeInterpolator,
    SealedNativeError,
    build_native,
    manifest_path,
)
from lsa.alt.sealed_tables import SealedKernelTables


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    if shutil.which("cc") is None:
        pytest.skip("explicit native build test requires a C compiler")
    path = tmp_path_factory.mktemp("native-build") / "sealed-interp.so"
    metadata = build_native(path)
    return path, metadata


def copy_library(compiled, tmp_path):
    original, metadata = compiled
    path = tmp_path / "copy.so"
    shutil.copy2(original, path)
    shutil.copy2(manifest_path(original), manifest_path(path))
    return path, metadata["binary_sha256"]


def test_explicit_build_is_immutable_and_has_fixed_identity(compiled):
    path, metadata = compiled
    assert (
        metadata["source_sha256"]
        == hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    )
    assert metadata["flags"] == list(COMPILE_FLAGS)
    assert metadata["binary_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    native = NativeInterpolator(path, metadata["binary_sha256"])
    identity = native.identity
    assert identity["binary_sha256"] == metadata["binary_sha256"]
    assert (
        identity["build_manifest_sha256"]
        == hashlib.sha256(manifest_path(path).read_bytes()).hexdigest()
    )
    identity["flags"].append("-ffast-math")
    assert native.identity["flags"] == list(COMPILE_FLAGS)
    with pytest.raises(FileExistsError):
        build_native(path)


@pytest.mark.parametrize(
    "u_min,step,length",
    [
        (-60.0, 0.02, 7001),
        (-2000.14, 0.02, 104008),
        (-1102.37, 0.02, 59120),
        (-63.75, 0.25, 260),
    ],
)
def test_bitwise_physical_node_and_near_node_parity(compiled, u_min, step, length):
    path, metadata = compiled
    native = NativeInterpolator(path, metadata["binary_sha256"])
    nodes = u_min + step * np.arange(length, dtype=np.float64)
    # Large baseline stresses centered arithmetic and coordinate quantization.
    values = np.asarray(1e9 + np.sin(nodes / 31.0) * 1e7 - 0.12 * nodes**2)
    selected = nodes[[0, 1, 2, 3, 4, length // 2, length - 8, length - 4, length - 1]]
    rng = np.random.default_rng(84720)
    u = np.unique(
        np.r_[
            selected,
            np.nextafter(selected, -np.inf),
            np.nextafter(selected, np.inf),
            rng.uniform(nodes[0], nodes[-1], 1000),
        ]
    )
    u = u[(u >= nodes[0]) & (u <= nodes[-1])]
    # Exercise the actual Python stencil without constructing a synthetic store.
    table = SimpleNamespace(
        grid_step=step, maximum_u=nodes[-1], _native_interpolator=None
    )
    col = SimpleNamespace(offset=0, length=length, u_min=u_min)
    expected = SealedKernelTables._anchor_values(table, 2, 0, u, {0: col}, values)
    actual = native.interpolate(values, u_min, step, u)
    assert actual.tobytes() == expected.tobytes()
    assert values.flags.writeable  # Native code has no ownership of caller data.
    assert native.interpolate(values, u_min, step, np.array([])).shape == (0,)


def test_tiny_rounded_final_node_extrapolation_matches_python(compiled):
    path, metadata = compiled
    native = NativeInterpolator(path, metadata["binary_sha256"])
    u_min, step, length = -1903.58, 0.02, 99180
    nodes = u_min + step * np.arange(length)
    values = 1e8 - 1e6 * np.logaddexp(0, nodes / 100)
    u = np.array([np.nextafter(nodes[-1], np.inf)])
    table = SimpleNamespace(grid_step=step, maximum_u=u[0], _native_interpolator=None)
    col = SimpleNamespace(offset=0, length=length, u_min=u_min)
    expected = SealedKernelTables._anchor_values(table, 2, 0, u, {0: col}, values)
    assert native.interpolate(values, u_min, step, u).tobytes() == expected.tobytes()


@pytest.mark.parametrize("queries", [[-0.01], [7.01], [float("inf")], [float("nan")]])
def test_uncovered_or_nonfinite_queries_rejected(compiled, queries):
    path, metadata = compiled
    native = NativeInterpolator(path, metadata["binary_sha256"])
    with pytest.raises(SealedNativeError):
        native.interpolate(np.arange(8.0), 0.0, 1.0, queries)


def test_invalid_values_and_grid_fail_closed(compiled):
    path, metadata = compiled
    native = NativeInterpolator(path, metadata["binary_sha256"])
    for values, u_min, step, queries in [
        (np.arange(7.0), 0.0, 1.0, [0.0]),
        (np.arange(8.0), 0.0, 0.0, [0.0]),
        (np.arange(8.0), 0.0, float("inf"), [0.0]),
        (np.full(8, np.nan), 0.0, 1.0, [0.0]),
        (np.arange(8.0), 1e100, 1.0, [1e100]),
        (np.arange(8.0) + 1j, 0.0, 1.0, [0.0]),
        (np.arange(8.0), 0.0, 1.0, [1j]),
    ]:
        with pytest.raises(SealedNativeError):
            native.interpolate(values, u_min, step, queries)


def test_bad_pin_and_missing_manifest_never_compile(compiled, tmp_path, monkeypatch):
    path, pin = copy_library(compiled, tmp_path)
    monkeypatch.setattr(
        "lsa.alt.sealed_native.subprocess.run",
        lambda *a, **kw: pytest.fail("serving invoked compiler"),
    )
    with pytest.raises(SealedNativeError):
        NativeInterpolator(path, "0" * 64)
    manifest_path(path).unlink()
    with pytest.raises(SealedNativeError):
        NativeInterpolator(path, pin)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "0" * 64),
        ("flags", ["-ffast-math"]),
        ("abi_version", 100),
        ("compiled_compiler", "another compiler"),
        ("platform", {"system": "unrelated", "machine": "other"}),
    ],
)
def test_manifest_mismatch_rejected(compiled, tmp_path, field, value):
    path, pin = copy_library(compiled, tmp_path)
    metadata = json.loads(manifest_path(path).read_text())
    metadata[field] = value
    manifest_path(path).write_text(json.dumps(metadata))
    with pytest.raises(SealedNativeError):
        NativeInterpolator(path, pin)


def test_binary_changed_before_load_rejected(compiled, tmp_path):
    path, pin = copy_library(compiled, tmp_path)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(SealedNativeError):
        NativeInterpolator(path, pin)


def test_manifest_changed_after_load_rejected(compiled, tmp_path):
    path, pin = copy_library(compiled, tmp_path)
    native = NativeInterpolator(path, pin)
    manifest_path(path).write_text(manifest_path(path).read_text() + " ")
    with pytest.raises(SealedNativeError):
        native.interpolate(np.arange(8.0), 0.0, 1.0, [0.0])
    with pytest.raises(SealedNativeError):
        _ = native.identity
