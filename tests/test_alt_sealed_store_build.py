"""Offline store identity, exact columns and interruption recovery contracts."""

import hashlib
import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lsa.alt import sealed_store_build as builder
from lsa.alt._vendor.pmwm import _runtime, mellin


def _plan(path, levels=(54,), counts=(0, 1, 2, 3)):
    plan = builder.create_plan(
        levels=levels, support_max_count=max(counts),
        anchors={level: list(counts) for level in levels},
    )
    builder.write_plan(plan, path)
    return plan


def _json(path):
    return json.loads(Path(path).read_text())


def test_plan_is_noncomputing_and_preserves_explicit_source_positions(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("planning must not evaluate a numerical kernel")

    monkeypatch.setattr(mellin, "exact_log_phi_column", forbidden)
    anchors = list(range(257)) + list(range(300, 1300, 100))
    source = tmp_path / "anchors.json"
    source.write_text(json.dumps({"levels": {"53": {"anchors": anchors}}}))
    original = source.read_bytes()
    plan = builder.create_plan(source, levels=[54, 138], support_max_count=350)
    assert plan["anchors"] == {"54": anchors[:266], "138": anchors[:266]}
    assert plan["anchors"]["54"][-1] == 1100
    assert plan["estimated_columns"] == 532
    assert plan["estimated_payload_bytes"] > 0
    destination = tmp_path / "planned"
    builder.write_plan(plan, destination)
    assert sorted(p.name for p in destination.iterdir()) == ["plan.json"]
    assert source.read_bytes() == original
    small = builder.create_plan(source, levels=[54], support_max_count=3)
    assert small["anchors"]["54"] == list(range(257))
    with pytest.raises(FileExistsError):
        builder.write_plan(plan, tmp_path)


def test_build_resume_seal_and_exact_column_format(tmp_path, monkeypatch):
    store = tmp_path / "store"
    plan = _plan(store, levels=(54, 138))
    first = builder.build_store(store, levels=[54])
    assert first["built_levels"] == [54]
    assert first["pending_levels"] == [138]
    with pytest.raises(ValueError, match="pending levels"):
        builder.seal_store(store)
    before = {p.name: builder.sha256(p) for p in store.glob("level_054.*")}

    def forbidden(*args, **kwargs):
        raise AssertionError("verified completed levels must not be recomputed")

    with monkeypatch.context() as patch:
        patch.setattr(mellin, "exact_log_phi_column", forbidden)
        resumed = builder.build_store(store, levels=[54])
    assert resumed["resumed_levels"] == [54]
    assert before == {p.name: builder.sha256(p) for p in store.glob("level_054.*")}
    builder.build_store(store, levels=[138])
    manifest = builder.seal_store(store)
    verified = builder.verify_store(store)
    assert verified["sealed"] is True
    assert manifest["files"]["plan.json"] == builder.sha256(store / "plan.json")
    assert verified["columns"] == 8
    assert verified["pending_levels"] == []
    assert builder.seal_store(store) == manifest
    with pytest.raises(ValueError, match="sealed stores"):
        builder.build_store(store)

    index = _json(store / "level_138.index.json")
    payload = np.fromfile(store / "level_138.bin", dtype="<f8")
    token = _runtime._settings.set({"PMM_BUILD_EXACT": "1"})
    try:
        for count in (0, 1, 2, 3):
            record = index["columns"][str(count)]
            lower, length = builder.column_geometry(plan, 138, count)
            assert lower == record["u_min"]
            grid = lower + plan["grid_step"] * np.arange(length)
            expected = mellin.exact_log_phi_column(count, 138, grid)
            column = payload[record["offset"]:record["offset"] + length]
            np.testing.assert_array_equal(column, expected)
            assert hashlib.sha256(column.tobytes()).hexdigest() == record["sha256"]
            assert grid[-1] == pytest.approx(80, abs=1e-12)
    finally:
        _runtime._settings.reset(token)


def test_failed_attempt_and_partial_publication_are_preserved(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _plan(store)
    real = mellin.exact_log_phi_column
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArithmeticError("injected construction failure")
        return real(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(mellin, "exact_log_phi_column", fail_second)
        with pytest.raises(ArithmeticError, match="injected"):
            builder.build_store(store)
    old_attempts = list((store / "attempts").iterdir())
    assert len(old_attempts) == 1
    assert (old_attempts[0] / "failed.json").is_file()
    interrupted_bytes = (old_attempts[0] / "level_054.bin").read_bytes()
    assert interrupted_bytes
    assert not (store / "level_054.complete.json").exists()
    # A crash between publishing a binary and its completion record leaves this.
    (store / "level_054.bin").write_bytes(b"unverified partial publication")
    result = builder.build_store(store)
    assert result["built_levels"] == [54]
    assert (old_attempts[0] / "level_054.bin").read_bytes() == interrupted_bytes
    preserved = list((store / "attempts").glob("*/previous-level_054.bin"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"unverified partial publication"
    builder.seal_store(store)


def test_corrupted_completed_column_is_rejected_not_rebuilt(tmp_path):
    store = tmp_path / "store"
    _plan(store, counts=(0,))
    builder.build_store(store)
    binary = store / "level_054.bin"
    with binary.open("r+b") as stream:
        stream.write(b"corrupt!")
    corrupt = binary.read_bytes()
    with pytest.raises(ValueError, match="column hash mismatch"):
        builder.build_store(store)
    with pytest.raises(ValueError, match="column hash mismatch"):
        builder.seal_store(store)
    assert binary.read_bytes() == corrupt


def test_source_or_plan_changes_reject_resume(tmp_path):
    store = tmp_path / "store"
    plan = _plan(store, counts=(0,))
    builder.build_store(store)
    plan["created_utc"] = "changed"
    (store / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="checkpoint identity"):
        builder.build_store(store)
    plan["source_sha256"]["lsa/alt/sealed_store_build.py"] = "0" * 64
    (store / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="source changed"):
        builder.build_store(store)


def test_parallel_level_ownership_and_sealed_manifest_integrity(tmp_path):
    store = tmp_path / "store"
    _plan(store, levels=(54, 138), counts=(0,))
    result = builder.build_store(store, workers=2)
    assert result["workers"] == 2
    assert result["built_levels"] == [54, 138]
    for level in (54, 138):
        record = _json(store / f"level_{level:03d}.complete.json")
        assert record["worker_count"] == 2
    manifest = builder.seal_store(store)
    manifest["files"]["level_138.index.json"] = "0" * 64
    (store / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        builder.verify_store(store)


def test_plan_validation_and_default_sealed_verification(tmp_path):
    store = tmp_path / "store"
    plan = _plan(store, counts=(0,))
    with pytest.raises(ValueError, match="not sealed"):
        builder.verify_store(store)
    assert builder.verify_store(store, require_sealed=False)["pending_levels"] == [54]
    with pytest.raises(ValueError, match="outside the frozen plan"):
        builder.build_store(store, levels=[138])
    with pytest.raises(ValueError, match="integer"):
        builder.build_store(store, workers=0)
    plan["left_drop"] = 59
    with pytest.raises(ValueError, match="at least 60"):
        builder.validate_plan(plan)
    plan["left_drop"] = 60
    plan["grid_step"] = 0.03
    with pytest.raises(ValueError, match="integer number of grid steps"):
        builder.validate_plan(plan)


def _hold_worker_lock(path, acquired, release):
    with builder._store_lock(path, level=54):
        acquired.set()
        if not release.wait(20):
            raise TimeoutError("parent failed to release the test worker")


def test_surviving_worker_excludes_replacement_coordinator(tmp_path):
    store = tmp_path / "store"
    _plan(store, counts=(0,))
    context = multiprocessing.get_context("spawn")
    acquired, release = context.Event(), context.Event()
    worker = context.Process(target=_hold_worker_lock, args=(store, acquired, release))
    try:
        with builder._store_lock(store):
            worker.start()
            assert acquired.wait(15)
        # Model a lost coordinator while its independently spawned worker lives.
        # The new coordinator acquires the store lock but cannot disturb that level.
        with pytest.raises(RuntimeError, match="level 54 lock"):
            builder.build_store(store)
        assert not (store / "attempts").exists()
    finally:
        release.set()
        worker.join(15)
        if worker.is_alive():
            worker.terminate()
            worker.join()
    assert worker.exitcode == 0
    assert builder.build_store(store)["built_levels"] == [54]


def test_slurm_wrapper_rejects_excess_workers_before_scheduler_calls():
    script = Path(__file__).resolve().parents[1] / "scripts/alt_build_kernel_store_slurm.sh"
    result = subprocess.run(
        ["/bin/bash", str(script)], capture_output=True, text=True, check=False,
        env={
            "ALT_REPO": "/unused", "ALT_EXPECTED_COMMIT": "0" * 40,
            "ALT_PYTHON": sys.executable, "ALT_KERNEL_STORE": "/unused",
            "ALT_KERNEL_PLAN_SHA256": "0" * 64,
            "ALT_BUILD_WORKERS": "3", "SLURM_JOB_ID": "1",
            "SLURM_CPUS_PER_TASK": "2", "PATH": "/usr/bin:/bin",
        },
    )
    assert result.returncode == 2
    assert "exceed allocated CPUs" in result.stderr
    assert "srun" not in result.stderr
