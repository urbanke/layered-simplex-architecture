"""Offline construction of immutable, accurately evaluated kernel columns.

Planning never evaluates a kernel. A build writes complete levels in isolated
attempt directories and publishes a verified completion record last. Numerical
admission is separate from construction and file-integrity verification.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import platform
import sys
import time
import uuid
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

FORMAT = "lsa-sealed-kernels-v1"
SCHEMA_VERSION = 1
BUILDER_SETTINGS = {
    "PMM_BUILD_EXACT": "1",
    "oversample": 8.0,
    "tail_sigmas": 14.0,
    "series_tolerance": 1e-13,
    "contour_tail_nats": 40.0,
}
_SOURCE_NAMES = (
    "sealed_store_build.py",
    "_vendor/pmwm/mellin.py",
    "_vendor/pmwm/_runtime.py",
)


def _now():
    return datetime.now(UTC).isoformat()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object_hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def sha256(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular file, not a symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    """Portable identities of the builder and its numerical implementation."""
    root = Path(__file__).resolve().parent
    return {f"lsa/alt/{name}": sha256(root / name) for name in _SOURCE_NAMES}


def _read_json(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"symlinks are not store files: {path}")
    return json.loads(path.read_text(encoding="utf8"))


def _atomic_json(path, value):
    """Publish a complete JSON file without replacing any existing file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def parse_levels(value):
    """Parse an explicit comma-separated list, including inclusive ranges."""
    if not isinstance(value, str):
        result = list(value)
    else:
        result = []
        for part in value.split(","):
            if "-" in part:
                low, high = (int(item) for item in part.split("-"))
                if high < low:
                    raise ValueError(f"reversed depth range: {part}")
                result.extend(range(low, high + 1))
            else:
                result.append(int(part))
    if not result:
        raise ValueError("at least one stored depth is required")
    for level in result:
        _integer(level, "depth", 2)
        if level > 138:
            raise ValueError("this ALT store design supports depths through 138")
    return sorted(set(result))


def column_geometry(plan, level, count):
    """Return (u_min, length), using an integer number of master-grid steps."""
    step = float(plan["grid_step"])
    lower = math.floor(
        (-float(plan["left_drop"]) - level * math.log(count + 1.0)) / step
    )
    upper = round(float(plan["u_max"]) / step)
    return lower * step, upper - lower + 1


def validate_plan(plan):
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("format") != FORMAT:
        raise ValueError("unsupported sealed kernel plan format")
    levels = parse_levels(plan["levels"])
    if levels != plan["levels"]:
        raise ValueError("plan levels must be sorted and unique")
    support = _integer(plan["support_max_count"], "support_max_count")
    if plan.get("interpolation_degree") != 7 or plan.get("count_degree") != 11:
        raise ValueError("this format requires interpolation degrees 7 and 11")
    for key in ("grid_step", "u_max", "left_drop"):
        if not math.isfinite(plan[key]) or plan[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if plan["left_drop"] < 60:
        raise ValueError("left_drop must provide at least 60 nats")
    if not math.isclose(
        plan["u_max"] / plan["grid_step"],
        round(plan["u_max"] / plan["grid_step"]), abs_tol=1e-9, rel_tol=0,
    ):
        raise ValueError("u_max must be an integer number of grid steps")
    if set(plan["anchors"]) != {str(level) for level in levels}:
        raise ValueError("anchor levels must exactly match the plan")
    for level in levels:
        counts = plan["anchors"][str(level)]
        if not counts or counts != sorted(set(counts)):
            raise ValueError("anchors must be sorted, unique and nonempty")
        for count in counts:
            _integer(count, "count anchor")
        if counts[0] != 0 or counts[-1] < support:
            raise ValueError("anchors must cover zero through support_max_count")
        if any(column_geometry(plan, level, count)[1] < 8 for count in counts):
            raise ValueError("every column needs at least eight grid points")
    if plan.get("builder_settings") != BUILDER_SETTINGS:
        raise ValueError("plan does not specify the fixed accurate builder settings")
    sources = plan.get("source_sha256", {})
    if set(sources) != {f"lsa/alt/{name}" for name in _SOURCE_NAMES}:
        raise ValueError("plan numerical source identities are incomplete")
    if any(not isinstance(v, str) or len(v) != 64 for v in sources.values()):
        raise ValueError("invalid source SHA256")
    return plan


def create_plan(
    anchor_source=None, *, levels=None, support_max_count=1045889,
    pad_anchors=8, anchors=None, grid_step=0.02, u_max=80.0, left_drop=60.0,
):
    """Plan a new store without constructing columns or changing its source.

    ``anchor_source`` names the original anchors.json or its containing directory.
    Its explicit positions are retained through eight anchors above the supported
    count. ``anchors`` supplies a complete explicit mapping for diagnostic stores.
    """
    levels = parse_levels(range(2, 139) if levels is None else levels)
    support_max_count = _integer(support_max_count, "support_max_count")
    _integer(pad_anchors, "pad_anchors", 1)
    if (anchor_source is None) == (anchors is None):
        raise ValueError("supply either anchor_source or explicit anchors")
    provenance = {"policy": "explicit-diagnostic-anchors"}
    if anchor_source is not None:
        anchor_source = Path(anchor_source)
        if anchor_source.is_dir():
            anchor_source = anchor_source / "anchors.json"
        source = _read_json(anchor_source)
        template = max(int(level) for level in source["levels"])
        mapped = {}
        for level in levels:
            source_level = level if str(level) in source["levels"] else template
            full = source["levels"][str(source_level)]["anchors"]
            if full != sorted(set(full)) or full[:257] != list(range(257)):
                raise ValueError("source anchors must retain the dense 0..256 floor")
            # The dense floor is part of the inherited design even when a
            # diagnostic plan declares a smaller public count domain.
            stop = max(257, bisect_right(full, support_max_count) + pad_anchors)
            if stop > len(full):
                raise ValueError("source anchor grid has insufficient upper padding")
            mapped[str(level)] = full[:stop]
        provenance = {
            "policy": "inherited-explicit-grid-with-upper-padding",
            "filename": anchor_source.name,
            "sha256": sha256(anchor_source),
            "template_level": template,
            "dense_below": 256,
            "pad_anchors": pad_anchors,
        }
    else:
        mapped = {str(level): list(anchors.get(level, anchors.get(str(level), [])))
                  for level in levels}
    plan = {
        "schema_version": SCHEMA_VERSION, "format": FORMAT,
        "created_utc": _now(), "grid_step": grid_step, "u_max": u_max,
        "left_drop": left_drop, "interpolation_degree": 7, "count_degree": 11,
        "levels": levels, "anchors": mapped,
        "support_max_count": support_max_count, "anchor_source": provenance,
        "builder_settings": dict(BUILDER_SETTINGS), "source_sha256": source_identity(),
        "dtype": "<f8", "numerical_admission": "required separately",
    }
    validate_plan(plan)
    plan["estimated_columns"] = sum(len(values) for values in mapped.values())
    plan["estimated_payload_bytes"] = 8 * sum(
        column_geometry(plan, level, count)[1]
        for level in levels for count in mapped[str(level)]
    )
    return plan


def write_plan(plan, path):
    """Write a plan into a new or empty directory; never touch an existing store."""
    validate_plan(plan)
    path = Path(path)
    if path.is_symlink() or (path.exists() and any(path.iterdir())):
        raise FileExistsError(f"plan destination must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    _atomic_json(path / "plan.json", plan)
    return path / "plan.json"


def _load_plan(path):
    return validate_plan(_read_json(Path(path) / "plan.json"))


def _assert_source(plan):
    if source_identity() != plan["source_sha256"]:
        raise ValueError("builder/numerical source changed since this plan was frozen")


def _names(level):
    stem = f"level_{level:03d}"
    return f"{stem}.bin", f"{stem}.index.json", f"{stem}.complete.json"


def _runtime_identity():
    import scipy
    return {
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        )},
    }


@contextmanager
def _store_lock(path, *, level=None):
    """Lock a coordinator or one level, including workers outliving their parent."""
    lock_name = ".build.lock" if level is None else f".level_{level:03d}.build.lock"
    lock = Path(path) / lock_name
    if lock.is_symlink():
        raise ValueError("store lock cannot be a symlink")
    with lock.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            subject = "store" if level is None else f"level {level}"
            raise RuntimeError(f"another builder or sealer holds the {subject} lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _verify_payload(path, plan, level):
    binary_name, index_name, _ = _names(level)
    path = Path(path)
    index = _read_json(path / index_name)
    required = {"schema_version": 1, "format": FORMAT, "level": level,
                "grid_step": plan["grid_step"], "u_max": plan["u_max"]}
    if any(index.get(key) != value for key, value in required.items()):
        raise ValueError(f"level {level} index metadata differs from plan")
    columns = index.get("columns", {})
    counts = plan["anchors"][str(level)]
    if set(columns) != {str(count) for count in counts}:
        raise ValueError(f"level {level} columns differ from planned anchors")
    binary = path / binary_name
    if binary.is_symlink() or not binary.is_file():
        raise ValueError(f"missing regular binary file for level {level}")
    offset = 0
    with binary.open("rb") as stream:
        for count in counts:
            record = columns[str(count)]
            lower, length = column_geometry(plan, level, count)
            if (record.get("offset") != offset or record.get("length") != length
                    or record.get("u_min") != lower):
                raise ValueError(f"level {level}, count {count}: invalid grid or offsets")
            payload = stream.read(length * 8)
            if (len(payload) != length * 8
                    or hashlib.sha256(payload).hexdigest() != record.get("sha256")):
                raise ValueError(f"level {level}, count {count}: column hash mismatch")
            if not np.isfinite(np.frombuffer(payload, dtype="<f8")).all():
                raise ValueError(f"level {level}, count {count}: nonfinite kernel values")
            offset += length
        if stream.read(1):
            raise ValueError(f"level {level}: unindexed trailing binary data")
    return {"values": offset, "bytes": offset * 8, "columns": len(counts),
            "files": {binary_name: sha256(binary), index_name: sha256(path / index_name)}}


def _verify_completed(path, plan, level, plan_hash):
    _, _, complete_name = _names(level)
    record = _read_json(Path(path) / complete_name)
    if (record.get("schema_version") != 1 or record.get("format") != FORMAT
            or record.get("level") != level or record.get("plan_sha256") != plan_hash
            or record.get("source_sha256") != plan["source_sha256"]
            or record.get("builder_settings") != plan["builder_settings"]):
        raise ValueError(f"level {level} checkpoint identity mismatch")
    verified = _verify_payload(path, plan, level)
    if any(record.get(key) != value for key, value in verified.items()):
        raise ValueError(f"level {level} checkpoint file hash or size mismatch")
    for key in ("seconds", "kernel_seconds"):
        if not isinstance(record.get(key), (int, float)) or not math.isfinite(
            record[key]
        ) or record[key] < 0:
            raise ValueError(f"level {level} checkpoint has invalid timing")
    return record


def _build_level(path_string, level, plan_hash, worker_count):
    # Spawned processes do not inherit their coordinator's flock. A worker can
    # survive a killed coordinator, so it must protect its own publication too.
    with _store_lock(path_string, level=level):
        return _build_locked_level(path_string, level, plan_hash, worker_count)


def _build_locked_level(path_string, level, plan_hash, worker_count):
    path = Path(path_string)
    plan = _load_plan(path)
    if sha256(path / "plan.json") != plan_hash:
        raise ValueError("plan changed during build")
    _assert_source(plan)
    binary_name, index_name, complete_name = _names(level)
    if (path / complete_name).exists():
        raise FileExistsError(f"completed level must not be rebuilt: {level}")
    attempt = path / "attempts" / f"level_{level:03d}-{uuid.uuid4().hex}"
    attempt.mkdir(parents=True)
    # An interrupted publication is not a completed checkpoint. Preserve every
    # leftover before starting this independent attempt; never accept it silently.
    for name in (binary_name, index_name):
        leftover = path / name
        if leftover.exists() or leftover.is_symlink():
            if leftover.is_symlink():
                raise ValueError("refusing a symlink among incomplete level files")
            os.rename(leftover, attempt / f"previous-{name}")
    started = _now()
    begin = time.perf_counter()
    index = {"schema_version": 1, "format": FORMAT, "level": level,
             "grid_step": plan["grid_step"], "u_max": plan["u_max"], "columns": {}}
    _atomic_json(attempt / "started.json", {
        "level": level, "plan_sha256": plan_hash, "started_utc": started,
        "source_sha256": plan["source_sha256"], "runtime": _runtime_identity(),
        "worker_count": worker_count,
    })
    from ._vendor.pmwm import _runtime, mellin
    token = _runtime._settings.set({"PMM_BUILD_EXACT": "1"})
    kernel_seconds = 0.0
    offset = 0
    try:
        with (attempt / binary_name).open("xb") as stream:
            for count in plan["anchors"][str(level)]:
                lower, length = column_geometry(plan, level, count)
                grid = lower + plan["grid_step"] * np.arange(length, dtype=np.float64)
                tick = time.perf_counter()
                values = np.asarray(mellin.exact_log_phi_column(
                    float(count), level, grid, oversample=8.0, tail_sigmas=14.0,
                    series_tolerance=1e-13, contour_tail_nats=40.0,
                ), dtype="<f8")
                kernel_seconds += time.perf_counter() - tick
                if values.shape != (length,) or not np.isfinite(values).all():
                    raise ValueError(f"invalid built column at L={level}, r={count}")
                payload = values.tobytes(order="C")
                stream.write(payload)
                index["columns"][str(count)] = {
                    "offset": offset, "length": length, "u_min": lower,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                offset += length
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_json(attempt / index_name, index)
        verified = _verify_payload(attempt, plan, level)
        _assert_source(plan)
        if sha256(path / "plan.json") != plan_hash:
            raise ValueError("plan changed during construction")
        record = {
            "schema_version": 1, "format": FORMAT, "level": level,
            "plan_sha256": plan_hash, "source_sha256": plan["source_sha256"],
            "builder_settings": plan["builder_settings"], "started_utc": started,
            "completed_utc": _now(), "seconds": time.perf_counter() - begin,
            "kernel_seconds": kernel_seconds, "worker_count": worker_count,
            "runtime": _runtime_identity(), "attempt": str(attempt.relative_to(path)),
            **verified,
        }
        # Hard links publish without replacing existing files; the independent
        # attempt remains available. Completion is the sole acceptance marker.
        os.link(attempt / binary_name, path / binary_name)
        os.link(attempt / index_name, path / index_name)
        _atomic_json(path / complete_name, record)
        return record
    except BaseException as exc:
        _atomic_json(attempt / "failed.json", {
            "failed_utc": _now(), "error": f"{type(exc).__name__}: {exc}",
            "plan_sha256": plan_hash, "level": level,
        })
        raise
    finally:
        _runtime._settings.reset(token)


def build_store(path, *, levels=None, workers=1, progress=None):
    """Build explicitly selected levels, resuming only verified completed ones."""
    path = Path(path).resolve()
    _integer(workers, "workers", 1)
    plan = _load_plan(path)
    _assert_source(plan)
    selected = plan["levels"] if levels is None else parse_levels(levels)
    if not set(selected) <= set(plan["levels"]):
        raise ValueError("selected levels are outside the frozen plan")
    with _store_lock(path):
        if (path / "manifest.json").exists():
            raise ValueError("sealed stores cannot be built or extended")
        plan_hash = sha256(path / "plan.json")
        # Verify all existing checkpoints, including levels outside this request.
        completed = []
        for level in plan["levels"]:
            if (path / _names(level)[2]).exists():
                _verify_completed(path, plan, level, plan_hash)
                completed.append(level)
        pending = [level for level in selected if level not in completed]
        worker_count = min(workers, len(pending))
        results = []
        if worker_count == 1:
            for level in pending:
                record = _build_level(str(path), level, plan_hash, worker_count)
                results.append(record)
                if progress:
                    progress(record)
        elif worker_count > 1:
            with ProcessPoolExecutor(
                max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [pool.submit(_build_level, str(path), level, plan_hash,
                                       worker_count) for level in pending]
                for future in as_completed(futures):
                    record = future.result()
                    results.append(record)
                    if progress:
                        progress(record)
        _assert_source(plan)
        if sha256(path / "plan.json") != plan_hash:
            raise ValueError("plan changed during build")
        all_completed = sorted(set(completed) | {row["level"] for row in results})
        return {"format": FORMAT, "plan_sha256": plan_hash,
                "built_levels": sorted(row["level"] for row in results),
                "resumed_levels": sorted(set(completed) & set(selected)),
                "completed_levels": all_completed,
                "pending_levels": sorted(set(plan["levels"]) - set(all_completed)),
                "workers": worker_count}


def verify_store(path, *, require_sealed=True):
    """Verify planned grids, all completed columns, and the immutable seal.

    With require_sealed=False, pending levels are reported. Existing completed
    levels must still pass every integrity check. This does not certify accuracy.
    """
    path = Path(path)
    plan = _load_plan(path)
    plan_hash = sha256(path / "plan.json")
    files = {"plan.json": plan_hash}
    completed = []
    records = []
    for level in plan["levels"]:
        complete_name = _names(level)[2]
        if (path / complete_name).exists():
            record = _verify_completed(path, plan, level, plan_hash)
            files.update(record["files"])
            files[complete_name] = sha256(path / complete_name)
            completed.append(level)
            records.append(record)
    manifest_path = path / "manifest.json"
    sealed = manifest_path.exists()
    pending = sorted(set(plan["levels"]) - set(completed))
    if require_sealed and not sealed:
        raise ValueError("store is not sealed")
    if sealed:
        manifest = _read_json(manifest_path)
        if (pending or manifest.get("schema_version") != 1
                or manifest.get("format") != FORMAT or manifest.get("sealed") is not True
                or manifest.get("plan_sha256") != plan_hash
                or manifest.get("levels") != plan["levels"]
                or manifest.get("files") != files):
            raise ValueError("sealed manifest does not match complete verified store")
    expected = {name for level in plan["levels"] for name in _names(level)}
    if any(item.name not in expected for item in path.glob("level_*")):
        raise ValueError("unplanned level files in store")
    return {"format": FORMAT, "sealed": sealed, "plan_sha256": plan_hash,
            "completed_levels": completed, "pending_levels": pending, "files": files,
            "columns": sum(row["columns"] for row in records),
            "payload_bytes": sum(row["bytes"] for row in records),
            "level_seconds_sum": sum(row["seconds"] for row in records),
            "kernel_seconds_sum": sum(row["kernel_seconds"] for row in records),
            "numerical_admission": "not established by integrity verification"}


def seal_store(path):
    """Seal only a complete verified plan; never rewrite a pre-existing seal."""
    path = Path(path)
    with _store_lock(path):
        plan = _load_plan(path)
        _assert_source(plan)
        if (path / "manifest.json").exists():
            verify_store(path)
            return _read_json(path / "manifest.json")
        verified = verify_store(path, require_sealed=False)
        if verified["pending_levels"]:
            raise ValueError(f"cannot seal: pending levels {verified['pending_levels']}")
        manifest = {
            "schema_version": 1, "format": FORMAT, "sealed": True,
            "sealed_utc": _now(), "plan_sha256": verified["plan_sha256"],
            "files": verified["files"], "levels": plan["levels"],
            "source_sha256": plan["source_sha256"],
            "builder_settings": plan["builder_settings"],
            "payload_bytes": verified["payload_bytes"],
            "columns": verified["columns"],
            "level_seconds_sum": verified["level_seconds_sum"],
            "kernel_seconds_sum": verified["kernel_seconds_sum"],
            "sealing_runtime": _runtime_identity(),
            "numerical_admission": "required separately",
        }
        _atomic_json(path / "manifest.json", manifest)
        verify_store(path)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_command = commands.add_parser("plan", help="write a plan; no kernel computation")
    plan_command.add_argument("--anchors", type=Path, required=True)
    plan_command.add_argument("--out", type=Path, required=True)
    plan_command.add_argument("--levels", default="2-138")
    plan_command.add_argument("--support-max-count", type=int, default=1045889)
    plan_command.add_argument("--pad-anchors", type=int, default=8)
    build = commands.add_parser("build", help="explicitly construct planned levels offline")
    build.add_argument("--store", type=Path, required=True)
    build.add_argument("--levels", help="subset of the frozen plan, e.g. 2,54,80,138")
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--seal", action="store_true", help="seal if the complete plan is built")
    seal = commands.add_parser("seal", help="verify and seal a complete plan")
    seal.add_argument("--store", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify a sealed store without computation")
    verify.add_argument("--store", type=Path, required=True)
    verify.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = create_plan(args.anchors, levels=args.levels,
                           support_max_count=args.support_max_count,
                           pad_anchors=args.pad_anchors)
        write_plan(plan, args.out)
        result = {"plan": str(args.out / "plan.json"),
                  "levels": plan["levels"], "columns": plan["estimated_columns"],
                  "estimated_payload_bytes": plan["estimated_payload_bytes"]}
    elif args.command == "build":
        def progress(record):
            print(json.dumps({"completed_level": record["level"],
                              "seconds": record["seconds"], "bytes": record["bytes"]}),
                  flush=True)
        result = build_store(args.store, levels=args.levels, workers=args.workers,
                             progress=progress)
        if args.seal:
            seal_store(args.store)
            result["sealed"] = True
    elif args.command == "seal":
        result = seal_store(args.store)
    else:
        result = verify_store(args.store, require_sealed=not args.allow_incomplete)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0
