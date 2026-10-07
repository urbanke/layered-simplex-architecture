"""Immutable, content-addressed records for ALT experiment executions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def utc_now():
    return datetime.now(UTC).isoformat()


def source_identity(root):
    root = Path(root).resolve()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    # Include untracked implementation files in development runs, not just HEAD.
    names = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    hashes = {
        name: sha256(root / name)
        for name in names
        if (root / name).is_file() and not name.startswith(("artifacts/", "output/"))
    }
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
        "files": hashes,
        "tree_sha256": canonical_hash(hashes),
    }


def physical_memory():
    if sys.platform != "darwin":
        return {"bytes": None, "status": "unavailable_on_this_platform"}
    try:
        result = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
        )
        return {"bytes": int(result.strip()), "status": "available"}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"bytes": None, "status": "unavailable_or_access_restricted"}


def environment():
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "mpmath", "pytest", "reportlab"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Whitelist numerical settings only: never serialize the entire environment.
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory": physical_memory(),
        "packages": packages,
        "threads": {k: os.environ.get(k) for k in keys},
    }


class Run:
    """A run directory is created once; failed runs remain inspectable."""

    def __init__(
        self,
        directory,
        *,
        repo,
        protocol,
        experiment,
        purpose,
        engine_identity=None,
        inputs=(),
        command=None,
    ):
        if purpose not in ("smoke", "validation", "production"):
            raise ValueError("purpose must be smoke, validation, or production")
        self.path = Path(directory).resolve()
        self.repo = Path(repo).resolve()
        identity = source_identity(repo)
        if purpose == "production":
            if identity["dirty"]:
                raise ValueError("production requires a clean committed source tree")
            if protocol.get("status") != "frozen":
                raise ValueError("production requires a frozen protocol")
        self.path.mkdir(parents=True, exist_ok=False)
        self.record = {
            "schema_version": 1,
            "run_id": self.path.name,
            "purpose": purpose,
            "experiment": experiment,
            "started_utc": utc_now(),
            "source": identity,
            "environment": environment(),
            "command": command or sys.argv,
            "protocol_sha256": canonical_hash(protocol),
            "engine": engine_identity,
            "inputs": [
                {"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in inputs
            ],
        }
        write_json(self.path / "protocol.json", protocol)
        write_json(self.path / "started.json", self.record)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        integrity_error = None
        final_source = source_identity(self.repo)
        if final_source["tree_sha256"] != self.record["source"]["tree_sha256"]:
            integrity_error = RuntimeError("source files changed during the run")
            self.record["source_at_finish"] = final_source
        for entry in self.record["inputs"]:
            if (
                not Path(entry["path"]).is_file()
                or sha256(entry["path"]) != entry["sha256"]
            ):
                integrity_error = RuntimeError("an input changed during the run")
                break
        if integrity_error is not None and error is None:
            kind, error = RuntimeError, integrity_error
        self.record.update(
            finished_utc=utc_now(), status="failed" if error else "complete"
        )
        if error:
            self.record["error"] = {"type": kind.__name__, "message": str(error)}
        self.record["outputs"] = {
            str(p.relative_to(self.path)): {
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(self.path.rglob("*"))
            if p.is_file()
        }
        write_json(self.path / "manifest.json", self.record)
        if integrity_error is not None and traceback is None:
            raise integrity_error
        return False


def verify_run(directory, *, require_complete=True):
    root = Path(directory).resolve()
    record = read_json(root / "manifest.json")
    if require_complete and record["status"] != "complete":
        raise ValueError(f"run is {record['status']}")
    expected = set(record["outputs"]) | {"manifest.json"}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"file inventory changed: {actual ^ expected}")
    for name, info in record["outputs"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            raise ValueError(f"unsafe output path: {name}")
        if sha256(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"checksum mismatch: {name}")
    if canonical_hash(read_json(root / "protocol.json")) != record["protocol_sha256"]:
        raise ValueError("protocol identity mismatch")
    return record
