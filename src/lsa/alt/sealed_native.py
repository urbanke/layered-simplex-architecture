"""Explicit, immutable native interpolation; prediction never invokes a compiler.

The native stencil preserves the physical float64 nodes and operation order of
``SealedKernelTables``. Analytic tails and count interpolation remain in the
reader. A library must be built explicitly, pinned by SHA256, and accompanied by
its immutable build manifest before it can be loaded.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FORMAT = "lsa-sealed-native-v1"
ABI_VERSION = 1
COMPILE_FLAGS = (
    "-O3",
    "-std=c11",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-fPIC",
    "-shared",
)
SOURCE_PATH = Path(__file__).with_name("sealed_interp.c")


class SealedNativeError(ValueError):
    """An explicitly selected native binary or interpolation query is invalid."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def manifest_path(path):
    return Path(str(path) + ".json")


def _json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise SealedNativeError(f"duplicate build-manifest key: {key}")
            out[key] = value
        return out

    result = json.loads(Path(path).read_bytes(), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise SealedNativeError("native build manifest must be an object")
    return result


def _load_library(path):
    lib = ctypes.CDLL(str(path))
    abi = lib.lsa_sealed_interp_abi
    abi.argtypes, abi.restype = [], ctypes.c_int
    source = lib.lsa_sealed_interp_source_sha256
    source.argtypes, source.restype = [], ctypes.c_char_p
    compiler = lib.lsa_sealed_interp_compiler
    compiler.argtypes, compiler.restype = [], ctypes.c_char_p
    if abi() != ABI_VERSION:
        raise SealedNativeError("native interpolation ABI differs")
    fn = lib.lsa_sealed_interp_column
    pointer, integer, real = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
    fn.argtypes = [
        pointer,
        integer,
        pointer,
        integer,
        real,
        real,
        real,
        pointer,
        pointer,
    ]
    fn.restype = ctypes.c_int
    return lib, fn, source().decode("ascii"), compiler().decode("utf-8")


class NativeInterpolator:
    """Load one explicitly pinned library; no build, fallback, or mutable buffers."""

    def __init__(self, path, sha256):
        self.path = Path(path).absolute()
        self.manifest_path = manifest_path(self.path)
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(c not in "0123456789abcdef" for c in sha256)
        ):
            raise SealedNativeError("native library requires a lowercase SHA256 pin")
        try:
            paths = (self.path, self.manifest_path, SOURCE_PATH, Path(__file__))
            self._fingerprints = {p: _fingerprint(p) for p in paths}
            metadata = _json(self.manifest_path)
            source_hash = _sha256(SOURCE_PATH)
            if (
                type(metadata.get("schema_version")) is not int
                or metadata["schema_version"] != 1
                or metadata.get("format") != FORMAT
                or metadata.get("abi_version") != ABI_VERSION
                or metadata.get("binary_sha256") != sha256
                or _sha256(self.path) != sha256
                or metadata.get("source_sha256") != source_hash
                or metadata.get("flags") != list(COMPILE_FLAGS)
                or metadata.get("platform")
                != {
                    "system": platform.system(),
                    "machine": platform.machine(),
                }
            ):
                raise SealedNativeError("native binary/build identity does not match")
            self._library, self._function, embedded_source, embedded_compiler = (
                _load_library(self.path)
            )
            if embedded_source != source_hash or embedded_compiler != metadata.get(
                "compiled_compiler"
            ):
                raise SealedNativeError(
                    "native binary source/compiler handshake differs"
                )
            self._identity = {
                "format": FORMAT,
                "abi_version": ABI_VERSION,
                "binary_sha256": sha256,
                "source_sha256": source_hash,
                "wrapper_sha256": _sha256(__file__),
                "build_manifest_sha256": _sha256(self.manifest_path),
                "flags": list(COMPILE_FLAGS),
                "platform": metadata["platform"],
                "compiler": metadata["compiler"],
                "compiled_compiler": embedded_compiler,
            }
            self.check_unchanged()
        except (OSError, KeyError, AttributeError, UnicodeError, ValueError) as exc:
            raise SealedNativeError(
                f"cannot open pinned native interpolator: {exc}"
            ) from exc

    @property
    def identity(self):
        """Return an independent JSON-compatible copy for immutable Run records."""
        self.check_unchanged()
        return json.loads(json.dumps(self._identity))

    def check_unchanged(self):
        try:
            if any(_fingerprint(p) != value for p, value in self._fingerprints.items()):
                raise SealedNativeError(
                    "native binary, manifest, or implementation changed"
                )
        except OSError as exc:
            raise SealedNativeError(
                "native binary, manifest, or implementation unavailable"
            ) from exc

    def interpolate(self, values, u_min, step, u):
        """Interpolate an owned/borrowed column at finite in-column coordinates.

        The enclosing reader enforces its exact declared upper support and handles
        certified left tails. This method allows only a few coordinate-rounding
        ULPs beyond the last stored physical node, just as the Python stencil.
        """
        self.check_unchanged()
        if np.iscomplexobj(values) or np.iscomplexobj(u):
            raise SealedNativeError("native interpolation requires real arrays")
        values = np.ascontiguousarray(values, dtype=np.float64)
        u = np.ascontiguousarray(u, dtype=np.float64)
        if values.ndim != 1 or len(values) < 8 or u.ndim != 1:
            raise SealedNativeError(
                "native interpolation requires a column and 1-D queries"
            )
        u_min, step = float(u_min), float(step)
        if not math.isfinite(u_min) or not math.isfinite(step) or step <= 0:
            raise SealedNativeError("native grid origin/spacing are invalid")
        span = step * (len(values) - 1)
        end = u_min + span
        if not math.isfinite(span) or not math.isfinite(end):
            raise SealedNativeError("native grid endpoints must be finite")
        upper_guard = end + 4 * max(math.ulp(u_min), math.ulp(span), math.ulp(end))
        if not math.isfinite(upper_guard):
            raise SealedNativeError("native grid endpoint has no finite safety guard")
        out = np.empty(len(u), dtype=np.float64)
        failed = ctypes.c_int64(-1)
        status = self._function(
            values.ctypes.data,
            len(values),
            u.ctypes.data,
            len(u),
            u_min,
            step,
            upper_guard,
            out.ctypes.data,
            ctypes.byref(failed),
        )
        self.check_unchanged()
        if status:
            raise SealedNativeError(
                f"native interpolation rejected query {failed.value} (status {status})"
            )
        return out


def _compiler_command(command, *, timeout):
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "no compiler diagnostic").strip()
        raise SealedNativeError(
            f"compiler command failed with exit {exc.returncode}: {detail}"
        ) from exc


def build_native(out, *, compiler="cc"):
    """Compile once to a NEW binary path, then publish its immutable manifest."""
    out = Path(out).absolute()
    sidecar = manifest_path(out)
    if out.exists() or out.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(
            "native binary/manifest already exists; choose a new path"
        )
    executable = shutil.which(str(compiler))
    if executable is None:
        raise SealedNativeError(f"compiler is unavailable: {compiler}")
    # Preserve argv[0]: compiler-dispatch symlinks such as cc -> ccache depend
    # on the selected name. Record and fingerprint the target separately.
    executable = str(Path(executable).absolute())
    resolved_executable = str(Path(executable).resolve(strict=True))
    source_before = _fingerprint(SOURCE_PATH)
    source_hash = _sha256(SOURCE_PATH)
    compiler_before = _fingerprint(executable)
    compiler_hash = _sha256(executable)
    version = _compiler_command([executable, "--version"], timeout=30).stdout.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sealed-native-", dir=out.parent) as temp:
        temp = Path(temp)
        binary = temp / out.name
        command = [
            executable,
            *COMPILE_FLAGS,
            f'-DLSA_SEALED_SOURCE_SHA256="{source_hash}"',
            str(SOURCE_PATH),
            "-o",
            str(binary),
            "-lm",
        ]
        result = _compiler_command(command, timeout=120)
        if (
            _fingerprint(SOURCE_PATH) != source_before
            or _fingerprint(executable) != compiler_before
            or str(Path(executable).resolve(strict=True)) != resolved_executable
        ):
            raise SealedNativeError("source/compiler changed during native build")
        _, _, embedded_source, embedded_compiler = _load_library(binary)
        if embedded_source != source_hash:
            raise SealedNativeError(
                "compiled source digest differs from requested source"
            )
        metadata = {
            "schema_version": 1,
            "format": FORMAT,
            "abi_version": ABI_VERSION,
            "source_sha256": source_hash,
            "binary_sha256": _sha256(binary),
            "flags": list(COMPILE_FLAGS),
            "compiler": {
                "executable": executable,
                "resolved_executable": resolved_executable,
                "sha256": compiler_hash,
                "version": version,
            },
            "compiled_compiler": embedded_compiler,
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        staged_manifest = temp / "manifest.json"
        staged_manifest.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        # Exclusive links prevent concurrent builds from replacing either file.
        # If interrupted between links, the incomplete pair is rejected at open.
        os.link(binary, out)
        os.link(staged_manifest, sidecar)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build an explicit pinned sealed interpolator"
    )
    parser.add_argument(
        "--out", required=True, help="NEW shared-library path, outside Git"
    )
    parser.add_argument(
        "--cc", default="cc", help="explicit compiler executable (default: cc)"
    )
    args = parser.parse_args(argv)
    metadata = build_native(args.out, compiler=args.cc)
    print(
        json.dumps(
            {
                "path": str(Path(args.out).absolute()),
                "sha256": metadata["binary_sha256"],
                "manifest": str(manifest_path(Path(args.out).absolute())),
            },
            indent=2,
        )
    )
    return 0
