"""Read-only, fully precomputed product-moment kernels.

Serving uses polynomial interpolation and an analytic far-left limit only.
There is no import of the contour evaluator and no table construction path.
The final evidence integration remains the responsibility of ``DepthEvaluator``.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.special import loggamma

FORMAT = "lsa-sealed-kernels-v1"
_STENCIL = 8
_COUNT_STENCIL = 12
LEFT_LIMIT_LOG_ERROR_BOUND = -math.log1p(-math.exp(-60.0))


def _compensated_sum(rows):
    """Neumaier summation across a short stencil, vectorized over queries."""
    total = np.zeros(rows.shape[1], dtype=np.float64)
    correction = np.zeros_like(total)
    for term in rows:
        updated = total + term
        correction += np.where(
            np.abs(total) >= np.abs(term), (total - updated) + term,
            (term - updated) + total,
        )
        total = updated
    return total + correction


def _count_coefficients(anchors, r):
    # Center at the query before taking logarithms. Subtracting two rounded
    # log(r+1) values near14 can create micro-nat errors for counts near a million.
    xs = np.array([math.log1p((a - r) / (r + 1.)) for a in anchors])
    weights = np.ones(len(anchors))
    for i in range(len(anchors)):
        for j in range(len(anchors)):
            if i != j:
                weights[i] /= xs[i] - xs[j]
    weights /= -xs
    return weights / math.fsum(weights)


def _count_interpolate(values, coefficients, center):
    origin = values[center]
    return origin + _compensated_sum(coefficients[:, None] * (values - origin))


class SealedTableError(ValueError):
    """A sealed store is malformed, changed, or cannot cover a query."""


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise SealedTableError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise SealedTableError(f"{name} must be an integer") from exc
    if result < minimum:
        raise SealedTableError(f"{name} must be at least {minimum}")
    return result


def _finite(value, name):
    if isinstance(value, bool):
        raise SealedTableError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SealedTableError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise SealedTableError(f"{name} must be finite")
    return result


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SealedTableError(f"duplicate JSON key {key!r}: {path.name}")
            result[key] = value
        return result

    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=unique)
    except (OSError, ValueError) as exc:
        raise SealedTableError(f"invalid JSON: {path.name}: {exc}") from exc
    if not isinstance(result, dict):
        raise SealedTableError(f"expected JSON object: {path.name}")
    return result


def _header(record, name):
    if (type(record.get("schema_version")) is not int
            or record["schema_version"] != 1 or record.get("format") != FORMAT):
        raise SealedTableError(f"unsupported sealed-store schema: {name}")


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in value)):
        raise SealedTableError(f"invalid SHA256 for {name}")
    return value.lower()


@dataclass(frozen=True)
class _Column:
    offset: int
    length: int
    u_min: float
    sha256: str | None


class SealedKernelTables:
    """Serve the immutable format described by ``plan.json`` and its seal.

    ``files_sha256`` may contain hashes already checked by ``DepthEvaluator``.
    Passing it avoids rereading large files for duplicate hashing; every entry
    required by the seal must match. Plan/manifest identity and all metadata,
    array bounds, finite-value checks and mutation checks remain independent.
    Without that mapping, all sealed file hashes are checked on opening.
    """

    def __init__(self, path: str | Path, *, files_sha256: Mapping | None = None):
        self.path = Path(path).resolve(strict=True)
        if not self.path.is_dir():
            raise SealedTableError("sealed-store path must be a directory")
        self._closed = False
        self._levels = {}
        self._fingerprints = {}
        for name in ("plan.json", "manifest.json"):
            self._regular_file(name)
        plan = _json(self.path / "plan.json")
        manifest = _json(self.path / "manifest.json")
        _header(plan, "plan.json")
        _header(manifest, "manifest.json")
        if manifest.get("sealed") is not True:
            raise SealedTableError("kernel store is not sealed")
        self.plan_sha256 = _sha256(self.path / "plan.json")
        if manifest.get("plan_sha256") != self.plan_sha256:
            raise SealedTableError("manifest does not bind this plan")
        self.grid_step = _finite(plan.get("grid_step"), "grid_step")
        self.maximum_u = _finite(plan.get("u_max"), "u_max")
        if self.grid_step <= 0:
            raise SealedTableError("grid_step must be positive")
        if _finite(plan.get("left_drop"), "left_drop") < 60:
            raise SealedTableError("left_drop must provide at least 60 nats")
        if plan.get("interpolation_degree") != 7 or plan.get("count_degree") != 11:
            raise SealedTableError("sealed reader requires interpolation degrees 7 and 11")
        self.supported_count = _integer(plan.get("support_max_count"), "support_max_count")
        levels = plan.get("levels")
        if not isinstance(levels, list) or not levels:
            raise SealedTableError("plan levels must be a nonempty list")
        self.coverage_depths = tuple(_integer(x, "depth", 2) for x in levels)
        if tuple(sorted(set(self.coverage_depths))) != self.coverage_depths:
            raise SealedTableError("plan levels must be unique and increasing")
        if manifest.get("levels") != levels:
            raise SealedTableError("sealed levels differ from the plan")
        anchors = plan.get("anchors")
        if not isinstance(anchors, dict) or set(anchors) != {str(x) for x in levels}:
            raise SealedTableError("plan anchors must cover exactly the declared levels")
        self._anchors = {}
        for L in levels:
            values = anchors[str(L)]
            if not isinstance(values, list) or not values:
                raise SealedTableError(f"invalid anchors at depth {L}")
            values = tuple(_integer(r, "anchor") for r in values)
            if tuple(sorted(set(values))) != values:
                raise SealedTableError(f"anchors must be unique and increasing at depth {L}")
            if values[0] != 0 or values[-1] < self.supported_count:
                raise SealedTableError(f"anchors do not cover the count domain at depth {L}")
            self._anchors[L] = values
        file_hashes = manifest.get("files")
        if not isinstance(file_hashes, dict):
            raise SealedTableError("manifest files must be a hash mapping")
        required = {f"level_{L:03d}.{suffix}" for L in levels
                    for suffix in ("bin", "index.json")}
        if not required <= file_hashes.keys():
            raise SealedTableError("manifest is missing required level files")
        checked = {}
        for name, digest in file_hashes.items():
            if name == "manifest.json":
                raise SealedTableError("manifest cannot include its own hash")
            file = self._regular_file(name)
            want = _digest(digest, name)
            if files_sha256 is None:
                actual = _sha256(file)
            else:
                actual = files_sha256.get(name)
            if actual != want:
                raise SealedTableError(f"sealed file hash mismatch: {name}")
            checked[name] = want
        if "plan.json" in checked and checked["plan.json"] != self.plan_sha256:
            raise SealedTableError("manifest plan file hash mismatch")
        if files_sha256 is not None:
            for name in ("plan.json", "manifest.json"):
                if files_sha256.get(name) != _sha256(self.path / name):
                    raise SealedTableError(f"pinned metadata hash mismatch: {name}")
        self.required_files = tuple(sorted(required | set(checked) | {"plan.json", "manifest.json"}))
        self.files_identity = MappingProxyType({**checked, "plan.json": self.plan_sha256,
                                                "manifest.json": _sha256(self.path / "manifest.json")})
        self.plan = _freeze(plan)
        self.manifest = _freeze(manifest)

    @staticmethod
    def _stat(path):
        st = path.stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

    def _regular_file(self, name):
        if not isinstance(name, str) or Path(name).name != name:
            raise SealedTableError("sealed file names must be basenames")
        file = self.path / name
        if file.is_symlink() or not file.is_file():
            raise SealedTableError(f"sealed file is missing or not regular: {name}")
        self._fingerprints[name] = self._stat(file)
        return file

    def _check_open(self, L=None):
        if self._closed:
            raise SealedTableError("sealed table reader is closed")
        names = ["plan.json", "manifest.json"]
        if L is not None:
            names.extend((f"level_{L:03d}.bin", f"level_{L:03d}.index.json"))
        for name in names:
            file = self.path / name
            try:
                unchanged = not file.is_symlink() and self._stat(file) == self._fingerprints[name]
            except OSError:
                unchanged = False
            if not unchanged:
                raise SealedTableError(f"sealed file changed after opening: {name}")

    def _depth(self, L):
        L = _integer(L, "depth", 2)
        if L not in self._anchors:
            raise SealedTableError(f"depth {L} is not covered by this store")
        self._check_open(L)
        return L

    def count_anchors(self, L):
        return self._anchors[self._depth(L)]

    def _load_level(self, L):
        L = self._depth(L)
        if L in self._levels:
            return self._levels[L]
        name = f"level_{L:03d}.index.json"
        index = _json(self.path / name)
        _header(index, name)
        if (_integer(index.get("level"), "index level", 2) != L
                or _finite(index.get("grid_step"), "index grid_step") != self.grid_step
                or _finite(index.get("u_max"), "index u_max") != self.maximum_u):
            raise SealedTableError(f"level/grid identity mismatch: {name}")
        rows = index.get("columns")
        if not isinstance(rows, dict) or set(rows) != {str(r) for r in self._anchors[L]}:
            raise SealedTableError(f"column index differs from planned anchors: {name}")
        columns = {}
        for r in self._anchors[L]:
            row = rows[str(r)]
            if not isinstance(row, dict):
                raise SealedTableError("column index entry must be an object")
            col = _Column(_integer(row.get("offset"), "offset"),
                          _integer(row.get("length"), "length", _STENCIL),
                          _finite(row.get("u_min"), "u_min"),
                          _digest(row["sha256"], "column") if "sha256" in row else None)
            end = col.u_min + self.grid_step * (col.length - 1)
            atol = 16 * np.finfo(float).eps * max(1., abs(col.u_min), abs(self.maximum_u))
            if col.u_min >= self.maximum_u or abs(end - self.maximum_u) > atol:
                raise SealedTableError(f"column grid endpoint mismatch at ({L}, {r})")
            if col.u_min + L * math.log(r + 1.) > -60. + atol:
                raise SealedTableError(f"column leaves an uncertified left gap at ({L}, {r})")
            columns[r] = col
        end = 0
        for col in sorted(columns.values(), key=lambda c: c.offset):
            if col.offset != end:
                raise SealedTableError(f"overlapping or noncontiguous columns at depth {L}")
            end += col.length
        file = self.path / f"level_{L:03d}.bin"
        if file.stat().st_size != 8 * end:
            raise SealedTableError(f"binary length differs from column index at depth {L}")
        data = np.memmap(file, dtype="<f8", mode="r", shape=(end,))
        try:
            for r, col in columns.items():
                values = data[col.offset:col.offset + col.length]
                if not np.all(np.isfinite(values)):
                    raise SealedTableError(f"nonfinite column values at ({L}, {r})")
                if col.sha256 and hashlib.sha256(memoryview(values).cast("B")).hexdigest() != col.sha256:
                    raise SealedTableError(f"column hash mismatch at ({L}, {r})")
            self._check_open(L)
        except BaseException:
            data._mmap.close()
            raise
        result = (MappingProxyType(columns), data)
        self._levels[L] = result
        return result

    def column_metadata(self, L, r):
        columns, _ = self._load_level(L)
        r = _integer(r, "anchor")
        if r not in columns:
            raise SealedTableError(f"count {r} is not an anchor")
        col = columns[r]
        return MappingProxyType({"offset": col.offset, "length": col.length,
                                 "u_min": col.u_min, "sha256": col.sha256})

    def ladder_anchors_for(self, L, r):
        L = self._depth(L)
        r = _integer(r, "count")
        if r > self.supported_count:
            raise SealedTableError(f"count {r} exceeds store support {self.supported_count}")
        anchors = self._anchors[L]
        j = int(np.searchsorted(anchors, r))
        if j < len(anchors) and anchors[j] == r:
            return (r,)
        if len(anchors) < _COUNT_STENCIL:
            raise SealedTableError("non-anchor query needs twelve count anchors")
        # At the dense/geometric join, adjacent ranks give a stencil such as
        # 252..256,267,289,..., which is badly conditioned. Select approximately
        # equal log spacings, using the dense floor to supply the lower nodes.
        tail = np.array([a for a in anchors if a > 256], dtype=np.float64)
        if len(tail) >= _COUNT_STENCIL:
            spacing = float(np.median(np.diff(np.log(tail + 1.))))
            origin = math.log(tail[-min(20, len(tail))] + 1.)
            first = math.floor((math.log(r + 1.) - origin) / spacing) - 5
            targets = origin + (first + np.arange(_COUNT_STENCIL)) * spacing
            logs = np.log(np.asarray(anchors, dtype=np.float64) + 1.)
            chosen = tuple(anchors[int(np.argmin(abs(logs - target)))]
                           for target in targets)
            if (len(set(chosen)) == _COUNT_STENCIL
                    and chosen[0] < r < chosen[-1]):
                return chosen
        lo = max(0, min(j - _COUNT_STENCIL // 2, len(anchors) - _COUNT_STENCIL))
        return anchors[lo:lo + _COUNT_STENCIL]

    def ensure_columns(self, L, r_values):
        """Check coverage and integrity; this method never constructs columns."""
        L = self._depth(L)
        for r in r_values:
            self.ladder_anchors_for(L, r)
        self._load_level(L)

    def _query_grid(self, u):
        if np.iscomplexobj(u):
            raise SealedTableError("query u must be real")
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        if u.ndim != 1 or not np.all(np.isfinite(u)):
            raise SealedTableError("query u must be a finite scalar or one-dimensional array")
        if np.any(u > self.maximum_u):
            raise SealedTableError(f"query exceeds sealed upper bound {self.maximum_u:g}")
        return u

    def _anchor_values(self, L, r, u, columns, data):
        if np.any(u > self.maximum_u):
            raise SealedTableError("shifted anchor query exceeds the sealed upper bound")
        col = columns[r]
        vals = data[col.offset:col.offset + col.length]
        out = np.empty(len(u))
        left = u < col.u_min
        if left.any():
            # 1-exp(-tY) <= tY implies a relative moment deficit <=
            # exp(u) E[Y^(r+1)]/E[Y^r] = exp(u + L log(r+1)).
            if np.any(u[left] + L * math.log(r + 1.) > -60.):
                raise SealedTableError(f"query enters uncertified left gap at ({L}, {r})")
            out[left] = L * float(loggamma(r + 1.))
        inside = ~left
        if inside.any():
            query = u[inside]
            s = (query - col.u_min) / self.grid_step
            starts = np.clip(np.floor(s).astype(np.int64) - 3, 0, len(vals) - _STENCIL)
            indices = starts[:, None] + np.arange(_STENCIL)[None, :]
            # Reconstruct the physical float64 coordinates used by the builder.
            # A nominal uniform index loses ~1e-13 in u on long columns, which
            # becomes ~1e-7 nats when a large-count kernel has slope near1e6.
            nodes = col.u_min + self.grid_step * indices.astype(np.float64)
            node_origin = nodes[:, 3]
            xs = (nodes - node_origin[:, None]) / self.grid_step
            xq = (query - node_origin) / self.grid_step
            dx = xq[:, None] - xs
            exact = query[:, None] == nodes
            weights = np.ones(xs.shape)
            for i in range(_STENCIL):
                for j in range(_STENCIL):
                    if i != j:
                        weights[:, i] /= xs[:, i] - xs[:, j]
            weights /= np.where(exact, 1., dx)
            window = vals[indices]
            value_origin = window[:, 3]
            numerator = _compensated_sum(
                (weights * (window - value_origin[:, None])).T
            )
            values = value_origin + numerator / _compensated_sum(weights.T)
            hit = exact.any(axis=1)
            if hit.any():
                values[hit] = window[np.arange(len(values)), np.argmax(exact, axis=1)][hit]
            out[inside] = values
        return out

    def log_phi_matrix(self, L, r_values, u):
        L = self._depth(L)
        rs = tuple(_integer(r, "count") for r in r_values)
        u = self._query_grid(u)
        per_r = [self.ladder_anchors_for(L, r) for r in rs]
        columns, data = self._load_level(L)
        raw = {}

        def fixed_values(a):
            if a not in raw:
                raw[a] = self._anchor_values(L, a, u, columns, data)
            return raw[a]

        out = np.empty((len(rs), len(u)))
        for row, (r, anchors) in enumerate(zip(rs, per_r)):
            if len(anchors) == 1:
                out[row] = fixed_values(r)
                continue
            coefficients = _count_coefficients(anchors, r)
            center = int(np.argmin([abs(a - r) for a in anchors]))
            shifts = [L * math.log1p((r - a) / (a + 1.)) for a in anchors]
            v = u + L * math.log(r + 1.)
            far_left = v <= -60.
            if far_left.any():
                # Bound the actual requested moment directly, avoiding count
                # interpolation of an already certified constant left limit.
                out[row, far_left] = L * float(loggamma(r + 1.))
            # Align the rapid small-t transition at common v=u+L*log(r+1).
            # Farther right, raw fixed-u values avoid subtracting and restoring
            # a huge gamma term. The handover remains subject to calibration;
            # it is not a uniform error guarantee. Never query beyond the store.
            aligned = ((v <= 2 * L) & ~far_left
                       & (u <= self.maximum_u - max(shifts)))
            if aligned.any():
                values = np.array([
                    self._anchor_values(L, a, u[aligned] + shift, columns, data)
                    - L * float(loggamma(a + 1.))
                    for a, shift in zip(anchors, shifts)
                ])
                out[row, aligned] = (
                    _count_interpolate(values, coefficients, center)
                    + L * float(loggamma(r + 1.))
                )
            fixed = ~(aligned | far_left)
            if fixed.any():
                values = np.array([fixed_values(a)[fixed] for a in anchors])
                out[row, fixed] = _count_interpolate(values, coefficients, center)
        if not np.all(np.isfinite(out)):
            raise SealedTableError("interpolation produced nonfinite values")
        self._check_open(L)
        return out

    def log_phi(self, L, r, u):
        return self.log_phi_matrix(L, (r,), u)[0]

    def level_tables(self, L, r_values, u_grid):
        from ._vendor.pmwm.layered import ProductMomentTables

        rs = tuple(sorted({_integer(r, "count") for r in r_values}))
        if not rs:
            raise SealedTableError("scan tables need at least one count")
        u = self._query_grid(u_grid)
        if len(u) < 2 or np.any(np.diff(u) <= 0):
            raise SealedTableError("scan grid must be strictly increasing")
        matrix = self.log_phi_matrix(L, rs, u)
        return ProductMomentTables.from_matrix(max_L=L, L=L, r_values=rs,
                                              u_grid=u, matrix=matrix)

    def close(self):
        if not self._closed:
            for _, data in self._levels.values():
                data._mmap.close()
            self._levels.clear()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()
