"""Explicit, traceable count-profile evaluation for the ALT depth model.

Natural logarithms are used for evidence. Prediction arrays contain uncorrected
probabilities; normalization error is checked and retained, never hidden.
The reference backend is independent nested positive quadrature at depth two.
The store backend uses pinned PMWM numerics, never builds or changes a store,
and requires a passed configuration-specific calibration for production use.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import platform
import threading
import warnings
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import IntegrationWarning, quad
from scipy.special import logsumexp


class NumericalError(RuntimeError):
    """An accuracy or immutable-input check failed."""


class UnsupportedDomain(ValueError):
    """A request is outside the explicitly supported numerical domain."""


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _profile(d, partition):
    d = _integer(d, "d", 1)
    parts = tuple(sorted((_integer(x, "count", 1) for x in partition), reverse=True))
    if len(parts) > d:
        raise ValueError("observed support exceeds d")
    return d, parts


def _depths(values):
    values = tuple(_integer(x, "depth") for x in values)
    if not values or len(set(values)) != len(values):
        raise ValueError("depths must be nonempty and distinct")
    return values


def _array(values):
    value = np.asarray(values, dtype=float)
    value.setflags(write=False)
    return value


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class StoreConfig:
    """Explicit candidate domain/settings, not an accuracy certificate.

    ``files_sha256`` must cover the manifest, anchor grid when used, and all
    level index/data files in the store. Paths are relative to ``path``.
    Saddle substitution is disabled unless ``saddle_min_depth`` is supplied.
    All requested depths must be within ``max_depth``; no depth truncation is
    used. A separate passed calibration enables production for this config.
    """

    path: str | Path
    files_sha256: Mapping[str, str]
    max_depth: int
    max_d: int
    max_n: int
    max_count: int
    ladder_every: int = 1
    ladder_degree: int = 11
    saddle_min_depth: int | None = None
    grid_step: float = 0.02
    minimum_grid_step: float = 0.0025
    u_max: float = 35.0
    scan_mode: str = "full"
    significance_gap: float = 40.0
    minimum_right_gap: float = 30.0
    series_tail_nats: float | None = None

    def __post_init__(self):
        for name in ("max_depth", "max_d", "max_n", "max_count"):
            _integer(getattr(self, name), name, 1)
        _integer(self.ladder_every, "ladder_every")
        _integer(self.ladder_degree, "ladder_degree", 1)
        if self.saddle_min_depth is not None:
            _integer(self.saddle_min_depth, "saddle_min_depth", 2)
        if self.max_depth > 70 and self.saddle_min_depth is None:
            raise UnsupportedDomain("stored-column design ends at depth 70")
        if self.scan_mode not in ("full", "sparse"):
            raise ValueError("scan_mode must be full or sparse")
        for name in (
            "grid_step",
            "minimum_grid_step",
            "significance_gap",
            "minimum_right_gap",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_grid_step > self.grid_step:
            raise ValueError("minimum_grid_step must not exceed grid_step")
        if not math.isfinite(self.u_max):
            raise ValueError("u_max must be finite")
        if self.series_tail_nats is not None and (
            not math.isfinite(self.series_tail_nats) or self.series_tail_nats <= 0
        ):
            raise ValueError("series_tail_nats must be positive or None")


@dataclass(frozen=True)
class EvidenceResult:
    depths: tuple[int, ...]
    log_evidence: np.ndarray
    diagnostics: dict

    @property
    def mixture_log_evidence(self):
        return float(logsumexp(self.log_evidence) - math.log(len(self.depths)))

    @property
    def posterior(self):
        return _array(np.exp(self.log_evidence - logsumexp(self.log_evidence)))


@dataclass(frozen=True)
class PredictionResult:
    depths: tuple[int, ...]
    component_probabilities: np.ndarray
    mixture_probabilities: np.ndarray
    posterior: np.ndarray
    component_log_evidence: np.ndarray
    diagnostics: dict

    @property
    def log_evidence(self):
        return self.component_log_evidence


@dataclass(frozen=True)
class CountPredictionResult:
    depths: tuple[int, ...]
    counts: tuple[int, ...]
    multiplicities: tuple[int, ...]
    component_probabilities: np.ndarray
    mixture_probabilities: np.ndarray
    posterior: np.ndarray
    component_log_evidence: np.ndarray
    diagnostics: dict

    @property
    def by_count(self):
        return dict(zip(self.counts, self.mixture_probabilities, strict=True))


_STORE_LOCK = threading.RLock()


class DepthEvaluator:
    """Depth-zero aware evaluator with explicit backend and diagnostics.

    ``DepthEvaluator(mode="reference")`` supports d<=32, N<=20, L<=2 and
    needs no store. Analytic depths zero/one also support larger profiles.
    ``mode="store"`` requires StoreConfig. Store reads and module settings
    are serialized inside a process; no ambient PMM_* settings are consumed.
    ``purpose="production"`` additionally requires a calibration mapping with
    status="passed" and the evaluator's exact configuration_sha256.
    """

    def __init__(
        self,
        *,
        mode="reference",
        store: StoreConfig | None = None,
        prediction_tolerance=1e-6,
        purpose="validation",
        calibration=None,
    ):
        if mode not in ("reference", "store"):
            raise ValueError("mode must be reference or store")
        if purpose not in ("validation", "production"):
            raise ValueError("purpose must be validation or production")
        if not math.isfinite(prediction_tolerance) or prediction_tolerance <= 0:
            raise ValueError("prediction_tolerance must be finite and positive")
        if (mode == "store") != (store is not None):
            raise ValueError("only store mode requires a StoreConfig")
        # Own the hash mapping rather than retain a caller's mutable dictionary.
        if store is not None:
            store = replace(store, files_sha256=dict(store.files_sha256))
        self.mode, self.store_config = mode, store
        self.prediction_tolerance = float(prediction_tolerance)
        self.purpose = purpose
        self._store = None
        self._fingerprints = {}
        self._reference_cache = {}
        config = None if store is None else asdict(store)
        if config is not None:
            config.pop("path")  # store identity is portable and content-based
            config["files_sha256"] = dict(store.files_sha256)
        vendor = Path(__file__).parent / "_vendor" / "pmwm" / "provenance.json"
        self.configuration = {
            "mode": mode,
            "store": config,
            "prediction_tolerance": self.prediction_tolerance,
            "implementation_sha256": _hash_file(__file__),
            "vendor_provenance_sha256": _hash_file(vendor),
            "vendor_source_sha256": {
                p.name: _hash_file(p) for p in sorted(vendor.parent.glob("*.py"))
            },
            "native_kernel": False,
            "depth_truncation": False,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "system": platform.system(),
                "machine": platform.machine(),
            },
        }
        self.configuration_sha256 = _json_hash(self.configuration)
        self.production_ready = bool(
            mode == "store"
            and calibration
            and calibration.get("status") == "passed"
            and calibration.get("configuration_sha256") == self.configuration_sha256
        )
        if purpose == "production" and not self.production_ready:
            raise NumericalError(
                "production requires passed calibration for this exact configuration"
            )
        if store is not None:
            self._open_store()

    def _open_store(self):
        config = self.store_config
        path = Path(config.path).resolve(strict=True)
        if not path.is_dir():
            raise ValueError("store path must be an existing directory")
        required = {"manifest.json"}
        if config.ladder_every:
            required.add("anchors.json")
        required.update(p.name for p in path.glob("level_*.bin"))
        required.update(p.name for p in path.glob("level_*.index.json"))
        hashes = dict(config.files_sha256)
        if not required <= hashes.keys():
            raise NumericalError(
                f"missing store content hashes: {sorted(required - hashes.keys())}"
            )
        for name, digest in hashes.items():
            file = path / name
            if Path(name).name != name or file.is_symlink() or not file.is_file():
                raise ValueError(
                    "store hash entries must name regular files directly in the store"
                )
            if _hash_file(file) != digest:
                raise NumericalError(f"store hash mismatch: {name}")
            self._fingerprints[file] = self._stat(file)
        from ._vendor.pmwm.universal_tables import UniversalTables

        class ReadOnlyTables(UniversalTables):
            def _save_manifest(self):
                raise NumericalError("store writes are disabled")

            def _append_column(self, *args, **kwargs):
                raise NumericalError("store writes are disabled")

            def _build_column(self, *args, **kwargs):
                raise NumericalError("store builds are disabled")

        self._store = ReadOnlyTables(path, read_only=True)

    @staticmethod
    def _stat(path):
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def _check_store_unchanged(self):
        for path, expected in self._fingerprints.items():
            if self._stat(path) != expected:
                raise NumericalError(
                    f"immutable store changed since verification: {path.name}"
                )

    def close(self):
        if self._store is not None:
            self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _store_context(self):
        from ._vendor.pmwm import _runtime
        from ._vendor.pmwm import universal_tables as ut

        config = self.store_config
        values = {
            "PMM_SCAN": config.scan_mode,
            "PMM_SCAN_LEGACY": "0",
            "PMM_BUILD_EXACT": "1",
            "PMM_INTERP_KERNEL": "0",
        }
        globals_ = {
            "_SADDLE_MIN_L": config.saddle_min_depth or 0,
            "_LADDER_F": 0.0,
            "_LADDER_EVERY": config.ladder_every,
            "_LADDER_DEGREE": config.ladder_degree,
            "_PHI_WAVE": 0.0,
            "_PHI_BIAS": 0.0,
            "SERIES_TAIL_NATS": config.series_tail_nats or math.inf,
        }
        with _STORE_LOCK:
            self._check_store_unchanged()
            old = {name: getattr(ut, name) for name in globals_}
            token = _runtime._settings.set(values)
            try:
                for name, value in globals_.items():
                    setattr(ut, name, value)
                yield
                self._check_store_unchanged()
            finally:
                for name, value in old.items():
                    setattr(ut, name, value)
                _runtime._settings.reset(token)

    def _check_domain(self, d, parts, depths):
        if self.mode == "reference":
            if any(L > 1 for L in depths) and (
                d > 32 or sum(parts) > 20 or max(depths) > 2
            ):
                raise UnsupportedDomain(
                    "reference quadrature supports d<=32, N<=20, depths<=2"
                )
            return
        config = self.store_config
        if (
            d > config.max_d
            or sum(parts) > config.max_n
            or max(parts, default=0) > config.max_count
            or max(depths) > config.max_depth
        ):
            raise UnsupportedDomain("request exceeds declared store evaluation domain")

    @staticmethod
    def _analytic(d, parts, L):
        n = sum(parts)
        if not n or d == 1:
            return 0.0, "empty-or-single-label"
        if L == 0 or n == 1:
            return -n * math.log(d), "uniform" if L == 0 else "symmetry"
        if L == 1:
            return (
                math.lgamma(d)
                - math.lgamma(d + n)
                + sum(math.lgamma(r + 1) for r in parts)
            ), "add-one"
        return None

    def _result(self, depths, values, components):
        return EvidenceResult(
            depths,
            _array(values),
            {
                "mode": self.mode,
                "configuration_sha256": self.configuration_sha256,
                "components": components,
                "units": "nats",
            },
        )

    def evidence(self, d, partition, max_index):
        return self.evidence_at_depths(
            d, partition, range(_integer(max_index, "max_index") + 1)
        )

    def evidence_at_depths(self, d, partition, depths):
        return self._evaluate(d, {0: partition}, _depths(depths))[0]

    def evaluate_profiles(self, d, profiles, max_index):
        """Reuse each level's rows across a mapping or sequence of profiles."""
        depths = tuple(range(_integer(max_index, "max_index") + 1))
        return self.evaluate_profiles_at_depths(d, profiles, depths)

    evidence_batch = evaluate_profiles

    def evaluate_profiles_at_depths(self, d, profiles, depths):
        mapping = isinstance(profiles, Mapping)
        values = profiles if mapping else dict(enumerate(profiles))
        result = self._evaluate(d, values, _depths(depths))
        return result if mapping else [result[i] for i in range(len(result))]

    def _evaluate(self, d, profiles, depths, families=None):
        clean = {key: _profile(d, p)[1] for key, p in profiles.items()}
        d = _integer(d, "d", 1)
        for p in clean.values():
            self._check_domain(d, p, depths)
        out = {key: [] for key in clean}
        diagnostics = {key: [] for key in clean}
        if not clean:
            return {}

        def accept(key, L, value, diagnostic):
            if not math.isfinite(value) or value > 1e-8:
                raise NumericalError(f"invalid log evidence at depth {L}: {value}")
            out[key].append(value)
            diagnostics[key].append({"depth": L, **diagnostic})

        def evaluate_levels():
            for L in depths:
                pending = {}
                for key, p in clean.items():
                    analytic = self._analytic(d, p, L)
                    if analytic is not None:
                        accept(
                            key,
                            L,
                            analytic[0],
                            {"method": analytic[1], "converged": True},
                        )
                    elif self.mode == "reference":
                        value, diag = self._reference_l2(d, p)
                        accept(key, L, value, diag)
                    else:
                        pending[key] = p
                if not pending:
                    continue
                from ._vendor.pmwm.layered import (
                    log_q_lambda_scan,
                    log_q_lambda_scan_family,
                )

                rs = {0, 1, 2}
                for p in pending.values():
                    for r in p:
                        rs.update((r, r + 1, r + 2))
                if families:
                    for base_key, aug_keys in families.items():
                        if base_key in pending:
                            for c in aug_keys:
                                rs.update(range(c, c + 4))
                config = self.store_config
                if config.saddle_min_depth is None or L < config.saddle_min_depth:
                    for name in (f"level_{L:02d}.bin", f"level_{L:02d}.index.json"):
                        if name not in config.files_sha256:
                            raise UnsupportedDomain(
                                f"no pinned store data for depth {L}"
                            )
                lo = min(-70.0, -L * math.log(max(rs) + 1.0) - 40.0)
                # All members at one level use the same grid. If any peak is
                # unresolved, discard the entire attempt and refine together:
                # mixing coarse parents and fine children corrupts ratios.
                step = config.grid_step
                refinements = 0
                while True:
                    grid = np.linspace(
                        lo,
                        config.u_max,
                        math.ceil((config.u_max - lo) / step) + 1,
                    )
                    tables = self._store.level_tables(L, sorted(rs), grid)
                    records = {}

                    def collect(
                        key,
                        depth,
                        value,
                        diagnostic,
                        *,
                        records=records,
                        spacing=float(grid[1] - grid[0]),
                        refinements=refinements,
                        step=step,
                    ):
                        diagnostic.update(
                            outer_grid_step=spacing,
                            grid_refinements=refinements,
                            requested_grid_step=step,
                        )
                        records[key] = (depth, value, diagnostic)

                    try:
                        served = set()
                        if families:
                            for base_key, aug_keys in families.items():
                                if base_key not in pending:
                                    continue
                                base_result, aug_results = log_q_lambda_scan_family(
                                    d=d,
                                    L=L,
                                    base_partition=clean[base_key],
                                    cs=tuple(aug_keys),
                                    tables=tables,
                                    significance_gap=config.significance_gap,
                                )
                                self._accept_scan(collect, base_key, L, base_result)
                                served.add(base_key)
                                for c, key in aug_keys.items():
                                    self._accept_scan(collect, key, L, aug_results[c])
                                    served.add(key)
                        for key, parts in pending.items():
                            if key not in served:
                                result = log_q_lambda_scan(
                                    d=d,
                                    L=L,
                                    partition=parts,
                                    tables=tables,
                                    significance_gap=config.significance_gap,
                                )
                                self._accept_scan(collect, key, L, result)
                        break
                    except NumericalError as exc:
                        if "NARROW" not in str(exc) or step <= config.minimum_grid_step:
                            raise
                        step = max(config.minimum_grid_step, step / 2)
                        refinements += 1
                for key, (depth, value, diagnostic) in records.items():
                    accept(key, depth, value, diagnostic)

        if self.mode == "store":
            with self._store_context():
                evaluate_levels()
        else:
            evaluate_levels()
        return {key: self._result(depths, out[key], diagnostics[key]) for key in clean}

    def _accept_scan(self, accept, key, L, result):
        if (
            not result.converged
            or "NARROW" in result.message
            or "unresolved" in result.message
            or (
                result.right_gap is not None
                and result.right_gap < self.store_config.minimum_right_gap
            )
        ):
            raise NumericalError(
                f"unresolved depth-{L} evidence: {result.message}; right_gap={result.right_gap}"
            )
        accept(
            key,
            L,
            result.log_q,
            {
                "method": result.method,
                "converged": result.converged,
                "message": result.message,
                "right_gap": result.right_gap,
                "left_gap": result.left_gap,
                "peaks": result.peaks,
                "kernel_branch": "direct-contour"
                if self.store_config.saddle_min_depth
                and L >= self.store_config.saddle_min_depth
                else "stored",
            },
        )

    def _reference_l2(self, d, parts):
        key = d, parts
        if key in self._reference_cache:
            return self._reference_cache[key]
        n = sum(parts)
        classes = Counter(parts)
        if d > len(parts):
            classes[0] = d - len(parts)
        max_inner_relative_error = 0.0

        def log_phi(r, u):
            # phi_r^(2)(t) = Gamma(r+1) int x^r e^-x/(1+t*x)^(r+1) dx.
            # x=exp(y), with analytic maximum and positive adaptive quadrature.
            nonlocal max_inner_relative_error
            t = math.exp(u)
            xp = 2.0 * (r + 1) / (1.0 + math.sqrt(1.0 + 4.0 * t * (r + 1)))
            yp = math.log(xp)

            def f(y):
                return (
                    (r + 1) * y - math.exp(y) - (r + 1) * float(np.logaddexp(0, u + y))
                )

            peak = f(yp)
            a, b = min(yp - 60, -u - 60), math.log(r + 1) + 6
            value, error = quad(
                lambda y: math.exp(f(y) - peak),
                a,
                b,
                points=[yp],
                epsabs=2e-11,
                epsrel=2e-11,
                limit=150,
            )
            relative = error / value
            max_inner_relative_error = max(max_inner_relative_error, relative)
            if value <= 0 or relative > 1e-8:
                raise NumericalError("reference kernel quadrature did not converge")
            return math.lgamma(r + 1) + peak + math.log(value)

        def log_integrand(u):
            return n * u + sum(count * log_phi(r, u) for r, count in classes.items())

        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            try:
                peak = max(log_integrand(float(u)) for u in np.linspace(-12, 12, 25))
                left_constant = 2 * sum(math.lgamma(r + 1) for r in parts)
                # Rigorous tails: exp(-tY)<=1 on the left; on the right
                # exp(-z)<= (a/e)^a z^-a with a=r+1/2 for every coordinate.
                right_constant = sum(
                    count
                    * ((r + 0.5) * math.log(r + 0.5) - (r + 0.5) + 2 * math.lgamma(0.5))
                    for r, count in classes.items()
                )
                rate = d / 2
                a = min(-20.0, (peak - 40 - left_constant + math.log(n)) / n)
                b = max(20.0, (right_constant - peak + 40 - math.log(rate)) / rate)
                value, error = quad(
                    lambda u: math.exp(log_integrand(u) - peak),
                    a,
                    b,
                    epsabs=2e-9,
                    epsrel=2e-9,
                    limit=150,
                )
            except IntegrationWarning as exc:
                raise NumericalError(f"reference quadrature warning: {exc}") from exc
        tail = math.exp(left_constant + n * a - math.log(n) - peak) + math.exp(
            right_constant - rate * b - math.log(rate) - peak
        )
        relative_error = (error + tail) / value
        if value <= 0 or relative_error > 1e-7:
            raise NumericalError("reference evidence quadrature did not converge")
        result = (
            peak + math.log(value) - math.lgamma(n),
            {
                "method": "independent-positive-quadrature-L2",
                "converged": True,
                "quadrature_relative_error_estimate": relative_error,
                "kernel_relative_error_estimate": max_inner_relative_error,
                "tail_relative_bound": tail / value,
                "u_interval": [a, b],
            },
        )
        self._reference_cache[key] = result
        return result

    def transition_log_evidence(self, d, transitions, *, depths):
        """Joint scans for just the observed next-count class in each transition.

        ``transitions[key] = (base_profile, previous_count_of_next_symbol)``.
        This is the same family evaluation as prediction_by_count, without
        materializing probabilities for unrelated count classes or labels.
        """
        expanded, families = {}, {}
        for key, (partition, count) in transitions.items():
            d, parts = _profile(d, partition)
            count = _integer(count, "previous_count")
            if (count == 0 and len(parts) == d) or (count and count not in parts):
                raise ValueError("next-symbol count class is absent")
            augmented = list(parts)
            if count:
                augmented.remove(count)
            augmented.append(count + 1)
            base_key, child_key = (key, "base"), (key, "child")
            expanded[base_key] = parts
            expanded[child_key] = tuple(sorted(augmented, reverse=True))
            families[base_key] = {count: child_key}
        results = self._evaluate(d, expanded, _depths(depths), families)
        return {
            key: (results[(key, "base")], results[(key, "child")])
            for key in transitions
        }

    def prediction_by_count(self, d, partition, *, depths):
        return self.prediction_by_count_batch(d, {0: partition}, depths=depths)[0]

    def prediction_by_count_batch(self, d, profiles, *, depths):
        """Evaluate base/augmented families, sharing each stored kernel matrix."""
        depths = _depths(depths)
        expanded, families, specifications = {}, {}, {}
        for key, partition in profiles.items():
            d, parts = _profile(d, partition)
            multiplicities = Counter(parts)
            if len(parts) < d:
                multiplicities[0] = d - len(parts)
            classes = tuple(sorted(multiplicities))
            base_key = (key, "base")
            expanded[base_key] = parts
            aug_keys = {}
            for c in classes:
                aug = list(parts)
                if c:
                    aug.remove(c)
                aug.append(c + 1)
                aug_key = (key, c)
                expanded[aug_key] = tuple(sorted(aug, reverse=True))
                aug_keys[c] = aug_key
            families[base_key] = aug_keys
            specifications[key] = (
                classes,
                tuple(multiplicities[c] for c in classes),
                base_key,
            )
        evidence = self._evaluate(d, expanded, depths, families)
        results = {}
        for key, (classes, multiplicities, base_key) in specifications.items():
            base = evidence[base_key]
            augmented = np.stack(
                [evidence[(key, c)].log_evidence for c in classes], axis=1
            )
            probabilities = np.exp(augmented - base.log_evidence[:, None])
            posterior = base.posterior
            mixed = np.exp(logsumexp(augmented, axis=0) - logsumexp(base.log_evidence))
            masses = probabilities @ np.asarray(multiplicities)
            mixed_mass = float(mixed @ np.asarray(multiplicities))
            diagnostics = {
                "configuration_sha256": self.configuration_sha256,
                "component_normalization": masses.tolist(),
                "mixture_normalization": mixed_mass,
                "maximum_normalization_error": float(
                    max(np.max(np.abs(masses - 1)), abs(mixed_mass - 1))
                ),
                "normalization_applied": False,
                "base": base.diagnostics,
                "augmented": {str(c): evidence[(key, c)].diagnostics for c in classes},
            }
            if (
                not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or diagnostics["maximum_normalization_error"]
                > self.prediction_tolerance
            ):
                raise NumericalError(
                    f"predictive normalization check failed: {diagnostics['maximum_normalization_error']}"
                )
            results[key] = CountPredictionResult(
                depths,
                classes,
                multiplicities,
                _array(probabilities),
                _array(mixed),
                posterior,
                base.log_evidence,
                diagnostics,
            )
        return results

    def predict(self, counts, *, depths):
        return self.predict_batch({0: counts}, depths=depths)[0]

    def predict_batch(self, counts_batch, *, depths):
        """Full alphabet arrays; profiles with different d are grouped separately."""
        checked = {
            key: tuple(_integer(c, "count") for c in counts)
            for key, counts in counts_batch.items()
        }
        groups = {}
        for key, counts in checked.items():
            if not counts:
                raise ValueError("counts must include at least one alphabet label")
            groups.setdefault(len(counts), {})[key] = tuple(c for c in counts if c)
        results = {}
        for d, profiles in groups.items():
            by_count = self.prediction_by_count_batch(d, profiles, depths=depths)
            for key, value in by_count.items():
                position = {c: j for j, c in enumerate(value.counts)}
                idx = [position[c] for c in checked[key]]
                results[key] = PredictionResult(
                    value.depths,
                    _array(value.component_probabilities[:, idx]),
                    _array(value.mixture_probabilities[idx]),
                    value.posterior,
                    value.component_log_evidence,
                    value.diagnostics,
                )
        return results
