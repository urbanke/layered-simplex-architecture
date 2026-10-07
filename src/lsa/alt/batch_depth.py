"""Bounded batching and exchangeable-profile caching around DepthEvaluator.

The numerical evaluator, model grid, prior weights, evidence and probabilities
are unchanged. A cohort shares kernel preparation through the existing batch
API. Only compact count-class predictions are cached; full alphabet arrays are
created when a caller requests its labelled prediction.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import validate_counts
from .depth import CountPredictionResult, EvidenceResult, PredictionResult


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _depths(values):
    grid = tuple(_integer(L, "depth", 0) for L in values)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("depths must be nonempty and distinct")
    return grid


def _readonly(values):
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ProfileKey:
    d: int
    positive_count_multiplicities: tuple[tuple[int, int], ...]
    depths: tuple[int, ...]

    @property
    def n(self):
        return sum(
            count * multiplicity
            for count, multiplicity in self.positive_count_multiplicities
        )

    @property
    def partition(self):
        return tuple(
            count
            for count, multiplicity in reversed(self.positive_count_multiplicities)
            for _ in range(multiplicity)
        )

    @property
    def sha256(self):
        payload = [self.d, self.positive_count_multiplicities, self.depths]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest()


def _profile_key(counts, depths):
    m = validate_counts(counts)
    values, multiplicities = np.unique(m[m > 0], return_counts=True)
    frequencies = tuple(
        (int(count), int(multiplicity))
        for count, multiplicity in zip(values, multiplicities, strict=True)
    )
    return m, ProfileKey(len(m), frequencies, depths)


class BatchedDepthEvaluator:
    """Prepare one bounded cohort, then serve ordinary ``predict`` calls.

    Example: ``prepare_predictions(twenty_count_vectors, depths=range(81))``
    evaluates all new profiles in chunks. ``predict`` remaps their count classes
    to the original label order. The default is an explicit error on a cache
    miss; ``on_miss='evaluate'`` opts into ordinary single-profile evaluation.
    A cohort larger than the cache raises instead of silently evicting members
    before the caller can consume them. Use several cohorts for longer runs.
    """

    def __init__(
        self,
        evaluator,
        *,
        chunk_size: int,
        max_cached_profiles: int,
        on_miss: str = "error",
    ):
        self.evaluator = evaluator
        self.chunk_size = _integer(chunk_size, "chunk_size")
        self.max_cached_profiles = _integer(max_cached_profiles, "max_cached_profiles")
        if on_miss not in ("error", "evaluate"):
            raise ValueError("on_miss must be 'error' or 'evaluate'")
        self.on_miss = on_miss
        self._cache: OrderedDict[ProfileKey, CountPredictionResult] = OrderedDict()
        self._closed = False
        self._cohort_number = 0
        self._statistics = {
            "prepared_samples": 0,
            "evaluated_profiles": 0,
            "kernel_batches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "preparation_seconds": 0.0,
        }
        self.configuration = {
            "base_evaluator": deepcopy(getattr(evaluator, "configuration", {})),
            "chunk_size": self.chunk_size,
            "max_cached_profiles": self.max_cached_profiles,
            "on_miss": on_miss,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "cache_key": "alphabet size, positive count multiplicities, ordered depth grid",
            "normalization_applied": False,
        }
        self.configuration_sha256 = hashlib.sha256(
            json.dumps(self.configuration, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("batch evaluator is closed")

    @property
    def production_ready(self):
        return bool(getattr(self.evaluator, "production_ready", False))

    def cache_info(self):
        return {
            **self._statistics,
            "cached_profiles": len(self._cache),
            "max_cached_profiles": self.max_cached_profiles,
            "cached_array_bytes": sum(
                value.component_probabilities.nbytes
                + value.mixture_probabilities.nbytes
                + value.posterior.nbytes
                + value.component_log_evidence.nbytes
                for value in self._cache.values()
            ),
            "memory_note": "array bytes exclude retained original diagnostics and Python object overhead",
        }

    def clear(self):
        self._cache.clear()

    def close(self):
        self.clear()
        self._closed = True
        self.evaluator.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _insert(self, key, result, protected):
        while key not in self._cache and len(self._cache) >= self.max_cached_profiles:
            victim = next(
                (candidate for candidate in self._cache if candidate not in protected),
                None,
            )
            if victim is None:
                raise RuntimeError("prepared cohort exceeds cache capacity")
            del self._cache[victim]
            self._statistics["evictions"] += 1
        self._cache[key] = result
        self._cache.move_to_end(key)

    def prepare_predictions(
        self,
        counts_batch: Mapping[Any, Sequence[int]] | Iterable[Sequence[int]],
        *,
        depths: Sequence[int],
    ) -> dict:
        self._check_open()
        grid = _depths(depths)
        values = (
            counts_batch.values() if isinstance(counts_batch, Mapping) else counts_batch
        )
        cohort: OrderedDict[ProfileKey, None] = OrderedDict()
        samples = 0
        for counts in values:
            _, key = _profile_key(counts, grid)
            cohort[key] = None
            samples += 1
            if len(cohort) > self.max_cached_profiles:
                raise ValueError(
                    "cohort has more unique profiles than cache capacity; consume smaller cohorts"
                )
        if not samples:
            raise ValueError("prediction cohort must not be empty")
        protected = set(cohort)
        groups = {}
        for key in cohort:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                groups.setdefault((key.d, key.n), []).append(key)
        started = time.perf_counter()
        evaluated, batches = 0, 0
        cohort_number = self._cohort_number
        self._cohort_number += 1
        for (d, n), keys in groups.items():
            for offset in range(0, len(keys), self.chunk_size):
                chunk = keys[offset : offset + self.chunk_size]
                profiles = {i: key.partition for i, key in enumerate(chunk)}
                results = self.evaluator.prediction_by_count_batch(
                    d, profiles, depths=grid
                )
                if set(results) != set(profiles):
                    raise ArithmeticError(
                        "batch evaluator returned incomplete or extra profiles"
                    )
                for i, key in enumerate(chunk):
                    value = results[i]
                    if tuple(value.depths) != grid:
                        raise ArithmeticError(
                            "batch evaluator changed requested depth order"
                        )
                    expected = dict(key.positive_count_multiplicities)
                    unseen = d - sum(expected.values())
                    if unseen:
                        expected[0] = unseen
                    if (
                        dict(zip(value.counts, value.multiplicities, strict=True))
                        != expected
                    ):
                        raise ArithmeticError(
                            "count-class result does not match prepared profile"
                        )
                    if value.component_probabilities.shape != (
                        len(grid),
                        len(expected),
                    ):
                        raise ArithmeticError("count-class prediction shape mismatch")
                    diagnostics = deepcopy(value.diagnostics)
                    diagnostics["batch_preparation"] = {
                        "cohort_number": cohort_number,
                        "chunk_number": batches,
                        "chunk_unique_profiles": len(chunk),
                        "d": d,
                        "n": n,
                        "profile_sha256": key.sha256,
                        "adapter_configuration_sha256": self.configuration_sha256,
                    }
                    compact = CountPredictionResult(
                        grid,
                        tuple(value.counts),
                        tuple(value.multiplicities),
                        _readonly(value.component_probabilities),
                        _readonly(value.mixture_probabilities),
                        _readonly(value.posterior),
                        _readonly(value.component_log_evidence),
                        diagnostics,
                    )
                    self._insert(key, compact, protected)
                evaluated += len(chunk)
                batches += 1
        seconds = time.perf_counter() - started
        self._statistics["prepared_samples"] += samples
        self._statistics["evaluated_profiles"] += evaluated
        self._statistics["kernel_batches"] += batches
        self._statistics["preparation_seconds"] += seconds
        if not protected <= self._cache.keys():
            raise RuntimeError("a prepared profile was evicted prematurely")
        return {
            "samples": samples,
            "unique_profiles": len(cohort),
            "newly_evaluated_profiles": evaluated,
            "kernel_batches": batches,
            "seconds": seconds,
            "cache": self.cache_info(),
        }

    prepare = prepare_predictions

    def predict(self, counts, *, depths):
        self._check_open()
        grid = _depths(depths)
        m, key = _profile_key(counts, grid)
        hit = key in self._cache
        if not hit:
            self._statistics["cache_misses"] += 1
            if self.on_miss == "error":
                raise KeyError(
                    "count profile/depth grid was not prepared (or has been evicted)"
                )
            self.prepare_predictions([m], depths=grid)
        else:
            self._statistics["cache_hits"] += 1
        compact = self._cache[key]
        self._cache.move_to_end(key)
        classes = np.asarray(compact.counts)
        columns = np.searchsorted(classes, m)
        if np.any(columns >= len(classes)) or np.any(classes[columns] != m):
            raise ArithmeticError("prepared count classes do not cover labelled counts")
        diagnostics = deepcopy(compact.diagnostics)
        diagnostics["batch_cache"] = {
            "hit": hit,
            "profile_sha256": key.sha256,
            "label_mapping": "exact observed count class",
            "normalization_applied": False,
        }
        return PredictionResult(
            grid,
            _readonly(compact.component_probabilities[:, columns]),
            _readonly(compact.mixture_probabilities[columns]),
            _readonly(compact.posterior),
            _readonly(compact.component_log_evidence),
            diagnostics,
        )


def iter_evidence_batches(
    evaluator,
    profiles: Iterable[tuple[Any, Sequence[int]]],
    *,
    d: int,
    depths: Sequence[int],
    chunk_size: int,
):
    """Yield every input profile's EvidenceResult in bounded chunks and order.

    Profiles are positive-count partitions, and inputs can be streamed from the
    saved sample file. Equal profiles within a chunk are evaluated once. IDs may
    repeat across chunks; records are never dropped. Original diagnostics are
    retained, with chunk metadata added. At most one chunk is held in memory.
    """
    d = _integer(d, "d")
    grid = _depths(depths)
    chunk_size = _integer(chunk_size, "chunk_size")
    iterator = iter(profiles)
    chunk_number = 0
    while chunk := list(itertools.islice(iterator, chunk_size)):
        unique, positions = OrderedDict(), []
        for identifier, parts in chunk:
            partition = tuple(
                sorted(
                    (_integer(value, "positive count") for value in parts), reverse=True
                )
            )
            if len(partition) > d:
                raise ValueError("profile support exceeds d")
            if partition not in unique:
                unique[partition] = len(unique)
            positions.append((identifier, unique[partition]))
        evaluated = evaluator.evaluate_profiles_at_depths(
            d, {index: parts for parts, index in unique.items()}, grid
        )
        if set(evaluated) != set(unique.values()):
            raise ArithmeticError(
                "batch evaluator returned incomplete or extra evidence profiles"
            )
        for identifier, index in positions:
            value = evaluated[index]
            if tuple(value.depths) != grid:
                raise ArithmeticError("batch evidence changed requested depth order")
            diagnostics = deepcopy(value.diagnostics)
            diagnostics["batch_preparation"] = {
                "chunk_number": chunk_number,
                "input_profiles": len(chunk),
                "unique_profiles": len(unique),
            }
            yield (
                identifier,
                EvidenceResult(grid, _readonly(value.log_evidence), diagnostics),
            )
        chunk_number += 1
