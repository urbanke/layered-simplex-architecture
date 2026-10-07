"""Saved common samples and per-trial records for the retained ALT benchmark.

This module has no numerical-engine import. Evaluators are injected explicitly;
an enabled method with a missing evaluator is an error. Each invocation is one
sample set, so the primary and power replication are distinct immutable runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .baselines import (
    BASELINE_METHODS,
    evaluate_baseline,
    kl_bits,
    validate_probabilities,
)

TARGET_IDS = (
    "uniform", "step", "zipf_1", "zipf_1.5", "zipf_2", "zipf_3", "zipf_4",
    "zipf_5", "geometric", "dirichlet_1", "dirichlet_half",
)
HISTORICAL_N_GRID = (1000, 2000, 3000, 5000, 7000, 10000, 14000, 20000)
PRIMARY_METHODS = (
    "add_one", "kt", "ristad", "good_turing_hybrid",
    "dirichlet_concentration_mixture", "absolute_discounting",
    "lsa_fixed", "lsa_depth_mixture", "oracle",
)
POWER_METHODS = (*PRIMARY_METHODS[:-1], "power_mixture", "oracle")
METHOD_IDS = (*BASELINE_METHODS, "lsa_fixed", "lsa_depth_mixture", "power_mixture")
SEED_SCHEME = "numpy.default_rng([seed, canonical_target_index, trial]); target then n_values in order"


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        # Diagnostics may include log(0). JSON never contains nonstandard NaN.
        return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    return value


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf8") as stream:
        json.dump(jsonable(payload), stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < (0 if zero else 1):
        raise ValueError(f"{name} is outside its domain")
    return int(value)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "d", "n_values", "trials", "seed", "sample_set_id", "targets", "methods",
        "depths", "fixed_depth", "powers", "normalize_numerical",
        "numerical_normalization_tolerance", "dirichlet_exponents",
        "dirichlet_target_policy", "sample_size_policy",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing explicit benchmark settings: {sorted(missing)}")
    out = dict(config)
    for key in ("d", "trials"):
        out[key] = _positive_int(out[key], key)
    out["seed"] = _positive_int(out["seed"], "seed", zero=True)
    for key in ("n_values", "depths", "powers"):
        values = [_positive_int(v, key, zero=key != "n_values") for v in out[key]]
        if len(set(values)) != len(values) or values != sorted(values):
            raise ValueError(f"{key} must be distinct and sorted")
        out[key] = values
    if not out["n_values"]:
        raise ValueError("n_values must not be empty")
    for key, allowed in (("targets", TARGET_IDS), ("methods", METHOD_IDS)):
        if not out[key] or len(set(out[key])) != len(out[key]):
            raise ValueError(f"{key} must be nonempty and unique")
        if set(out[key]) - set(allowed):
            raise ValueError(f"unknown {key}: {sorted(set(out[key]) - set(allowed))}")
        out[key] = list(out[key])
    if "step" in out["targets"] and out["d"] % 2:
        raise ValueError("the half-and-half step target requires even d")
    if not isinstance(out["sample_set_id"], str) or not out["sample_set_id"].strip():
        raise ValueError("sample_set_id must be a nonempty string")
    if out["dirichlet_target_policy"] != "redraw_per_trial_shared_across_n":
        raise ValueError("unsupported Dirichlet target policy; declare redraw_per_trial_shared_across_n")
    if out["sample_size_policy"] != "independent_multinomial_per_n":
        raise ValueError("unsupported sample-size policy; no implicit nested-prefix samples")
    if type(out["normalize_numerical"]) is not bool:
        raise ValueError("normalize_numerical must be an explicit bool")
    tolerance = float(out["numerical_normalization_tolerance"])
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("numerical_normalization_tolerance must be positive and finite")
    out["numerical_normalization_tolerance"] = tolerance
    if any(method.startswith("lsa_") for method in out["methods"]) and not out["depths"]:
        raise ValueError("LSA methods require an explicit nonempty depth grid")
    if "lsa_fixed" in out["methods"] and out["fixed_depth"] not in out["depths"]:
        raise ValueError("fixed_depth is absent from the evaluated depth grid")
    if "power_mixture" in out["methods"] and not out["powers"]:
        raise ValueError("power mixture requires an explicit nonempty power grid")
    exponents = list(out["dirichlet_exponents"])
    if not exponents or len(set(exponents)) != len(exponents):
        raise ValueError("Dirichlet exponent grid must be explicit, nonempty and unique")
    if any(isinstance(j, bool) or not isinstance(j, (int, np.integer)) for j in exponents):
        raise ValueError("Dirichlet exponents must be integers")
    out["dirichlet_exponents"] = list(map(int, exponents))
    return out


def make_target(name: str, d: int, rng: np.random.Generator) -> np.ndarray:
    if name == "uniform":
        p = np.full(d, 1.0 / d)
    elif name == "step":
        if d % 2:
            raise ValueError("step target requires even d")
        p = np.r_[np.full(d // 2, 1 / (2 * d)), np.full(d // 2, 3 / (2 * d))]
    elif name.startswith("zipf_") and name in TARGET_IDS:
        p = np.arange(1, d + 1, dtype=float) ** (-float(name.split("_")[1]))
        p /= p.sum()
    elif name == "geometric":
        p = np.exp(np.arange(1, d + 1) * np.log(0.998))
        p /= p.sum()
    elif name in ("dirichlet_1", "dirichlet_half"):
        p = rng.dirichlet(np.full(d, 1.0 if name == "dirichlet_1" else 0.5))
    else:
        raise ValueError(f"unknown target {name!r}")
    return validate_probabilities(p, size=d)


def _sampling_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in (
        "d", "n_values", "trials", "seed", "sample_set_id", "targets",
        "dirichlet_target_policy", "sample_size_policy",
    )}


def prepare_samples(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Persist targets and every labelled count vector before scoring any method.

    One compressed file per target/trial keeps memory bounded. File hashes and
    RNG coordinates are recorded. Existing artifacts are never overwritten.
    """
    config = validate_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    files = []
    for target in config["targets"]:
        for trial in range(config["trials"]):
            coordinates = [config["seed"], TARGET_IDS.index(target), trial]
            rng = np.random.default_rng(coordinates)
            p = make_target(target, config["d"], rng)
            arrays = {"target": p}
            for n in config["n_values"]:
                arrays[f"counts_{n}"] = rng.multinomial(n, p)
            path = output_dir / f"{target}-trial-{trial:04d}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            files.append({
                "target_id": target, "trial": trial, "rng_coordinates": coordinates,
                "path": path.name, "sha256": sha256_file(path),
            })
    manifest = {
        "schema_version": 1, "sampling": _sampling_spec(config),
        "seed_scheme": SEED_SCHEME, "numpy_version": np.__version__, "files": files,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _attr(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        if name not in result:
            raise ValueError(f"evaluator result lacks {name}")
        return result[name]
    if not hasattr(result, name):
        raise ValueError(f"evaluator result lacks {name}")
    return getattr(result, name)


def _numeric_prediction(q: Any, config: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    q = np.asarray(q, dtype=float)
    if q.shape != (config["d"],) or np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ArithmeticError("numerical predictor is nonfinite, negative, or has wrong shape")
    mass = float(q.sum())
    error = abs(mass - 1)
    if mass <= 0 or error > config["numerical_normalization_tolerance"]:
        raise ArithmeticError(f"raw predictive normalization error {error:.8g} exceeds gate")
    if config["normalize_numerical"]:
        q = q / mass
    validate_probabilities(q, size=config["d"])
    return q, {"raw_mass": mass, "raw_mass_error": error,
               "normalization_applied": config["normalize_numerical"]}


def loss_record(value: float | None, *, reason: str | None = None) -> dict[str, Any]:
    if value is None:
        return {"status": "undefined", "bits": None, "reason": reason}
    if np.isnan(value) or value == -np.inf:
        raise ArithmeticError("invalid KL value")
    if np.isposinf(value):
        return {"status": "infinite", "bits": None, "reason": reason or "zero_predictive_probability"}
    return {"status": "finite", "bits": float(value), "reason": reason}


def paired_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict:
    if left["status"] != "finite" or right["status"] != "finite":
        return {"status": "undefined", "bits": None,
                "reason": f"nonfinite_pair:{left['status']}:{right['status']}"}
    return {"status": "finite", "bits": left["bits"] - right["bits"], "reason": None}


def summarize_losses(losses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every trial contributes; no finite-only average disguises AD failures."""
    if not losses:
        raise ValueError("cannot summarize no trials")
    statuses = {name: sum(x["status"] == name for x in losses)
                for name in ("finite", "infinite", "undefined")}
    if sum(statuses.values()) != len(losses):
        raise ValueError("unknown loss status")
    result = {"trials": len(losses), "counts": statuses, "mean_bits": None, "se_bits": None}
    if statuses["undefined"]:
        result["status"] = "undefined"
    elif statuses["infinite"]:
        result["status"] = "infinite"
    else:
        values = np.asarray([x["bits"] for x in losses])
        result.update(status="finite", mean_bits=float(values.mean()),
                      se_bits=float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None)
    return result


def _evaluate_trial(config: Mapping[str, Any], p: np.ndarray, counts: np.ndarray,
                    depth_evaluator: Any, power_evaluator: Any) -> dict:
    losses, diagnostics, posteriors = {}, {}, {}
    for method in config["methods"]:
        if method not in BASELINE_METHODS:
            continue
        prediction = evaluate_baseline(method, counts, target=p,
                                       dirichlet_exponents=config["dirichlet_exponents"])
        losses[method] = loss_record(
            None if prediction.probabilities is None else kl_bits(p, prediction.probabilities),
            reason=prediction.reason,
        )
        diagnostics[method] = prediction.diagnostics
        if "posterior" in prediction.diagnostics:
            posteriors[method] = {
                "indices": config["dirichlet_exponents"],
                "weights": prediction.diagnostics["posterior"],
            }
    derived = {}
    for family, evaluator, grid_key, mixture_method in (
        ("depth", depth_evaluator, "depths", "lsa_depth_mixture"),
        ("power", power_evaluator, "powers", "power_mixture"),
    ):
        needed = mixture_method in config["methods"] or (
            family == "depth" and "lsa_fixed" in config["methods"])
        if not needed:
            continue
        grid = config[grid_key]
        result = evaluator.predict(counts, **{grid_key: grid})
        components = np.asarray(_attr(result, "component_probabilities"), dtype=float)
        posterior = np.asarray(_attr(result, "posterior"), dtype=float)
        log_evidence = np.asarray(_attr(result, "component_log_evidence"), dtype=float)
        if components.shape != (len(grid), config["d"]) or log_evidence.shape != (len(grid),):
            raise ArithmeticError("evaluator returned incompatible component dimensions")
        if np.any(~np.isfinite(log_evidence)):
            raise ArithmeticError("every requested model must have finite log evidence")
        validate_probabilities(posterior, size=len(grid))
        expected_posterior = np.exp(log_evidence - logsumexp(log_evidence))
        if not np.allclose(posterior, expected_posterior, rtol=1e-8, atol=1e-12):
            raise ArithmeticError("posterior does not match the declared equal-prior evidence rule")
        # Retain every component's raw mass, not just the dominant mixture.
        component_diagnostics = []
        for row in components:
            _, diag = _numeric_prediction(row, config)
            component_diagnostics.append(diag)
        diagnostics[family] = {
            "evaluator": _attr(result, "diagnostics"), "indices": grid,
            "component_normalization": component_diagnostics,
            "component_log_evidence_nats": log_evidence.tolist(),
        }
        posteriors[family] = {"indices": grid, "weights": posterior.tolist()}
        mixture = np.asarray(_attr(result, "mixture_probabilities"), dtype=float)
        if not np.allclose(mixture, posterior @ components, rtol=1e-8, atol=1e-14):
            raise ArithmeticError("mixture prediction is inconsistent with its components")
        if mixture_method in config["methods"]:
            q, diagnostics[mixture_method] = _numeric_prediction(mixture, config)
            losses[mixture_method] = loss_record(kl_bits(p, q))
        if family == "depth" and "lsa_fixed" in config["methods"]:
            q, diagnostics["lsa_fixed"] = _numeric_prediction(components[grid.index(config["fixed_depth"])], config)
            losses["lsa_fixed"] = loss_record(kl_bits(p, q))
        if family == "depth" and 0 in grid and len(grid) > 1:
            positive = np.asarray(grid) > 0
            weights = np.exp(log_evidence[positive] - logsumexp(log_evidence[positive]))
            q, diag = _numeric_prediction(weights @ components[positive], config)
            derived["positive_depth_mixture"] = {
                "loss": loss_record(kl_bits(p, q)), "normalization": diag,
                "indices": list(np.asarray(grid)[positive]), "posterior": weights.tolist(),
            }
    if set(losses) != set(config["methods"]):
        raise RuntimeError("an enabled method was not evaluated")
    paired = {
        f"{left}_minus_{right}": paired_difference(losses[left], losses[right])
        for i, left in enumerate(config["methods"])
        for right in config["methods"][i + 1:]
    }
    if "power_mixture" in losses and "lsa_depth_mixture" in losses:
        paired["power_mixture_minus_lsa_depth_mixture"] = paired_difference(
            losses["power_mixture"], losses["lsa_depth_mixture"])
    if "positive_depth_mixture" in derived and "lsa_depth_mixture" in losses:
        paired["lsa_depth_mixture_minus_positive_depth_mixture"] = paired_difference(
            losses["lsa_depth_mixture"], derived["positive_depth_mixture"]["loss"])
    return {"losses": losses, "posteriors": posteriors, "diagnostics": diagnostics,
            "paired_differences": paired, "derived": derived}


def aggregate_records(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict:
    cells = {}
    for target in config["targets"]:
        cells[target] = {}
        for n in config["n_values"]:
            trials = [r for r in records if r["target_id"] == target and r["n"] == n]
            if sorted(r["trial"] for r in trials) != list(range(config["trials"])):
                raise ValueError(f"incomplete or duplicate trials for {target}, n={n}")
            cell = {"methods": {method: summarize_losses([r["losses"][method] for r in trials])
                                for method in config["methods"]}}
            pair_names = set(trials[0]["paired_differences"])
            if any(set(r["paired_differences"]) != pair_names for r in trials):
                raise ValueError("inconsistent paired comparison coverage")
            cell["paired_differences"] = {
                name: summarize_losses([r["paired_differences"][name] for r in trials])
                for name in sorted(pair_names)
            }
            cell["mean_posteriors"] = {}
            for family, first in trials[0]["posteriors"].items():
                for r in trials:
                    if r["posteriors"][family]["indices"] != first["indices"]:
                        raise ValueError("posterior grids differ across trials")
                cell["mean_posteriors"][family] = {
                    "indices": first["indices"],
                    "weights": np.mean([r["posteriors"][family]["weights"] for r in trials], axis=0).tolist(),
                }
            cell["derived"] = {}
            if "positive_depth_mixture" in trials[0]["derived"]:
                cell["derived"]["positive_depth_mixture"] = summarize_losses(
                    [r["derived"]["positive_depth_mixture"]["loss"] for r in trials])
            cells[target][str(n)] = cell
    return {"schema_version": 1, "config": config, "units": "next-symbol KL bits",
            "uncertainty": "sample standard deviation / sqrt(trials), ddof=1; null for one trial",
            "nonfinite_policy": "undefined if any trial undefined, else infinite if any trial infinite; no omission",
            "targets": cells}


def run_benchmark(config: Mapping[str, Any], output_dir: str | Path, *,
                  depth_evaluator: Any, power_evaluator: Any = None,
                  samples_dir: str | Path | None = None) -> dict:
    """Score one immutable sample set; optionally reuse its saved count files.

    Evaluator.predict returns component_probabilities [models,d],
    mixture_probabilities [d], posterior, component_log_evidence (nats),
    and diagnostics. It accepts explicit depths/powers, respectively.
    """
    config = validate_config(config)
    if any(m.startswith("lsa_") for m in config["methods"]) and depth_evaluator is None:
        raise ValueError("enabled LSA method has no depth evaluator")
    if "power_mixture" in config["methods"] and power_evaluator is None:
        raise ValueError("enabled power mixture has no power evaluator")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse overwrite before doing work; root runtime may pre-create the folder.
    for name in ("benchmark-config.json", "trials.jsonl", "summary.json"):
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)
    _write_json(output_dir / "benchmark-config.json", config)
    if samples_dir is None:
        samples_dir = output_dir / "samples"
        sample_manifest = prepare_samples(config, samples_dir)
    else:
        samples_dir = Path(samples_dir)
        sample_manifest = json.loads((samples_dir / "manifest.json").read_text())
        if sample_manifest["sampling"] != _sampling_spec(config):
            raise ValueError("saved samples do not match this declared sampling protocol")
    records = []
    started = time.perf_counter()
    with (output_dir / "trials.jsonl").open("x", encoding="utf8") as stream:
        for entry in sample_manifest["files"]:
            path = Path(samples_dir) / entry["path"]
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"sample checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as sample:
                p = sample["target"]
                for n in config["n_values"]:
                    counts = sample[f"counts_{n}"]
                    if counts.shape != (config["d"],) or int(counts.sum()) != n:
                        raise ValueError("saved count vector does not match n,d")
                    trial_start = time.perf_counter()
                    record = _evaluate_trial(config, p, counts, depth_evaluator, power_evaluator)
                    record.update(
                        sample_set_id=config["sample_set_id"], target_id=entry["target_id"],
                        trial=entry["trial"], n=n, sample_file=entry["path"],
                        sample_sha256=entry["sha256"], seconds=time.perf_counter() - trial_start,
                    )
                    record = jsonable(record)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    # Detailed quadrature logs can be large; retain them on
                    # disk and keep only aggregation inputs in memory.
                    records.append({k: v for k, v in record.items() if k != "diagnostics"})
    summary = aggregate_records(records, config)
    summary.update(seconds=time.perf_counter() - started,
                   sample_manifest_sha256=sha256_file(Path(samples_dir) / "manifest.json"),
                   trial_records_sha256=sha256_file(output_dir / "trials.jsonl"))
    _write_json(output_dir / "summary.json", summary)
    return summary
