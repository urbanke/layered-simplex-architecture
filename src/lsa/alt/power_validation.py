"""Saved-profile calibration of the single-layer E**w model, not production.

One independent count vector per declared target compares every requested power
at two explicit settings. Predictions are never normalized. Sequence evidence
differences and target-weighted predictive KL changes are reported in bits.
Passing these finite profiles is not a certificate for all possible profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gammaln, logsumexp

from .artifacts import canonical_hash, environment, sha256, utc_now, write_json
from .benchmark import TARGET_IDS, make_target
from .powers import PowerEvaluator, PowerIntegrationError, PowerSettings


def w2_closed_form_reference(d, partition, *, mode, window, decimal_digits=80):
    """Independent w=2 kernels from erfc plus high-precision moment recurrence.

    With a=exp(v), psi_r=a*J_(2r)(a), J_k=int z^k exp(-z*z-a*z) dz.
    J_0 is closed form, J_1=(1-a*J_0)/2 and
    J_(k+2)=((k+1)*J_k-a*J_(k+1))/2. This forward recurrence is bounded
    to counts<=20, |v|<=6 and >=80 decimal digits to control cancellation.
    Adaptive outer quadrature is independently refined on a wider interval.
    """
    partition = tuple(int(r) for r in partition if r)
    n = sum(partition)
    if not n or max(partition) > 20 or decimal_digits < 80:
        raise ValueError("closed-form reference requires 1<=max count<=20 and >=80 digits")
    freq = Counter(partition)
    if len(partition) < d:
        freq[0] = d - len(partition)
    values = tuple(sorted(freq))
    max_order = max(values) + 1
    left, right = float(window[0]), float(window[1])
    if not -6 <= left - 0.25 < mode < right + 0.25 <= 6:
        raise ValueError("closed-form reference outside its stable v-window")

    def logs(v, digits):
        with mp.workdps(digits):
            a = mp.exp(mp.mpf(float(v)))
            j0 = mp.sqrt(mp.pi) / 2 * mp.exp(a*a/4) * mp.erfc(a/2)
            moments = [j0, (1-a*j0)/2]
            for k in range(2*max_order-1):
                moments.append(((k+1)*moments[k]-a*moments[k+1])/2)
            if any(moments[2*r] <= 0 for r in range(max_order+1)):
                raise PowerIntegrationError("closed-form moment recurrence lost positivity")
            kernel = {r: float(mp.log(a*moments[2*r])) for r in range(max_order+1)}
        base = sum(freq[r]*kernel[r] for r in values)
        return np.array([base] + [base+kernel[r+1]-kernel[r]-math.log(n) for r in values])

    shifts = logs(mode, decimal_digits)

    def integrate(lo, hi, digits, tolerance):
        result, error, info = quad_vec(
            lambda v: np.exp(logs(v, digits)-shifts), lo, hi,
            epsabs=tolerance, epsrel=tolerance, points=[mode], full_output=True,
            norm="max", limit=400)
        if not info.success or np.any(result <= 0) or np.any(~np.isfinite(result)):
            raise PowerIntegrationError("independent closed-form outer quadrature failed")
        return result, float(error/np.min(result)), int(info.neval)

    coarse, _, _ = integrate(left, right, decimal_digits, 2e-10)
    fine, error, evaluations = integrate(left-0.25, right+0.25, decimal_digits+20, 2e-11)
    refinement = float(np.max(np.abs(np.log(fine/coarse))))
    if max(error, refinement) > 2e-8:
        raise PowerIntegrationError("independent closed-form reference did not converge")
    log_integrals = np.log(fine)+shifts
    q = np.exp(log_integrals[1:]-log_integrals[0])
    mass = float(sum(freq[r]*q[i] for i, r in enumerate(values)))
    if abs(mass-1) > 2e-7:
        raise PowerIntegrationError("independent closed-form reference normalization failed")
    return {
        "log_evidence": float(log_integrals[0]+math.log(2)-gammaln(n)),
        "count_values": list(values), "probabilities": q.tolist(),
        "diagnostics": {"decimal_digits": [decimal_digits, decimal_digits+20],
                        "refinement_log_difference": refinement,
                        "outer_relative_error_estimate": error,
                        "outer_evaluations": evaluations, "raw_normalization": mass,
                        "v_window": [left-0.25, right+0.25]},
    }


def _settings(config, key):
    return PowerSettings(**config[key])


def _validate_config(config):
    config = json.loads(json.dumps(config, allow_nan=False))
    for key in ("d", "n", "seed", "workers"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid {key}")
    targets, powers = config["targets"], config["powers"]
    if not targets or len(set(targets)) != len(targets) or set(targets)-set(TARGET_IDS):
        raise ValueError("targets must be a nonempty unique subset of benchmark targets")
    if (not powers or len(set(powers)) != len(powers)
            or any(type(w) is not int or not 0 <= w <= 80 for w in powers)):
        raise ValueError("powers must be distinct integers in [0,80]")
    if set(config["independent_w2_targets"])-set(targets):
        raise ValueError("independent targets must be in the saved-profile set")
    if config["independent_w2_targets"] and 2 not in powers:
        raise ValueError("independent w2 comparison requires power 2")
    for key in ("default_settings", "strict_settings"):
        settings = _settings(config, key)
        if config["d"] > settings.max_alphabet or config["n"] > settings.max_sample_size:
            raise ValueError("pilot exceeds evaluator settings domain")
        config[key] = asdict(settings)
    for key in ("max_log_evidence_change_bits", "max_predictive_kl_change_bits"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"invalid tolerance {key}")
    return config


def _posterior_summary(powers, logs, components, target_mass):
    posterior = np.exp(logs-logsumexp(logs))
    q = posterior @ components
    positive = posterior > 0
    entropy = float(-np.sum(posterior[positive]*np.log(posterior[positive])))
    return {
        "posterior": posterior.tolist(), "mixture_probabilities": q.tolist(),
        "mixture_log_evidence": float(logsumexp(logs)-math.log(len(powers))),
        "posterior_mean_power": float(posterior @ np.asarray(powers)),
        "posterior_mode_power": int(powers[int(np.argmax(posterior))]),
        "posterior_effective_components": math.exp(entropy),
        "posterior_mass_powers_41_to_80": float(posterior[np.asarray(powers) >= 41].sum()),
        "predictive_cross_entropy_bits": float(-target_mass @ np.log2(q)),
    }


def _profile_job(arguments):
    config, path_string, target_id, expected_hash = arguments
    path = Path(path_string)
    started = time.perf_counter()
    if sha256(path / "sample.npz") != expected_hash:
        raise ValueError("saved sample changed before calibration")
    with np.load(path / "sample.npz", allow_pickle=False) as data:
        counts, target = data["counts"], data["target"]
    values = np.unique(counts)
    class_mass = np.array([target[counts == r].sum() for r in values])
    multiplicities = np.array([(counts == r).sum() for r in values])
    partition = counts[counts > 0]
    powers = config["powers"]
    by_setting, failures = {}, []
    with (path / "components.jsonl").open("x", encoding="utf8") as stream:
        for label in ("default", "strict"):
            evaluator = PowerEvaluator(_settings(config, f"{label}_settings"))
            rows = []
            for power in powers:
                start = time.perf_counter()
                record = {"setting": label, "power": power}
                try:
                    result = evaluator.prediction_by_count(config["d"], partition, powers=[power])
                    if tuple(values) != result.count_values:
                        raise ValueError("count-class mapping mismatch")
                    record.update(status="passed", log_evidence=float(result.component_log_evidence[0]),
                                  probabilities=result.component_probabilities[0].tolist(),
                                  diagnostics=result.diagnostics["components"][0],
                                  raw_normalization=result.diagnostics["component_normalization"][0])
                except (PowerIntegrationError, ValueError, ArithmeticError) as error:
                    record.update(status="failed", error_type=type(error).__name__, error=str(error))
                    failures.append({"setting": label, "power": power, "error": str(error)})
                record["seconds"] = time.perf_counter()-start
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
                rows.append(record)
            by_setting[label] = rows
    summary = {"target": target_id, "status": "failed" if failures else "passed",
               "failures": failures, "count_values": values.tolist(),
               "class_multiplicities": multiplicities.tolist(),
               "class_target_mass": class_mass.tolist(), "normalization_applied": False,
               "sample_sha256": expected_hash, "powers": powers}
    if not failures:
        arrays = {}
        for label, rows in by_setting.items():
            logs = np.array([row["log_evidence"] for row in rows])
            probs = np.array([row["probabilities"] for row in rows])
            arrays[label] = (logs, probs)
            summary[label] = _posterior_summary(powers, logs, probs, class_mass)
            summary[label]["max_raw_normalization_error"] = float(np.max(np.abs(probs @ multiplicities-1)))
        default_logs, default_q = arrays["default"]
        strict_logs, strict_q = arrays["strict"]
        evidence_change = np.abs(default_logs-strict_logs)/math.log(2)
        kl_change = np.abs(np.log2(default_q/strict_q) @ class_mass)
        default, strict = summary["default"], summary["strict"]
        mix_kl_change = float(abs(class_mass @ np.log2(
            np.array(default["mixture_probabilities"])/np.array(strict["mixture_probabilities"]))))
        summary["comparison"] = {
            "component_log_evidence_change_bits": evidence_change.tolist(),
            "component_predictive_kl_change_bits": kl_change.tolist(),
            "max_log_evidence_change_bits": float(np.max(evidence_change)),
            "max_predictive_kl_change_bits": float(np.max(kl_change)),
            "mixture_predictive_kl_change_bits": mix_kl_change,
            "posterior_total_variation": float(np.abs(np.array(default["posterior"])-strict["posterior"]).sum()/2),
            "max_class_log_predictive_change_bits": float(np.max(np.abs(np.log2(default_q/strict_q)))),
        }
        if (evidence_change.max() > config["max_log_evidence_change_bits"]
                or max(kl_change.max(), mix_kl_change) > config["max_predictive_kl_change_bits"]):
            summary["status"] = "failed"
            summary["failures"].append({"error": "default/strict refinement exceeds declared tolerance"})
        if target_id in config["independent_w2_targets"]:
            ref_start = time.perf_counter()
            index = powers.index(2)
            component = by_setting["strict"][index]
            try:
                reference = w2_closed_form_reference(
                    config["d"], partition, mode=component["diagnostics"]["v_mode"],
                    window=component["diagnostics"]["v_window"])
                reference["logevidence_difference_bits"] = abs(reference["log_evidence"]-strict_logs[index])/math.log(2)
                reference["predictive_kl_difference_bits"] = float(abs(class_mass @ np.log2(
                    np.array(reference["probabilities"])/strict_q[index])))
                reference["status"] = "passed" if (
                    reference["logevidence_difference_bits"] <= config["max_log_evidence_change_bits"]
                    and reference["predictive_kl_difference_bits"] <= config["max_predictive_kl_change_bits"]
                ) else "failed"
            except (PowerIntegrationError, ValueError, ArithmeticError) as error:
                reference = {"status": "failed", "error": str(error)}
            reference["seconds"] = time.perf_counter()-ref_start
            summary["independent_w2"] = reference
            if reference["status"] != "passed":
                summary["status"] = "failed"
                summary["failures"].append({"error": "independent w2 comparison failed"})
    summary["sample_unchanged"] = sha256(path / "sample.npz") == expected_hash
    if not summary["sample_unchanged"]:
        summary["status"] = "failed"
    summary["seconds"] = time.perf_counter()-started
    summary["components_sha256"] = sha256(path / "components.jsonl")
    write_json(path / "summary.json", summary)
    print(f"power calibration: {target_id}: {summary['status']} ({summary['seconds']:.1f}s)", flush=True)
    return summary


def run_power_validation(config, output_dir):
    """Run a finite saved-profile calibration; never use the whole-repo Run wrapper."""
    config = _validate_config(config)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    package = Path(__file__).parent
    sources = {name: package / name for name in (
        "powers.py", "power_validation.py", "benchmark.py", "baselines.py", "artifacts.py")}
    initial_hashes = {name: sha256(path) for name, path in sources.items()}
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "started.json", {
        "started_utc": utc_now(), "purpose": "validation", "source_files": initial_hashes,
        "config_sha256": canonical_hash(config), "environment": environment(),
        "seed_scheme": "numpy.default_rng([seed, canonical_target_index, 0]); target then multinomial",
        "scope": "one saved profile per target; finite pilot, not all-profile certification",
    })
    jobs, profiles = [], []
    for target_id in config["targets"]:
        path = output_dir / target_id
        path.mkdir()
        coordinates = [config["seed"], TARGET_IDS.index(target_id), 0]
        rng = np.random.default_rng(coordinates)
        target = make_target(target_id, config["d"], rng)
        counts = rng.multinomial(config["n"], target)
        with (path / "sample.npz").open("xb") as stream:
            np.savez_compressed(stream, target=target, counts=counts)
        digest = sha256(path / "sample.npz")
        profiles.append({"target": target_id, "rng_coordinates": coordinates,
                         "path": f"{target_id}/sample.npz", "sha256": digest,
                         "observed_symbols": int(np.count_nonzero(counts)), "max_count": int(counts.max())})
        jobs.append((config, str(path), target_id, digest))
    write_json(output_dir / "samples.json", {"profiles": profiles})
    if config["workers"] == 1:
        results = list(map(_profile_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            results = list(pool.map(_profile_job, jobs))
    final_hashes = {name: sha256(path) for name, path in sources.items()}
    source_unchanged = initial_hashes == final_hashes
    status = "passed" if source_unchanged and all(r["status"] == "passed" for r in results) else "failed"
    summary = {"schema_version": 1, "status": status, "purpose": "validation",
               "finished_utc": utc_now(), "seconds": time.perf_counter()-started,
               "source_files_at_start": initial_hashes, "source_files_at_end": final_hashes,
               "source_unchanged": source_unchanged, "config_sha256": canonical_hash(config),
               "domain": {"d": config["d"], "n": config["n"], "powers": config["powers"],
                          "targets": config["targets"], "profiles_per_target": 1},
               "normalization_applied": False, "production_domain_certified": False,
               "profiles": [{"target": r["target"], "status": r["status"], "seconds": r["seconds"],
                             "summary_sha256": sha256(output_dir / r["target"] / "summary.json"),
                             "comparison": r.get("comparison"), "failures": r["failures"]} for r in results]}
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "files.json", {
        str(path.relative_to(output_dir)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.rglob("*")) if path.is_file()})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    result = run_power_validation(json.loads(Path(arguments.config).read_text()), arguments.out)
    print(json.dumps({"status": result["status"], "seconds": result["seconds"]}), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)
