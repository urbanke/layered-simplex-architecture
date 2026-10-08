"""Paired profile validation of sealed kernels against corrected direct kernels.

Both engines receive identical saved counts, targets, depths, outer settings,
and all required augmented count profiles. This checks the change of kernel
provider; independent high-precision and outer-refinement checks are separate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import Run, canonical_hash, read_json, sha256, verify_run, write_json
from .baselines import validate_probabilities
from .benchmark import jsonable, make_target
from .bible import load_corpus
from .depth import DepthEvaluator, StoreConfig
from .depth_validation import comparison

OUTER_SETTINGS = (
    "grid_step", "minimum_grid_step", "u_max", "maximum_u_max",
    "upper_window_increment", "scan_mode", "significance_gap", "minimum_right_gap",
)

REFERENCE_PROTOCOL = {
    "schema_version": 1,
    "provider": "corrected-direct-contour",
    "analytic_depths": [0, 1],
    "minimum_direct_depth": 2,
}


def validate_reference_protocol(config):
    """Require the explicit all-direct reference; historical hybrid runs stay distinct."""
    if (
        type(config.get("schema_version")) is not int
        or config["schema_version"] != 2
        or canonical_hash(config.get("reference")) != canonical_hash(REFERENCE_PROTOCOL)
    ):
        raise ValueError(
            "sealed profile protocol v2 requires the declared corrected direct "
            "reference from depth2"
        )


def validate_config(config):
    validate_reference_protocol(config)
    if config.get("purpose") != "validation":
        raise ValueError("sealed profile checks require purpose=validation")
    for key, maximum in (("loss_tolerance_bits", 1e-5), ("raw_mass_tolerance", 1e-7)):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= maximum:
            raise ValueError(f"{key} must be positive and no greater than {maximum:g}")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("at least one fixed profile case is required")
    ids = set()
    for case in cases:
        if not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids:
            raise ValueError("case IDs must be nonempty and unique")
        ids.add(case["id"])
        for key in ("d", "n"):
            if type(case.get(key)) is not int or case[key] < 1:
                raise ValueError(f"case {key} must be a positive integer")
        depths = case.get("depths")
        if (not isinstance(depths, list) or not depths or len(set(depths)) != len(depths)
                or any(type(L) is not int or not 0 <= L <= 138 for L in depths)):
            raise ValueError("case depths must be distinct integers in 0..138")
        if type(case.get("predictive")) is not bool:
            raise ValueError("case predictive must be explicit")
        if case.get("kind") not in ("synthetic", "bible", "explicit"):
            raise ValueError("unknown profile case kind")
        if case["kind"] == "synthetic" and not case.get("seed_coordinates"):
            raise ValueError("synthetic cases require explicit seed_coordinates")
    return config


def _sample(case, repo):
    corpus = None
    if case["kind"] == "synthetic":
        rng = np.random.default_rng(case["seed_coordinates"])
        target = make_target(case["target"], case["d"], rng)
        counts = rng.multinomial(case["n"], target)
    else:
        counts = np.zeros(case["d"], dtype=np.int64)
        if case["kind"] == "bible":
            tokens, corpus = load_corpus(case, repo)
            if case["n"] > len(tokens):
                raise ValueError("Bible prefix exceeds pinned corpus")
            parts = list(Counter(tokens[:case["n"]]).values())
        else:
            parts = case.get("profile", [])
            if not parts or any(type(r) is not int or r < 1 for r in parts):
                raise ValueError("explicit profile must contain positive integer counts")
        if len(parts) > case["d"] or sum(parts) != case["n"]:
            raise ValueError("profile does not match its declared alphabet and sample size")
        counts[:len(parts)] = parts
        target = (np.full(case["d"], 1. / case["d"])
                  if case.get("target", "empirical") == "uniform" else counts / case["n"])
        if case["kind"] == "explicit" and case.get("target", "empirical") not in ("uniform", "empirical"):
            raise ValueError("explicit profiles support uniform or empirical target masses")
    validate_probabilities(target)
    return counts, target, corpus


def _family(counts):
    profile = tuple(sorted((int(c) for c in counts if c), reverse=True))
    classes, multiplicities = np.unique(counts, return_counts=True)
    augmented = {}
    for value in classes:
        r = int(value)
        child = list(profile)
        if r:
            child.remove(r)
        child.append(r + 1)
        augmented[str(r)] = sorted(child, reverse=True)
    return profile, tuple(int(x) for x in classes), tuple(int(x) for x in multiplicities), augmented


def compare_profiles(legacy, candidate, *, n, class_mass=None, multiplicities=None):
    if legacy.depths != candidate.depths:
        raise ValueError("paired depth grids differ")
    values = comparison(legacy, candidate, n=n, class_mass=class_mass,
                        multiplicities=multiplicities)
    la = getattr(legacy, "component_log_evidence", legacy.log_evidence
                 if hasattr(legacy, "log_evidence") else None)
    lb = getattr(candidate, "component_log_evidence", candidate.log_evidence
                 if hasattr(candidate, "log_evidence") else None)
    values["component_log_evidence_change_bits"] = ((lb - la) / math.log(2)).tolist()
    values["component_codelength_change_bits_per_token"] = ((la - lb) / (n * math.log(2))).tolist()
    if class_mass is not None:
        if legacy.counts != candidate.counts or legacy.multiplicities != candidate.multiplicities:
            raise ValueError("paired count classes or multiplicities differ")
        mult, mass = np.asarray(multiplicities), np.asarray(class_mass)
        raw_a, raw_b = legacy.component_probabilities, candidate.component_probabilities
        mix_a, mix_b = legacy.mixture_probabilities, candidate.mixture_probabilities
        if any(np.any(q <= 0) or not np.isfinite(q).all() for q in (raw_a, raw_b, mix_a, mix_b)):
            raise ArithmeticError("numerical predictions must remain finite and positive")
        za, zb, zma, zmb = raw_a @ mult, raw_b @ mult, float(mix_a @ mult), float(mix_b @ mult)
        qa, qb, ma, mb = raw_a / za[:, None], raw_b / zb[:, None], mix_a / zma, mix_b / zmb
        delta = ((np.log(qa) - np.log(qb)) @ mass) / math.log(2)
        mixed = float((np.log(ma) - np.log(mb)) @ mass) / math.log(2)
        # These are the already-evaluated family evidence ratios, inverted in
        # log space; no separate evaluation or different child grid is used.
        aug_a = la[:, None] + np.log(raw_a)
        aug_b = lb[:, None] + np.log(raw_b)
        values.update(
            component_predictive_kl_change_bits=delta.tolist(),
            signed_mixture_predictive_kl_change_bits=mixed,
            legacy_component_raw_mass=za.tolist(), candidate_component_raw_mass=zb.tolist(),
            legacy_mixture_raw_mass=zma, candidate_mixture_raw_mass=zmb,
            maximum_augmented_evidence_difference_bits=float(np.abs(aug_b - aug_a).max() / math.log(2)),
            maximum_augmented_codelength_difference_bits_per_token=float(np.abs(aug_b - aug_a).max() / ((n + 1) * math.log(2))),
            maximum_normalized_component_probability_change=float(np.abs(qa - qb).max()),
            maximum_normalized_mixture_probability_change=float(np.abs(ma - mb).max()),
        )
    return values


def assess_measurements(values, *, predictive, config):
    gates = {"maximum_codelength_difference_bits_per_token": config["loss_tolerance_bits"],
             "mixture_codelength_difference_bits_per_token": config["loss_tolerance_bits"]}
    if predictive:
        gates.update({"maximum_predictive_kl_change_bits": config["loss_tolerance_bits"],
                      "mixture_predictive_kl_change_bits": config["loss_tolerance_bits"],
                      "maximum_augmented_codelength_difference_bits_per_token": config["loss_tolerance_bits"],
                      "maximum_raw_mass_error": config["raw_mass_tolerance"]})
    return {key: bool(isinstance(values.get(key), (int, float))
                      and math.isfinite(values[key]) and 0 <= values[key] <= limit)
            for key, limit in gates.items()}


def _engine(options):
    if options.get("mode", "store") != "store":
        raise ValueError("both profile providers must use mode=store")
    return DepthEvaluator(mode="store", store=StoreConfig(**options["store"]),
                          prediction_tolerance=options.get("prediction_tolerance", 1e-3))


def run_sealed_profile_validation(config, output_dir, *, candidate_engine, legacy_engine,
                                  repo, inputs=(), command=None):
    validate_config(config)
    candidate_store, legacy_store = StoreConfig(**candidate_engine["store"]), StoreConfig(**legacy_engine["store"])
    if candidate_store.format != "sealed" or legacy_store.format != "legacy":
        raise ValueError("comparison requires explicit sealed and legacy providers")
    if legacy_store.saddle_min_depth != REFERENCE_PROTOCOL["minimum_direct_depth"]:
        raise ValueError("reference engine must select corrected direct kernels from depth2")
    if any(getattr(candidate_store, key) != getattr(legacy_store, key) for key in OUTER_SETTINGS):
        raise ValueError("paired providers require identical outer integration settings")
    store_inputs = [Path(s.path) / name for s in (candidate_store, legacy_store)
                    for name in s.files_sha256]
    corpus_inputs = [Path(repo) / case[key] for case in config["cases"]
                     if case["kind"] == "bible" for key in ("corpus", "manifest")]
    all_inputs = sorted({Path(p).resolve() for p in (*inputs, *store_inputs, *corpus_inputs)})
    with _engine(candidate_engine) as candidate, _engine(legacy_engine) as legacy:
        identities = {"candidate": candidate.configuration, "legacy": legacy.configuration}
        with Run(output_dir, repo=repo, protocol={"suite": "sealed_profile", "config": config},
                 experiment="calibration_sealed_profile", purpose="validation",
                 engine_identity=identities, inputs=all_inputs, command=command) as run:
            root = run.path / "data"
            root.mkdir()
            write_json(root / "config.json", config)
            write_json(root / "candidate-engine.json", candidate.configuration)
            write_json(root / "legacy-engine.json", legacy.configuration)
            cases = []
            for i, case in enumerate(config["cases"]):
                counts, target, corpus = _sample(case, repo)
                profile, classes, mult, augmented = _family(counts)
                sample = root / f"case-{i:03d}.npz"
                np.savez_compressed(sample, counts=counts, target=target)
                record = {"id": case["id"], "case": case, "sample_sha256": sha256(sample),
                          "counts_sha256": canonical_hash(counts.tolist()),
                          "target_sha256": canonical_hash(target.tolist()),
                          "profile": list(profile), "count_classes": list(classes),
                          "multiplicities": list(mult),
                          "augmented_profiles": augmented if case["predictive"] else {}}
                write_json(root / f"family-{i:03d}.json", record)
                if corpus is not None:
                    write_json(root / f"corpus-{i:03d}.json", corpus)
                cases.append((case, counts, target, record))
            results = []
            for i, (case, counts, target, record) in enumerate(cases):
                started = time.monotonic()
                print(f"sealed profile {i+1}/{len(cases)}: {case['id']}", flush=True)
                profile = record["profile"]
                try:
                    timings, evaluated = {}, {}
                    for name, engine in (("legacy", legacy), ("candidate", candidate)):
                        tick = time.monotonic()
                        evaluated[name] = (engine.prediction_by_count(case["d"], profile, depths=case["depths"])
                                           if case["predictive"] else engine.evidence_at_depths(case["d"], profile, case["depths"]))
                        timings[name] = time.monotonic() - tick
                    a, b = evaluated["legacy"], evaluated["candidate"]
                    mass = [float(target[counts == c].sum()) for c in a.counts] if case["predictive"] else None
                    values = compare_profiles(a, b, n=case["n"], class_mass=mass,
                                              multiplicities=a.multiplicities if case["predictive"] else None)
                    gates = assess_measurements(values, predictive=case["predictive"], config=config)
                    record.update(status="passed" if all(gates.values()) else "failed",
                                  measurements=values, gates=gates, provider_seconds=timings,
                                  class_target_mass=mass)
                    with gzip.open(root / f"evaluation-{i:03d}.json.gz", "xt") as stream:
                        json.dump(jsonable(evaluated), stream, allow_nan=False)
                except (ArithmeticError, RuntimeError, ValueError) as exc:
                    record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                record["seconds"] = time.monotonic() - started
                write_json(root / f"result-{i:03d}.json", record)
                results.append(record)
                print(f"{record['status']} ({record['seconds']:.3f}s)", flush=True)
            candidate._check_store_unchanged()
            legacy._check_store_unchanged()
            summary = {
                "status": "passed" if all(row["status"] == "passed" for row in results) else "failed",
                "config_sha256": canonical_hash(config), "cases": results,
                "reference": config["reference"],
                "candidate_configuration_sha256": canonical_hash(candidate.configuration),
                "legacy_configuration_sha256": canonical_hash(legacy.configuration),
                "candidate_evaluator_configuration_sha256": candidate.configuration_sha256,
                "legacy_evaluator_configuration_sha256": legacy.configuration_sha256,
                "source_unchanged": True, "store_unchanged": True, "unavailable_cases": [],
                "scope": "declared identical profiles, augmented count classes and mixtures; change of kernel provider at identical outer settings",
                "independent_high_precision_reference": False,
                "normalization": "raw masses retained; both predictions normalized before target-weighted KL comparison",
                "augmented_evidence": "reconstructed from evaluated base log evidence and raw family evidence ratios",
                "units": {"evidence": "total bits", "codelength": "bits/token", "predictive_kl": "bits", "raw_mass": "probability mass"},
                "integrity": "source_unchanged and store_unchanged require the containing Run to finish and verify successfully",
                "production_certified": False,
            }
            write_json(root / "summary.json", summary)
            write_json(run.path / "result.json", summary)
            if summary["status"] != "passed":
                raise ArithmeticError("sealed profile validation failed; saved results retain every case")
    verify_run(output_dir)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-engine", type=Path, required=True)
    parser.add_argument("--legacy-engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    result = run_sealed_profile_validation(
        read_json(args.config), args.out, candidate_engine=read_json(args.candidate_engine),
        legacy_engine=read_json(args.legacy_engine), repo=args.repo,
        inputs=[args.config, args.candidate_engine, args.legacy_engine],
    )
    print(json.dumps({"status": result["status"], "cases": len(result["cases"])}))
    return 0 if result["status"] == "passed" else 1
