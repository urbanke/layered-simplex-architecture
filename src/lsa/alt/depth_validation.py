"""Saved paper-domain depth profiles under declared numerical refinements."""

from __future__ import annotations

import gzip
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .artifacts import canonical_hash, sha256, write_json
from .baselines import validate_probabilities
from .benchmark import TARGET_IDS, jsonable, make_target
from .bible import load_corpus
from .depth import DepthEvaluator, StoreConfig


def comparison(default, refined, *, n, class_mass=None, multiplicities=None):
    if hasattr(default, "component_log_evidence"):
        logs_a, logs_b = default.component_log_evidence, refined.component_log_evidence
    else:
        logs_a, logs_b = default.log_evidence, refined.log_evidence
    evidence = np.asarray(logs_a) - np.asarray(logs_b)
    result = {
        "maximum_evidence_difference_bits": float(
            np.max(np.abs(evidence)) / math.log(2)
        ),
        "maximum_codelength_difference_bits_per_token": float(
            np.max(np.abs(evidence)) / (n * math.log(2))
        ),
        "mixture_codelength_difference_bits_per_token": float(
            abs(logsumexp(logs_a) - logsumexp(logs_b)) / (n * math.log(2))
        ),
    }
    if class_mass is not None:
        mult = np.asarray(multiplicities)
        qa, qb = default.component_probabilities, refined.component_probabilities
        za, zb = qa @ mult, qb @ mult
        ma, mb = default.mixture_probabilities, refined.mixture_probabilities
        zma, zmb = float(ma @ mult), float(mb @ mult)
        active = np.asarray(class_mass) > 0
        # The paper declares final normalization. First retain its raw mass,
        # then compare the reported loss using that identical convention.
        qa, qb = qa / za[:, None], qb / zb[:, None]
        ma, mb = ma / zma, mb / zmb
        if np.any(qa[:, active] <= 0) or np.any(qb[:, active] <= 0):
            raise ArithmeticError("positive target mass has zero numerical prediction")
        delta = (
            (np.log(qa[:, active]) - np.log(qb[:, active]))
            @ np.asarray(class_mass)[active]
            / math.log(2)
        )
        mixed = float(
            np.dot(
                np.asarray(class_mass)[active], np.log(ma[active]) - np.log(mb[active])
            )
            / math.log(2)
        )
        result.update(
            maximum_predictive_kl_change_bits=float(np.max(np.abs(delta))),
            mixture_predictive_kl_change_bits=abs(mixed),
            maximum_raw_mass_error=float(
                max(
                    np.max(abs(za - 1)), np.max(abs(zb - 1)), abs(zma - 1), abs(zmb - 1)
                )
            ),
            maximum_posterior_change=float(
                np.max(abs(default.posterior - refined.posterior))
            ),
        )
    return result


def run_depth_validation(config, output_dir, *, engine_config, repo):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    store = StoreConfig(**engine_config["store"])
    stricter = replace(store, **config["refinement"])
    sources = [Path(__file__), Path(__file__).with_name("depth.py")]
    initial_hashes = {p.name: sha256(p) for p in sources}
    write_json(root / "implementation.json", initial_hashes)
    cases = []
    # Save every draw before expensive numerical work starts.
    for index, case in enumerate(config["cases"]):
        d, n = case["d"], case["n"]
        if case["kind"] == "synthetic":
            rng = np.random.default_rng(
                case.get(
                    "seed_coordinates",
                    [config["seed"], TARGET_IDS.index(case["target"]), index],
                )
            )
            target = make_target(case["target"], d, rng)
            validate_probabilities(target)
            counts = rng.multinomial(n, target)
        else:
            tokens, corpus = load_corpus(case, repo)
            if n > len(tokens):
                raise ValueError("prefix exceeds corpus")
            from collections import Counter

            mult = Counter(tokens[:n])
            if len(mult) > d:
                raise ValueError("vocabulary exceeds declared alphabet")
            counts = np.zeros(d, dtype=np.int64)
            counts[: len(mult)] = list(mult.values())
            target = counts / n
            write_json(root / f"corpus-{index}.json", corpus)
        np.savez_compressed(
            root / f"case-{index:03d}.npz", counts=counts, target=target
        )
        cases.append((case, counts, target))
    summaries = []
    with (
        DepthEvaluator(
            mode="store", store=store, prediction_tolerance=config["raw_mass_tolerance"]
        ) as base,
        DepthEvaluator(
            mode="store",
            store=stricter,
            prediction_tolerance=config["raw_mass_tolerance"],
        ) as fine,
    ):
        write_json(root / "default-engine.json", base.configuration)
        write_json(root / "refined-engine.json", fine.configuration)
        for index, (case, counts, target) in enumerate(cases):
            started = time.monotonic()
            print(f"depth case {index + 1}/{len(cases)}: {case['id']}", flush=True)
            profile = tuple(counts[counts > 0])
            record = {
                "id": case["id"],
                "case": case,
                "sample_sha256": sha256(root / f"case-{index:03d}.npz"),
            }
            try:
                if case["predictive"]:
                    a = base.prediction_by_count(
                        case["d"], profile, depths=case["depths"]
                    )
                    b = fine.prediction_by_count(
                        case["d"], profile, depths=case["depths"]
                    )
                    if a.counts != b.counts:
                        raise ValueError("prediction class mismatch")
                    mass = [float(target[counts == c].sum()) for c in a.counts]
                    values = comparison(
                        a,
                        b,
                        n=case["n"],
                        class_mass=mass,
                        multiplicities=a.multiplicities,
                    )
                else:
                    a = base.evidence_at_depths(case["d"], profile, case["depths"])
                    b = fine.evidence_at_depths(case["d"], profile, case["depths"])
                    values = comparison(a, b, n=case["n"])
                passed = (
                    values["maximum_codelength_difference_bits_per_token"]
                    <= config["loss_tolerance_bits"]
                )
                if case["predictive"]:
                    passed &= (
                        values["maximum_predictive_kl_change_bits"]
                        <= config["loss_tolerance_bits"]
                        and values["maximum_raw_mass_error"]
                        <= config["raw_mass_tolerance"]
                    )
                record.update(
                    status="passed" if passed else "failed", measurements=values
                )
                with gzip.open(
                    root / f"evaluation-{index:03d}.json.gz", "xt"
                ) as stream:
                    json.dump(
                        jsonable({"default": a, "refined": b}), stream, allow_nan=False
                    )
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                record.update(
                    status="failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            record["seconds"] = time.monotonic() - started
            write_json(root / f"result-{index:03d}.json", record)
            summaries.append(record)
            print(
                record["status"],
                round(record["seconds"], 3),
                record.get("measurements", record.get("error")),
                flush=True,
            )
    if {p.name: sha256(p) for p in sources} != initial_hashes:
        raise RuntimeError("calibration source changed during execution")
    summary = {
        "status": "passed"
        if all(r["status"] == "passed" for r in summaries)
        else "failed",
        "cases": summaries,
        "config_sha256": canonical_hash(config),
        "scope": "declared finite cases under outer-grid/window refinement; kernel references are separate",
    }
    write_json(root / "summary.json", summary)
    return summary
