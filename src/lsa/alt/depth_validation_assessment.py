"""Assess saved depth checks and supplement missing *actual* grid halvings.

The original run is read only. Each assessment freezes its completed inputs;
unfinished cases stay pending and cannot yield a passed overall assessment.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from .artifacts import canonical_hash, sha256, write_json
from .baselines import validate_counts, validate_probabilities
from .benchmark import TARGET_IDS, jsonable
from .depth import DepthEvaluator, StoreConfig
from .depth_validation import comparison


def gate_measurements(values, config, *, predictive):
    """Gate all reported losses; mass and numerical error have separate units."""
    checks = {
        "maximum_codelength_difference_bits_per_token": config["loss_tolerance_bits"],
        "mixture_codelength_difference_bits_per_token": config["loss_tolerance_bits"],
    }
    if predictive:
        checks.update(
            maximum_predictive_kl_change_bits=config["loss_tolerance_bits"],
            mixture_predictive_kl_change_bits=config["loss_tolerance_bits"],
            maximum_raw_mass_error=config["raw_mass_tolerance"],
        )
    failures = []
    for name, tolerance in checks.items():
        value = values.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or not math.isfinite(tolerance)
            or tolerance <= 0
            or value > tolerance
        ):
            failures.append(name)
    for name, value in values.items():
        if (
            not isinstance(value, (int, float)) or not math.isfinite(value)
        ) and name not in failures:
            failures.append(name)
    return {"status": "failed" if failures else "passed", "failed_checks": failures}


def _components(result):
    diagnostics = result["diagnostics"]
    return diagnostics.get("base", diagnostics)["components"]


def grid_assessment(default, refined):
    """Inspect the measured spacing, allowing the tiny linspace rounding error."""
    if default["depths"] != refined["depths"]:
        raise ValueError("saved depth grids differ")
    a = {row["depth"]: row for row in _components(default)}
    b = {row["depth"]: row for row in _components(refined)}
    result = []
    for depth in default["depths"]:
        da, db = a[depth], b[depth]
        x, y = da.get("outer_grid_step"), db.get("outer_grid_step")
        if x is None and y is None and depth <= 1:
            result.append({"depth": depth, "status": "analytic"})
            continue
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (x, y)
        ):
            raise ValueError(f"missing/invalid measured grid spacing at depth {depth}")
        result.append(
            {
                "depth": depth,
                "default_step": x,
                "refined_step": y,
                "status": "halved" if y <= 0.51 * x else "pending",
                "supplement_step": min(x, y) / 2,
            }
        )
    return result


def _object(result):
    value = dict(result)
    for key in (
        "log_evidence",
        "component_log_evidence",
        "component_probabilities",
        "mixture_probabilities",
        "posterior",
    ):
        if key in value:
            value[key] = np.asarray(value[key], dtype=float)
    return SimpleNamespace(**value)


def _measure(default, refined, case, counts, target):
    kwargs = {}
    if case["predictive"]:
        if (
            default["counts"] != refined["counts"]
            or default["multiplicities"] != refined["multiplicities"]
        ):
            raise ValueError("saved prediction classes/multiplicities differ")
        kwargs = {
            "class_mass": [float(target[counts == c].sum()) for c in default["counts"]],
            "multiplicities": default["multiplicities"],
        }
    return comparison(_object(default), _object(refined), n=case["n"], **kwargs)


def _merge(refined, supplemental, *, predictive):
    """Replace selected components, then restore the full declared mixture."""
    merged = copy.deepcopy(refined)
    key = "component_log_evidence" if predictive else "log_evidence"
    for j, depth in enumerate(supplemental["depths"]):
        i = merged["depths"].index(depth)
        merged[key][i] = supplemental[key][j]
        if predictive:
            if (
                merged["counts"] != supplemental["counts"]
                or merged["multiplicities"] != supplemental["multiplicities"]
            ):
                raise ValueError("supplemental count classes differ")
            merged["component_probabilities"][i] = supplemental[
                "component_probabilities"
            ][j]
    # Keep actual base-grid diagnostics for each replaced component. These
    # rows can have different numerical configurations, so no single engine
    # identity is assigned to this derived, full-grid comparison object.
    rows = {r["depth"]: r for r in _components(supplemental)}
    diagnostics = merged["diagnostics"].get("base", merged["diagnostics"])
    diagnostics["components"] = [
        rows.get(r["depth"], r) for r in diagnostics["components"]
    ]
    if predictive:
        logs = np.asarray(merged[key])
        weights = np.exp(logs - logsumexp(logs))
        merged["posterior"] = weights.tolist()
        merged["mixture_probabilities"] = (
            weights @ np.asarray(merged["component_probabilities"])
        ).tolist()
    return merged


def _snapshot(source, destination, records):
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(data)
    if sha256(source) != digest:
        raise RuntimeError(f"input changed during snapshot: {source}")
    records.append(
        {"path": str(source.resolve()), "sha256": digest, "snapshot": str(destination)}
    )


def assess_depth_run(
    run_dir, output_dir, *, engine_config, run_supplemental=False, config_path=None
):
    """Freeze completed shard rows, gate them, and optionally halve missing grids.

    ``engine_config`` supplies the explicit immutable-store path. Its content
    hashes/settings must match the saved original engine; supplemental settings
    come from that run's saved refined engine. No new samples are drawn.
    """
    run_dir, out = Path(run_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    inputs = []
    declared_config = (
        Path(config_path) if config_path is not None else run_dir / "config.json"
    )
    _snapshot(declared_config, out / "inputs/config.json", inputs)
    config = json.loads((out / "inputs/config.json").read_text())
    expected = {}
    for index, raw_case in enumerate(config["cases"]):
        case = copy.deepcopy(raw_case)
        if case["kind"] == "synthetic":
            case.setdefault(
                "seed_coordinates",
                [config["seed"], TARGET_IDS.index(case["target"]), index],
            )
        expected[case["id"]] = case
    if not expected or len(expected) != len(config["cases"]):
        raise ValueError("expected case IDs must be nonempty and unique")
    write_json(
        out / "request.json",
        jsonable(
            {
                "run_dir": str(run_dir.resolve()),
                "run_supplemental": run_supplemental,
                "engine_config": engine_config,
            }
        ),
    )
    source_files = [
        Path(__file__),
        Path(__file__).with_name("depth_validation.py"),
        Path(__file__).with_name("depth.py"),
        Path(__file__).with_name("sealed_tables.py"),
        Path(__file__).with_name("sealed_native.py"),
        Path(__file__).with_name("sealed_interp.c"),
    ]
    source_files += sorted((Path(__file__).parent / "_vendor/pmwm").glob("*.py"))
    source_files.append(Path(__file__).parent / "_vendor/pmwm/provenance.json")
    source_hashes = {str(p): sha256(p) for p in source_files}
    for source in source_files:
        _snapshot(
            source,
            out / "source-capsule" / source.relative_to(Path(__file__).parent),
            inputs,
        )
    write_json(out / "implementation.json", source_hashes)
    roots = sorted(run_dir.glob("shard-*")) or [run_dir]
    completed_paths = {shard: sorted(shard.glob("result-*.json")) for shard in roots}
    records, seen = [], set()
    native = None
    for shard in roots:
        for result_path in completed_paths[shard]:
            # A live writer may have created a file before completing its JSON.
            try:
                record = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                continue
            case = record["case"]
            if (
                case["id"] in seen
                or case["id"] not in expected
                or case != expected[case["id"]]
            ):
                raise ValueError("saved case is duplicated, unexpected, or changed")
            seen.add(case["id"])
            index = result_path.stem.split("-")[1]
            relative = shard.relative_to(run_dir)
            names = [
                result_path.name,
                "config.json",
                "implementation.json",
                "default-engine.json",
                "refined-engine.json",
                f"case-{index}.npz",
            ]
            if record["status"] == "passed":
                names.append(f"evaluation-{index}.json.gz")
            for name in names:
                target = out / "inputs" / relative / name
                if not target.exists():
                    _snapshot(shard / name, target, inputs)
            saved = out / "inputs" / relative
            hashes = json.loads((saved / "implementation.json").read_text())
            if any(
                sha256(Path(__file__).with_name(name)) != digest
                for name, digest in hashes.items()
            ):
                raise RuntimeError(
                    "original validation source no longer matches its recorded hash"
                )
            if sha256(saved / f"case-{index}.npz") != record["sample_sha256"]:
                raise ValueError("saved sample hash does not match result")
            base_engine = json.loads((saved / "default-engine.json").read_text())
            fine_engine = json.loads((saved / "refined-engine.json").read_text())
            supplied = StoreConfig(**engine_config["store"])
            supplied_settings = asdict(supplied)
            supplied_settings.pop("path")
            if supplied.native_library_path is not None:
                supplied_settings["native_library_path"] = "/host-local-native"
            if supplied_settings != base_engine["store"]:
                raise ValueError(
                    "explicit engine does not match saved default store settings"
                )
            if supplied.format == "sealed":
                provider = sha256(Path(__file__).with_name("sealed_tables.py"))
                if any(
                    saved_engine.get("sealed_provider_sha256") != provider
                    for saved_engine in (base_engine, fine_engine)
                ):
                    raise ValueError(
                        "saved sealed provider differs from current source"
                    )
            if supplied.interpolation_backend == "native":
                if native is None:
                    from .sealed_native import NativeInterpolator

                    native = NativeInterpolator(
                        supplied.native_library_path, supplied.native_library_sha256
                    )
                    for source in (native.path, native.manifest_path):
                        _snapshot(source, out / "native-inputs" / source.name, inputs)
                if any(
                    saved_engine.get("sealed_native_identity") != native.identity
                    for saved_engine in (base_engine, fine_engine)
                ):
                    raise ValueError(
                        "saved native interpolation identity differs from supplied library"
                    )
            elif any(
                saved_engine.get("sealed_native_identity") is not None
                for saved_engine in (base_engine, fine_engine)
            ):
                raise ValueError(
                    "saved native identity differs from the supplied Python provider"
                )
            assessed = {
                "id": case["id"],
                "case": case,
                "original_status": record["status"],
                "sample_sha256": record["sample_sha256"],
            }
            if record["status"] != "passed":
                assessed.update(
                    status="failed",
                    reason="original case failed",
                    original_error=record.get("error"),
                )
                records.append(assessed)
                continue
            with gzip.open(saved / f"evaluation-{index}.json.gz", "rt") as stream:
                evaluations = json.load(stream)
            default, refined = evaluations["default"], evaluations["refined"]
            with np.load(saved / f"case-{index}.npz", allow_pickle=False) as sample:
                counts, target = sample["counts"], sample["target"]
            counts = validate_counts(counts)
            target = validate_probabilities(target, size=case["d"])
            if counts.shape != (case["d"],) or int(counts.sum()) != case["n"]:
                raise ValueError("saved counts disagree with declared case")
            values = _measure(default, refined, case, counts, target)
            assessed.update(
                measurements=values,
                acceptance=gate_measurements(
                    values, config, predictive=case["predictive"]
                ),
            )
            grids = grid_assessment(default, refined)
            needed = [r for r in grids if r["status"] == "pending"]
            assessed["original_grid_checks"] = grids
            if (
                run_supplemental
                and needed
                and assessed["acceptance"]["status"] == "passed"
            ):
                merged = copy.deepcopy(refined)
                groups = {}
                for row in needed:
                    groups.setdefault(row["supplement_step"], []).append(row["depth"])
                assessed["supplements"] = []
                for group_index, (step, depths) in enumerate(groups.items()):
                    settings = dict(
                        fine_engine["store"], path=engine_config["store"]["path"]
                    )
                    if supplied.interpolation_backend == "native":
                        # Saved identities use a portable sentinel. Execution
                        # restores only the supplied, independently pinned path.
                        settings["native_library_path"] = supplied.native_library_path
                    settings["grid_step"] = step
                    settings["minimum_grid_step"] = min(
                        settings["minimum_grid_step"], step
                    )
                    prefix = f"supplement-{case['id']}-{group_index:02d}"
                    try:
                        with DepthEvaluator(
                            mode="store",
                            store=StoreConfig(**settings),
                            prediction_tolerance=config["raw_mass_tolerance"],
                        ) as evaluator:
                            # Refuse a changed numerical implementation, even if
                            # saved scalar results could still be read.
                            for key in (
                                "implementation_sha256",
                                "vendor_provenance_sha256",
                                "vendor_source_sha256",
                                "runtime",
                                "sealed_provider_sha256",
                                "sealed_native_identity",
                            ):
                                if evaluator.configuration.get(key) != fine_engine.get(
                                    key
                                ):
                                    raise RuntimeError(
                                        f"supplemental engine differs: {key}"
                                    )
                            write_json(
                                out / f"{prefix}-engine.json", evaluator.configuration
                            )
                            profile = tuple(counts[counts > 0])
                            result = (
                                evaluator.prediction_by_count(
                                    case["d"], profile, depths=depths
                                )
                                if case["predictive"]
                                else evaluator.evidence_at_depths(
                                    case["d"], profile, depths
                                )
                            )
                            supplemental = jsonable(result)
                        with gzip.open(out / f"{prefix}.json.gz", "xt") as stream:
                            json.dump(supplemental, stream, allow_nan=False)
                        merged = _merge(
                            merged, supplemental, predictive=case["predictive"]
                        )
                        assessed["supplements"].append(
                            {
                                "depths": depths,
                                "requested_step": step,
                                "result": f"{prefix}.json.gz",
                                "status": "evaluated",
                            }
                        )
                    except (ArithmeticError, RuntimeError, ValueError) as exc:
                        assessed["supplements"].append(
                            {
                                "depths": depths,
                                "requested_step": step,
                                "status": "failed",
                                "error": str(exc),
                            }
                        )
                        break
                supplemental_values = _measure(refined, merged, case, counts, target)
                values = _measure(default, merged, case, counts, target)
                assessed.update(
                    measurements=values,
                    acceptance=gate_measurements(
                        values, config, predictive=case["predictive"]
                    ),
                    supplemental_measurements=supplemental_values,
                    supplemental_acceptance=gate_measurements(
                        supplemental_values, config, predictive=case["predictive"]
                    ),
                )
                grids = grid_assessment(default, merged)
            assessed["final_grid_checks"] = grids
            failed = (
                assessed["acceptance"]["status"] == "failed"
                or assessed.get("supplemental_acceptance", {}).get("status") == "failed"
                or any(s["status"] == "failed" for s in assessed.get("supplements", []))
            )
            assessed["status"] = (
                "failed"
                if failed
                else (
                    "pending"
                    if any(r["status"] == "pending" for r in grids)
                    else "passed"
                )
            )
            write_json(out / f"assessment-{case['id']}.json", assessed)
            records.append(assessed)
    if native is not None:
        native.check_unchanged()
    pending_ids = [c for c in expected if c not in seen]
    unchanged = all(sha256(Path(p)) == digest for p, digest in source_hashes.items())
    summary = {
        "status": "failed"
        if not unchanged or any(r["status"] == "failed" for r in records)
        else (
            "pending"
            if pending_ids or any(r["status"] == "pending" for r in records)
            else "passed"
        ),
        "expected_cases": len(expected),
        "completed_cases_read": len(records),
        "pending_case_ids": pending_ids,
        "cases": records,
        "source_unchanged": unchanged,
        "original_config_sha256": canonical_hash(config),
        "scope": "Frozen completed inputs; all loss/mass gates and actual grid halving. A finite-case numerical check, not a whole-domain error bound.",
    }
    write_json(out / "input-manifest.json", inputs)
    write_json(out / "summary.json", summary)
    return summary
