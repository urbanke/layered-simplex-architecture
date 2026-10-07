"""Verify distributed shards and restore the ordinary campaign artifact layout.

Merging never samples or evaluates a model. Raw diagnostic records are streamed;
only the small fields needed by the existing aggregators are held in memory.
"""

from __future__ import annotations

import gzip
import json
import shutil
import tempfile
from itertools import zip_longest
from pathlib import Path

import numpy as np

from .artifacts import (
    Run,
    canonical_hash,
    read_json,
    sha256,
    source_identity,
    verify_run,
    write_json,
)
from .benchmark import (
    SEED_SCHEME,
    TARGET_IDS,
    _batch_sample_entries,
    _sampling_spec,
    aggregate_records,
    trial_ids,
    validate_config,
)
from .scaling import cells, selected_cells, summarize_scaling

SCALING_EXPERIMENTS = ("spectrum", "factorial", "depth_scaling")


def _rows(path):
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    with opener(path, "rt", encoding="utf8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + "\n")


def _safe_file(root, name):
    path = root / name
    if (
        not isinstance(name, str)
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"unsafe shard file: {name}")
    if not path.is_file():
        raise ValueError(f"missing shard file: {path}")
    return path


def _provenance(shard):
    return {"id": shard["job"]["id"], "manifest_sha256": shard["manifest_sha256"]}


def _campaign_admission(plan, root):
    admission_path = root / "admission.json"
    if not admission_path.exists() and plan["purpose"] != "production":
        return {"admission": None, "calibration": None, "inputs": []}
    admission = read_json(_safe_file(root, "admission.json"))
    if admission.get("purpose") != plan["purpose"]:
        raise ValueError("campaign admission purpose differs from the plan")
    calibration, inputs = None, [admission_path]
    if admission.get("calibration_sha256") is not None:
        path = _safe_file(root, "parent-calibration.json")
        calibration = read_json(path)
        if canonical_hash(calibration) != admission["calibration_sha256"]:
            raise ValueError("parent calibration differs from campaign admission")
        inputs.append(path)
    elif plan["purpose"] == "production":
        raise ValueError("production campaign admission lacks parent calibration")
    return {"admission": admission, "calibration": calibration, "inputs": inputs}


def _verified_shards(plan, runs_dir, repo):
    # Import lazily: the controller also exposes this merge entry point.
    from .distributed import (
        job_protocol,
        validate_plan,
        validate_record_engine,
        validate_record_runtime,
        validate_saved_admission,
    )

    validate_plan(plan)
    current = source_identity(repo)
    if (
        current["tree_sha256"] != plan["source_tree_sha256"]
        or current["commit"] != plan["source_commit"]
    ):
        raise ValueError("merge source tree/commit differs from the campaign plan")
    runs_root = Path(runs_dir).resolve()
    runtime = read_json(_safe_file(runs_root, "runtime.json"))
    if any(
        runtime.get(key) != plan[key]
        for key in ("engine_options_sha256", "power_settings_sha256")
    ):
        raise ValueError("campaign runtime settings differ from the plan")
    admission = _campaign_admission(plan, runs_root)
    jobs_root = runs_root / "jobs"
    expected = {job["id"] for job in plan["jobs"]}
    if not jobs_root.is_dir() or jobs_root.is_symlink():
        raise ValueError("missing or unsafe campaign jobs directory")
    actual = {path.name for path in jobs_root.iterdir()}
    if actual != expected:
        raise ValueError(
            f"incomplete or unexpected shard set: {sorted(actual ^ expected)}"
        )
    result = {name: [] for name in plan["protocol"]["experiments"]}
    engines = {}
    for job in plan["jobs"]:
        path = jobs_root / job["id"]
        if (
            not path.is_dir()
            or path.is_symlink()
            or any(p.is_symlink() for p in path.rglob("*"))
        ):
            raise ValueError(f"unsafe shard directory: {job['id']}")
        record = verify_run(path)
        name = job["experiment"]
        expected_protocol = job_protocol(plan, job)
        if record["experiment"] != name or record["purpose"] != plan["purpose"]:
            raise ValueError(f"shard experiment/purpose differs from plan: {job['id']}")
        if (
            record["source"]["tree_sha256"] != plan["source_tree_sha256"]
            or record["source"]["commit"] != plan["source_commit"]
        ):
            raise ValueError(f"shard source differs from plan: {job['id']}")
        if (
            record["protocol_sha256"] != canonical_hash(expected_protocol)
            or read_json(path / "protocol.json") != expected_protocol
        ):
            raise ValueError(f"shard protocol differs from plan: {job['id']}")
        engine = record.get("engine")
        validate_record_engine(plan, name, engine, runtime=runtime)
        validate_record_runtime(record, runtime)
        validate_saved_admission(plan, job, record, admission["calibration"])
        if name in engines and engines[name] != engine:
            raise ValueError(f"inconsistent shard engine identity for {name}")
        engines[name] = engine
        result[name].append(
            {
                "job": job,
                "path": path,
                "record": record,
                "manifest_sha256": sha256(path / "manifest.json"),
            }
        )
    return result, engines, runtime, admission


def _benchmark_samples(shards, config, destination):
    """Assemble one target/trial at a time, including separately saved n cells."""
    sources = {}
    numpy_versions = set()
    for shard in shards:
        settings = validate_config(shard["job"]["config"])
        data = shard["path"] / "data"
        if read_json(data / "benchmark-config.json") != settings:
            raise ValueError("benchmark shard config differs from its protocol")
        manifest = read_json(data / "samples/manifest.json")
        if manifest["sampling"] != _sampling_spec(settings):
            raise ValueError(
                "benchmark shard sample specification differs from its protocol"
            )
        numpy_versions.add(manifest["numpy_version"])
        entries = _batch_sample_entries(manifest, settings)
        sample_paths = set()
        shard["sample_entries"] = {}
        for entry in entries:
            path = _safe_file(data / "samples", entry["path"])
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"sample checksum mismatch: {path}")
            sample_paths.add(path.name)
            key = (entry["target_id"], entry["trial"])
            shard["sample_entries"][key] = entry
            sources.setdefault(key, []).append((shard, settings, entry, path))
        actual = {p.name for p in (data / "samples").iterdir()}
        if actual != sample_paths | {"manifest.json"}:
            raise ValueError("unexpected benchmark sample files")
    expected = {
        (target, trial) for target in config["targets"] for trial in trial_ids(config)
    }
    if set(sources) != expected:
        raise ValueError("incomplete or unexpected benchmark samples")
    destination.mkdir()
    entries, by_trial = [], {}
    for target in config["targets"]:
        for trial in trial_ids(config):
            arrays, provenance = {}, []
            fragments = sources[(target, trial)]
            for shard, settings, original, path in fragments:
                with np.load(path, allow_pickle=False) as saved:
                    expected_keys = {
                        "target",
                        *(f"counts_{n}" for n in settings["n_values"]),
                    }
                    if set(saved.files) != expected_keys:
                        raise ValueError(
                            "benchmark sample arrays differ from the shard grid"
                        )
                    p = saved["target"]
                    if (
                        p.shape != (config["d"],)
                        or np.any(~np.isfinite(p))
                        or np.any(p < 0)
                        or not np.isclose(p.sum(), 1)
                    ):
                        raise ValueError("invalid benchmark target array")
                    if "target" in arrays and not np.array_equal(arrays["target"], p):
                        raise ValueError(
                            "different target arrays for the same global benchmark trial"
                        )
                    arrays["target"] = p
                    for n in settings["n_values"]:
                        key = f"counts_{n}"
                        counts = saved[key]
                        if key in arrays:
                            raise ValueError("duplicate benchmark sample cell")
                        if (
                            counts.shape != (config["d"],)
                            or not np.issubdtype(counts.dtype, np.integer)
                            or np.any(counts < 0)
                            or int(counts.sum()) != n
                        ):
                            raise ValueError("benchmark sample counts differ from n,d")
                        arrays[key] = counts
                provenance.append(
                    {
                        **_provenance(shard),
                        "path": original["path"],
                        "sha256": original["sha256"],
                        "n_values": settings["n_values"],
                    }
                )
            if set(arrays) != {"target", *(f"counts_{n}" for n in config["n_values"])}:
                raise ValueError("incomplete benchmark sample-size coverage")
            path = destination / f"{target}-trial-{trial:04d}.npz"
            if len(fragments) == 1:
                shutil.copyfile(fragments[0][3], path)
            else:
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
            entry = {
                "target_id": target,
                "trial": trial,
                "rng_coordinates": [config["seed"], TARGET_IDS.index(target), trial],
                "path": path.name,
                "sha256": sha256(path),
                "source_samples": provenance,
            }
            entries.append(entry)
            by_trial[(target, trial)] = entry
    manifest = {
        "schema_version": 1,
        "sampling": _sampling_spec(config),
        "seed_scheme": SEED_SCHEME,
        "numpy_version": next(iter(numpy_versions))
        if len(numpy_versions) == 1
        else None,
        "source_numpy_versions": sorted(numpy_versions),
        "files": entries,
    }
    write_json(destination / "manifest.json", manifest)
    return by_trial


def _merge_benchmark(shards, config, destination, batch_size, *, trial_blocks=10):
    config = validate_config(config)
    destination.mkdir()
    write_json(destination / "benchmark-config.json", config)
    samples = _benchmark_samples(shards, config, destination / "samples")
    records = []
    with (destination / "trials.jsonl").open("x", encoding="utf8") as stream:
        for shard in shards:
            settings = shard["job"]["config"]
            expected = {
                (target, trial, n)
                for target in settings["targets"]
                for trial in trial_ids(settings)
                for n in settings["n_values"]
            }
            seen = set()
            for row in _rows(shard["path"] / "data/trials.jsonl"):
                identity = (row["target_id"], row["trial"], row["n"])
                if identity not in expected or identity in seen:
                    raise ValueError(
                        "incomplete, unexpected or duplicate benchmark shard trial"
                    )
                seen.add(identity)
                old_sample = shard["sample_entries"][identity[:2]]
                if (
                    row["sample_set_id"] != config["sample_set_id"]
                    or row["sample_file"] != old_sample["path"]
                    or row["sample_sha256"] != old_sample["sha256"]
                ):
                    raise ValueError(
                        "benchmark trial references a different saved sample"
                    )
                if set(row["losses"]) != set(config["methods"]):
                    raise ValueError(
                        "benchmark trial method coverage differs from protocol"
                    )
                sample = samples[identity[:2]]
                row["source_job"] = {
                    **_provenance(shard),
                    "sample_file": row["sample_file"],
                    "sample_sha256": row["sample_sha256"],
                }
                row.update(sample_file=sample["path"], sample_sha256=sample["sha256"])
                _write_row(stream, row)
                records.append(
                    {key: value for key, value in row.items() if key != "diagnostics"}
                )
            if seen != expected:
                raise ValueError("incomplete benchmark shard trial coverage")
    summary = aggregate_records(records, config)
    summary.update(
        sample_manifest_sha256=sha256(destination / "samples/manifest.json"),
        trial_records_sha256=sha256(destination / "trials.jsonl"),
    )
    batching = [
        shard for shard in shards if (shard["path"] / "data/batching.jsonl").is_file()
    ]
    if batching:
        with (destination / "batching.jsonl").open("x", encoding="utf8") as stream:
            for shard in batching:
                for row in _rows(shard["path"] / "data/batching.jsonl"):
                    _write_row(stream, {**row, "source_job": _provenance(shard)})
        summary["batching"] = {
            "batch_size": batch_size,
            "source_jobs": len(batching),
            "preparation_records_file": "batching.jsonl",
            "preparation_records_sha256": sha256(destination / "batching.jsonl"),
        }
    write_json(destination / "summary.json", summary)
    from .block_diagnostics import benchmark_block_diagnostics

    write_json(
        destination / "block-diagnostics.json",
        benchmark_block_diagnostics(
            records,
            config,
            block_count=trial_blocks,
        ),
    )


def _scaling_sources(name, shards):
    by_cell = {}
    for shard in shards:
        config = shard["job"]["config"]
        root = shard["path"] / "data"
        if read_json(root / "config.json") != {"kind": name, **config}:
            raise ValueError("scaling shard config differs from its protocol")
        selected = selected_cells(name, config)
        for prefix, extension in (
            ("target", "npz"),
            ("samples", "jsonl.gz"),
            ("trials", "jsonl.gz"),
        ):
            expected = {f"{prefix}-{index:04d}.{extension}" for index, _ in selected}
            if {p.name for p in root.glob(f"{prefix}-*.{extension}")} != expected:
                raise ValueError("incomplete or unexpected scaling shard cell files")
        for cell_id, _ in selected:
            by_cell.setdefault(cell_id, []).append(shard)
    return by_cell


def _validate_draw(draw, cell_id, cell, trial):
    if (
        draw["cell_id"] != cell_id
        or draw["trial"] != trial
        or any(draw[key] != value for key, value in cell.items())
    ):
        raise ValueError(
            "scaling trial metadata differs from the declared global cell/trial"
        )
    labels, counts = draw["occupied_labels"], draw["counts"]
    if (
        len(labels) != len(counts)
        or labels != sorted(set(labels))
        or any(type(label) is not int or not 0 <= label < cell["d"] for label in labels)
    ):
        raise ValueError("invalid scaling sample labels")
    if (
        any(type(count) is not int or count <= 0 for count in counts)
        or sum(counts) != cell["n"]
        or draw["profile"] != sorted(counts, reverse=True)
    ):
        raise ValueError("invalid scaling sample counts/profile")


def _merge_scaling(name, shards, config, destination, batch_size):
    by_cell = _scaling_sources(name, shards)
    complete_cells = list(cells(name, config))
    if set(by_cell) != set(range(len(complete_cells))):
        raise ValueError("incomplete or unexpected scaling cells")
    destination.mkdir()
    full_config = {"kind": name, **config}
    write_json(destination / "config.json", full_config)
    write_json(
        destination / "execution.json",
        {"batch_size": batch_size, "merged_source_jobs": len(shards)},
    )
    # summarize_scaling deliberately uses the established aggregation formula.
    # Its input projection excludes large quadrature diagnostics and draw labels.
    with tempfile.TemporaryDirectory(
        prefix=".merge-summary-", dir=destination.parent
    ) as temporary:
        compact = Path(temporary)
        write_json(compact / "config.json", full_config)
        for cell_id, cell in enumerate(complete_cells):
            ordered = sorted(
                by_cell[cell_id],
                key=lambda shard: shard["job"]["config"].get("trial_start", 0),
            )
            target_name = f"target-{cell_id:04d}.npz"
            sample_name = f"samples-{cell_id:04d}.jsonl.gz"
            trial_name = f"trials-{cell_id:04d}.jsonl.gz"
            target = None
            next_trial = config.get("trial_start", 0)
            with (
                gzip.open(
                    destination / sample_name, "xt", encoding="utf8"
                ) as sample_stream,
                gzip.open(
                    destination / trial_name, "xt", encoding="utf8"
                ) as trial_stream,
                gzip.open(
                    compact / trial_name, "xt", encoding="utf8"
                ) as compact_stream,
            ):
                for shard in ordered:
                    root = shard["path"] / "data"
                    settings = shard["job"]["config"]
                    if settings.get("trial_start", 0) != next_trial:
                        raise ValueError(
                            "incomplete or duplicate scaling trial intervals"
                        )
                    with np.load(root / target_name, allow_pickle=False) as saved:
                        if set(saved.files) != {"probabilities"}:
                            raise ValueError("unexpected scaling target arrays")
                        p = saved["probabilities"]
                    if (
                        p.shape != (cell["d"],)
                        or np.any(~np.isfinite(p))
                        or np.any(p < 0)
                        or not np.isclose(p.sum(), 1)
                    ):
                        raise ValueError("invalid scaling target array")
                    if target is None:
                        target = p
                        shutil.copyfile(root / target_name, destination / target_name)
                    elif not np.array_equal(target, p):
                        raise ValueError(
                            "different scaling target arrays for one global cell"
                        )
                    count = 0
                    for draw, row in zip_longest(
                        _rows(root / sample_name), _rows(root / trial_name)
                    ):
                        if draw is None or row is None or count >= settings["trials"]:
                            raise ValueError(
                                "incomplete or duplicate scaling shard samples/trials"
                            )
                        _validate_draw(draw, cell_id, cell, next_trial)
                        if (
                            any(row.get(key) != value for key, value in draw.items())
                            or row.get("sample_source") != sample_name
                        ):
                            raise ValueError(
                                "scaling trial differs from its saved common sample"
                            )
                        source = _provenance(shard)
                        _write_row(sample_stream, {**draw, "source_job": source})
                        _write_row(trial_stream, {**row, "source_job": source})
                        keys = {
                            "cell_id",
                            "trial",
                            "profile",
                            "discovered",
                            "regret_bits",
                            "mixture_regret_bits",
                            "entropy_bits",
                            "expected_discovered",
                            "naming_bits",
                            *cell,
                        }
                        _write_row(compact_stream, {key: row[key] for key in keys})
                        count += 1
                        next_trial += 1
                    if count != settings["trials"]:
                        raise ValueError("incomplete scaling shard trial coverage")
            if next_trial != config.get("trial_start", 0) + config["trials"]:
                raise ValueError("incomplete scaling trial coverage")
        summary = summarize_scaling(compact)
    write_json(destination / "summary.json", summary)


def merge_campaign(plan, runs_dir, out, *, repo):
    """Restore complete, report-compatible Runs after validating every shard.

    The exact planned run set lives at ``runs_dir/jobs/<id>``. Completed runs
    are never overwritten. All source manifests are pinned as Run inputs, and
    source jobs are reverified before each merged experiment is completed.
    """
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    shards_by_name, engines, runtime, admission = _verified_shards(plan, runs_dir, repo)
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "protocol_sha256": plan["protocol_sha256"],
        "experiments": {},
    }
    for name, config in plan["protocol"]["experiments"].items():
        shards = shards_by_name[name]
        inputs = [shard["path"] / "manifest.json" for shard in shards]
        inputs.append(Path(runs_dir).resolve() / "runtime.json")
        inputs.extend(admission["inputs"])
        with Run(
            out / name,
            repo=repo,
            protocol=plan["protocol"],
            experiment=name,
            purpose=plan["purpose"],
            engine_identity=engines[name],
            inputs=inputs,
            command=["distributed-merge", str(Path(runs_dir).resolve()), str(out)],
        ) as run:
            write_json(run.path / "source-runtime.json", runtime)
            if admission["admission"] is not None:
                write_json(run.path / "source-admission.json", admission["admission"])
            if admission["calibration"] is not None:
                write_json(
                    run.path / "source-calibration.json", admission["calibration"]
                )
            write_json(
                run.path / "source-jobs.json",
                {
                    "schema_version": 1,
                    "plan_sha256": canonical_hash(plan),
                    "jobs": {
                        shard["job"]["id"]: {
                            "manifest": str(shard["path"] / "manifest.json"),
                            "manifest_sha256": shard["manifest_sha256"],
                            "protocol_sha256": shard["record"]["protocol_sha256"],
                            "config": shard["job"]["config"],
                        }
                        for shard in shards
                    },
                },
            )
            if name.startswith("benchmark_"):
                _merge_benchmark(
                    shards,
                    config,
                    run.path / "data",
                    plan["batch_size"],
                    trial_blocks=plan["protocol"]
                    .get("uncertainty", {})
                    .get("trial_blocks", 10),
                )
            elif name in SCALING_EXPERIMENTS:
                _merge_scaling(
                    name, shards, config, run.path / "data", plan["batch_size"]
                )
            else:
                if len(shards) != 1 or shards[0]["job"]["config"] != config:
                    raise ValueError(
                        f"singleton experiment has inconsistent shards: {name}"
                    )
                shutil.copytree(shards[0]["path"] / "data", run.path / "data")
            for shard in shards:
                verify_run(shard["path"])
        record = verify_run(out / name)
        result["experiments"][name] = {
            "path": str(out / name),
            "manifest_sha256": sha256(out / name / "manifest.json"),
            "jobs": len(shards),
            "status": record["status"],
        }
    write_json(out / "merge.json", result)
    return result
