"""Archive the completed frozen pilot without changing its checkout or records."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from lsa.alt.artifacts import source_identity, verify_run

RUN_ID = "timing-uncertainty-d1e4d26-001"
COMMIT = "d1e4d2693610595d43434a6079e2162176e3c149"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not archive members: {path}")
        if path.is_file():
            result[str(path.relative_to(directory))] = path.read_bytes()
    return result


def package(frozen, working, out):
    frozen, working, out = frozen.resolve(), working.resolve(), out.resolve()
    run = frozen / "output/alt2027" / RUN_ID
    analysis = run.with_name(RUN_ID + "-analysis")
    record = verify_run(run)
    source_before = source_identity(frozen)
    if (
        record["purpose"] != "validation"
        or record["status"] != "complete"
        or record["source"]["commit"] != COMMIT
        or record["source"]["dirty"]
        or source_before != record["source"]
    ):
        raise ValueError("pilot source/status is not the verified clean frozen run")
    protocol = json.loads((run / "protocol.json").read_text())
    config = protocol["experiments"]["benchmark_primary"]
    if (
        config["targets"] != ["uniform", "zipf_5"]
        or config["n_values"] != [1000]
        or config["trials"] != 20
        or config["depths"] != list(range(81))
        or config["fixed_depth"] != 22
        or len(config["methods"]) != 9
    ):
        raise ValueError("unexpected pilot scientific settings")
    rows = [
        json.loads(line)
        for line in (run / "data/trials.jsonl").read_text().splitlines()
    ]
    identities = {(row["target_id"], row["trial"], row["n"]) for row in rows}
    expected = {
        (target, trial, 1000) for target in config["targets"] for trial in range(20)
    }
    if len(rows) != 40 or identities != expected:
        raise ValueError("unexpected or duplicate trial identities")
    for row in rows:
        if (
            set(row["losses"]) != set(config["methods"])
            or row["posteriors"]["depth"]["indices"] != list(range(81))
            or len(row["diagnostics"]["depth"]["component_log_evidence_nats"]) != 81
        ):
            raise ValueError("a trial lacks a declared method or depth")
    run_files, analysis_files = inventory(run), inventory(analysis)
    members = {RUN_ID + "/run/" + key: data for key, data in run_files.items()}
    members.update(
        {RUN_ID + "/analysis/" + key: data for key, data in analysis_files.items()}
    )
    observed = {}

    def include(path, destination, expected_hash=None):
        data = path.read_bytes()
        if expected_hash is not None and digest(data) != expected_hash:
            raise ValueError(f"recorded input/source changed: {path}")
        members[RUN_ID + "/" + destination] = data
        observed[str(path)] = digest(data)

    for item in record["inputs"]:
        include(Path(item["path"]), "inputs/" + Path(item["path"]).name, item["sha256"])
    analysis_metadata = json.loads(analysis_files["timing-and-scope.json"])
    include(
        frozen / "output/alt2027/summarize-timing-uncertainty.py",
        "analysis/summarize-timing-uncertainty.py",
        analysis_metadata["analysis_script_sha256"],
    )
    if analysis_metadata["run_manifest_sha256"] != digest(run_files["manifest.json"]):
        raise ValueError("analysis is linked to a different run")
    include(run.with_suffix(".log"), "packaging/original-cli.log")
    source_paths = [
        name
        for name in record["source"]["files"]
        if (name.startswith("src/lsa/") and name.endswith(".py"))
        or name
        in (
            "requirements-alt.lock",
            "pyproject.toml",
            "scripts/alt_experiments.py",
            "src/lsa/alt/_vendor/pmwm/provenance.json",
            "experiments/alt2027/protocol.json",
        )
    ]
    for name in source_paths:
        include(
            frozen / name, "run-source-capsule/" + name, record["source"]["files"][name]
        )
    current_protocol = json.loads(
        (working / "experiments/alt2027/protocol.json").read_text()
    )
    current_trials = {
        name: current_protocol["experiments"][name]["trials"]
        for name in ("benchmark_primary", "benchmark_powers")
    }
    metadata = {
        "schema_version": 1,
        "archive_id": RUN_ID,
        "packaged_utc": datetime.now(UTC).isoformat(),
        "purpose": "completed timing and paired-uncertainty validation pilot; no production certification",
        "original_run": str(run),
        "original_analysis": str(analysis),
        "source_commit": COMMIT,
        "source_tree_sha256": record["source"]["tree_sha256"],
        "source_scope": "Scientific Python sources, CLI launcher, environment lock, project metadata, vendor provenance and base protocol verified against original Run source hashes at packaging. The original Run retains the complete tracked-file identity.",
        "source_capsule_files": len(source_paths),
        "original_files_modified": False,
        "working_benchmark_trial_counts_observed_at_packaging": current_trials,
        "external_store": "Not copied; original effective engine record and saved engine input retain complete file hashes.",
        "analysis_scope": "All18 method cells and all72 unordered paired comparisons; nonfinite trials retained.",
        "validation_checks": {
            "run_manifest_and_all_outputs": True,
            "all40_trial_identities": True,
            "all81_depths": True,
            "all9_methods": True,
            "input_bytes_match_recorded_hashes": True,
        },
    }
    members[RUN_ID + "/packaging/archive-metadata.json"] = encoded(metadata)
    include(out / "README.md", "packaging/README.md")
    include(Path(__file__).resolve(), "packaging/build_archive.py")
    archive = out / (RUN_ID + ".tar.gz")
    with (
        archive.open("xb") as stream,
        gzip.GzipFile(
            filename="", fileobj=stream, mode="wb", mtime=0, compresslevel=9
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for path, data in sorted(members.items()):
            info = tarfile.TarInfo(path)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    manifest = {
        "archive": archive.name,
        "archive_sha256": digest(archive.read_bytes()),
        "archive_bytes": archive.stat().st_size,
        "files": {
            path: {"sha256": digest(data), "bytes": len(data)}
            for path, data in sorted(members.items())
        },
    }
    with tarfile.open(archive, "r:gz") as tar:
        if set(tar.getnames()) != set(manifest["files"]):
            raise ValueError("archive inventory differs")
        for member in tar:
            item = manifest["files"][member.name]
            if (
                not member.isfile()
                or member.size != item["bytes"]
                or digest(tar.extractfile(member).read()) != item["sha256"]
            ):
                raise ValueError("archive member checksum differs")
    if (
        run_files != inventory(run)
        or analysis_files != inventory(analysis)
        or source_identity(frozen) != source_before
        or any(
            digest(Path(path).read_bytes()) != value for path, value in observed.items()
        )
    ):
        raise ValueError(
            "an original source, run, analysis, or input changed during packaging"
        )
    manifest_path = out / (RUN_ID + ".manifest.json")
    with manifest_path.open("xb") as stream:
        stream.write(encoded(manifest))
    index = {
        "schema_version": 1,
        "archives": [
            {
                "archive": archive.name,
                "sha256": manifest["archive_sha256"],
                "bytes": manifest["archive_bytes"],
                "file_count": len(members),
                "manifest": manifest_path.name,
                "manifest_sha256": digest(manifest_path.read_bytes()),
            }
        ],
        "all_archive_payload_checksums_verified": True,
        "frozen_checkout_and_original_records_unchanged": True,
        "production_certified": False,
    }
    with (out / "archive-index.json").open("xb") as stream:
        stream.write(encoded(index))
    with (out / "SHA256SUMS").open("x") as stream:
        stream.write(
            "".join(
                f"{digest(path.read_bytes())}  {path.name}\n"
                for path in sorted(out.iterdir())
                if path.is_file() and path.name != "SHA256SUMS"
            )
        )
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package(args.frozen, args.working, args.out)
