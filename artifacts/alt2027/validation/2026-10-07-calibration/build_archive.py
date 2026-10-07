"""Package the five completed 2026-10-07 calibration runs without altering them."""

import argparse
import ast
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(path.read_text())


def inventory(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink excluded from immutable archive: {path}")
        if path.is_file():
            files[str(path.relative_to(directory))] = path.read_bytes()
    return files


def check_recorded_files(directory, records):
    for name, item in records.items():
        expected = item if isinstance(item, str) else item["sha256"]
        data = (directory / name).read_bytes()
        if digest(data) != expected or (isinstance(item, dict) and len(data) != item["bytes"]):
            raise ValueError(f"recorded artifact changed: {directory / name}")


def compare_source(root, capsule, current, expected):
    copied, actual = capsule.read_bytes(), (root / current).read_bytes()
    if digest(copied) != expected:
        raise ValueError(f"existing source capsule hash mismatch: {capsule}")
    return {"repository_path": current, "recorded_sha256": expected,
            "capsule_sha256": digest(copied), "current_sha256": digest(actual),
            "capsule_matches_record": True, "current_matches_record": digest(actual) == expected}


def selected_ast(data, names):
    nodes = {}
    for node in ast.parse(data).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            nodes[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    nodes[target.id] = ast.dump(node, include_attributes=False)
    if set(nodes) != set(names):
        raise ValueError("requested source symbols were not found")
    return nodes


def package(root, output):
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    moment = datetime.now(UTC).isoformat()
    (output / "build_archive.py").write_bytes(Path(__file__).read_bytes())
    runs = [
        ("power-calibration-001", "power-calibration-001"),
        ("prior-validation-002", "prior-validation-002"),
        ("kernel-calibration-20261007-declared-grid", "kernel-calibration-20261007-declared-grid"),
        ("chain-pilot-20261007-all-depths", "chain-pilot-20261007/all-depths"),
        ("batch-depth-pilot-001", "batch-depth-pilot-001"),
    ]
    index = {"schema_version": 1, "packaged_utc": moment,
             "purpose": "immutable validation archive; no production certification",
             "pending_excluded": [{"source_run": "output/alt2027/depth-calibration-20261007",
                                    "status": "pending", "included": False,
                                    "reason": "Active profile-depth calibration at packaging; no final results included."}],
             "archives": []}
    for name, relative in runs:
        original = root / "output/alt2027" / relative
        raw = inventory(original)
        initial = {key: digest(data) for key, data in raw.items()}
        supplement, sources = {}, []
        metadata = {"archive_id": name, "original_run": str(original.relative_to(root)),
                    "packaged_utc": moment, "original_files_modified": False,
                    "source_comparison": sources, "production_certified": False}
        if name.startswith("power-"):
            check_recorded_files(original, read_json(original / "files.json"))
            start = read_json(original / "started.json")["source_files"]
            finish = read_json(original / "summary.json")["source_files_at_end"]
            if start != finish:
                raise ValueError("power source identity changed during run")
            for filename, expected in start.items():
                repo_path = "src/lsa/alt/" + filename
                current = (root / repo_path).read_bytes()
                archived = current
                retrieval = "current bytes verified against run-start and run-finish hashes"
                if digest(current) != expected:
                    if filename != "benchmark.py":
                        raise ValueError(f"unexpected post-run source change: {filename}")
                    archived = subprocess.check_output(
                        ["git", "show", "e51c449:" + repo_path], cwd=root)
                    if selected_ast(archived, ["make_target", "TARGET_IDS"]) != selected_ast(current, ["make_target", "TARGET_IDS"]):
                        raise ValueError("power sampling dependencies changed after pilot")
                    retrieval = "git e51c449 blob verified against run-start and run-finish hashes"
                    supplement["current-at-packaging/" + repo_path] = current
                    metadata["later_execution_integration"] = {
                        "file": repo_path, "note": "Benchmark cohort batching was integrated after this completed power pilot.",
                        "unchanged_imported_symbols_verified_by_ast": ["make_target", "TARGET_IDS"],
                        "numerical_power_source_unchanged": True}
                if digest(archived) != expected:
                    raise ValueError("could not recover exact power source bytes")
                supplement["run-source-capsule/" + repo_path] = archived
                sources.append({"repository_path": repo_path, "recorded_sha256": expected,
                                "capsule_sha256": digest(archived), "current_sha256": digest(current),
                                "current_matches_record": digest(current) == expected,
                                "retrieval": retrieval})
            metadata["source_scope"] = "Original run fingerprints, with exact bytes recovered and verified at packaging."
        elif name.startswith("prior-"):
            check_recorded_files(original, read_json(original / "summary.json")["artifacts"])
            for path, expected in read_json(original / "module-fingerprints.json").items():
                sources.append(compare_source(root, original / "sources" / path,
                                              "src/lsa/alt/" + path, expected))
            metadata["source_scope"] = "Existing original module-fingerprint capsule; only its explicitly fingerprinted files."
        elif name.startswith("kernel-"):
            for path, expected in read_json(original / "source-capsule.json").items():
                sources.append(compare_source(root, original / path,
                                              path.removeprefix("source-capsule/"), expected))
            metadata["source_scope"] = "Existing original kernel source capsule; all comparisons retained including failing legacy/stored-row diagnostics."
            metadata["scientific_status"] = "All declared corrected-direct gates pass; stored-row strict-gate caveat preserved in assessment.json."
        elif name.startswith("batch-"):
            check_recorded_files(original, read_json(original / "artifact-hashes.json"))
            for path, expected in read_json(original / "source-hashes.json").items():
                sources.append(compare_source(root, original / "sources" / path, path, expected))
            metadata["source_scope"] = "Existing original batch pilot source capsule."
        elif name.startswith("chain-"):
            engine = read_json(original / "engine.json")
            engine_sources = {"src/lsa/alt/depth.py": engine["implementation_sha256"],
                              "src/lsa/alt/_vendor/pmwm/provenance.json": engine["vendor_provenance_sha256"]}
            engine_sources.update({"src/lsa/alt/_vendor/pmwm/"+p: h
                                   for p, h in engine["vendor_source_sha256"].items()})
            for repo_path, expected in engine_sources.items():
                data = (root / repo_path).read_bytes()
                if digest(data) != expected:
                    raise ValueError("chain engine differs from recorded identity")
                supplement["engine-source-capsule/"+repo_path] = data
                sources.append({"repository_path": repo_path, "recorded_sha256": expected,
                                "current_sha256": digest(data), "current_matches_record": True,
                                "identity_origin": "original engine.json"})
            for filename in ("chain_validation.py", "bible.py", "benchmark.py", "baselines.py", "artifacts.py"):
                repo_path = "src/lsa/alt/"+filename
                data = (root / repo_path).read_bytes()
                supplement["source-at-packaging/"+repo_path] = data
                sources.append({"repository_path": repo_path, "sha256_at_packaging": digest(data),
                                "identity_origin": "observed at archive packaging only",
                                "run_start_hash_available": False})
            runner = root / "output/alt2027/chain-pilot-20261007/full.py"
            supplement["source-at-packaging/runner/full.py"] = runner.read_bytes()
            corpus = read_json(original / "corpus.json")
            if digest((root / corpus["compressed_file"]).read_bytes()) != corpus["compressed_sha256"]:
                raise ValueError("canonical corpus no longer matches chain record")
            metadata["source_scope"] = (
                "Engine bytes match the original engine identity. The chain module and helper hashes "
                "were first archived at packaging; they are not claimed to have been measured at run start.")
            metadata["external_inputs"] = (
                "The canonical corpus is tracked separately in data/ and its compressed hash was verified. "
                "Large numerical stores are not copied; original engine.json retains their per-file identities.")
        if any(item.get("current_matches_record") is False for item in sources) and not name.startswith("power-"):
            raise ValueError(f"unexpected current source mismatch in {name}")
        summary = read_json(original / "summary.json")
        metadata["original_summary_status"] = summary.get("status", "see original kernel assessment")
        members = {name + "/run/" + key: value for key, value in raw.items()}
        members.update({name + "/packaging/" + key: value for key, value in supplement.items()})
        members[name + "/packaging/archive-metadata.json"] = json_bytes(metadata)
        archive = output / (name + ".tar.gz")
        with archive.open("xb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path, data in sorted(members.items()):
                    info = tarfile.TarInfo(path)
                    info.size, info.mtime, info.mode = len(data), 0, 0o644
                    tar.addfile(info, io.BytesIO(data))
        manifest = {"archive": archive.name, "archive_sha256": digest(archive.read_bytes()),
                    "archive_bytes": archive.stat().st_size,
                    "files": {path: {"sha256": digest(data), "bytes": len(data)} for path, data in sorted(members.items())}}
        with tarfile.open(archive, "r:gz") as tar:
            if set(tar.getnames()) != set(manifest["files"]):
                raise ValueError("archive inventory verification failed")
            for member in tar:
                record = manifest["files"][member.name]
                if not member.isfile() or member.size != record["bytes"] or digest(tar.extractfile(member).read()) != record["sha256"]:
                    raise ValueError("archive payload checksum verification failed")
        if initial != {key: digest(value) for key, value in inventory(original).items()}:
            raise ValueError(f"original run changed during packaging: {name}")
        manifest_name = name + ".manifest.json"
        (output / manifest_name).write_bytes(json_bytes(manifest))
        index["archives"].append({"archive": archive.name, "sha256": manifest["archive_sha256"],
                                  "bytes": manifest["archive_bytes"], "file_count": len(members),
                                  "manifest": manifest_name, "manifest_sha256": digest((output / manifest_name).read_bytes()),
                                  "original_run": metadata["original_run"],
                                  "source_scope": metadata["source_scope"],
                                  "source_current_mismatches": [s for s in sources if s.get("current_matches_record") is False],
                                  "source_at_packaging_only": [s for s in sources if s.get("run_start_hash_available") is False]})
        print(name, len(members), archive.stat().st_size, flush=True)
    index["compressed_archive_bytes"] = sum(a["bytes"] for a in index["archives"])
    index["all_archive_payload_checksums_verified"] = True
    index["all_original_run_files_unchanged_during_packaging"] = True
    (output / "archive-index.json").write_bytes(json_bytes(index))
    sums = "".join(f"{digest(p.read_bytes())}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file())
    (output / "SHA256SUMS").write_text(sums)
    print("total compressed bytes", index["compressed_archive_bytes"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package(args.repo, args.out)
