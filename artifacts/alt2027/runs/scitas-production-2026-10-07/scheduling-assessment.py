"""Verify scheduling evidence and select 4/8/16 nodes; never submit jobs.

This is a resource gate, not numerical production admission. Its estimates use
finite representative profiles, with explicit conservative extrapolation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import heapq
import importlib.metadata
import json
import math
from pathlib import Path
import platform

from lsa.alt.artifacts import canonical_hash, read_json, sha256, source_identity, utc_now, verify_run, write_json
from lsa.alt.distributed import power_options, validate_plan
from lsa.alt.scaling import cells

CASES = {
    "primary-zipf1p5-20": ("benchmark_primary", {"targets": ["zipf_1.5"], "trials": 20, "trial_start": 0}),
    "primary-uniform-20": ("benchmark_primary", {"targets": ["uniform"], "trials": 20, "trial_start": 0}),
    "spectrum-d1e6-alpha1p5-20": ("spectrum", {"cell_ids": [27], "trials": 20, "trial_start": 0}),
    "depth-alpha2-20": ("depth_scaling", {"cell_ids": [0], "trials": 20, "trial_start": 0}),
    "factorial-d1e5-n1e4-alpha1p5-100": ("factorial", {"cell_ids": [96], "trials": 100, "trial_start": 0}),
    "bible-full": ("bible", {}),
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def positive(value, label):
    require(isinstance(value, (float, int)) and math.isfinite(value) and value > 0, label)
    return float(value)

def assess(args, report):
    inputs = report["input_files"]
    def read(path):
        path = Path(path)
        value = read_json(path)
        inputs[str(path.resolve())] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        return value

    plan = validate_plan(read(args.plan))
    report["plan_sha256"] = canonical_hash(plan)
    require(plan["purpose"] == "production", "expected production plan")
    require((plan["block_size"], plan["factorial_block_size"], plan["batch_size"]) == (20, 500, 20), "unassessed partition or batch size")
    expected_counts = {"architecture": 1, "benchmark_primary": 550, "benchmark_powers": 550,
                       "bible": 1, "bible_secondary": 1, "depth_scaling": 150,
                       "factorial": 1000, "spectrum": 1650, "validation": 1}
    require(dict(Counter(j["experiment"] for j in plan["jobs"])) == expected_counts, "unassessed campaign size")
    source = source_identity(args.repo)
    require(not source["dirty"] and source["commit"] == plan["source_commit"]
            and source["tree_sha256"] == plan["source_tree_sha256"], "source differs from frozen plan")
    report["source_commit"] = source["commit"]
    require(platform.system() == "Linux", "resource gate must run in the pinned Linux runtime")
    computational = {n: h for n, h in source["files"].items()
                     if (n.startswith("src/lsa/alt/") and n != "src/lsa/alt/admission.py")
                     or n == "requirements-alt.lock"}
    require(computational, "empty computational source set")
    def check_source(old):
        require(all(old.get(n) == h for n, h in computational.items()), "pilot computational source differs from plan")
    def check_environment(saved):
        require(saved["platform"].startswith("Linux") and saved["machine"] == platform.machine()
                and saved["python"].split()[0] == platform.python_version(), "timing runtime or host platform differs")
        for package in ("numpy", "scipy", "mpmath"):
            require(saved["packages"][package] == importlib.metadata.version(package), f"timing {package} version differs")
        thread_keys = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"}
        require(set(saved["threads"]) == thread_keys and all(v == "1" for v in saved["threads"].values()), "nested timing threads enabled or missing")

    pilots = {}
    for name, (experiment, selectors) in CASES.items():
        root = args.pilot_root / name
        result, request = read(root / "pilot-result.json"), read(root / "pilot-request.json")
        require(result["status"] == "passed" and not result.get("error"), f"failed pilot: {name}")
        record = verify_run(root / "run")
        read(root / "run/manifest.json")
        require(record["purpose"] == "validation" and record["experiment"] == experiment, f"wrong pilot purpose: {name}")
        check_source(record["source"]["files"])
        require(request["source"] == record["source"] and request["cpu_count"] == 1, f"pilot source or CPU mismatch: {name}")
        require(request["batch_size"] == 20 and request["purpose"] == "validation", f"pilot request mismatch: {name}")
        check_environment(record["environment"])
        actual = read(root / "run/protocol.json")
        expected = copy.deepcopy(plan["protocol"])
        config = copy.deepcopy(expected["experiments"][experiment])
        config.update(selectors)
        if experiment.startswith("benchmark_"):
            config["sampling_n_values"] = expected["experiments"][experiment]["n_values"]
        expected["experiments"] = {experiment: config}
        require(actual.get("status") in ("implementation", "frozen"), "unexpected pilot protocol state")
        expected["status"] = actual["status"]
        require(canonical_hash(actual) == canonical_hash(expected), f"pilot scientific settings differ: {name}")
        depth = record["engine"]["depth"]
        options = {k: depth[k] for k in ("mode", "store", "prediction_tolerance")}
        require(canonical_hash(options) == plan["engine_options_sha256"], f"pilot engine differs: {name}")
        runtime = depth["runtime"]
        import numpy, scipy
        require(runtime == {"python": platform.python_version(), "numpy": numpy.__version__,
                            "scipy": scipy.__version__, "system": platform.system(), "machine": platform.machine()}, "pilot runtime differs")
        seconds = positive(result["wall_seconds"], "invalid pilot elapsed time")
        rss = positive(result["peak_rss_kib"], "invalid pilot peak RSS")
        require(rss < 5 * 1024**2, f"pilot exceeds 5 GiB RSS: {name}")
        require(seconds < 4 * 3600, f"pilot exceeds 4 h limit: {name}")
        pilots[name] = {"wall_seconds": seconds, "peak_rss_kib": rss,
                        "run_output_bytes": sum(v["bytes"] for v in record["outputs"].values()),
                        "run_manifest_sha256": sha256(root / "run/manifest.json")}
        if experiment == "benchmark_primary":
            path = root / "run/data/batching.jsonl"
            batches = [json.loads(line) for line in path.read_text().splitlines()]
            inputs[str(path.resolve())] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            require([b["n"] for b in batches] == config["n_values"] and all(len(b["samples"]) == 20 for b in batches), "incomplete pilot batches")
            times = {str(b["n"]): positive(b["preparation"]["seconds"], "invalid cohort timing") for b in batches}
            pilots[name]["preparation_seconds_by_n"] = times
            pilots[name]["whole_run_nonpreparation_seconds"] = max(0, seconds - sum(times.values()))
    report["pilots"] = pilots

    late = args.pilot_root / "sampling-late-offsets"
    replay, request = read(late / "result.json"), read(late / "request.json")
    require(replay["status"] == "passed", "late-offset timing failed")
    check_source(request["source"]["files"])
    check_environment(request["environment"])
    require(request["purpose"] == "validation", "late-offset purpose mismatch")
    require({(r["experiment"], r["cell_id"], r["trial_start"], r["trials"]) for r in replay["rows"]}
            == {("spectrum", 22, 980, 20), ("spectrum", 27, 980, 20), ("factorial", 96, 4500, 500)}, "late-offset coverage differs")
    replay_cost = {}
    for row in replay["rows"]:
        expected_cell = list(cells(row["experiment"], plan["protocol"]["experiments"][row["experiment"]]))[row["cell_id"]]
        require(row["cell"] == expected_cell, "late-offset cell differs")
        path = late / row["last_count_file"]
        require(path.resolve().is_relative_to(late.resolve()) and sha256(path) == row["last_count_file_sha256"], "late-offset sample hash differs")
        inputs[str(path.resolve())] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        elapsed = positive(row["replay_seconds"], "invalid replay time")
        replay_cost[row["experiment"]] = max(replay_cost.get(row["experiment"], 0), elapsed)
    report["late_replay_seconds"] = replay_cost

    power_record = verify_run(args.power_root)
    read(args.power_root / "manifest.json")
    require(power_record["purpose"] == "validation", "power timing purpose differs")
    check_source(power_record["source"]["files"])
    check_environment(power_record["environment"])
    power_data = args.power_root / "data"
    summary, config = read(power_data / "summary.json"), read(power_data / "config.json")
    require(summary["status"] == "passed" and summary["source_unchanged"], "corrected power calibration did not pass")
    require(canonical_hash(power_options(config["default_settings"])) == plan["power_settings_sha256"], "power settings differ from plan")
    benchmark = plan["protocol"]["experiments"]["benchmark_powers"]
    require(config["targets"] == benchmark["targets"] and config["powers"] == benchmark["powers"]
            and config["d"] == benchmark["d"] and benchmark["n_values"] == [config["n"]], "power timing domain differs")
    power_seconds = {}
    for target in benchmark["targets"]:
        path = power_data / target / "components.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        inputs[str(path.resolve())] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        selected = [r for r in rows if r["setting"] == "default"]
        require([r["power"] for r in selected] == benchmark["powers"]
                and all(r["status"] == "passed" for r in selected), f"incomplete default powers: {target}")
        power_seconds[target] = sum(positive(r["seconds"], "invalid power timing") for r in selected)
    report["power_default_seconds_per_profile"] = power_seconds

    primary = max(pilots[n]["wall_seconds"] for n in ("primary-uniform-20", "primary-zipf1p5-20"))
    power_depth = max(pilots[n]["preparation_seconds_by_n"]["1000"] + pilots[n]["whole_run_nonpreparation_seconds"]
                      for n in ("primary-uniform-20", "primary-zipf1p5-20"))
    base_cost = {"benchmark_primary": primary,
                 "spectrum": pilots["spectrum-d1e6-alpha1p5-20"]["wall_seconds"] + replay_cost["spectrum"],
                 "depth_scaling": pilots["depth-alpha2-20"]["wall_seconds"] + max(replay_cost.values()),
                 "factorial": pilots["factorial-d1e5-n1e4-alpha1p5-100"]["wall_seconds"] * 5 + replay_cost["factorial"],
                 "bible": pilots["bible-full"]["wall_seconds"],
                 "bible_secondary": pilots["bible-full"]["wall_seconds"] * 1.01,
                 "architecture": 600.0, "validation": 600.0}
    jobs = []
    totals = defaultdict(float)
    for job in plan["jobs"]:
        family, local = job["experiment"], job["config"]
        if family == "benchmark_powers":
            base = power_seconds[local["targets"][0]] * local["trials"] + power_depth * local["trials"] / 20
        elif family in ("benchmark_primary", "spectrum", "depth_scaling"):
            base = base_cost[family] * local["trials"] / 20
        elif family == "factorial":
            base = base_cost[family] * local["trials"] / 500
        else:
            base = base_cost[family]
        inflated = 2 * base + 60
        require(inflated + 3600 < 24 * 3600, f"projected single job exceeds walltime: {job['id']}")
        jobs.append({"id": job["id"], "experiment": family, "measured_basis_seconds": base,
                     "budget_seconds": inflated})
        totals[family] += inflated
    report["model"] = {
        "time_multiplier": 2.0, "additional_seconds_per_job": 60,
        "reserve_seconds_per_controller": 3600, "cpus_per_node": 72,
        "memory_gib_per_node": 440, "walltime_seconds": 86400,
        "primary_basis": "Slower full 20-trial, eight-n primary pilot assigned to every primary target",
        "power_depth_basis_seconds_per_20_trials": power_depth,
        "spectrum_basis": "Full 139-depth d=1e6 alpha1.5 pilot plus worst measured late replay, assigned to every spectrum cell",
        "factorial_basis": "Five times the 100-profile large-cell elapsed time plus worst late factorial replay",
        "unmeasured_singletons": "Architecture and validation each receive 600 s before 2x inflation; secondary Bible receives 1.01x full primary Bible elapsed",
        "limitation": "Finite representative profiles plus conservative headroom; runtime is a projection, not an upper bound for every possible sample.",
    }
    report["jobs"] = jobs
    report["projected_core_hours_by_experiment"] = {k: v/3600 for k, v in totals.items()}
    report["peak_measured_pilot_rss_gib"] = max(p["peak_rss_kib"] for p in pilots.values()) / 1024**2
    report["logical_store_hash_bytes"] = 3902 * 2802407082
    projections = []
    for count in (4, 8, 16):
        controllers = []
        for index in range(count):
            assigned = jobs[index::count]
            load = sum(j["budget_seconds"] for j in assigned)
            longest = max(j["budget_seconds"] for j in assigned)
            slots = [0.0] * 72
            for job in assigned:
                earliest = heapq.heappop(slots)
                heapq.heappush(slots, earliest + job["budget_seconds"])
            # A conservative identical-machine list-scheduling bound, with
            # explicit tail plus one hour reserved for controller/I/O overhead.
            bound = load/72 + longest + 3600
            controllers.append({"worker_index": index, "job_count": len(assigned),
                                "budget_core_hours": load/3600, "longest_job_hours": longest/3600,
                                "simulated_queue_hours_with_reserve": (max(slots)+3600)/3600,
                                "conservative_bound_hours": bound/3600})
        maximum = max(c["conservative_bound_hours"] for c in controllers)
        projections.append({"nodes": count, "controllers": controllers,
                            "conservative_makespan_hours": maximum, "eligible": maximum < 24})
    report["projections"] = projections
    report["selected_nodes"] = next((p["nodes"] for p in projections if p["eligible"]), None)
    report["status"] = "passed" if report["selected_nodes"] else "pending"
    report["reason"] = "Verified scheduling evidence fits the conservative resource budget" if report["selected_nodes"] else "No assessed node count fits the 24-hour budget; revise scheduling before launch"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "repo", "pilot-root", "power-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = {"schema_version": 1, "purpose": "scheduling_assessment", "production_admission": False,
              "status": "pending", "selected_nodes": None, "input_files": {},
              "script_sha256": sha256(__file__), "assessed_utc": utc_now()}
    try:
        assess(args, report)
    except FileNotFoundError as error:
        report.update(status="pending", reason=f"Required completed input missing: {error}")
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "assessment.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "selected_nodes", "plan_sha256", "reason")}))
    raise SystemExit(0 if report["status"] == "passed" else 2 if report["status"] == "pending" else 1)

if __name__ == "__main__":
    main()
