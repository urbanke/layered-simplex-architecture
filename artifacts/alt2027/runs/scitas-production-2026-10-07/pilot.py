"""Production-grid scheduling pilot; exact protocol sample IDs, validation only."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from lsa.alt.artifacts import environment, read_json, sha256, source_identity, utc_now, write_json
from lsa.alt.cli import run_one

COMMIT = "f0a5ebd0a57d46caeb9bf45407c062f37b8d64bc"
BASE = Path("/scratch/urbanke/lsa-alt2027")
REPO = BASE / "checkouts" / COMMIT
ROOT = BASE / "runs/throughput-f0a5ebd-20261007-001"
CASES = [
    ("primary-zipf1p5-20", "benchmark_primary", {"targets": ["zipf_1.5"], "trials": 20, "trial_start": 0}),
    ("primary-uniform-20", "benchmark_primary", {"targets": ["uniform"], "trials": 20, "trial_start": 0}),
    ("spectrum-d1e6-alpha1p5-20", "spectrum", {"cell_ids": [27], "trials": 20, "trial_start": 0}),
    ("depth-alpha2-20", "depth_scaling", {"cell_ids": [0], "trials": 20, "trial_start": 0}),
    ("factorial-d1e5-n1e4-alpha1p5-100", "factorial", {"cell_ids": [96], "trials": 100, "trial_start": 0}),
    ("bible-full", "bible", {}),
]

def main():
    case_id = int(sys.argv[1])
    name, experiment, selectors = CASES[case_id]
    nodes = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).splitlines()
    assert socket.gethostname().split(".")[0] in {n.split(".")[0] for n in nodes}
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == 1
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == COMMIT
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    root = ROOT / name
    root.mkdir(parents=True, exist_ok=False)
    protocol = read_json(REPO / "experiments/alt2027/protocol.json")
    config = copy.deepcopy(protocol["experiments"][experiment])
    config.update(selectors)
    if experiment.startswith("benchmark_"):
        config["sampling_n_values"] = protocol["experiments"][experiment]["n_values"]
    protocol["experiments"] = {experiment: config}
    protocol_path = root / "protocol.json"
    write_json(protocol_path, protocol)
    engine = BASE / "setup/jed-engine.json"
    args = SimpleNamespace(repo=REPO, purpose="validation", protocol=protocol_path,
                           engine_config=engine, power_settings=None, calibration=None, batch_size=20)
    write_json(root / "pilot-request.json", {
        "purpose": "validation", "production_admission": False, "case": name,
        "experiment": experiment, "selectors": selectors, "batch_size": 20,
        "source": source_identity(REPO), "driver_sha256": sha256(__file__),
        "engine_sha256": sha256(engine), "protocol_sha256": sha256(protocol_path),
        "parent_protocol_sha256": sha256(REPO / "experiments/alt2027/protocol.json"),
        "environment": environment(), "job_id": os.environ["SLURM_JOB_ID"],
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"), "host": socket.gethostname(),
        "started_utc": utc_now(), "cpu_count": 1,
        "sampling_note": "Original global trial IDs and complete original sampling_n_values; no scientific protocol revision.",
    })
    start = time.perf_counter()
    status, error = "passed", None
    try:
        run_one(experiment, protocol, args, root / "run")
    except Exception:
        status, error = "failed", traceback.format_exc()
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result = {"case": name, "status": status, "error": error,
                  "wall_seconds": time.perf_counter() - start, "user_cpu_seconds": usage.ru_utime,
                  "system_cpu_seconds": usage.ru_stime, "peak_rss_kib": usage.ru_maxrss,
                  "finished_utc": utc_now(), "production_admission": False}
        write_json(root / "pilot-result.json", result)
        print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
