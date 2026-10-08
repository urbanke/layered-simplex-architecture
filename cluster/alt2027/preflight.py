"""Read-only complete store identity and fixed tiny numerical checks."""

import json
import math
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


def reference_preflight():
    import numpy as np

    from lsa.alt.depth import DepthEvaluator

    with DepthEvaluator(mode="reference") as evaluator:
        evidence = evaluator.evidence_at_depths(2, (2, 1), (0, 1, 2))
        prediction = evaluator.prediction_by_count(2, (2, 1), depths=(0, 1, 2))
    checks = [
        {
            "name": "L0_log_evidence",
            "error_nats": abs(evidence.log_evidence[0] + 3 * math.log(2)),
            "tolerance": 1e-12,
        },
        {
            "name": "L1_log_evidence",
            "error_nats": abs(evidence.log_evidence[1] - math.log(1 / 12)),
            "tolerance": 1e-12,
        },
        {
            "name": "L0_L1_L2_predictive_mass",
            "absolute_error": float(
                np.max(
                    np.abs(
                        prediction.component_probabilities
                        @ np.asarray(prediction.multiplicities)
                        - 1
                    )
                )
            ),
            "tolerance": 2e-8,
        },
    ]
    for check in checks:
        error = check.get("error_nats", check.get("absolute_error"))
        check["status"] = (
            "passed"
            if math.isfinite(error) and error <= check["tolerance"]
            else "failed"
        )
    return {
        "status": "passed"
        if all(c["status"] == "passed" for c in checks)
        else "failed",
        "purpose": "validation",
        "checks": checks,
        "d": 2,
        "partition": [2, 1],
        "depths": [0, 1, 2],
        "production_certified": False,
        "log_evidence": evidence.log_evidence.tolist(),
        "component_probabilities": prediction.component_probabilities.tolist(),
        "scope": "Tiny independent reference checks only; no production admission.",
    }


def store_preflight(repo, engine, reference):
    """Verify all pinned files on the allocated node, then compare tiny cases."""
    import numpy as np
    import scipy

    from lsa.alt.artifacts import canonical_hash, sha256
    from lsa.alt.depth import DepthEvaluator, StoreConfig

    candidate_path = Path(repo) / "experiments/alt2027/store-candidate.json"
    candidate = json.loads(candidate_path.read_text())
    expected = candidate["files_sha256"]
    if len(expected) != 106:
        raise ValueError("this preflight requires the pinned 106-file store")
    if (
        engine.get("mode") != "store"
        or engine.get("store", {}).get("files_sha256") != expected
    ):
        raise ValueError("engine must identify exactly the committed 106-file store")
    store = StoreConfig(**engine["store"])
    root = Path(store.path)
    if not root.is_absolute():
        raise ValueError("store path must be absolute")
    started = time.perf_counter()
    # The adapter verifies every complete file digest, refuses symlinks and
    # missing/unpinned level files, and disables every store write/build path.
    with DepthEvaluator(
        mode="store",
        store=store,
        prediction_tolerance=engine.get("prediction_tolerance", 1e-3),
    ) as evaluator:
        checked_seconds = time.perf_counter() - started
        total_bytes = sum((root / name).stat().st_size for name in expected)
        if total_bytes != candidate["bytes"]:
            raise ValueError("verified store byte count differs from pinned candidate")
        evidence = evaluator.evidence_at_depths(2, (2, 1), (0, 1, 2))
        prediction = evaluator.prediction_by_count(2, (2, 1), depths=(0, 1, 2))
        configuration = evaluator.configuration
        configuration_sha256 = evaluator.configuration_sha256
    checks = [
        {
            "name": "store_reference_L0_L1_L2_log_evidence",
            "error_nats": float(
                np.max(
                    np.abs(
                        evidence.log_evidence - np.asarray(reference["log_evidence"])
                    )
                )
            ),
            "tolerance": 1e-5,
        },
        {
            "name": "store_reference_L0_L1_L2_prediction",
            "absolute_error": float(
                np.max(
                    np.abs(
                        prediction.component_probabilities
                        - np.asarray(reference["component_probabilities"])
                    )
                )
            ),
            "tolerance": 1e-5,
        },
        {
            "name": "store_L0_L1_L2_predictive_mass",
            "absolute_error": float(
                np.max(
                    np.abs(
                        prediction.component_probabilities
                        @ np.asarray(prediction.multiplicities)
                        - 1
                    )
                )
            ),
            "tolerance": 1e-5,
        },
    ]
    for check in checks:
        error = check.get("error_nats", check.get("absolute_error"))
        check["status"] = (
            "passed"
            if math.isfinite(error) and error <= check["tolerance"]
            else "failed"
        )
    library_builds = {}
    for name, library in (("numpy", np), ("scipy", scipy)):
        output = StringIO()
        with redirect_stdout(output):
            library.show_config()
        library_builds[name] = output.getvalue()
    return {
        "status": "passed"
        if reference["status"] == "passed"
        and all(check["status"] == "passed" for check in checks)
        else "failed",
        "purpose": "validation",
        "production_certified": False,
        "reference": reference,
        "store": {
            "path": str(root.resolve()),
            "store_id": candidate["store_id"],
            "candidate_specification_sha256": sha256(candidate_path),
            "files_verified": len(expected),
            "bytes_verified": total_bytes,
            "files_sha256": expected,
            "content_identity_sha256": canonical_hash(expected),
            "verification_seconds": checked_seconds,
        },
        "engine_configuration": configuration,
        "engine_configuration_sha256": configuration_sha256,
        "numerical_library_builds": library_builds,
        "checks": checks,
        "scope": "All 106 pinned store files verified read-only; tiny d=2, N=3, depths 0-2 checks only. No plan jobs were run. High-depth and full-domain platform calibration and production admission remain separate.",
    }
