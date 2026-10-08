"""Batched, independently scanned sequence/conditional-evidence chain checks."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .artifacts import canonical_hash, write_json
from .benchmark import jsonable
from .bible import load_corpus


def check_sequence(evaluator, d, ids, depths, *, chunk_size=100):
    """Compare family-based predictions with separate batch-profile integrals.

    Each chunk reuses moment rows but evaluates the ordinary profile integral
    independently from the next-count family construction. No renormalization
    or probability clipping is applied. All depths have equal prior weights.
    """
    ids = np.asarray(ids)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= d):
        raise ValueError("sequence contains invalid labels")
    if not np.issubdtype(ids.dtype, np.integer) or chunk_size < 1:
        raise ValueError("integer labels and positive chunk size required")
    counts = Counter()
    prefixes = {0: ()}
    transitions = {}
    for t, symbol in enumerate(ids):
        c = counts[int(symbol)]
        transitions[t] = (prefixes[t], c)
        counts[int(symbol)] += 1
        prefixes[t + 1] = tuple(sorted(counts.values(), reverse=True))
    rows = []
    cumulative = 0.0
    log_prior_size = math.log(len(depths))
    for start in range(0, len(ids), chunk_size):
        stop = min(start + chunk_size, len(ids))
        ordinary = evaluator.evaluate_profiles_at_depths(
            d, {t: prefixes[t] for t in range(start, stop + 1)}, depths
        )
        family = evaluator.transition_log_evidence(
            d, {t: transitions[t] for t in range(start, stop)}, depths=depths
        )
        for t in range(start, stop):
            parent, child = family[t]
            log_parent = np.asarray(parent.log_evidence)
            log_child = np.asarray(child.log_evidence)
            component_q = np.exp(log_child - log_parent)
            posterior = np.exp(log_parent - logsumexp(log_parent))
            predicted = float(np.dot(posterior, component_q))
            if not math.isfinite(predicted) or not 0 < predicted <= 1:
                raise ArithmeticError("invalid family next-token probability")
            ratio = float(
                np.exp(
                    logsumexp(ordinary[t + 1].log_evidence)
                    - logsumexp(ordinary[t].log_evidence)
                )
            )
            cumulative -= math.log2(predicted)
            batch = float(
                -(logsumexp(ordinary[t + 1].log_evidence) - log_prior_size)
                / math.log(2)
            )
            rows.append(
                {
                    "t": t + 1,
                    "symbol": int(ids[t]),
                    "previous_count": transitions[t][1],
                    "probability": predicted,
                    "independent_evidence_ratio": ratio,
                    "probability_error": abs(predicted - ratio),
                    "cumulative_bits": cumulative,
                    "batch_bits": batch,
                    "chain_error_bits": abs(cumulative - batch),
                    "family_parent_log_evidence_nats": log_parent.tolist(),
                    "family_child_log_evidence_nats": log_child.tolist(),
                    "ordinary_parent_log_evidence_nats": ordinary[
                        t
                    ].log_evidence.tolist(),
                    "ordinary_child_log_evidence_nats": ordinary[
                        t + 1
                    ].log_evidence.tolist(),
                }
            )
    return rows


def run_chain_validation(config, output_dir, *, evaluator, repo):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    write_json(root / "engine.json", evaluator.configuration)
    tokens, corpus = load_corpus(config, repo)
    if config["steps"] > len(tokens):
        raise ValueError("declared prefix exceeds corpus")
    vocabulary = {}
    ids = np.asarray(
        [vocabulary.setdefault(t, len(vocabulary)) for t in tokens[: config["steps"]]],
        dtype=np.int64,
    )
    write_json(root / "dictionary.json", vocabulary)
    write_json(root / "corpus.json", corpus)
    np.savez_compressed(root / "token-ids.npz", ids=ids)
    start = time.monotonic()
    rows = check_sequence(
        evaluator, config["d"], ids, config["depths"], chunk_size=config["chunk_size"]
    )
    write_json(root / "transitions.json", jsonable(rows))
    max_prob = max(r["probability_error"] for r in rows)
    max_chain = max(r["chain_error_bits"] for r in rows)
    summary = {
        "status": "passed"
        if max_prob <= config["probability_tolerance"]
        and max_chain <= config["chain_tolerance_bits"]
        else "failed",
        "steps": len(rows),
        "depths": config["depths"],
        "d": config["d"],
        "max_probability_error": max_prob,
        "max_chain_error_bits": max_chain,
        "final_chain_error_bits": rows[-1]["chain_error_bits"],
        "seconds": time.monotonic() - start,
        "probability_tolerance": config["probability_tolerance"],
        "chain_tolerance_bits": config["chain_tolerance_bits"],
        "configuration_sha256": evaluator.configuration_sha256,
        "config_sha256": canonical_hash(config),
    }
    write_json(root / "summary.json", summary)
    return summary
