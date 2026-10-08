"""Fixed trial-block diagnostics for saved common-sample benchmarks.

Blocks depend only on the complete experiment's global trial IDs. They never
select trials, change pooled estimates, or trigger an outcome-dependent stop.
"""

from __future__ import annotations

from .benchmark import aggregate_records, trial_ids, validate_config


def _loss_summaries(summary):
    return {
        target: {
            n: {key: cell[key] for key in ("methods", "paired_differences")}
            for n, cell in ns.items()
        }
        for target, ns in summary["targets"].items()
    }


def benchmark_block_diagnostics(records, config, *, block_count=10):
    """Summarize every method and stored pair in a fixed partition of trials.

    Global trial IDs form contiguous, balanced blocks. When the trial count is
    not divisible by the requested block count, the earliest blocks have one
    extra trial. Small smoke runs use at most one block per trial. The existing
    benchmark aggregator supplies exactly the same nonfinite and SE rules for
    the pooled records and every block.
    """
    if (
        isinstance(block_count, bool)
        or not isinstance(block_count, int)
        or block_count < 1
    ):
        raise ValueError("block_count must be a positive integer")
    config = validate_config(config)
    methods = config["methods"]
    expected_pairs = {
        f"{left}_minus_{right}"
        for index, left in enumerate(methods)
        for right in methods[index + 1 :]
    }
    for row in records:
        if set(row["losses"]) != set(methods):
            raise ValueError("block diagnostics require every declared method")
        if not expected_pairs <= row["paired_differences"].keys():
            raise ValueError("block diagnostics require every declared method pair")
    # This checks complete target/n/global-trial coverage, including duplicates,
    # before any block is reported. Large raw diagnostics need not be supplied.
    pooled = aggregate_records(records, config)
    effective = min(config["trials"], block_count)
    size, extra = divmod(config["trials"], effective)
    start = trial_ids(config).start
    blocks = []
    for index in range(effective):
        length = size + (index < extra)
        stop = start + length
        selected = [row for row in records if start <= row["trial"] < stop]
        block = aggregate_records(
            selected, {**config, "trial_start": start, "trials": length}
        )
        blocks.append(
            {
                "block_id": index,
                "trial_start": start,
                "trial_stop_exclusive": stop,
                "trials": length,
                "targets": _loss_summaries(block),
            }
        )
        start = stop
    return {
        "schema_version": 1,
        "kind": "benchmark_trial_block_diagnostics",
        "diagnostic_only": True,
        "config": config,
        "units": pooled["units"],
        "uncertainty": pooled["uncertainty"],
        "nonfinite_policy": pooled["nonfinite_policy"],
        "requested_blocks": block_count,
        "block_count": effective,
        "partition_rule": "contiguous global trial IDs; balanced sizes; earlier blocks receive any remainder",
        "pooled_targets": _loss_summaries(pooled),
        "blocks": blocks,
    }
