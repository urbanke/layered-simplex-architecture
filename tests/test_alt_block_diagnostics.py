"""Fixed common-trial blocks expose tails and preserve symmetric comparisons."""

import copy

import pytest

from lsa.alt.benchmark import aggregate_records, loss_record, paired_difference
from lsa.alt.block_diagnostics import benchmark_block_diagnostics


def config(trials=100, *, trial_start=0):
    return {
        "d": 8,
        "n_values": [4, 8],
        "trials": trials,
        "trial_start": trial_start,
        "seed": 5,
        "sample_set_id": "block-fixture",
        "targets": ["uniform", "step"],
        "methods": ["add_one", "kt"],
        "depths": [],
        "fixed_depth": 1,
        "powers": [],
        "normalize_numerical": True,
        "numerical_normalization_tolerance": 1e-3,
        "dirichlet_exponents": [-2, 0, 2],
        "dirichlet_target_policy": "redraw_per_trial_shared_across_n",
        "sample_size_policy": "independent_multinomial_per_n",
    }


def records(settings, deltas=None):
    deltas = deltas if deltas is not None else [0] * settings["trials"]
    result = []
    for target_index, target in enumerate(settings["targets"]):
        for offset, difference in enumerate(deltas):
            for n in settings["n_values"]:
                baseline = 100 + 10 * target_index + n
                left, right = loss_record(baseline + difference), loss_record(baseline)
                result.append(
                    {
                        "target_id": target,
                        "trial": settings["trial_start"] + offset,
                        "n": n,
                        "losses": {"add_one": left, "kt": right},
                        "paired_differences": {
                            "add_one_minus_kt": paired_difference(left, right)
                        },
                        "posteriors": {},
                        "derived": {},
                    }
                )
    return result


def pair(summary, *, target="uniform", n="4"):
    return summary[target][n]["paired_differences"]["add_one_minus_kt"]


def test_rare_outlier_is_visible_in_fixed_block_without_changing_pooled_summary():
    settings = config()
    saved = records(settings, [0] * 99 + [100])
    original = copy.deepcopy(saved)
    pooled_before = aggregate_records(saved, settings)
    diagnostic = benchmark_block_diagnostics(saved, settings)
    assert diagnostic["requested_blocks"] == diagnostic["block_count"] == 10
    assert diagnostic["diagnostic_only"] is True
    assert [
        (block["trial_start"], block["trial_stop_exclusive"])
        for block in diagnostic["blocks"]
    ] == [(i, i + 10) for i in range(0, 100, 10)]
    assert pair(diagnostic["pooled_targets"])["mean_bits"] == 1
    assert pair(diagnostic["pooled_targets"])["se_bits"] == pytest.approx(1)
    assert [pair(block["targets"])["mean_bits"] for block in diagnostic["blocks"]] == [
        0
    ] * 9 + [10]
    assert pair(diagnostic["blocks"][-1]["targets"])["se_bits"] == pytest.approx(10)
    assert aggregate_records(saved, settings) == pooled_before
    assert saved == original


def test_favorable_and_adverse_differences_use_identical_blocks_and_uncertainty():
    settings = config()
    deltas = [0] * 89 + [2] * 10 + [100]
    adverse = benchmark_block_diagnostics(records(settings, deltas), settings)
    favorable = benchmark_block_diagnostics(
        records(settings, [-value for value in deltas]), settings
    )
    for left, right in zip(
        [adverse["pooled_targets"], *(b["targets"] for b in adverse["blocks"])],
        [favorable["pooled_targets"], *(b["targets"] for b in favorable["blocks"])],
        strict=True,
    ):
        assert pair(left)["mean_bits"] == -pair(right)["mean_bits"]
        assert pair(left)["se_bits"] == pair(right)["se_bits"]
        assert pair(left)["counts"] == pair(right)["counts"]
    assert [
        (b["trial_start"], b["trial_stop_exclusive"]) for b in adverse["blocks"]
    ] == [(b["trial_start"], b["trial_stop_exclusive"]) for b in favorable["blocks"]]


def test_nonfinite_trials_are_counted_in_every_method_and_pair():
    settings = config(20)
    saved = records(settings)
    for row in saved:
        if row["trial"] == 1:
            row["losses"]["add_one"] = loss_record(float("inf"))
        if row["trial"] == 3:
            row["losses"]["kt"] = loss_record(None, reason="undefined fixture")
        row["paired_differences"]["add_one_minus_kt"] = paired_difference(
            row["losses"]["add_one"], row["losses"]["kt"]
        )
    diagnostic = benchmark_block_diagnostics(saved, settings)
    for target in settings["targets"]:
        for n in map(str, settings["n_values"]):
            pooled = diagnostic["pooled_targets"][target][n]
            assert pooled["methods"]["add_one"]["counts"] == {
                "finite": 19,
                "infinite": 1,
                "undefined": 0,
            }
            assert pooled["methods"]["add_one"]["status"] == "infinite"
            assert pooled["methods"]["kt"]["counts"] == {
                "finite": 19,
                "infinite": 0,
                "undefined": 1,
            }
            assert pooled["methods"]["kt"]["status"] == "undefined"
            assert pair(diagnostic["pooled_targets"], target=target, n=n)["counts"] == {
                "finite": 18,
                "infinite": 0,
                "undefined": 2,
            }
            assert (
                pair(diagnostic["pooled_targets"], target=target, n=n)["mean_bits"]
                is None
            )
            assert (
                pair(diagnostic["blocks"][0]["targets"], target=target, n=n)["counts"][
                    "undefined"
                ]
                == 1
            )
            assert (
                pair(diagnostic["blocks"][1]["targets"], target=target, n=n)["counts"][
                    "undefined"
                ]
                == 1
            )


def test_global_ids_and_all_cells_survive_order_changes_and_small_smoke_blocks():
    settings = config(3, trial_start=17)
    saved = records(settings, [0, 1, 2])
    diagnostic = benchmark_block_diagnostics(saved[::-1], settings)
    assert diagnostic == benchmark_block_diagnostics(saved, settings)
    assert diagnostic["requested_blocks"] == 10 and diagnostic["block_count"] == 3
    assert [block["trial_start"] for block in diagnostic["blocks"]] == [17, 18, 19]
    for block in diagnostic["blocks"]:
        assert block["trials"] == 1
        assert set(block["targets"]) == set(settings["targets"])
        for target_index, target in enumerate(settings["targets"]):
            assert set(block["targets"][target]) == {"4", "8"}
            for n in settings["n_values"]:
                methods = block["targets"][target][str(n)]["methods"]
                assert methods["kt"]["mean_bits"] == 100 + 10 * target_index + n
                assert methods["kt"]["se_bits"] is None
                assert pair(block["targets"], target=target, n=str(n))["trials"] == 1


def test_nondivisible_trial_count_uses_balanced_fixed_blocks():
    settings = config(23, trial_start=9)
    diagnostic = benchmark_block_diagnostics(records(settings), settings)
    assert [block["trials"] for block in diagnostic["blocks"]] == [3] * 3 + [2] * 7
    assert [
        trial
        for block in diagnostic["blocks"]
        for trial in range(block["trial_start"], block["trial_stop_exclusive"])
    ] == list(range(9, 32))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_trial",
        "duplicate_trial",
        "wrong_target",
        "wrong_n",
        "missing_method",
        "missing_pair",
    ],
)
def test_incomplete_or_inconsistent_coverage_fails(mutation):
    settings = config(3)
    saved = records(settings)
    if mutation == "missing_trial":
        saved.pop()
    elif mutation == "duplicate_trial":
        saved.append(copy.deepcopy(saved[0]))
    elif mutation == "wrong_target":
        saved[0]["target_id"] = "geometric"
    elif mutation == "wrong_n":
        saved[0]["n"] = 99
    elif mutation == "missing_method":
        del saved[0]["losses"]["kt"]
    else:
        # Omitting the pair from every row must still fail, rather than quietly
        # hiding a comparison throughout the pooled and block summaries.
        for row in saved:
            row["paired_differences"] = {}
    with pytest.raises(ValueError, match="identities|every declared"):
        benchmark_block_diagnostics(saved, settings)


@pytest.mark.parametrize("blocks", [0, -1, True, 2.5])
def test_invalid_requested_block_count_fails(blocks):
    settings = config(3)
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_block_diagnostics(records(settings), settings, block_count=blocks)
