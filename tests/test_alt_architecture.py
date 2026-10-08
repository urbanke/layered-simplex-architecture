"""The displayed bars come from the saved draw, including the hidden layers."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from lsa.alt.architecture import run_architecture

TEMPLATE = (Path(__file__).parents[1] / "manuscript/snapshots/2026-10-07-alt-start"
            / "figures/alt/construction_tikz.tex")
CONFIG = {"d": 24, "depth": 4, "seed": 7, "artifact": "fig:lsa"}


def test_saved_layers_product_and_visible_bars_match(tmp_path):
    root = tmp_path / "draw"
    result = run_architecture(CONFIG, root, template_path=TEMPLATE)
    with np.load(root / "draws.npz", allow_pickle=False) as data:
        layers, theta = data["layers"], data["theta"]
        assert layers.shape == (4, 24)
        np.testing.assert_allclose(layers.sum(axis=1), 1, atol=2e-15)
        direct = layers.prod(axis=0)
        direct /= direct.sum()
        np.testing.assert_allclose(theta, direct, atol=2e-15)
        # Exact RNG recipe is part of the declared protocol.
        np.testing.assert_array_equal(layers, np.random.default_rng(7).dirichlet(np.ones(24), size=4))
    text = (root / "construction_tikz.tex").read_text()
    arrays = re.findall(r"\\foreach \\i/\\h in \{([^}]+)\}", text)
    expected_rows = np.vstack([layers[0], layers[1], layers[-1], theta])
    for array, row in zip(arrays, expected_rows, strict=True):
        values = [float(item.split("/")[1]) for item in array.split(",")]
        assert values == [float(f"{28 * x:.2f}") for x in row]
    assert result["winner_coordinate_one_based"] == int(np.argmax(theta)) + 1 == 8
    assert "(8.31,8.5) -- (8.31,53)" in text
    assert "(8.31,-18) -- (8.31,-3.5)" in text
    record = json.loads((root / "draw.json").read_text())
    assert record["winner_mass"] == theta[7]


def test_only_template_data_changes_and_run_is_immutable(tmp_path):
    one, two = tmp_path / "a", tmp_path / "b"
    run_architecture(CONFIG, one, template_path=TEMPLATE)
    run_architecture(CONFIG, two, template_path=TEMPLATE)
    assert (one / "construction_tikz.tex").read_bytes() == (two / "construction_tikz.tex").read_bytes()
    assert (one / "draw.json").read_bytes() == (two / "draw.json").read_bytes()

    def without_data(s):
        s = re.sub(r"(\\foreach \\i/\\h in \{)[^}]+", r"\1DATA", s)
        return re.sub(r"^(.*\\draw\[black!35,densely dotted)[^\n]+$", r"\1GUIDE", s,
                      flags=re.MULTILINE)

    assert without_data(TEMPLATE.read_text()) == without_data((one / "construction_tikz.tex").read_text())
    with pytest.raises(FileExistsError):
        run_architecture(CONFIG, one, template_path=TEMPLATE)


def test_fixed_geometry_rejects_an_incompatible_alphabet(tmp_path):
    with pytest.raises(ValueError, match="d=24"):
        run_architecture({**CONFIG, "d": 12}, tmp_path / "bad", template_path=TEMPLATE)
