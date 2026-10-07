"""Reproducible prior draw rendered in the manuscript's existing TikZ form."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .artifacts import sha256, write_json

_BARS = re.compile(r"(\\foreach \\i/\\h in \{)([^}]+)(\})")
_GUIDES = re.compile(r"^.*\\draw\[black!35,densely dotted[^\n]+$", re.MULTILINE)
_HEIGHT_SCALE = 28.0  # Existing schematic: a unit of probability is 28 mm.


def run_architecture(config, output_dir, *, template_path):
    """Save every simplex layer and replace only data in the fixed schematic.

    The existing 24-coordinate geometry is preserved. Layers 1, 2, and L are
    displayed; the product uses *all* L layers, including the omitted rows.
    Full-precision draws are saved separately from two-decimal displayed bars.
    """
    d, depth, seed = config["d"], config["depth"], config["seed"]
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (d, depth, seed)):
        raise ValueError("d, depth, and seed must be integers")
    if d != 24 or depth < 3 or seed < 0:
        raise ValueError("the fixed schematic requires d=24, depth>=3, and seed>=0")
    template_path = Path(template_path)
    source = template_path.read_text()
    matches = list(_BARS.finditer(source))
    if len(matches) != 4 or any(len(m[2].split(",")) != d for m in matches):
        raise ValueError("expected exactly four 24-coordinate bar rows in the template")
    if len(_GUIDES.findall(source)) != 2:
        raise ValueError("expected two dotted winner-guide segments")

    rng = np.random.default_rng(seed)
    layers = rng.dirichlet(np.ones(d), size=depth)
    log_weights = np.log(layers).sum(axis=0)
    theta = np.exp(log_weights - logsumexp(log_weights))
    winner = int(np.argmax(theta))
    shown = np.vstack((layers[0], layers[1], layers[-1], theta))
    heights = shown * _HEIGHT_SCALE
    # Keep bars within the existing row gaps/braces instead of quietly changing
    # the figure's scale or geometry for an unsuitable new configuration.
    if np.any(np.max(heights, axis=1) > np.array([8.0, 13.0, 13.0, 12.0])):
        raise ValueError("draw exceeds the fixed schematic geometry; choose an explicit new design")
    arrays = iter(",".join(f"{i + 1}/{height:.2f}" for i, height in enumerate(row))
                  for row in heights)
    rendered = _BARS.sub(lambda m: m[1] + next(arrays) + m[3], source)
    guide_x = winner + 1 + 0.31  # Center of the original width-0.62 bar.
    rendered = _GUIDES.sub(
        lambda m: re.sub(r"(?<=\()[-+]?\d+(?:\.\d+)?(?=,)",
                         f"{guide_x:.2f}", m[0]), rendered,
    )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    np.savez_compressed(root / "draws.npz", layers=layers, theta=theta,
                        log_unnormalized_product=log_weights,
                        displayed_layer_indices=np.array([0, 1, depth - 1]))
    (root / "construction_tikz.tex").write_text(rendered)
    draw = {"layers": layers.tolist(), "theta": theta.tolist(),
            "displayed_layers_one_based": [1, 2, depth],
            "height_scale_mm": _HEIGHT_SCALE,
            "displayed_heights_mm": [[float(f"{x:.2f}") for x in row] for row in heights],
            "winner_coordinate_one_based": winner + 1, "winner_mass": float(theta[winner]),
            "winner_guide_x": guide_x}
    write_json(root / "draw.json", draw)
    summary = {"config": config, "generator": "numpy.default_rng.dirichlet",
               "template_sha256": sha256(template_path),
               "winner_coordinate_one_based": winner + 1,
               "winner_mass": float(theta[winner]),
               "outputs": {name: {"sha256": sha256(root / name)} for name in
                           ("config.json", "draws.npz", "draw.json", "construction_tikz.tex")}}
    write_json(root / "summary.json", summary)
    return summary
