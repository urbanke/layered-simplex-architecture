"""Production calibration must bind the complete engine, not only depth."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsa.alt import cli
from lsa.alt.artifacts import canonical_hash, sha256
from lsa.alt.powers import PowerSettings


class AcceptedCalibration(Exception):
    """Stop before any run, output artifact, or numerical evaluation."""


def test_changed_power_settings_cannot_reuse_passed_calibration(tmp_path, monkeypatch):
    depth_configuration = {"backend": "fake depth for gate-only test"}
    monkeypatch.setattr(cli, "make_depth", lambda args: SimpleNamespace(
        configuration=depth_configuration, close=lambda: None))
    monkeypatch.setattr(cli, "source_identity", lambda repo: {"tree_sha256": "frozen-source"})

    def reached_run(*args, **kwargs):
        raise AcceptedCalibration()

    monkeypatch.setattr(cli, "Run", reached_run)
    protocol = {"status": "frozen", "experiments": {"benchmark_powers": {}}}
    engine = {
        "depth": depth_configuration,
        "power": {"settings": asdict(PowerSettings()),
                  "implementation_sha256": sha256(Path(cli.__file__).with_name("powers.py"))},
    }
    calibration = {
        "status": "passed", "protocol_sha256": canonical_hash(protocol),
        "covered_experiments": ["benchmark_powers"], "required_checks_complete": True,
        "source_tree_sha256": "frozen-source",
        "engine_sha256_by_experiment": {"benchmark_powers": canonical_hash(engine)},
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    args = SimpleNamespace(repo=tmp_path, purpose="production", calibration=calibration_path,
                           power_settings=None, protocol=tmp_path / "protocol.json", engine_config=None)
    # Positive control: a fully matching certificate reaches the run boundary.
    with pytest.raises(AcceptedCalibration):
        cli.run_one("benchmark_powers", protocol, args, tmp_path / "unused")

    power_path = tmp_path / "changed-power-settings.json"
    power_path.write_text(json.dumps({"outer_relative_tolerance": 4e-8}))
    args.power_settings = power_path
    with pytest.raises(ValueError, match="calibration does not cover"):
        cli.run_one("benchmark_powers", protocol, args, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()

    # An unchanged engine with a different scientific source also fails.
    args.power_settings = None
    monkeypatch.setattr(cli, "source_identity", lambda repo: {"tree_sha256": "changed-source"})
    with pytest.raises(ValueError, match="calibration does not cover"):
        cli.run_one("benchmark_powers", protocol, args, tmp_path / "unused")
