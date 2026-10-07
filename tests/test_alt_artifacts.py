import subprocess

import pytest

from lsa.alt.artifacts import Run, verify_run


def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "source.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=root,
        check=True,
    )
    return root


def test_tampering_and_added_outputs_are_detected(tmp_path):
    repo = repository(tmp_path)
    out = tmp_path / "run"
    with Run(out, repo=repo, protocol={"a": 1}, experiment="example", purpose="smoke"):
        (out / "values.json").write_text("[1,2]")
    verify_run(out)
    (out / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="inventory"):
        verify_run(out)
    (out / "extra").unlink()
    (out / "values.json").write_text("[3,4]")
    with pytest.raises(ValueError, match="checksum"):
        verify_run(out)


def test_failed_run_is_preserved_and_cannot_be_reused(tmp_path):
    repo = repository(tmp_path)
    out = tmp_path / "run"
    with (
        pytest.raises(RuntimeError),
        Run(out, repo=repo, protocol={}, experiment="x", purpose="validation"),
    ):
        (out / "offending-sample").write_text("sample")
        raise RuntimeError("quadrature failed")
    assert verify_run(out, require_complete=False)["status"] == "failed"
    with pytest.raises(ValueError, match="failed"):
        verify_run(out)
    with pytest.raises(FileExistsError):
        Run(out, repo=repo, protocol={}, experiment="x", purpose="validation")


def test_production_requires_freeze_and_clean_source(tmp_path):
    repo = repository(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        Run(
            tmp_path / "one",
            repo=repo,
            protocol={},
            experiment="x",
            purpose="production",
        )


def test_source_changes_during_run_are_recorded_as_failure(tmp_path):
    repo = repository(tmp_path)
    out = tmp_path / "run"
    with (
        pytest.raises(RuntimeError, match="source files changed"),
        Run(out, repo=repo, protocol={}, experiment="x", purpose="validation"),
    ):
        (repo / "source.py").write_text("value = 2\n")
    assert verify_run(out, require_complete=False)["status"] == "failed"
    (repo / "source.py").write_text("changed")
    with pytest.raises(ValueError, match="clean"):
        Run(
            tmp_path / "two",
            repo=repo,
            protocol={"status": "frozen"},
            experiment="x",
            purpose="production",
        )
