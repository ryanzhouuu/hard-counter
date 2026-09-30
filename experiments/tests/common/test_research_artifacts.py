from pathlib import Path

import pytest
from experiments.common.artifacts import (
    file_record,
    finish_run,
    load_run,
    publication,
    resume_matches,
)
from experiments.tests.common.artifact_fixture import manifest


def test_atomic_publish_overwrite_and_exact_resume(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    with publication(destination) as stage:
        assert not destination.exists()
        (stage / "weights.bin").write_bytes(b"synthetic")
        run = manifest((file_record(stage / "weights.bin", stage),))
        finish_run(stage, run)
    assert load_run(destination) == run
    assert resume_matches(destination, run.config_sha256, (), tmp_path)
    assert not resume_matches(destination, "b" * 64, (), tmp_path)
    with pytest.raises(FileExistsError), publication(destination):
        pass
    (destination / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_run(destination)


def test_failure_cleans_owned_stage_and_preserves_visible_failed_status(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    with pytest.raises(RuntimeError), publication(destination) as stage:
        (stage / "partial").write_bytes(b"partial")
        raise RuntimeError("optimization failed")
    assert list(tmp_path.iterdir()) == []
    with publication(destination) as stage:
        (stage / "failure.json").write_text("{}")
        failed = manifest((file_record(stage / "failure.json", stage),)).model_copy(
            update={
                "status": "failed",
                "failures": ("optimization failed",),
            }
        )
        finish_run(stage, failed)
    assert load_run(destination).status == "failed"
    assert not resume_matches(destination, failed.config_sha256, (), tmp_path)
