from pathlib import Path

import pytest
from experiments.common.contracts import StudyConfig, Variant
from experiments.common.session import prepare_session
from experiments.model_diagnostics.artifacts import (
    TrialInputs,
    TrialSpec,
    bind_inputs,
    finish_trial,
    load_trial,
    trial_stage,
)


def inputs(tmp_path: Path) -> TrialInputs:
    session = prepare_session(
        StudyConfig(study_id="diagnostic", variants=(Variant(variant_id="baseline"),)),
        tmp_path / "source",
        synthetic=True,
        dataset=None,
        protocol=None,
        schema=None,
        cache=None,
        mechanics=None,
        row_cap=64,
    )
    return bind_inputs(session, TrialSpec(), Path.cwd())


def test_trial_publication_detects_corruption_and_changed_spec(tmp_path: Path) -> None:
    expected = inputs(tmp_path)
    destination = tmp_path / "trial"
    with trial_stage(destination) as stage:
        (stage / "report.json").write_text("{}")
        finish_trial(stage, expected, "complete", 1)
    assert load_trial(destination, expected).status == "complete"
    with pytest.raises(FileExistsError), trial_stage(destination):
        pass
    changed = expected.model_copy(update={"spec": TrialSpec(seed=1)})
    with pytest.raises(ValueError, match="changed"):
        load_trial(destination, changed)
    (destination / "report.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        load_trial(destination, expected)


def test_trial_retains_recorded_failure_and_cleans_interruption(tmp_path: Path) -> None:
    expected = inputs(tmp_path)
    destination = tmp_path / "failed"
    with trial_stage(destination) as stage:
        (stage / "failure.json").write_text("{}")
        finish_trial(stage, expected, "time_limited", 2, "deadline")
    assert load_trial(destination).failure == "deadline"
    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError), trial_stage(interrupted) as stage:
        (stage / "partial").write_text("incomplete")
        raise RuntimeError("interrupted")
    assert not interrupted.exists()
    assert not interrupted.with_name("interrupted.lock").exists()
    assert not list(tmp_path.glob(".diagnostic-*"))
