import json
from pathlib import Path

import pytest
from experiments.common.session import Session
from experiments.model_diagnostics import batch
from experiments.model_diagnostics.artifacts import (
    TrialSpec,
    bind_inputs,
    finish_trial,
    trial_stage,
)
from experiments.tests.model_diagnostics.test_trial import session


def test_batch_freezes_choices_resumes_and_verifies_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = session(tmp_path)
    count = 0

    def fake_run(directory: Path, current: Session, spec: TrialSpec, root: Path):
        nonlocal count
        expected = bind_inputs(current, spec, root)
        if directory.exists():
            return batch.load_trial(directory, expected)
        count += 1
        with trial_stage(directory) as stage:
            (stage / "report.json").write_text(json.dumps({"selection_loss": 0.69}))
            manifest = finish_trial(stage, expected, "complete", 0.1)
        return manifest

    monkeypatch.setattr(batch, "run_trial", fake_run)
    directory = tmp_path / "batch"
    first = batch.run_batch(directory, current, Path.cwd(), progress=lambda _: None)
    assert first["status"] == "complete"
    assert count == 26
    choices = (directory / "player-selection.json").read_bytes()
    resumed = batch.run_batch(directory, current, Path.cwd(), progress=lambda _: None)
    assert resumed["status"] == "complete"
    assert count == 26
    assert (directory / "player-selection.json").read_bytes() == choices
    (directory / "trials/reference/report.json").write_text('{"selection_loss":0}')
    corrupt = batch.run_batch(directory, current, Path.cwd(), progress=lambda _: None)
    assert corrupt["status"] == "incomplete"
    assert "hash mismatch" in str(corrupt["reason"])


def test_budget_never_starts_an_unfinishable_fit(tmp_path: Path) -> None:
    current = session(tmp_path)
    directory = tmp_path / "batch"
    result = batch.run_batch(directory, current, Path.cwd(), budget_seconds=0.01)
    assert result["status"] == "budget_exhausted"
    assert not (directory / "trials").exists()
    with pytest.raises(ValueError, match="frozen"):
        batch.run_batch(directory, current, Path.cwd(), budget_seconds=1000)
    assert not (directory / ".batch.lock").exists()
