import json
from pathlib import Path

import pytest
from experiments.common.session import Session
from experiments.model_diagnostics import batch
from experiments.model_diagnostics import trial as trial_module
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


@pytest.mark.parametrize("prior_elapsed", [0, 75])
def test_interrupted_fit_runtime_counts_against_resume_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_elapsed: int
) -> None:
    current = session(tmp_path)
    directory = tmp_path / "batch"
    directory.mkdir()
    state_path = directory / "progress.json"
    if prior_elapsed:
        state_path.write_text(json.dumps({"elapsed_seconds": prior_elapsed}))
    elapsed = 0.0
    fit_calls = 0

    def interrupt(*args: object, **kwargs: object) -> None:
        nonlocal elapsed, fit_calls
        fit_calls += 1
        elapsed += 250
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "monotonic", lambda: elapsed)
    monkeypatch.setattr(trial_module, "fit_research", interrupt)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(directory, current, Path.cwd(), budget_seconds=400, progress=lambda _: None)

    state = json.loads(state_path.read_text())
    assert state["status"] == "interrupted"
    assert state["elapsed_seconds"] == prior_elapsed + 250
    assert not (directory / ".batch.lock").exists()
    assert not list((directory / "trials").iterdir())

    resumed = batch.run_batch(
        directory, current, Path.cwd(), budget_seconds=400, progress=lambda _: None
    )
    assert resumed["status"] == "budget_exhausted"
    assert resumed["elapsed_seconds"] == prior_elapsed + 250
    assert fit_calls == 1
