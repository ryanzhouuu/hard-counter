from pathlib import Path

import numpy as np
import pytest
from experiments.common.contracts import OptimizerConfig, StudyConfig, Variant
from experiments.common.fit import fit_research, predict_logits
from experiments.common.session import Session, prepare_session
from experiments.matchup_features.model import ResearchModel
from experiments.model_diagnostics import trial
from experiments.model_diagnostics.artifacts import TrialSpec
from experiments.model_diagnostics.model import ModelSpec, recipe


def session(tmp_path: Path) -> Session:
    return prepare_session(
        StudyConfig(study_id="diagnostic", variants=(Variant(variant_id="baseline"),)),
        tmp_path / "source",
        synthetic=True,
        dataset=None,
        protocol=None,
        schema=None,
        cache=None,
        mechanics=None,
        row_cap=128,
    )


def test_shuffle_changes_only_training_labels_consistently(tmp_path: Path) -> None:
    original = session(tmp_path).access
    shuffled = trial.shuffled_training(original, 3)
    assert shuffled.read("refit", "fit") != original.read("refit", "fit")
    assert shuffled.read("refit", "fit") == trial.shuffled_training(original, 3).read(
        "refit", "fit"
    )
    refit = {r.event_key: r.label for r in shuffled.read("refit", "fit")}
    for role in ("selection_fit", "watch"):
        rows = shuffled.read(role, "fit")
        assert sum(r.label for r in rows) == sum(r.label for r in original.read(role, "fit"))
        assert all(refit[r.event_key] == r.label for r in rows)
    assert shuffled.read("calibration", "calibrate") == original.read("calibration", "calibrate")
    assert shuffled.read("development", "compare") == original.read("development", "compare")


def test_trial_reload_resume_and_failure_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = session(tmp_path)
    spec = TrialSpec(optimizer=OptimizerConfig(max_epochs=2))
    directory = tmp_path / "fit"
    first = trial.run_trial(directory, current, spec, Path.cwd())
    assert first.status in ("complete", "calibration_failed")
    assert trial.run_trial(directory, current, spec, Path.cwd()) == first
    model = trial.load_model(directory, current, Path.cwd())
    dev = current.access.read("development", "compare")
    assert np.isfinite(predict_logits(model, dev, np.empty((len(dev), 0)))).all()

    def timeout(*args: object, **kwargs: object) -> None:
        raise TimeoutError("budget exhausted")

    monkeypatch.setattr(trial, "fit_research", timeout)
    failed = trial.run_trial(tmp_path / "timeout", current, spec, Path.cwd())
    assert failed.status == "time_limited"
    assert failed.failure == "budget exhausted"


def test_default_fit_reproduces_existing_training_loop(tmp_path: Path) -> None:
    current = session(tmp_path)
    count = len(current.schema.identity_vocab)
    builder, factory = recipe(ModelSpec(), count)
    config = OptimizerConfig(max_epochs=3)
    before = fit_research(
        current.access,
        builder,
        lambda rows, columns: ResearchModel(count, 0),
        config,
        seed=0,
        penalty=0,
        scale_features=False,
    )
    after = fit_research(
        current.access,
        builder,
        factory,
        config,
        seed=0,
        penalty=0,
        scale_features=False,
    )
    assert before.selected_epoch == after.selected_epoch
    np.testing.assert_array_equal(before.watch_losses, after.watch_losses)
