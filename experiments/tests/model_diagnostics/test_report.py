from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from experiments.common.checkpoints import predictions
from experiments.common.predictions import write_predictions
from experiments.common.session import Session
from experiments.common.statistics import score_predictions
from experiments.model_diagnostics import batch
from experiments.model_diagnostics.artifacts import (
    TrialSpec,
    bind_inputs,
    finish_trial,
    trial_stage,
)
from experiments.model_diagnostics.report import summarize, write_summary
from experiments.tests.model_diagnostics.test_trial import session
from pydantic_core import to_jsonable_python

from clash_sos.domain.canonical_dataset import canonical_json_bytes


def test_verified_summary_includes_seed_comparisons_and_uniform_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = session(tmp_path)

    def fake_run(directory: Path, current: Session, spec: TrialSpec, root: Path):
        dev = current.access.read("development", "compare")
        scored = predictions(dev, np.full(len(dev), 0.1), 1)
        metrics = asdict(score_predictions(scored))
        names = ("actual-development", "shared-matchup-development", "actual-development-raw")
        with trial_stage(directory) as stage:
            for name in names:
                write_predictions(stage / f"{name}.json", scored)
            report = {
                "selection_loss": 0.69,
                "selected_epoch": 1,
                "fit_seconds": 0.1,
                "metrics": {name: metrics for name in names},
                "temperatures": {},
                "player_effects": None,
            }
            (stage / "report.json").write_bytes(canonical_json_bytes(to_jsonable_python(report)))
            return finish_trial(stage, bind_inputs(current, spec, root), "complete", 0.1)

    monkeypatch.setattr(batch, "run_trial", fake_run)
    destination = tmp_path / "batch"
    assert (
        batch.run_batch(destination, current, Path.cwd(), progress=lambda _: None)["status"]
        == "complete"
    )
    report = summarize(destination)
    assert len(report["table"]) == 26  # type: ignore[arg-type]
    assert len(report["paired_by_seed"]) == 12  # type: ignore[arg-type]
    assert write_summary(destination).is_file()
    first = (destination / "summary.md").read_bytes()
    write_summary(destination)
    assert (destination / "summary.md").read_bytes() == first
    (destination / "trials/reference/report.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        summarize(destination)
