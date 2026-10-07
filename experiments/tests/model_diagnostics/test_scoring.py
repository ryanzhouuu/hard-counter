from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from experiments.model_diagnostics.scoring import evaluate
from experiments.tests.player_adjustment.test_player_history import row


def test_calibration_is_past_only_and_shared_output_removes_skill(tmp_path: Path) -> None:
    calibration = tuple(row(i, label=int(i % 4 != 0)) for i in range(80))
    development = tuple(row(i + 100, label=i % 2) for i in range(40))
    actual = (np.full(80, 0.7), np.full(40, 0.7))
    matchup = (np.full(80, 0.2), np.full(40, 0.2))
    first = evaluate(tmp_path, calibration, development, actual, matchup, calibration)
    assert first.selection_loss is not None
    assert first.actual_temperature is not None
    assert first.matchup_temperature is not None
    assert first.actual_temperature != first.matchup_temperature
    shared = first.predictions["shared-matchup-development"][0]
    assert shared.logit == pytest.approx(0.2)
    assert shared.probability == pytest.approx(1 / (1 + np.exp(-0.2 / first.actual_temperature)))
    changed = tmp_path / "changed"
    changed.mkdir()
    second = evaluate(
        changed,
        calibration,
        tuple(replace(r, label=1) for r in development),
        actual,
        matchup,
        calibration,
    )
    assert second.actual_temperature == first.actual_temperature
    assert second.matchup_temperature == first.matchup_temperature
    assert second.selection_loss != first.selection_loss


def test_failed_calibration_never_becomes_a_calibrated_output(tmp_path: Path) -> None:
    calibration = tuple(row(i, label=i % 2) for i in range(8))
    development = tuple(row(i + 10) for i in range(8))
    zeros = (np.zeros(8), np.zeros(8))
    result = evaluate(tmp_path, calibration, development, zeros, zeros, calibration)
    assert result.selection_loss is None
    assert set(result.predictions) == {
        "actual-calibration-raw",
        "actual-development-raw",
        "matchup-calibration-raw",
        "matchup-development-raw",
    }
    assert (tmp_path / "actual-calibration-diagnostics.json").exists()
    assert not (tmp_path / "actual-development.json").exists()


def test_scoring_refuses_future_calibration(tmp_path: Path) -> None:
    zeros = (np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError, match="precede"):
        evaluate(tmp_path, (row(2),), (row(1),), zeros, zeros, (row(0),))
