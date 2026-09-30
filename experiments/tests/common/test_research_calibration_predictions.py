from dataclasses import replace
from math import log
from pathlib import Path
from types import SimpleNamespace

import experiments.common.calibration as calibration
import pytest
from experiments.common.calibration import (
    calibrate_predictions,
    ensemble_logits,
    fit_temperature,
    sigmoid,
)
from experiments.common.predictions import pair_predictions, read_predictions, write_predictions
from experiments.tests.common.reporting_fixtures import prediction


def test_known_temperature_recovery_and_complementarity() -> None:
    logits = (2 * log(4),) * 100 + (-2 * log(4),) * 100
    labels = (1,) * 80 + (0,) * 20 + (0,) * 80 + (1,) * 20
    fit = fit_temperature(logits, labels)
    assert fit.temperature == pytest.approx(2, rel=1e-5)
    assert fit.calibrated_loss < fit.raw_loss
    assert fit.row_count == 200
    assert sigmoid(5, fit.temperature) + sigmoid(-5, fit.temperature) == pytest.approx(1)
    assert sigmoid(-10000) == 0
    assert sigmoid(10000) == 1


@pytest.mark.parametrize(
    "logits,labels",
    [
        ((), ()),
        ((1,), ()),
        ((float("inf"),), (1,)),
        ((1,), (2,)),
        ((0, 0), (0, 1)),
    ],
)
def test_invalid_calibration_inputs(logits: tuple[float, ...], labels: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        fit_temperature(logits, labels)


def test_boundary_solution_and_optimizer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="boundary"):
        fit_temperature((1, 2), (1, 1))

    def failed_optimizer(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=False, x=0.0, fun=0.0)

    monkeypatch.setattr(calibration, "minimize_scalar", failed_optimizer)
    with pytest.raises(ValueError, match="optimizer failed"):
        fit_temperature((1, 2), (0, 1))


def test_calibrating_later_rows_never_changes_fit_or_raw_scores() -> None:
    fit = fit_temperature((log(4),) * 10, (1,) * 8 + (0,) * 2)
    later = replace(prediction(0), logit=2)
    first = calibrate_predictions((later,), fit.temperature)
    second = calibrate_predictions((replace(later, label=0),), fit.temperature)
    assert first[0].probability == second[0].probability
    assert first[0].raw_probability == sigmoid(2)
    assert fit.temperature == pytest.approx(1, rel=1e-5)


def test_ensemble_averages_probabilities_before_logits() -> None:
    assert ensemble_logits(((0.2, 0.4), (0.8, 0.8))) == pytest.approx((0, log(1.5)))
    for values in ((), ((),), ((0.0,),), ((0.2,), (0.3, 0.4)), ((float("nan"),),)):
        with pytest.raises(ValueError):
            ensemble_logits(values)


def test_prediction_serialization_and_reordered_join(tmp_path: Path) -> None:
    rows = (prediction(1, 0.7), prediction(0, 0.8))
    path = tmp_path / "predictions.json"
    write_predictions(path, iter(rows))
    restored = read_predictions(path)
    assert restored == tuple(reversed(rows))
    assert pair_predictions(rows, restored) == tuple((row, row) for row in restored)
    path.write_text(path.read_text().replace('"logit":0', '"unknown":0,"logit":0'))
    with pytest.raises(ValueError):
        read_predictions(path)


def test_paired_join_rejects_duplicates_missing_labels_and_orientation() -> None:
    first, second = prediction(0), prediction(1)
    invalid = (
        (first, first),
        (first,),
        (first, replace(second, event_key=first.event_key)),
        (first, replace(second, label=0)),
        (first, replace(second, player_a="B", player_b="A")),
        (first, replace(second, event_key="different")),
    )
    for candidate in invalid:
        with pytest.raises(ValueError):
            pair_predictions(candidate, (first, second))


@pytest.mark.parametrize(
    "change",
    [
        {"probability": float("nan")},
        {"probability": 1.1},
        {"label": 2},
        {"logit": float("inf")},
        {"player_b": "A"},
        {"event_key": ""},
    ],
)
def test_prediction_validation(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(prediction(0), **change)


def test_zero_based_canonical_row_number_is_valid() -> None:
    row = prediction(0)
    assert replace(row, row_key=(*row.row_key[:3], 0)).row_key[3] == 0
    with pytest.raises(ValueError, match="nonnegative"):
        replace(row, row_key=(*row.row_key[:3], -1))
