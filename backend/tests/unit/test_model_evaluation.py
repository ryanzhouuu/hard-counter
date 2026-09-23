"""Compare streaming reports with established probability metrics."""

from typing import Any, cast

import pytest

from clash_sos.domain.model_artifact import brier_score, expected_calibration_error, log_loss
from clash_sos.domain.model_evaluation import MatchupEvaluationAccumulator


def test_streaming_scores_and_predeclared_slices() -> None:
    labels = (1, 0, 1, 0, 1)
    probabilities = (0.8, 0.2, 0.5, 0.5, 1.0)
    supports = ((0, 0, 0), (1, 0, 0), (3, 5, 1), (25, 30, 1), (2, 2, 0))
    accumulator = MatchupEvaluationAccumulator()
    differences = [
        accumulator.update(
            label=label,
            probability=probability,
            deck_a_support=a,
            deck_b_support=b,
            unordered_pair_support=pair,
            comparator_probability=0.5,
        )
        for label, probability, (a, b, pair) in zip(labels, probabilities, supports, strict=True)
    ]
    report = accumulator.finalize()
    overall = report.overall
    assert overall.metrics.row_count == 5
    assert overall.metrics.log_loss == pytest.approx(log_loss(labels, probabilities))
    assert overall.metrics.brier_score == pytest.approx(brier_score(labels, probabilities))
    assert overall.metrics.expected_calibration_error == pytest.approx(
        expected_calibration_error(labels, probabilities)
    )
    assert overall.accuracy_half_ties == pytest.approx(0.8)
    assert overall.tie_count == 2
    assert sum(item.count for item in overall.reliability) == 5
    assert overall.reliability[-1].count == 1
    assert report.groups["deck:both_unseen"].metrics.row_count == 1
    assert report.groups["deck:one_unseen"].metrics.row_count == 1
    assert report.groups["deck:both_seen"].metrics.row_count == 3
    assert report.groups["pair:seen_pair"].metrics.row_count == 2
    assert report.groups["support:support_20_plus"].metrics.row_count == 1
    assert sum(value for value in differences if value is not None) == pytest.approx(
        5 * (overall.metrics.log_loss - log_loss(labels, (0.5,) * 5))
    )


@pytest.mark.parametrize(
    "values",
    [
        {"label": 2},
        {"probability": float("nan")},
        {"probability": -0.1},
        {"deck_a_support": -1},
        {"comparator_probability": float("inf")},
    ],
)
def test_evaluation_rejects_invalid_rows(values: dict[str, float | int]) -> None:
    row = {
        "label": 1,
        "probability": 0.6,
        "deck_a_support": 1,
        "deck_b_support": 1,
        "unordered_pair_support": 1,
        **values,
    }
    with pytest.raises(ValueError):
        MatchupEvaluationAccumulator().update(**cast(Any, row))


def test_evaluation_requires_rows() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MatchupEvaluationAccumulator().finalize()
