import pytest
from pydantic import ValidationError

from clash_sos.domain.analytics import MatchupPrediction, PredictionProvenance, PredictionState
from clash_sos.domain.canonical import RecordIssue


def provenance() -> PredictionProvenance:
    return PredictionProvenance(
        model_version="model:v1",
        dataset_version="dataset:v1",
        card_catalog_version="catalog:v1",
        balance_era_id="2026-06",
    )


def test_available_prediction_requires_probability_and_provenance() -> None:
    prediction = MatchupPrediction(
        state=PredictionState.AVAILABLE,
        side_a_win_probability=0.25,
        provenance=provenance(),
    )

    assert prediction.side_a_win_probability == 0.25


@pytest.mark.parametrize(
    "payload",
    [
        {"state": PredictionState.AVAILABLE, "side_a_win_probability": 0.25},
        {
            "state": PredictionState.AVAILABLE,
            "side_a_win_probability": 0.25,
            "provenance": provenance(),
            "issue": RecordIssue.STALE_BALANCE_ERA,
        },
        {
            "state": PredictionState.UNAVAILABLE,
            "side_a_win_probability": 0.25,
        },
    ],
)
def test_prediction_rejects_inconsistent_availability(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MatchupPrediction.model_validate(payload)


def test_provenance_requires_versioned_values() -> None:
    with pytest.raises(ValidationError):
        PredictionProvenance(
            model_version="",
            dataset_version="dataset:v1",
            card_catalog_version="catalog:v1",
            balance_era_id="2026-06",
        )
