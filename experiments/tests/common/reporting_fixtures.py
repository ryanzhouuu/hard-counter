from datetime import UTC, datetime, timedelta

from experiments.common.predictions import Prediction

from clash_sos.domain.analytics import PredictionProvenance


def prediction(
    index: int,
    probability: float = 0.5,
    *,
    player_a: str = "A",
    player_b: str = "B",
    label: int = 1,
    timestamp: datetime | None = None,
) -> Prediction:
    stamp = timestamp or datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index)
    return Prediction(
        row_key=(stamp, f"f{index:04}", "source", index + 1),
        event_key=f"e{index}",
        timestamp=stamp,
        player_a=player_a,
        player_b=player_b,
        label=label,
        logit=0,
        probability=probability,
    )


def provenance(model: str) -> PredictionProvenance:
    return PredictionProvenance(
        model_version=model,
        dataset_version="synthetic",
        card_catalog_version="synthetic",
        balance_era_id="synthetic",
    )
