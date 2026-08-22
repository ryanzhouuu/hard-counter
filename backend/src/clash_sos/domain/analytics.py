"""Contracts shared by matchup predictions and rolling analytics."""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from clash_sos.domain.canonical import (
    Battle,
    DomainModel,
    PlayerId,
    RecordDisposition,
    RecordIssue,
    RecordState,
)


class PredictionState(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class PredictionProvenance(DomainModel):
    model_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    card_catalog_version: str = Field(min_length=1)
    balance_era_id: str = Field(min_length=1)


class MatchupPrediction(DomainModel):
    state: PredictionState
    side_a_win_probability: float | None = Field(default=None, ge=0, le=1)
    provenance: PredictionProvenance | None = None
    issue: RecordIssue | None = None

    @model_validator(mode="after")
    def validate_prediction(self) -> Self:
        if self.state is PredictionState.AVAILABLE:
            if self.side_a_win_probability is None or self.provenance is None:
                raise ValueError("available predictions require probability and provenance")
            if self.issue is not None:
                raise ValueError("available predictions cannot carry an issue")
        elif self.side_a_win_probability is not None or self.provenance is not None:
            raise ValueError("unavailable predictions cannot carry probability or provenance")
        return self


def valid_record_disposition() -> RecordDisposition:
    return RecordDisposition(state=RecordState.VALID)


class BattleAnalysisRecord(DomainModel):
    battle: Battle
    balance_era_id: str | None = None
    card_catalog_version: str | None = None
    disposition: RecordDisposition = Field(default_factory=valid_record_disposition)
    prediction: MatchupPrediction | None = None

    @model_validator(mode="after")
    def validate_provenance_annotations(self) -> Self:
        if self.disposition.state is RecordState.VALID and (
            self.balance_era_id is None or self.card_catalog_version is None
        ):
            raise ValueError(
                "valid analysis records require balance-era and card-catalog annotations"
            )
        return self


class AnalysisState(StrEnum):
    AVAILABLE = "available"
    INSUFFICIENT_DATA = "insufficient_data"


class ExclusionReasonCount(DomainModel):
    issue: RecordIssue
    count: int = Field(gt=0)


class RollingSoSObservation(DomainModel):
    timestamp: datetime
    battle_fingerprint: str = Field(min_length=1)
    player_win_probability: float = Field(ge=0, le=1)
    actual_win: int = Field(ge=0, le=1)
    difficulty: float = Field(ge=0, le=1)
    provenance: PredictionProvenance


class RollingSoSResult(DomainModel):
    status: AnalysisState
    target_player: PlayerId
    requested_window: int = Field(gt=0)
    eligible_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    exclusion_reasons: tuple[ExclusionReasonCount, ...] = ()
    window: tuple[RollingSoSObservation, ...] = ()
    strength_of_schedule: float | None = Field(default=None, ge=0, le=1)
    expected_wins: float | None = Field(default=None, ge=0)
    actual_wins: int | None = Field(default=None, ge=0)
    performance_above_expectation: float | None = None
    provenance: tuple[PredictionProvenance, ...] = ()

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        metrics = (
            self.strength_of_schedule,
            self.expected_wins,
            self.actual_wins,
            self.performance_above_expectation,
        )
        if self.status is AnalysisState.AVAILABLE:
            if len(self.window) != self.requested_window or any(
                metric is None for metric in metrics
            ):
                raise ValueError("available results require a complete window and metrics")
            if not self.provenance:
                raise ValueError("available results require prediction provenance")
        elif any(metric is not None for metric in metrics) or self.window:
            raise ValueError("insufficient results cannot carry primary metrics")
        return self
