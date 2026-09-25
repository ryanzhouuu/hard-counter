from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import AnyHttpUrl, ValidationError

from clash_sos.application.rolling_sos import calculate_rolling_sos, summarize_observations
from clash_sos.domain.analytics import (
    AnalysisState,
    BattleAnalysisRecord,
    MatchupPrediction,
    PredictionProvenance,
    PredictionState,
    RollingSoSObservation,
)
from clash_sos.domain.canonical import (
    BalanceChange,
    BalanceEra,
    Battle,
    BattleOutcome,
    BattleSide,
    CardForm,
    CardId,
    CardRef,
    Deck,
    EraBoundaryEvidence,
    EraBoundaryPolicy,
    PlayerId,
    RecordDisposition,
    RecordIssue,
    RecordState,
)

TARGET = PlayerId("#TARGET1")


def evidence(stated_date: datetime, policy: EraBoundaryPolicy) -> EraBoundaryEvidence:
    return EraBoundaryEvidence(
        summary="A documented balance update.",
        reference=AnyHttpUrl("https://example.test/balance-notes/june"),
        stated_date=stated_date,
        precision="day",
        boundary_policy=policy,
    )


ERA = BalanceEra(
    era_id="2026-06",
    valid_from=datetime(2026, 6, 1, tzinfo=UTC),
    valid_to=datetime(2026, 7, 1, tzinfo=UTC),
    card_catalog_version="catalog:2026-06",
    changed_cards=(
        BalanceChange(card_id=CardId("knight"), change_type="buff", summary="More hitpoints"),
    ),
    start_evidence=evidence(
        datetime(2026, 5, 31, tzinfo=UTC), EraBoundaryPolicy.DAY_AFTER_STATED_DATE
    ),
    end_evidence=evidence(datetime(2026, 7, 1, tzinfo=UTC), EraBoundaryPolicy.AT_STATED_DATE),
)
PROVENANCE = PredictionProvenance(
    model_version="model:v1",
    dataset_version="dataset:v1",
    card_catalog_version=ERA.card_catalog_version,
    balance_era_id=ERA.era_id,
)


def deck(offset: int) -> Deck:
    return Deck(
        cards=tuple(
            CardRef(card_id=CardId(f"card-{offset + index}"), form=CardForm.BASE)
            for index in range(8)
        )
    )


def battle(
    index: int,
    probability: float,
    outcome: BattleOutcome = BattleOutcome.SIDE_A_WIN,
    *,
    target_on_side_a: bool = True,
    timestamp: datetime | None = None,
    mode: str = "Ranked1v1_NewArena",
    card_level: int = 16,
) -> BattleAnalysisRecord:
    target_side = BattleSide(
        player_id=TARGET,
        deck=deck(index * 2),
        card_levels=(card_level,) * 8,
    )
    opponent_side = BattleSide(
        player_id=PlayerId(f"#OPP{index + 1}"),
        deck=deck(index * 2 + 1),
        card_levels=(card_level,) * 8,
    )
    side_a, side_b = (
        (target_side, opponent_side) if target_on_side_a else (opponent_side, target_side)
    )
    record = BattleAnalysisRecord(
        battle=Battle(
            side_a=side_a,
            side_b=side_b,
            outcome=outcome,
            timestamp=timestamp or datetime(2026, 6, 10, tzinfo=UTC) + timedelta(minutes=index),
            mode=mode,
            source_id=f"source:row-{index}",
        ),
        balance_era_id=ERA.era_id,
        card_catalog_version=ERA.card_catalog_version,
        prediction=MatchupPrediction(
            state=PredictionState.AVAILABLE,
            side_a_win_probability=probability,
            provenance=PROVENANCE,
        ),
    )
    return record


def unavailable(record: BattleAnalysisRecord) -> BattleAnalysisRecord:
    return record.model_copy(
        update={
            "prediction": MatchupPrediction(state=PredictionState.UNAVAILABLE),
        }
    )


def test_calculates_the_approved_rolling_aggregates() -> None:
    records = [
        battle(0, 0.25, BattleOutcome.SIDE_A_WIN),
        battle(1, 0.50, BattleOutcome.SIDE_B_WIN),
        battle(2, 0.75, BattleOutcome.SIDE_A_WIN),
    ]

    result = calculate_rolling_sos(TARGET, records, window_size=3)

    assert result.status is AnalysisState.AVAILABLE
    assert result.strength_of_schedule == pytest.approx(0.50)
    assert result.expected_wins == pytest.approx(1.50)
    assert result.actual_wins == 2
    assert result.performance_above_expectation == pytest.approx(0.50)


def test_summarizes_deck_only_predictions_outside_training_era() -> None:
    observations = [
        RollingSoSObservation(
            timestamp=datetime(2026, 9, 24, hour=index, tzinfo=UTC),
            battle_fingerprint=f"live-{index}",
            player_win_probability=probability,
            actual_win=actual_win,
            difficulty=1 - probability,
            provenance=PROVENANCE,
        )
        for index, probability, actual_win in [(1, 0.25, 1), (0, 0.75, 0)]
    ]

    result = summarize_observations(TARGET, observations, window_size=2, excluded_count=1)

    assert result.status is AnalysisState.AVAILABLE
    assert [item.battle_fingerprint for item in result.window] == ["live-0", "live-1"]
    assert result.expected_wins == pytest.approx(1.0)
    assert result.actual_wins == 1
    assert result.strength_of_schedule == pytest.approx(0.5)
    assert result.excluded_count == 1
    assert result.provenance == (PROVENANCE,)


def test_uses_the_final_n_sorted_eligible_records() -> None:
    records = [
        battle(3, 0.90, timestamp=datetime(2026, 6, 10, 3, tzinfo=UTC)),
        battle(0, 0.10, timestamp=datetime(2026, 6, 10, 0, tzinfo=UTC)),
        battle(2, 0.70, timestamp=datetime(2026, 6, 10, 2, tzinfo=UTC)),
        battle(1, 0.30, timestamp=datetime(2026, 6, 10, 1, tzinfo=UTC)),
    ]

    result = calculate_rolling_sos(TARGET, records, window_size=2)

    assert result.eligible_count == 4
    assert [observation.player_win_probability for observation in result.window] == [0.70, 0.90]
    assert result.expected_wins == pytest.approx(1.60)


def test_excluded_records_do_not_consume_slots_and_are_reported() -> None:
    unsupported = battle(1, 0.5, mode="TrophyRoad")
    non_max = battle(2, 0.5, card_level=15)
    draw = battle(3, 0.5, BattleOutcome.DRAW)
    stale = battle(4, 0.5).model_copy(
        update={
            "prediction": MatchupPrediction(
                state=PredictionState.AVAILABLE,
                side_a_win_probability=0.5,
                provenance=PROVENANCE.model_copy(update={"balance_era_id": "2026-05"}),
            )
        }
    )
    records = [battle(0, 0.5), unsupported, non_max, draw, stale, unavailable(battle(5, 0.5))]

    result = calculate_rolling_sos(TARGET, records, window_size=2)

    assert result.status is AnalysisState.INSUFFICIENT_DATA
    assert result.eligible_count == 1
    assert result.excluded_count == 5
    assert result.window == ()
    assert result.expected_wins is None
    assert result.provenance == (PROVENANCE,)
    assert {item.issue: item.count for item in result.exclusion_reasons} == {
        RecordIssue.DRAW_OUTCOME: 1,
        RecordIssue.NON_MAX_CARD_LEVEL: 1,
        RecordIssue.STALE_BALANCE_ERA: 1,
        RecordIssue.UNAVAILABLE_MODEL_COVERAGE: 1,
        RecordIssue.UNSUPPORTED_MODE: 1,
    }


def test_incompatible_prediction_catalog_is_stale() -> None:
    record = battle(0, 0.5).model_copy(
        update={
            "prediction": MatchupPrediction(
                state=PredictionState.AVAILABLE,
                side_a_win_probability=0.5,
                provenance=PROVENANCE.model_copy(
                    update={"card_catalog_version": "catalog:2026-05"}
                ),
            )
        }
    )

    result = calculate_rolling_sos(TARGET, [record], window_size=1)

    assert result.status is AnalysisState.INSUFFICIENT_DATA
    assert result.exclusion_reasons[0].issue is RecordIssue.STALE_BALANCE_ERA


def test_valid_records_require_era_and_catalog_annotations() -> None:
    base = battle(0, 0.5)

    with pytest.raises(ValidationError, match="balance-era and card-catalog"):
        BattleAnalysisRecord(battle=base.battle, prediction=base.prediction)

    quarantined = BattleAnalysisRecord(
        battle=base.battle,
        disposition=RecordDisposition(
            state=RecordState.QUARANTINED, issues=(RecordIssue.UNKNOWN_CARD,)
        ),
    )
    assert quarantined.balance_era_id is None
    assert quarantined.card_catalog_version is None


def test_target_side_orientation_inverts_probability_and_actual_result() -> None:
    record = battle(
        0,
        0.25,
        BattleOutcome.SIDE_A_WIN,
        target_on_side_a=False,
    )

    result = calculate_rolling_sos(TARGET, [record], window_size=1)

    assert result.window[0].player_win_probability == pytest.approx(0.75)
    assert result.window[0].actual_win == 0
    assert result.window[0].difficulty == pytest.approx(0.25)


@given(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
def test_player_perspective_is_symmetric(probability: float) -> None:
    record = battle(0, probability, BattleOutcome.SIDE_A_WIN)

    side_a_result = calculate_rolling_sos(TARGET, [record], window_size=1)
    side_b_result = calculate_rolling_sos(record.battle.side_b.player_id, [record], window_size=1)

    assert side_a_result.window[0].player_win_probability == pytest.approx(probability)
    assert side_b_result.window[0].player_win_probability == pytest.approx(1 - probability)
    assert side_a_result.window[0].actual_win == 1
    assert side_b_result.window[0].actual_win == 0


def test_non_valid_disposition_is_excluded_without_using_prediction() -> None:
    record = battle(0, 0.5).model_copy(
        update={
            "disposition": RecordDisposition(
                state=RecordState.QUARANTINED,
                issues=(RecordIssue.UNKNOWN_CARD,),
            )
        }
    )

    result = calculate_rolling_sos(TARGET, [record], window_size=1)

    assert result.status is AnalysisState.INSUFFICIENT_DATA
    assert len(result.exclusion_reasons) == 1
    assert result.exclusion_reasons[0].issue is RecordIssue.UNKNOWN_CARD
    assert result.exclusion_reasons[0].count == 1


def test_window_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        calculate_rolling_sos(TARGET, [], window_size=0)
