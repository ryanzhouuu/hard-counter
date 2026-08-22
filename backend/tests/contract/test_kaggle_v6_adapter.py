from datetime import UTC, datetime, timedelta, timezone

import pytest

from clash_sos.application.dataset_audit import BattleIdentityIndex
from clash_sos.domain.canonical import (
    BattleOutcome,
    CardForm,
    RecordIssue,
    RecordState,
)
from clash_sos.infrastructure.kaggle_v6.adapter import (
    KaggleRowError,
    SourceRowLocation,
    adapt_row,
    adapt_valid_row,
)
from clash_sos.infrastructure.kaggle_v6.balance_eras import KAGGLE_V6_ERA_REGISTRY
from clash_sos.infrastructure.kaggle_v6.schema import (
    KAGGLE_V6_SCHEMA,
    IncompatibleKaggleSchemaError,
    KaggleSchemaColumn,
    validate_kaggle_v6_schema,
)
from clash_sos.infrastructure.kaggle_v6.source import (
    KAGGLE_V6_OBSERVED_TIMESTAMP_MAX_UTC,
    KAGGLE_V6_OBSERVED_TIMESTAMP_MIN_UTC,
    KAGGLE_V6_SOURCE_ID,
)


def valid_row() -> dict[str, object]:
    row: dict[str, object] = {
        "winner_id": "#winner",
        "loser_id": "#loser",
        "time": datetime(2026, 6, 21, 8, tzinfo=timezone(timedelta(hours=-4))),
        "game_mode": "Ranked1v1_NewArena",
    }
    for side, offset in (("winner", 0), ("loser", 8)):
        for index in range(8):
            row[f"{side}_card_{index}"] = offset + index
            row[f"{side}_card_{index}_level"] = 16
    return row


def location() -> SourceRowLocation:
    return SourceRowLocation(archive_member="part.parquet", row_number=12)


def test_valid_winner_first_row_adapts_to_canonical_battle() -> None:
    battle = adapt_valid_row(valid_row())

    assert battle.source_id == KAGGLE_V6_SOURCE_ID
    assert battle.outcome is BattleOutcome.SIDE_A_WIN
    assert battle.side_a.player_id.value == "#WINNER"
    assert battle.side_b.player_id.value == "#LOSER"
    assert battle.side_a.deck.cards[0].card_id.value == "knight"
    assert battle.timestamp == datetime(2026, 6, 21, 12, tzinfo=UTC)


def test_source_coverage_endpoints_do_not_define_balance_eras() -> None:
    may = KAGGLE_V6_ERA_REGISTRY.lookup(KAGGLE_V6_OBSERVED_TIMESTAMP_MIN_UTC)
    june = KAGGLE_V6_ERA_REGISTRY.lookup(KAGGLE_V6_OBSERVED_TIMESTAMP_MAX_UTC)

    assert may is not None
    assert may.era_id == "2026-05"
    assert may.valid_from != KAGGLE_V6_OBSERVED_TIMESTAMP_MIN_UTC
    assert june is not None
    assert june.era_id == "2026-06"
    assert june.valid_to != KAGGLE_V6_OBSERVED_TIMESTAMP_MAX_UTC

    transition_day = datetime(2026, 6, 1, 12, tzinfo=UTC)
    assert KAGGLE_V6_ERA_REGISTRY.lookup(transition_day) is None


def test_adapter_preserves_deck_and_level_order() -> None:
    row = valid_row()
    row["winner_card_0"] = 121
    row["winner_card_0_level"] = 15

    battle = adapt_valid_row(row)

    assert battle.side_a.deck.cards[0].form is CardForm.EVOLUTION
    assert battle.side_a.card_levels[0] == 15


@pytest.mark.parametrize("value", [None, "", 42])
def test_adapter_rejects_malformed_scalar_fields(value: object) -> None:
    row = valid_row()
    row["winner_id"] = value

    with pytest.raises(KaggleRowError) as captured:
        adapt_valid_row(row)

    assert captured.value.issue is RecordIssue.MALFORMED_ROW


@pytest.mark.parametrize("value", [None, "not-a-time", datetime(2026, 6, 21)])
def test_adapter_rejects_malformed_timestamps(value: object) -> None:
    row = valid_row()
    row["time"] = value

    with pytest.raises(KaggleRowError) as captured:
        adapt_valid_row(row)

    assert captured.value.issue is RecordIssue.MALFORMED_TIMESTAMP


def test_adapter_rejects_missing_columns() -> None:
    row = valid_row()
    del row["loser_card_7_level"]

    with pytest.raises(KaggleRowError, match="missing columns") as captured:
        adapt_valid_row(row)

    assert captured.value.issue is RecordIssue.MALFORMED_ROW


def test_exact_nullable_source_schema_is_accepted() -> None:
    fingerprint = validate_kaggle_v6_schema(KAGGLE_V6_SCHEMA)

    assert len(KAGGLE_V6_SCHEMA) == 36
    assert len(fingerprint) == 64
    assert all(column.nullable for column in KAGGLE_V6_SCHEMA)


def test_incompatible_source_schema_is_rejected() -> None:
    columns = list(KAGGLE_V6_SCHEMA)
    columns[0] = KaggleSchemaColumn("winner_id", "VARCHAR", False)

    with pytest.raises(IncompatibleKaggleSchemaError):
        validate_kaggle_v6_schema(columns)


def test_valid_row_has_valid_disposition_and_provenance() -> None:
    record = adapt_row(valid_row(), location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert record.battle is not None
    assert record.location.row_number == 12
    assert record.balance_era_id == "2026-06"
    assert record.disposition.state is RecordState.VALID


@pytest.mark.parametrize(
    "field,value,issue",
    [
        ("winner_card_0", None, RecordIssue.INCOMPLETE_DECK),
        ("winner_card_0", 176, RecordIssue.UNKNOWN_CARD),
        ("winner_card_1", 0, RecordIssue.REPEATED_CARD),
    ],
)
def test_unadaptable_decks_are_quarantined(field: str, value: object, issue: RecordIssue) -> None:
    row = valid_row()
    row[field] = value

    record = adapt_row(row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert record.battle is None
    assert record.disposition.state is RecordState.QUARANTINED
    assert record.disposition.issues == (issue,)


def test_malformed_rows_are_invalid() -> None:
    row = valid_row()
    row["time"] = None

    record = adapt_row(row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert record.battle is None
    assert record.disposition.state is RecordState.INVALID
    assert record.disposition.issues == (RecordIssue.MALFORMED_TIMESTAMP,)


@pytest.mark.parametrize(
    "field,value,issue",
    [
        ("game_mode", "Ladder", RecordIssue.UNSUPPORTED_MODE),
        ("loser_card_7_level", 15, RecordIssue.NON_MAX_CARD_LEVEL),
    ],
)
def test_out_of_population_rows_are_unsupported(
    field: str, value: object, issue: RecordIssue
) -> None:
    row = valid_row()
    row[field] = value

    record = adapt_row(row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert record.battle is not None
    assert record.disposition.state is RecordState.UNSUPPORTED
    assert record.disposition.issues == (issue,)


def test_unsupported_mode_and_level_are_both_reported() -> None:
    row = valid_row()
    row["game_mode"] = "Ladder"
    row["winner_card_0_level"] = 15

    record = adapt_row(row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert record.disposition.issues == (
        RecordIssue.UNSUPPORTED_MODE,
        RecordIssue.NON_MAX_CARD_LEVEL,
    )


def test_adapted_rows_stage_the_registry_era_for_their_timestamp() -> None:
    may_row = valid_row()
    may_row["time"] = datetime(2026, 5, 23, tzinfo=UTC)

    june = adapt_row(valid_row(), location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)
    may = adapt_row(may_row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert june.balance_era_id == "2026-06"
    assert may.balance_era_id == "2026-05"


def test_out_of_era_rows_stage_without_an_era_and_stay_adaptable() -> None:
    row = valid_row()
    row["time"] = datetime(2026, 6, 1, 12, tzinfo=UTC)

    record = adapt_row(row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert record.battle is not None
    assert record.balance_era_id is None
    assert record.disposition.state is RecordState.UNSUPPORTED
    assert record.disposition.issues == (RecordIssue.STALE_BALANCE_ERA,)


def test_stale_era_combines_with_other_population_issues() -> None:
    row = valid_row()
    row["time"] = datetime(2026, 6, 1, 12, tzinfo=UTC)
    row["game_mode"] = "Ladder"

    record = adapt_row(row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert record.disposition.issues == (
        RecordIssue.UNSUPPORTED_MODE,
        RecordIssue.STALE_BALANCE_ERA,
    )


@pytest.mark.parametrize("level", [0, 17])
def test_scalar_card_level_violations_remain_invalid(level: int) -> None:
    row = valid_row()
    row["winner_card_0_level"] = level

    record = adapt_row(row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert record.battle is None
    assert record.disposition.state is RecordState.INVALID
    assert record.disposition.issues == (RecordIssue.MALFORMED_ROW,)


def test_identical_players_are_detected_before_battle_construction() -> None:
    row = valid_row()
    row["loser_id"] = "#winner"

    with pytest.raises(KaggleRowError) as captured:
        adapt_valid_row(row)

    assert captured.value.issue is RecordIssue.IDENTICAL_PLAYERS

    record = adapt_row(row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)

    assert record.battle is None
    assert record.balance_era_id is None
    assert record.disposition.state is RecordState.QUARANTINED
    assert record.disposition.issues == (RecordIssue.IDENTICAL_PLAYERS,)


def test_duplicates_across_source_locations_are_detected() -> None:
    index = BattleIdentityIndex()
    first = adapt_row(valid_row(), location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)
    second_location = SourceRowLocation(archive_member="other.parquet", row_number=2)
    second = adapt_row(valid_row(), location=second_location, era_registry=KAGGLE_V6_ERA_REGISTRY)
    assert first.battle is not None
    assert second.battle is not None

    assert index.observe(first.battle, location=first.location.identity_key) is None
    duplicate = index.observe(second.battle, location=second.location.identity_key)

    assert duplicate is not None
    assert duplicate.issues == (RecordIssue.DUPLICATE_BATTLE,)


def test_swapped_winner_record_is_a_conflict() -> None:
    index = BattleIdentityIndex()
    first = adapt_row(valid_row(), location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY)
    swapped_row = valid_row()
    for suffix in ("id", *(f"card_{index}" for index in range(8))):
        swapped_row[f"winner_{suffix}"], swapped_row[f"loser_{suffix}"] = (
            swapped_row[f"loser_{suffix}"],
            swapped_row[f"winner_{suffix}"],
        )
    assert first.battle is not None
    conflict_record = adapt_row(
        swapped_row, location=location(), era_registry=KAGGLE_V6_ERA_REGISTRY
    )
    assert conflict_record.battle is not None

    assert index.observe(first.battle, location="first") is None
    conflict = index.observe(conflict_record.battle, location="second")

    assert conflict is not None
    assert conflict.issues == (RecordIssue.CONFLICTING_BATTLE,)
