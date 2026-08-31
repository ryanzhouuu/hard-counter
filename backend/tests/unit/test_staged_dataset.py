from datetime import UTC, datetime
from hashlib import sha256

import pytest
from pydantic import ValidationError

from clash_sos.domain.canonical import BattleOutcome, RecordIssue, RecordState
from clash_sos.domain.canonical_dataset import deck_content_hash
from clash_sos.domain.staged_dataset import (
    POPULATION_OBSERVATION_ISSUES,
    STAGED_SCHEMA,
    UNADAPTABLE_SCHEMA,
    StagedBattleRow,
    UnadaptableRow,
    logical_row_stream_hash,
)
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_SOURCE_ID

SHA256 = "a" * 64
CARD_IDS = tuple(f"card-{index}" for index in range(8))
CARD_FORMS = ("base",) * 8
CARD_LEVELS = (16,) * 8
DECK_HASH = deck_content_hash(CARD_IDS, CARD_FORMS, CARD_LEVELS)


def staged_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_id": KAGGLE_V6_SOURCE_ID,
        "timestamp": datetime(2026, 6, 21, 12, tzinfo=UTC),
        "mode": "Ranked1v1_NewArena",
        "balance_era_id": "2026-06",
        "outcome": BattleOutcome.SIDE_A_WIN,
        "event_key": SHA256,
        "fingerprint": SHA256,
        "side_a_player_id": "#WINNER",
        "side_b_player_id": "#LOSER",
        "side_a_card_ids": CARD_IDS,
        "side_a_card_forms": CARD_FORMS,
        "side_a_card_levels": CARD_LEVELS,
        "side_a_deck_hash": DECK_HASH,
        "side_b_card_ids": CARD_IDS,
        "side_b_card_forms": CARD_FORMS,
        "side_b_card_levels": CARD_LEVELS,
        "side_b_deck_hash": DECK_HASH,
        "observation_issues": (),
        "archive_member": "part.parquet",
        "row_number": 0,
    }
    payload.update(overrides)
    return payload


def test_staged_schema_allows_nullable_era_only() -> None:
    assert [column.name for column in STAGED_SCHEMA if column.nullable] == ["balance_era_id"]
    assert len(STAGED_SCHEMA) == 20


def test_unadaptable_schema_has_four_required_columns() -> None:
    assert [column.name for column in UNADAPTABLE_SCHEMA] == [
        "archive_member",
        "row_number",
        "state",
        "issues",
    ]
    assert all(column.nullable is False for column in UNADAPTABLE_SCHEMA)


def test_staged_row_allows_unsupported_mode_low_level_and_null_era() -> None:
    levels = (16, 16, 16, 16, 16, 16, 16, 15)
    row = StagedBattleRow.model_validate(
        staged_payload(
            mode="Ladder",
            balance_era_id=None,
            side_a_card_levels=levels,
            side_a_deck_hash=deck_content_hash(CARD_IDS, CARD_FORMS, levels),
            observation_issues=(
                RecordIssue.NON_MAX_CARD_LEVEL,
                RecordIssue.STALE_BALANCE_ERA,
                RecordIssue.UNSUPPORTED_MODE,
            ),
        )
    )
    assert row.balance_era_id is None
    assert row.observation_issues == POPULATION_OBSERVATION_ISSUES


def test_staged_row_rejects_unsorted_or_non_population_issues() -> None:
    with pytest.raises(ValidationError):
        StagedBattleRow.model_validate(
            staged_payload(
                observation_issues=(
                    RecordIssue.UNSUPPORTED_MODE,
                    RecordIssue.NON_MAX_CARD_LEVEL,
                ),
            )
        )
    with pytest.raises(ValidationError):
        StagedBattleRow.model_validate(
            staged_payload(observation_issues=(RecordIssue.MALFORMED_ROW,))
        )


def test_staged_row_rejects_level_outside_one_to_sixteen() -> None:
    levels = (16,) * 7 + (0,)
    with pytest.raises(ValidationError):
        StagedBattleRow.model_validate(
            staged_payload(
                side_a_card_levels=levels,
                side_a_deck_hash=deck_content_hash(CARD_IDS, CARD_FORMS, levels),
            )
        )


def test_unadaptable_row_rejects_valid_state_and_detail() -> None:
    with pytest.raises(ValidationError):
        UnadaptableRow.model_validate(
            {
                "archive_member": "part.parquet",
                "row_number": 0,
                "state": RecordState.VALID,
                "issues": (RecordIssue.MALFORMED_ROW,),
            }
        )
    with pytest.raises(ValidationError):
        UnadaptableRow.model_validate(
            {
                "archive_member": "part.parquet",
                "row_number": 0,
                "state": RecordState.INVALID,
                "issues": (RecordIssue.MALFORMED_ROW,),
                "detail": "traceback",
            }
        )


def test_stream_hash_is_order_sensitive_and_stable() -> None:
    first = StagedBattleRow.model_validate(staged_payload(row_number=0))
    second = StagedBattleRow.model_validate(staged_payload(row_number=1, mode="Ladder"))
    assert logical_row_stream_hash((first, second)) == logical_row_stream_hash((first, second))
    assert logical_row_stream_hash((first, second)) != logical_row_stream_hash((second, first))
    assert logical_row_stream_hash(()) == sha256().hexdigest()


def test_staged_row_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError):
        StagedBattleRow.model_validate(staged_payload(timestamp=datetime(2026, 6, 21, 12)))


def test_staged_row_rejects_negative_row_number() -> None:
    with pytest.raises(ValidationError):
        StagedBattleRow.model_validate(staged_payload(row_number=-1))


def test_unadaptable_row_rejects_unsorted_issues() -> None:
    with pytest.raises(ValidationError):
        UnadaptableRow.model_validate(
            {
                "archive_member": "part.parquet",
                "row_number": 0,
                "state": RecordState.INVALID,
                "issues": (RecordIssue.MALFORMED_TIMESTAMP, RecordIssue.MALFORMED_ROW),
            }
        )


def test_unadaptable_row_rejects_empty_issues() -> None:
    with pytest.raises(ValidationError):
        UnadaptableRow.model_validate(
            {
                "archive_member": "part.parquet",
                "row_number": 0,
                "state": RecordState.INVALID,
                "issues": (),
            }
        )


def test_staged_row_rejects_mismatched_deck_hash() -> None:
    with pytest.raises(ValidationError, match="deck hash"):
        StagedBattleRow.model_validate(staged_payload(side_a_deck_hash=SHA256))


def test_staged_row_rejects_identical_players() -> None:
    with pytest.raises(ValidationError, match="distinct players"):
        StagedBattleRow.model_validate(staged_payload(side_b_player_id="#WINNER"))
