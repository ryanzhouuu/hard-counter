from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from clash_sos.domain.canonical import BattleOutcome
from clash_sos.domain.canonical_dataset import (
    ACCEPTED_MODE,
    CANONICAL_SCHEMA,
    CANONICAL_SCHEMA_VERSION,
    CanonicalBattleRow,
    canonical_row_sort_key,
    canonical_schema_fingerprint,
    deck_content_hash,
    logical_canonical_content_hash,
    stream_logical_canonical_content_hash,
)
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_SOURCE_ID

SHA256 = "a" * 64
CARD_IDS = tuple(f"card-{index}" for index in range(8))
CARD_FORMS = ("base",) * 8
CARD_LEVELS = (16,) * 8
DECK_HASH = deck_content_hash(CARD_IDS, CARD_FORMS, CARD_LEVELS)


def valid_row_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_version": "kaggle-v6-ranked16-v1",
        "source_id": KAGGLE_V6_SOURCE_ID,
        "timestamp": datetime(2026, 6, 21, 12, tzinfo=UTC),
        "mode": ACCEPTED_MODE,
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
        "archive_member": "part.parquet",
        "row_number": 12,
    }
    payload.update(overrides)
    return payload


def canonical_row(**overrides: object) -> CanonicalBattleRow:
    return CanonicalBattleRow.model_validate(valid_row_payload(**overrides))


def test_canonical_schema_version_literal() -> None:
    assert CANONICAL_SCHEMA_VERSION == "kaggle-v6-ranked16-schema:v1"


def test_canonical_schema_has_twenty_non_nullable_columns() -> None:
    assert len(CANONICAL_SCHEMA) == 20
    assert all(column.nullable is False for column in CANONICAL_SCHEMA)
    assert CANONICAL_SCHEMA[0].name == "dataset_version"
    assert CANONICAL_SCHEMA[-1].name == "row_number"


def test_canonical_schema_fingerprint_is_stable() -> None:
    assert canonical_schema_fingerprint() == canonical_schema_fingerprint()
    assert len(canonical_schema_fingerprint()) == 64


def test_canonical_row_rejects_kaggle_column_names() -> None:
    with pytest.raises(ValidationError):
        CanonicalBattleRow.model_validate({**valid_row_payload(), "winner_id": "#X"})


def test_row_number_is_zero_based() -> None:
    CanonicalBattleRow.model_validate({**valid_row_payload(), "row_number": 0})
    with pytest.raises(ValidationError):
        CanonicalBattleRow.model_validate({**valid_row_payload(), "row_number": -1})


def test_canonical_row_requires_ranked_mode_and_side_a_win() -> None:
    with pytest.raises(ValidationError):
        canonical_row(mode="Ladder")
    with pytest.raises(ValidationError):
        canonical_row(outcome=BattleOutcome.SIDE_B_WIN)


def test_canonical_row_requires_all_sixteen_levels() -> None:
    levels = (16, 16, 16, 16, 16, 16, 16, 15)
    with pytest.raises(ValidationError, match="level 16"):
        canonical_row(
            side_a_card_levels=levels,
            side_a_deck_hash=deck_content_hash(CARD_IDS, CARD_FORMS, levels),
        )


def test_canonical_row_requires_aligned_eight_card_arrays() -> None:
    with pytest.raises(ValidationError, match="eight aligned"):
        canonical_row(side_a_card_ids=CARD_IDS[:7])


def test_canonical_row_requires_matching_deck_hash() -> None:
    with pytest.raises(ValidationError, match="deck hash"):
        canonical_row(side_a_deck_hash=SHA256)


def test_canonical_row_requires_distinct_players() -> None:
    with pytest.raises(ValidationError, match="distinct players"):
        canonical_row(side_b_player_id="#WINNER")


def test_canonical_row_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        canonical_row(timestamp=datetime(2026, 6, 21, 12))


def test_deck_content_hash_ignores_source_order() -> None:
    reversed_ids = tuple(reversed(CARD_IDS))
    reversed_forms = tuple(reversed(CARD_FORMS))
    reversed_levels = tuple(reversed(CARD_LEVELS))
    assert deck_content_hash(CARD_IDS, CARD_FORMS, CARD_LEVELS) == deck_content_hash(
        reversed_ids, reversed_forms, reversed_levels
    )


def test_canonical_row_sort_key_orders_by_timestamp_then_identity() -> None:
    earlier = canonical_row(
        timestamp=datetime(2026, 6, 20, tzinfo=UTC),
        fingerprint="b" * 64,
        archive_member="a.parquet",
        row_number=1,
    )
    later = canonical_row(
        timestamp=datetime(2026, 6, 21, tzinfo=UTC),
        fingerprint="a" * 64,
        archive_member="a.parquet",
        row_number=0,
    )
    assert canonical_row_sort_key(earlier) < canonical_row_sort_key(later)


def test_logical_hash_is_stable_for_equivalent_row_order() -> None:
    first = canonical_row(fingerprint="b" * 64, row_number=1)
    second = canonical_row(
        timestamp=datetime(2026, 6, 22, tzinfo=UTC),
        fingerprint="a" * 64,
        archive_member="other.parquet",
        row_number=0,
    )
    rows = (first, second)
    assert logical_canonical_content_hash(rows) == logical_canonical_content_hash(
        list(reversed(rows))
    )
    assert len(logical_canonical_content_hash(rows)) == 64


def test_logical_hash_changes_when_row_content_changes() -> None:
    baseline = logical_canonical_content_hash((canonical_row(),))
    changed = logical_canonical_content_hash((canonical_row(balance_era_id="2026-05"),))
    assert baseline != changed
    assert len(baseline) == 64


def test_stream_logical_hash_matches_empty_array() -> None:
    assert stream_logical_canonical_content_hash(()) == logical_canonical_content_hash(())
    assert len(stream_logical_canonical_content_hash(())) == 64


def test_stream_logical_hash_matches_sorted_two_row_payload() -> None:
    first = canonical_row(fingerprint="b" * 64, row_number=1)
    second = canonical_row(
        timestamp=datetime(2026, 6, 22, tzinfo=UTC),
        fingerprint="a" * 64,
        archive_member="other.parquet",
        row_number=0,
    )
    shuffled = (second, first)
    ordered = tuple(sorted(shuffled, key=canonical_row_sort_key))
    streamed = stream_logical_canonical_content_hash(ordered)
    assert streamed == logical_canonical_content_hash(shuffled)


def test_stream_logical_hash_changes_when_row_content_changes() -> None:
    baseline = stream_logical_canonical_content_hash((canonical_row(),))
    changed = stream_logical_canonical_content_hash((canonical_row(balance_era_id="2026-05"),))
    assert baseline != changed
    assert len(baseline) == 64
