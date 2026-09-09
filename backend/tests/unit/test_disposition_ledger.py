from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from clash_sos.domain.canonical import BattleOutcome, RecordIssue, RecordState
from clash_sos.domain.disposition_ledger import (
    DISPOSITION_SCHEMA,
    DispositionRow,
    summarize_disposition_rows,
)
from clash_sos.domain.processed_manifest import INGESTION_ISSUES, RECORD_STATES
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_SOURCE_ID

SHA256 = "a" * 64
FINGERPRINT_B = "b" * 64


def valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "archive_member": "a.parquet",
        "row_number": 0,
        "state": RecordState.VALID,
        "issues": (),
        "source_id": KAGGLE_V6_SOURCE_ID,
        "timestamp": datetime(2026, 6, 21, 12, tzinfo=UTC),
        "mode": "Ranked1v1_NewArena",
        "balance_era_id": "2026-06",
        "outcome": BattleOutcome.SIDE_A_WIN,
        "event_key": SHA256,
        "fingerprint": SHA256,
        "side_a_player_id": "#WINNER",
        "side_b_player_id": "#LOSER",
        "representative_archive_member": "a.parquet",
        "representative_row_number": 0,
    }
    payload.update(overrides)
    return payload


def unadaptable_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "archive_member": "a.parquet",
        "row_number": 3,
        "state": RecordState.INVALID,
        "issues": (RecordIssue.MALFORMED_TIMESTAMP,),
    }
    payload.update(overrides)
    return payload


def test_disposition_schema_nullability_matches_unified_ledger() -> None:
    names = [column.name for column in DISPOSITION_SCHEMA]
    nullable = [column.name for column in DISPOSITION_SCHEMA if column.nullable]
    assert names[:4] == ["archive_member", "row_number", "state", "issues"]
    assert "event_key" in nullable
    assert "balance_era_id" in nullable
    assert "representative_archive_member" in nullable
    assert "representative_row_number" in nullable
    assert "archive_member" not in nullable
    assert "state" not in nullable
    assert "issues" not in nullable


def test_valid_adaptable_row_carries_identity_and_self_representative() -> None:
    row = DispositionRow.model_validate(valid_payload())
    assert row.state is RecordState.VALID
    assert row.issues == ()
    assert row.event_key == SHA256
    assert row.representative_archive_member == "a.parquet"
    assert row.representative_row_number == 0


def test_unadaptable_row_leaves_identity_and_representative_null() -> None:
    row = DispositionRow.model_validate(unadaptable_payload())
    assert row.event_key is None
    assert row.source_id is None
    assert row.timestamp is None
    assert row.representative_archive_member is None
    assert row.representative_row_number is None


def test_issues_must_be_unique_and_sorted() -> None:
    with pytest.raises(ValidationError, match="sorted"):
        DispositionRow.model_validate(
            unadaptable_payload(
                issues=(RecordIssue.MALFORMED_TIMESTAMP, RecordIssue.MALFORMED_ROW),
            )
        )
    with pytest.raises(ValidationError, match="unique"):
        DispositionRow.model_validate(
            unadaptable_payload(issues=(RecordIssue.MALFORMED_ROW, RecordIssue.MALFORMED_ROW))
        )


def test_valid_rows_cannot_carry_issues() -> None:
    with pytest.raises(ValidationError, match="valid"):
        DispositionRow.model_validate(valid_payload(issues=(RecordIssue.UNSUPPORTED_MODE,)))


def test_non_valid_rows_require_an_issue() -> None:
    with pytest.raises(ValidationError, match="issue"):
        DispositionRow.model_validate(unadaptable_payload(issues=()))


def test_insufficient_data_is_not_an_ingestion_state() -> None:
    with pytest.raises(ValidationError):
        DispositionRow.model_validate(unadaptable_payload(state=RecordState.INSUFFICIENT_DATA))


def test_representative_fields_must_be_paired() -> None:
    with pytest.raises(ValidationError, match="representative"):
        DispositionRow.model_validate(valid_payload(representative_row_number=None))
    with pytest.raises(ValidationError, match="representative"):
        DispositionRow.model_validate(
            unadaptable_payload(representative_archive_member="a.parquet")
        )


def test_unadaptable_rows_cannot_carry_identity() -> None:
    with pytest.raises(ValidationError, match="identity"):
        DispositionRow.model_validate(unadaptable_payload(event_key=SHA256))


def test_adaptable_rows_require_identity_fields() -> None:
    with pytest.raises(ValidationError, match="identity"):
        DispositionRow.model_validate(valid_payload(event_key=None))


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        DispositionRow.model_validate(valid_payload(timestamp=datetime(2026, 6, 21, 12)))


def test_summarize_disposition_rows_reconciles_states_and_issues() -> None:
    rows = (
        DispositionRow.model_validate(valid_payload()),
        DispositionRow.model_validate(
            valid_payload(
                row_number=1,
                state=RecordState.QUARANTINED,
                issues=(RecordIssue.DUPLICATE_BATTLE,),
                representative_row_number=0,
            )
        ),
        DispositionRow.model_validate(
            valid_payload(
                archive_member="b.parquet",
                row_number=0,
                state=RecordState.QUARANTINED,
                issues=(RecordIssue.CONFLICTING_BATTLE, RecordIssue.UNSUPPORTED_MODE),
                event_key=SHA256,
                fingerprint=FINGERPRINT_B,
                representative_archive_member=None,
                representative_row_number=None,
            )
        ),
        DispositionRow.model_validate(unadaptable_payload()),
        DispositionRow.model_validate(
            valid_payload(
                archive_member="c.parquet",
                state=RecordState.UNSUPPORTED,
                issues=(RecordIssue.STALE_BALANCE_ERA,),
                balance_era_id=None,
            )
        ),
    )
    summary = summarize_disposition_rows(rows, source_row_count=5)
    assert tuple(count.state for count in summary.states) == RECORD_STATES
    assert tuple(count.issue for count in summary.issues) == INGESTION_ISSUES
    assert summary.source_row_count == 5
    state_map = {count.state: count.count for count in summary.states}
    issue_map = {count.issue: count.count for count in summary.issues}
    assert state_map[RecordState.VALID] == 1
    assert state_map[RecordState.QUARANTINED] == 2
    assert state_map[RecordState.INVALID] == 1
    assert state_map[RecordState.UNSUPPORTED] == 1
    assert state_map[RecordState.INSUFFICIENT_DATA] == 0
    assert issue_map[RecordIssue.DUPLICATE_BATTLE] == 1
    assert issue_map[RecordIssue.CONFLICTING_BATTLE] == 1
    assert issue_map[RecordIssue.UNSUPPORTED_MODE] == 1
    assert issue_map[RecordIssue.STALE_BALANCE_ERA] == 1
    assert issue_map[RecordIssue.MALFORMED_TIMESTAMP] == 1
    assert summary.duplicate_row_count == 1
    assert summary.conflict_row_count == 1


def test_summarize_rejects_count_mismatch() -> None:
    rows = (DispositionRow.model_validate(valid_payload()),)
    with pytest.raises(ValueError, match="source_row_count"):
        summarize_disposition_rows(rows, source_row_count=2)
