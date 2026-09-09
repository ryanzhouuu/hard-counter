from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from hypothesis import given
from hypothesis import strategies as st

from clash_sos.application.dataset_grouping import (
    KaggleV6GroupingError,
    disposition_from_unadaptable,
    finalize_event_group,
    group_staged_dataset,
)
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.canonical import BattleOutcome, RecordIssue, RecordState
from clash_sos.domain.canonical_dataset import deck_content_hash
from clash_sos.domain.disposition_ledger import DISPOSITION_SCHEMA
from clash_sos.domain.staged_dataset import StagedBattleRow, UnadaptableRow
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_SOURCE_ID
from clash_sos.infrastructure.kaggle_v6.staging_io import write_staged_part, write_unadaptable_part

EVENT_KEY = "a" * 64
FINGERPRINT_A = "b" * 64
FINGERPRINT_B = "c" * 64
CARD_IDS = tuple(f"card-{index}" for index in range(8))
CARD_FORMS = ("base",) * 8
CARD_LEVELS = (16,) * 8
DECK_HASH = deck_content_hash(CARD_IDS, CARD_FORMS, CARD_LEVELS)


def staged_row(**overrides: object) -> StagedBattleRow:
    payload: dict[str, object] = {
        "source_id": KAGGLE_V6_SOURCE_ID,
        "timestamp": datetime(2026, 6, 21, 12, tzinfo=UTC),
        "mode": "Ranked1v1_NewArena",
        "balance_era_id": "2026-06",
        "outcome": BattleOutcome.SIDE_A_WIN,
        "event_key": EVENT_KEY,
        "fingerprint": FINGERPRINT_A,
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
        "archive_member": "a.parquet",
        "row_number": 0,
    }
    payload.update(overrides)
    return StagedBattleRow.model_validate(payload)


def test_unique_valid_row_is_accepted_as_self_representative() -> None:
    row = staged_row()
    result = finalize_event_group((row,))
    assert len(result) == 1
    assert result[0].state is RecordState.VALID
    assert result[0].issues == ()
    assert result[0].representative_archive_member == "a.parquet"
    assert result[0].representative_row_number == 0
    assert result[0].event_key == EVENT_KEY


def test_duplicate_keeps_earliest_location_regardless_of_input_order() -> None:
    later = staged_row(archive_member="b.parquet", row_number=4)
    earlier = staged_row()
    result = finalize_event_group((later, earlier))
    by_location = {(row.archive_member, row.row_number): row for row in result}
    assert by_location[("a.parquet", 0)].state is RecordState.VALID
    duplicate = by_location[("b.parquet", 4)]
    assert duplicate.state is RecordState.QUARANTINED
    assert duplicate.issues == (RecordIssue.DUPLICATE_BATTLE,)
    assert duplicate.representative_archive_member == "a.parquet"
    assert duplicate.representative_row_number == 0


def test_conflict_quarantines_every_row_without_a_representative() -> None:
    first = staged_row()
    conflict = staged_row(
        archive_member="b.parquet",
        row_number=1,
        fingerprint=FINGERPRINT_B,
        side_a_player_id="#LOSER",
        side_b_player_id="#WINNER",
    )
    result = finalize_event_group((first, conflict))
    assert {row.state for row in result} == {RecordState.QUARANTINED}
    assert all(row.issues == (RecordIssue.CONFLICTING_BATTLE,) for row in result)
    assert all(row.representative_archive_member is None for row in result)
    assert all(row.representative_row_number is None for row in result)


def test_duplicate_outside_population_accepts_no_valid_row() -> None:
    earlier = staged_row(
        mode="Ladder",
        observation_issues=(RecordIssue.UNSUPPORTED_MODE,),
    )
    later = staged_row(
        archive_member="b.parquet",
        row_number=2,
        mode="Ladder",
        observation_issues=(RecordIssue.UNSUPPORTED_MODE,),
    )
    result = finalize_event_group((later, earlier))
    by_location = {(row.archive_member, row.row_number): row for row in result}
    representative = by_location[("a.parquet", 0)]
    assert representative.state is RecordState.UNSUPPORTED
    assert representative.issues == (RecordIssue.UNSUPPORTED_MODE,)
    duplicate = by_location[("b.parquet", 2)]
    assert duplicate.state is RecordState.QUARANTINED
    assert duplicate.issues == (RecordIssue.DUPLICATE_BATTLE, RecordIssue.UNSUPPORTED_MODE)
    assert not any(row.state is RecordState.VALID for row in result)


def test_conflict_retains_observation_issues_in_sorted_order() -> None:
    first = staged_row(
        observation_issues=(RecordIssue.NON_MAX_CARD_LEVEL, RecordIssue.UNSUPPORTED_MODE),
        mode="Ladder",
    )
    conflict = staged_row(
        archive_member="b.parquet",
        fingerprint=FINGERPRINT_B,
        observation_issues=(RecordIssue.NON_MAX_CARD_LEVEL, RecordIssue.UNSUPPORTED_MODE),
        mode="Ladder",
    )
    result = finalize_event_group((conflict, first))
    expected = (
        RecordIssue.CONFLICTING_BATTLE,
        RecordIssue.NON_MAX_CARD_LEVEL,
        RecordIssue.UNSUPPORTED_MODE,
    )
    assert all(row.issues == expected for row in result)


def test_unadaptable_row_passthrough_copies_state_without_identity() -> None:
    row = UnadaptableRow(
        archive_member="a.parquet",
        row_number=3,
        state=RecordState.QUARANTINED,
        issues=(RecordIssue.IDENTICAL_PLAYERS,),
    )
    disposition = disposition_from_unadaptable(row)
    assert disposition.state is RecordState.QUARANTINED
    assert disposition.issues == (RecordIssue.IDENTICAL_PLAYERS,)
    assert disposition.event_key is None
    assert disposition.representative_archive_member is None


@given(st.permutations((0, 1, 2)))
def test_source_order_permutation_keeps_earliest_duplicate_representative(
    order: tuple[int, ...],
) -> None:
    rows = (
        staged_row(archive_member="c.parquet", row_number=9),
        staged_row(archive_member="a.parquet", row_number=3),
        staged_row(archive_member="b.parquet", row_number=0),
    )
    result = finalize_event_group(tuple(rows[index] for index in order))
    by_location = {(row.archive_member, row.row_number): row for row in result}
    assert by_location[("a.parquet", 3)].state is RecordState.VALID
    assert by_location[("b.parquet", 0)].issues == (RecordIssue.DUPLICATE_BATTLE,)
    assert by_location[("c.parquet", 9)].representative_row_number == 3


@given(st.sampled_from((FINGERPRINT_A, FINGERPRINT_B)))
def test_side_swapped_conflict_has_no_accepted_representative(left_fingerprint: str) -> None:
    right_fingerprint = FINGERPRINT_B if left_fingerprint == FINGERPRINT_A else FINGERPRINT_A
    left = staged_row(fingerprint=left_fingerprint)
    right = staged_row(
        archive_member="b.parquet",
        fingerprint=right_fingerprint,
        side_a_player_id="#LOSER",
        side_b_player_id="#WINNER",
    )
    result = finalize_event_group((right, left))
    assert all(row.state is RecordState.QUARANTINED for row in result)
    assert all(RecordIssue.CONFLICTING_BATTLE in row.issues for row in result)
    assert all(row.representative_archive_member is None for row in result)


def _write_workspace(
    directory: Path,
    *,
    staged: tuple[StagedBattleRow, ...] = (),
    unadaptable: tuple[UnadaptableRow, ...] = (),
) -> Path:
    workspace = directory / "workspace"
    (workspace / "staging").mkdir(parents=True)
    (workspace / "unadaptable").mkdir(parents=True)
    if staged:
        write_staged_part(
            workspace / "staging" / "a" / "part-00000.parquet",
            staged,
            row_group_rows=131072,
        )
    if unadaptable:
        write_unadaptable_part(
            workspace / "unadaptable" / "a" / "part-00000.parquet",
            unadaptable,
            row_group_rows=131072,
        )
    return workspace


def _read_disposition_rows(output_dir: Path) -> list[tuple[str, int, str, list[str], str | None]]:
    paths = sorted((output_dir / "dispositions").glob("*.parquet"))
    connection = duckdb.connect()
    try:
        query = " UNION ALL ".join(
            "SELECT archive_member, row_number, state, issues, event_key FROM read_parquet(?)"
            for _ in paths
        )
        rows = connection.execute(query, [str(path) for path in paths]).fetchall()
        return [
            (str(member), int(number), str(state), list(issues), None if key is None else str(key))
            for member, number, state, issues, key in rows
        ]
    finally:
        connection.close()


def test_duckdb_grouping_matches_python_rules_and_passthrough(tmp_path: Path) -> None:
    staged = (
        staged_row(),
        staged_row(archive_member="b.parquet", row_number=4),
        staged_row(
            archive_member="c.parquet",
            row_number=1,
            event_key="d" * 64,
            fingerprint="e" * 64,
            mode="Ladder",
            observation_issues=(RecordIssue.UNSUPPORTED_MODE,),
        ),
    )
    unadaptable = (
        UnadaptableRow(
            archive_member="a.parquet",
            row_number=3,
            state=RecordState.INVALID,
            issues=(RecordIssue.MALFORMED_ROW,),
        ),
    )
    workspace = _write_workspace(tmp_path, staged=staged, unadaptable=unadaptable)
    result = group_staged_dataset(
        workspace,
        workspace,
        source_row_count=4,
        config=StagingConfig(threads=1, memory_limit="256MB"),
        temp_directory=tmp_path / "tmp",
    )
    expected = {
        (row.archive_member, row.row_number): (
            row.state.value,
            [issue.value for issue in row.issues],
        )
        for row in (
            *finalize_event_group(staged[:2]),
            *finalize_event_group((staged[2],)),
            disposition_from_unadaptable(unadaptable[0]),
        )
    }
    observed = {
        (member, number): (state, issues)
        for member, number, state, issues, _key in _read_disposition_rows(workspace)
    }
    assert observed == expected
    assert result.summary.source_row_count == 4
    assert result.summary.duplicate_row_count == 1
    files = {path.name for path in result.disposition_files}
    assert files == {"a.parquet", "b.parquet", "c.parquet"}
    connection = duckdb.connect()
    try:
        mixed = next(path for path in result.disposition_files if path.name == "a.parquet")
        columns = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(mixed)]
        ).fetchall()
        assert [str(row[0]) for row in columns] == [column.name for column in DISPOSITION_SCHEMA]
    finally:
        connection.close()


def test_grouping_fails_closed_on_count_mismatch(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path, staged=(staged_row(),))
    with pytest.raises(KaggleV6GroupingError, match="source_row_count"):
        group_staged_dataset(
            workspace,
            workspace,
            source_row_count=2,
            config=StagingConfig(threads=1, memory_limit="256MB"),
            temp_directory=tmp_path / "tmp",
        )
    assert not (workspace / "dispositions").exists()
    assert list((workspace / "staging").rglob("*.parquet"))


def test_grouping_refuses_existing_disposition_output(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path, staged=(staged_row(),))
    (workspace / "dispositions").mkdir()
    with pytest.raises(KaggleV6GroupingError, match="already exists"):
        group_staged_dataset(
            workspace,
            workspace,
            source_row_count=1,
            config=StagingConfig(threads=1, memory_limit="256MB"),
            temp_directory=tmp_path / "tmp",
        )
