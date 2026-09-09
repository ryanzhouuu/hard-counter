from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import duckdb
import pytest

from clash_sos.application.dataset_grouping import group_staged_dataset
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.canonical import BattleOutcome, RecordIssue, RecordState
from clash_sos.domain.canonical_dataset import CANONICAL_SCHEMA, deck_content_hash
from clash_sos.domain.processed_manifest import DEFAULT_DATASET_VERSION
from clash_sos.domain.staged_dataset import StagedBattleRow, UnadaptableRow
from clash_sos.infrastructure.kaggle_v6.grouping_io import list_parquet_files
from clash_sos.infrastructure.kaggle_v6.materialize_io import (
    KaggleV6MaterializeError,
    read_canonical_schema,
    write_canonical_parquet,
)
from clash_sos.infrastructure.kaggle_v6.source import KAGGLE_V6_SOURCE_ID
from clash_sos.infrastructure.kaggle_v6.staging_io import (
    connect_staging_duckdb,
    write_staged_part,
    write_unadaptable_part,
)

EVENT_KEY = "a" * 64
FINGERPRINT_A = "b" * 64
FINGERPRINT_B = "c" * 64
FINGERPRINT_C = "d" * 64
CARD_IDS = ("card-3", "card-1", "card-7", "card-0", "card-2", "card-4", "card-6", "card-5")
CARD_FORMS = ("base",) * 8
CARD_LEVELS = (16,) * 8
DECK_HASH = deck_content_hash(CARD_IDS, CARD_FORMS, CARD_LEVELS)
CONFIG = StagingConfig(threads=1, memory_limit="256MB")


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


def _write_workspace(
    directory: Path,
    *,
    staged: tuple[StagedBattleRow, ...] = (),
    unadaptable: tuple[UnadaptableRow, ...] = (),
) -> Path:
    workspace = directory / "workspace"
    (workspace / "staging").mkdir(parents=True)
    (workspace / "unadaptable").mkdir(parents=True)
    staged_by_member: dict[str, list[StagedBattleRow]] = {}
    for row in staged:
        staged_by_member.setdefault(row.archive_member, []).append(row)
    for member, rows in staged_by_member.items():
        write_staged_part(
            workspace / "staging" / PurePosixPath(member).stem / "part-00000.parquet",
            rows,
            row_group_rows=131072,
        )
    unadaptable_by_member: dict[str, list[UnadaptableRow]] = {}
    for row in unadaptable:
        unadaptable_by_member.setdefault(row.archive_member, []).append(row)
    for member, rows in unadaptable_by_member.items():
        write_unadaptable_part(
            workspace / "unadaptable" / PurePosixPath(member).stem / "part-00000.parquet",
            rows,
            row_group_rows=131072,
        )
    return workspace


def _group(workspace: Path, tmp_path: Path, row_count: int) -> None:
    group_staged_dataset(
        workspace,
        workspace,
        source_row_count=row_count,
        config=CONFIG,
        temp_directory=tmp_path / "group-tmp",
    )


def _write_canonical(workspace: Path, tmp_path: Path, output: Path) -> int:
    connection = connect_staging_duckdb(
        memory_limit=CONFIG.memory_limit,
        threads=1,
        temp_directory=tmp_path / "materialize-tmp",
    )
    try:
        return write_canonical_parquet(
            connection,
            staging_files=list_parquet_files(workspace / "staging"),
            disposition_files=list_parquet_files(workspace / "dispositions"),
            output_path=output,
            dataset_version=DEFAULT_DATASET_VERSION,
            row_group_rows=CONFIG.parquet_row_group_rows,
        )
    finally:
        connection.close()


def test_write_canonical_schema_and_zstd_compression(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path, staged=(staged_row(),))
    _group(workspace, tmp_path, 1)
    output = workspace / "canonical.parquet"
    count = _write_canonical(workspace, tmp_path, output)
    assert count == 1
    connection = duckdb.connect()
    try:
        schema = read_canonical_schema(connection, output)
        observed = tuple((column.name, column.physical_type) for column in schema)
        expected = tuple((column.name, column.physical_type) for column in CANONICAL_SCHEMA)
        assert observed == expected
        meta = connection.execute(
            """
            SELECT COALESCE(SUM(stats_null_count), 0),
                   LIST(DISTINCT compression ORDER BY compression)
            FROM parquet_metadata(?)
            """,
            [str(output)],
        ).fetchone()
        assert meta == (0, ["ZSTD"])
    finally:
        connection.close()


def test_write_canonical_sorts_and_keeps_only_valid_rows(tmp_path: Path) -> None:
    later = staged_row(
        timestamp=datetime(2026, 6, 22, 12, tzinfo=UTC),
        event_key="e" * 64,
        fingerprint=FINGERPRINT_C,
        archive_member="b.parquet",
        row_number=1,
        side_b_player_id="#LATER",
    )
    duplicate = staged_row(archive_member="b.parquet", row_number=4)
    unsupported = staged_row(
        archive_member="c.parquet",
        row_number=2,
        event_key="f" * 64,
        fingerprint="e" * 64,
        mode="Ladder",
        observation_issues=(RecordIssue.UNSUPPORTED_MODE,),
        side_b_player_id="#LADDER",
    )
    conflict_a = staged_row(
        archive_member="c.parquet",
        row_number=3,
        event_key="1" * 64,
        fingerprint=FINGERPRINT_A,
        side_b_player_id="#CA",
    )
    conflict_b = staged_row(
        archive_member="c.parquet",
        row_number=4,
        event_key="1" * 64,
        fingerprint=FINGERPRINT_B,
        side_a_player_id="#LOSER",
        side_b_player_id="#WINNER",
    )
    unadaptable = UnadaptableRow(
        archive_member="a.parquet",
        row_number=9,
        state=RecordState.INVALID,
        issues=(RecordIssue.MALFORMED_ROW,),
    )
    workspace = _write_workspace(
        tmp_path,
        staged=(later, staged_row(), duplicate, unsupported, conflict_a, conflict_b),
        unadaptable=(unadaptable,),
    )
    _group(workspace, tmp_path, 7)
    output = workspace / "canonical.parquet"
    count = _write_canonical(workspace, tmp_path, output)
    assert count == 2
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            """
            SELECT archive_member, row_number, fingerprint, dataset_version
            FROM read_parquet(?)
            """,
            [str(output)],
        ).fetchall()
    finally:
        connection.close()
    assert rows == [
        ("a.parquet", 0, FINGERPRINT_A, DEFAULT_DATASET_VERSION),
        ("b.parquet", 1, FINGERPRINT_C, DEFAULT_DATASET_VERSION),
    ]


def test_write_canonical_preserves_deck_order_and_hashes(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path, staged=(staged_row(),))
    _group(workspace, tmp_path, 1)
    output = workspace / "canonical.parquet"
    _write_canonical(workspace, tmp_path, output)
    connection = duckdb.connect()
    try:
        row = connection.execute(
            """
            SELECT side_a_card_ids, side_a_card_forms, side_a_card_levels,
                   side_a_deck_hash, side_b_deck_hash
            FROM read_parquet(?)
            """,
            [str(output)],
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    card_ids = tuple(str(value) for value in row[0])
    forms = tuple(str(value) for value in row[1])
    levels = tuple(int(value) for value in row[2])
    hashes = (str(row[3]), str(row[4]))
    expected = (CARD_IDS, CARD_FORMS, CARD_LEVELS, (DECK_HASH, DECK_HASH))
    assert (card_ids, forms, levels, hashes) == expected
    assert hashes[0] == deck_content_hash(card_ids, forms, levels)


def test_write_canonical_refuses_existing_output(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path, staged=(staged_row(),))
    _group(workspace, tmp_path, 1)
    output = workspace / "canonical.parquet"
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(KaggleV6MaterializeError, match="already exists"):
        _write_canonical(workspace, tmp_path, output)
    assert output.read_text(encoding="utf-8") == "sentinel"
    assert list((workspace / "staging").rglob("*.parquet"))
