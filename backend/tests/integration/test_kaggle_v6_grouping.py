import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import duckdb

from clash_sos.application.dataset_grouping import group_staged_dataset
from clash_sos.application.dataset_staging import StagingConfig, stage_kaggle_v6
from clash_sos.domain.canonical import RecordIssue, RecordState
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.schema import CARD_COLUMNS, LEVEL_COLUMNS


def kaggle_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "winner_id": "#AAA",
        "loser_id": "#BBB",
        "time": datetime(2026, 6, 21, 16, tzinfo=UTC),
        "game_mode": "Ranked1v1_NewArena",
    }
    for index, name in enumerate(CARD_COLUMNS):
        row[name] = index
        row[f"{name}_level"] = 16
    row.update(overrides)
    return row


ROW_COLUMNS = (
    "winner_id",
    "loser_id",
    "time",
    "game_mode",
    *CARD_COLUMNS,
    *LEVEL_COLUMNS,
)


def write_kaggle_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "winner_id VARCHAR",
        "loser_id VARCHAR",
        "time TIMESTAMPTZ",
        "game_mode VARCHAR",
        *(f"{name} UTINYINT" for name in CARD_COLUMNS),
        *(f"{name} UTINYINT" for name in LEVEL_COLUMNS),
    ]
    connection = duckdb.connect()
    try:
        connection.execute(f"CREATE TABLE battles ({', '.join(columns)})")
        placeholders = ", ".join("?" for _ in ROW_COLUMNS)
        for row in rows:
            values = [row[name] for name in ROW_COLUMNS]
            connection.execute(f"INSERT INTO battles VALUES ({placeholders})", values)
        connection.execute("COPY battles TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()


def swapped_sides(row: dict[str, object]) -> dict[str, object]:
    swapped = dict(row)
    swapped["winner_id"] = row["loser_id"]
    swapped["loser_id"] = row["winner_id"]
    for index in range(8):
        swapped[f"winner_card_{index}"] = row[f"loser_card_{index}"]
        swapped[f"loser_card_{index}"] = row[f"winner_card_{index}"]
        swapped[f"winner_card_{index}_level"] = row[f"loser_card_{index}_level"]
        swapped[f"loser_card_{index}_level"] = row[f"winner_card_{index}_level"]
    return swapped


def create_grouping_archive(directory: Path) -> Path:
    conflict = kaggle_row(
        time=datetime(2026, 6, 22, 12, tzinfo=UTC),
        winner_id="#CCC",
        loser_id="#DDD",
    )
    a_path = directory / "a.parquet"
    b_path = directory / "b.parquet"
    write_kaggle_parquet(
        a_path,
        [
            kaggle_row(),
            kaggle_row(game_mode="Ladder"),
            kaggle_row(loser_card_7_level=15),
            kaggle_row(time=None),
            kaggle_row(winner_id="#SAME", loser_id="#SAME"),
            kaggle_row(winner_card_0_level=0),
            conflict,
        ],
    )
    write_kaggle_parquet(
        b_path,
        [
            kaggle_row(winner_card_0=176),
            kaggle_row(winner_card_0=None),
            kaggle_row(winner_card_1=0),
            kaggle_row(time=datetime(2026, 6, 1, 12, tzinfo=UTC)),
            kaggle_row(),
            swapped_sides(conflict),
        ],
    )
    mapping = directory / "cardToID.json"
    mapping.write_text(
        json.dumps({entry.source_name: entry.source_id for entry in KAGGLE_V6_CARDS.entries}),
        encoding="utf-8",
    )
    archive_path = directory / "source.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.write(b_path, "b.parquet")
        archive.write(a_path, "a.parquet")
        archive.write(mapping, "cardToID.json")
    return archive_path


def read_ledger(output_dir: Path) -> list[tuple[str, int, str, list[str]]]:
    paths = sorted((output_dir / "dispositions").glob("*.parquet"))
    connection = duckdb.connect()
    try:
        query = " UNION ALL ".join(
            "SELECT archive_member, row_number, state, issues FROM read_parquet(?)" for _ in paths
        )
        rows = connection.execute(query, [str(path) for path in paths]).fetchall()
        return [
            (str(member), int(number), str(state), list(issues))
            for member, number, state, issues in rows
        ]
    finally:
        connection.close()


def test_stage_then_group_reconciles_synthetic_archive(tmp_path: Path) -> None:
    archive_path = create_grouping_archive(tmp_path)
    workspace = tmp_path / "workspace"
    config = StagingConfig(batch_rows=2, threads=1, memory_limit="256MB")
    staged = stage_kaggle_v6(
        archive_path,
        workspace,
        temp_directory=tmp_path / "stage-tmp",
        config=config,
    )
    grouped = group_staged_dataset(
        workspace,
        workspace,
        source_row_count=staged.source_row_count,
        config=config,
        temp_directory=tmp_path / "group-tmp",
    )

    assert staged.source_row_count == 13
    assert grouped.row_count == 13
    assert grouped.summary.source_row_count == 13
    state_map = {count.state: count.count for count in grouped.summary.states}
    issue_map = {count.issue: count.count for count in grouped.summary.issues}
    assert sum(state_map.values()) == 13
    assert state_map[RecordState.VALID] == 1
    assert state_map[RecordState.UNSUPPORTED] == 3
    assert state_map[RecordState.INVALID] == 2
    assert state_map[RecordState.QUARANTINED] == 7
    assert state_map[RecordState.INSUFFICIENT_DATA] == 0
    assert issue_map[RecordIssue.DUPLICATE_BATTLE] == 1
    assert issue_map[RecordIssue.CONFLICTING_BATTLE] == 2
    assert grouped.summary.duplicate_row_count == 1
    assert grouped.summary.conflict_row_count == 2

    ledger = {
        (member, number): (state, issues)
        for member, number, state, issues in read_ledger(workspace)
    }
    assert len(ledger) == 13
    assert ledger[("a.parquet", 0)] == ("valid", [])
    assert ledger[("b.parquet", 4)][0] == "quarantined"
    assert "duplicate_battle" in ledger[("b.parquet", 4)][1]
    assert ledger[("a.parquet", 1)] == ("unsupported", ["unsupported_mode"])
    assert "non_max_card_level" in ledger[("a.parquet", 2)][1]
    assert ledger[("b.parquet", 3)][0] == "unsupported"
    assert "stale_balance_era" in ledger[("b.parquet", 3)][1]
    assert ledger[("a.parquet", 6)][0] == "quarantined"
    assert ledger[("b.parquet", 5)][0] == "quarantined"
    assert "conflicting_battle" in ledger[("a.parquet", 6)][1]
    assert "conflicting_battle" in ledger[("b.parquet", 5)][1]
    assert not any(
        state == "valid" and "conflicting_battle" in issues for _, (state, issues) in ledger.items()
    )
    assert not any(state == "valid" and issues for _, (state, issues) in ledger.items())


def test_duplicate_representative_is_valid_only_without_observation_issues(tmp_path: Path) -> None:
    a_path = tmp_path / "a.parquet"
    b_path = tmp_path / "b.parquet"
    write_kaggle_parquet(a_path, [kaggle_row(game_mode="Ladder")])
    write_kaggle_parquet(b_path, [kaggle_row(game_mode="Ladder")])
    mapping = tmp_path / "cardToID.json"
    mapping.write_text(
        json.dumps({entry.source_name: entry.source_id for entry in KAGGLE_V6_CARDS.entries}),
        encoding="utf-8",
    )
    archive_path = tmp_path / "source.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.write(b_path, "b.parquet")
        archive.write(a_path, "a.parquet")
        archive.write(mapping, "cardToID.json")
    workspace = tmp_path / "workspace"
    config = StagingConfig(batch_rows=2, threads=1, memory_limit="256MB")
    staged = stage_kaggle_v6(
        archive_path,
        workspace,
        temp_directory=tmp_path / "tmp",
        config=config,
    )
    grouped = group_staged_dataset(
        workspace,
        workspace,
        source_row_count=staged.source_row_count,
        config=config,
        temp_directory=tmp_path / "group-tmp",
    )
    state_map = {count.state: count.count for count in grouped.summary.states}
    assert state_map[RecordState.VALID] == 0
    assert state_map[RecordState.UNSUPPORTED] == 1
    assert state_map[RecordState.QUARANTINED] == 1
    ledger = {
        (member, number): (state, issues)
        for member, number, state, issues in read_ledger(workspace)
    }
    assert ledger[("a.parquet", 0)][0] == "unsupported"
    assert "unsupported_mode" in ledger[("a.parquet", 0)][1]
    assert ledger[("b.parquet", 0)][0] == "quarantined"
    assert "duplicate_battle" in ledger[("b.parquet", 0)][1]
