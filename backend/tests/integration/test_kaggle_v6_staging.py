import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import duckdb
import pytest

from clash_sos.application.dataset_staging import StagingConfig, stage_kaggle_v6
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS
from clash_sos.infrastructure.kaggle_v6.schema import CARD_COLUMNS, LEVEL_COLUMNS
from clash_sos.infrastructure.kaggle_v6.source import (
    KAGGLE_V6_ARCHIVE_SHA256,
    KAGGLE_V6_ARCHIVE_SIZE,
)
from clash_sos.infrastructure.kaggle_v6.staging_io import (
    KaggleV6StagingError,
    list_parquet_members,
)


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


def create_staging_archive(directory: Path, *, incompatible_schema: bool = False) -> Path:
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
        ],
    )
    if incompatible_schema:
        incompatible = directory / "bad.parquet"
        columns = [
            "winner_id VARCHAR",
            "loser_id VARCHAR",
            "time TIMESTAMPTZ",
            "game_mode VARCHAR",
            *(f"{name} USMALLINT" for name in CARD_COLUMNS),
            *(f"{name} UTINYINT" for name in LEVEL_COLUMNS),
        ]
        connection = duckdb.connect()
        try:
            connection.execute(f"CREATE TABLE battles ({', '.join(columns)})")
            values = [kaggle_row()[name] for name in ROW_COLUMNS]
            placeholders = ", ".join("?" for _ in values)
            connection.execute(f"INSERT INTO battles VALUES ({placeholders})", values)
            connection.execute("COPY battles TO ? (FORMAT PARQUET)", [str(incompatible)])
        finally:
            connection.close()
        member_paths = [incompatible, a_path, b_path]
    else:
        member_paths = [a_path, b_path]
    mapping = directory / "cardToID.json"
    mapping.write_text(
        json.dumps({entry.source_name: entry.source_id for entry in KAGGLE_V6_CARDS.entries}),
        encoding="utf-8",
    )
    archive_path = directory / "source.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        if incompatible_schema:
            archive.write(member_paths[0], member_paths[0].name)
        archive.write(b_path, "b.parquet")
        archive.write(a_path, "a.parquet")
        archive.write(mapping, "cardToID.json")
    return archive_path


def read_staged_rows(workspace: Path) -> list[tuple[str, int, str, list[str]]]:
    connection = duckdb.connect()
    try:
        paths = sorted((workspace / "staging").rglob("*.parquet"))
        query = " UNION ALL ".join(
            "SELECT archive_member, row_number, mode, observation_issues FROM read_parquet(?)"
            for _ in paths
        )
        rows = connection.execute(query, [str(path) for path in paths]).fetchall()
        return [(str(a), int(b), str(c), list(d)) for a, b, c, d in rows]
    finally:
        connection.close()


def read_unadaptable_rows(
    workspace: Path,
) -> list[tuple[str, int, str, list[str]]]:
    connection = duckdb.connect()
    try:
        paths = sorted((workspace / "unadaptable").rglob("*.parquet"))
        query = " UNION ALL ".join(
            "SELECT archive_member, row_number, state, issues FROM read_parquet(?)" for _ in paths
        )
        rows = connection.execute(query, [str(path) for path in paths]).fetchall()
        return [(str(a), int(b), str(c), list(d)) for a, b, c, d in rows]
    finally:
        connection.close()


def test_stage_synthetic_archive_routes_and_reconciles(tmp_path: Path) -> None:
    archive_path = create_staging_archive(tmp_path)
    workspace = tmp_path / "workspace"
    config = StagingConfig(batch_rows=2, threads=1, memory_limit="256MB")

    result = stage_kaggle_v6(
        archive_path,
        workspace,
        temp_directory=tmp_path / "work",
        config=config,
    )

    assert result.source_row_count == 11
    assert result.staged_row_count == 5
    assert result.unadaptable_row_count == 6
    assert result.members == ("a.parquet", "b.parquet")

    staged = read_staged_rows(workspace)
    assert staged == sorted(staged, key=lambda row: (row[0], row[1]))
    assert [row[1] for row in staged if row[0] == "a.parquet"] == [0, 1, 2]
    assert [row[1] for row in staged if row[0] == "b.parquet"] == [3, 4]

    ladder = next(row for row in staged if row[1] == 1)
    assert ladder[2] == "Ladder"
    assert "unsupported_mode" in ladder[3]

    stale = next(row for row in staged if row[0] == "b.parquet" and row[1] == 3)
    assert "stale_balance_era" in stale[3]

    low_level = next(row for row in staged if row[1] == 2)
    assert "non_max_card_level" in low_level[3]

    unadaptable = {(row[0], row[1]): (row[2], row[3]) for row in read_unadaptable_rows(workspace)}
    assert unadaptable[("a.parquet", 3)] == ("invalid", ["malformed_timestamp"])
    assert unadaptable[("a.parquet", 4)] == ("quarantined", ["identical_players"])
    assert unadaptable[("a.parquet", 5)] == ("invalid", ["malformed_row"])
    assert unadaptable[("b.parquet", 0)] == ("quarantined", ["unknown_card"])
    assert unadaptable[("b.parquet", 1)] == ("quarantined", ["incomplete_deck"])
    assert unadaptable[("b.parquet", 2)] == ("quarantined", ["repeated_card"])

    connection = duckdb.connect()
    try:
        unadaptable_path = next((workspace / "unadaptable").rglob("*.parquet"))
        columns = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(unadaptable_path)]
        ).fetchall()
        assert [str(row[0]) for row in columns] == [
            "archive_member",
            "row_number",
            "state",
            "issues",
        ]
    finally:
        connection.close()

    staging_parts = sorted((workspace / "staging" / "a").glob("part-*.parquet"))
    assert [part.name for part in staging_parts] == ["part-00000.parquet", "part-00001.parquet"]


def test_stage_repeated_run_matches_logical_hashes(tmp_path: Path) -> None:
    archive_path = create_staging_archive(tmp_path)
    config = StagingConfig(batch_rows=2, threads=1, memory_limit="256MB")
    first = stage_kaggle_v6(
        archive_path,
        tmp_path / "first",
        temp_directory=tmp_path / "work-a",
        config=config,
    )
    second = stage_kaggle_v6(
        archive_path,
        tmp_path / "second",
        temp_directory=tmp_path / "work-b",
        config=config,
    )
    assert first.logical_staging_sha256 == second.logical_staging_sha256
    assert first.logical_unadaptable_sha256 == second.logical_unadaptable_sha256


def test_extracted_member_is_released_before_next_member(tmp_path: Path) -> None:
    archive_path = create_staging_archive(tmp_path)
    released: list[Path] = []

    @contextmanager
    def tracking_extracted(
        archive: ZipFile, member: ZipInfo, work_directory: Path, chunk_size: int
    ) -> Generator[tuple[Path, str]]:
        from clash_sos.infrastructure.kaggle_v6.audit_io import extracted_member as real_extracted

        with real_extracted(archive, member, work_directory, chunk_size) as extraction:
            path, digest = extraction
            if str(path).endswith(".parquet"):
                for previous in released:
                    assert not previous.exists()
                released.append(path)
            yield path, digest

    with patch(
        "clash_sos.application.dataset_staging.extracted_member",
        side_effect=tracking_extracted,
    ):
        stage_kaggle_v6(
            archive_path,
            tmp_path / "workspace",
            temp_directory=tmp_path / "work",
            config=StagingConfig(batch_rows=2, threads=1, memory_limit="256MB"),
        )


def test_stage_failure_removes_workspace(tmp_path: Path) -> None:
    from clash_sos.infrastructure.kaggle_v6 import staging_io

    archive_path = create_staging_archive(tmp_path)
    workspace = tmp_path / "workspace"
    original = staging_io.write_staged_part

    def failing_write(path: Path, rows: object, **kwargs: object) -> None:
        original(path, rows, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("injected")

    with (
        patch(
            "clash_sos.application.dataset_staging.write_staged_part",
            side_effect=failing_write,
        ),
        pytest.raises(RuntimeError, match="injected"),
    ):
        stage_kaggle_v6(
            archive_path,
            workspace,
            temp_directory=tmp_path / "work",
            config=StagingConfig(batch_rows=2, threads=1, memory_limit="256MB"),
        )
    assert not workspace.exists()
    assert not list(tmp_path.glob("clash-sos-stage-*"))


def test_unknown_member_name_fails(tmp_path: Path) -> None:
    archive_path = create_staging_archive(tmp_path)
    workspace = tmp_path / "workspace"
    with pytest.raises(KaggleV6StagingError, match="unknown"):
        stage_kaggle_v6(
            archive_path,
            workspace,
            temp_directory=tmp_path / "work",
            members=("missing.parquet",),
            config=StagingConfig(threads=1, memory_limit="256MB"),
        )
    assert not workspace.exists()


def test_existing_workspace_fails(tmp_path: Path) -> None:
    archive_path = create_staging_archive(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sentinel = workspace / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(KaggleV6StagingError, match="already exists"):
        stage_kaggle_v6(
            archive_path,
            workspace,
            temp_directory=tmp_path / "work",
            config=StagingConfig(threads=1, memory_limit="256MB"),
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_incompatible_schema_fails(tmp_path: Path) -> None:
    archive_path = create_staging_archive(tmp_path, incompatible_schema=True)
    workspace = tmp_path / "workspace"
    with pytest.raises(KaggleV6StagingError, match="incompatible schema"):
        stage_kaggle_v6(
            archive_path,
            workspace,
            temp_directory=tmp_path / "work",
            members=("bad.parquet",),
            config=StagingConfig(threads=1, memory_limit="256MB"),
        )
    assert not workspace.exists()


def test_representative_member_throughput(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = os.environ.get("CLASH_SOS_KAGGLE_ARCHIVE")
    manifest = Path(
        os.environ.get("CLASH_SOS_KAGGLE_MANIFEST", "data/metadata/kaggle-v6-dataset.json")
    )
    if not archive:
        pytest.skip("set CLASH_SOS_KAGGLE_ARCHIVE to run the representative member checkpoint")
    if not manifest.is_file():
        pytest.skip("run clash-sos dataset audit-kaggle-v6 first so a raw manifest exists")
    archive_path = Path(archive)
    with ZipFile(archive_path) as zip_file:
        first = list_parquet_members(zip_file)[0]
    result = stage_kaggle_v6(
        archive_path,
        tmp_path / "workspace",
        temp_directory=tmp_path / "tmp",
        members=(first,),
        raw_manifest_path=manifest,
        expected_archive_size=KAGGLE_V6_ARCHIVE_SIZE,
        expected_archive_sha256=KAGGLE_V6_ARCHIVE_SHA256,
    )
    projected_runtime = 23 * (
        result.source_row_count / result.rows_per_second if result.rows_per_second else 0.0
    )
    projected_disk = 23 * result.temp_disk_bytes
    print(
        f"member={first} rows={result.source_row_count} rps={result.rows_per_second:.1f} "
        f"temp_disk_bytes={result.temp_disk_bytes} projected_full_seconds={projected_runtime:.0f} "
        f"projected_staging_disk_bytes={projected_disk}"
    )
    assert result.source_row_count == result.staged_row_count + result.unadaptable_row_count
    assert result.source_row_count > 0
