from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from clash_sos.application.dataset_splits import write_temporal_split
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.dataset_splits import SPLIT_SCHEMA, assign_temporal_partition
from clash_sos.infrastructure.kaggle_v6.split_io import KaggleV6SplitError

TRAIN_END = datetime(2026, 6, 10, tzinfo=UTC)
VALIDATION_END = datetime(2026, 6, 20, tzinfo=UTC)
FINGERPRINT = "a" * 64
CONFIG = StagingConfig(threads=1, memory_limit="256MB")


def write_identity_parquet(path: Path, rows: tuple[tuple[datetime, str, str, int], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE identities (
                timestamp TIMESTAMPTZ,
                fingerprint VARCHAR,
                archive_member VARCHAR,
                row_number BIGINT
            )
            """
        )
        connection.executemany("INSERT INTO identities VALUES (?, ?, ?, ?)", list(rows))
        connection.execute("COPY identities TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()


def test_assign_temporal_partition_uses_half_open_cutovers() -> None:
    before = TRAIN_END - timedelta(seconds=1)
    assert (
        assign_temporal_partition(before, train_end=TRAIN_END, validation_end=VALIDATION_END)
        == "train"
    )
    assert (
        assign_temporal_partition(TRAIN_END, train_end=TRAIN_END, validation_end=VALIDATION_END)
        == "validation"
    )
    assert (
        assign_temporal_partition(
            VALIDATION_END, train_end=TRAIN_END, validation_end=VALIDATION_END
        )
        == "test"
    )


def test_write_temporal_split_assigns_boundaries_and_schema(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    write_identity_parquet(
        canonical,
        (
            (datetime(2026, 6, 9, 23, tzinfo=UTC), FINGERPRINT, "b.parquet", 1),
            (TRAIN_END, "b" * 64, "a.parquet", 0),
            (VALIDATION_END, "c" * 64, "a.parquet", 2),
        ),
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    result = write_temporal_split(
        canonical,
        first,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
        config=CONFIG,
        temp_directory=tmp_path / "tmp",
    )
    write_temporal_split(
        canonical,
        second,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
        config=CONFIG,
        temp_directory=tmp_path / "tmp-b",
    )
    assert first.read_bytes() == second.read_bytes()
    assert [partition.partition for partition in result.partitions] == [
        "train",
        "validation",
        "test",
    ]
    assert [partition.row_count for partition in result.partitions] == [1, 1, 1]
    connection = duckdb.connect()
    try:
        columns = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(first)]
        ).fetchall()
        rows = connection.execute(
            """
            SELECT archive_member, row_number, partition
            FROM read_parquet(?)
            ORDER BY timestamp, fingerprint, archive_member, row_number
            """,
            [str(first)],
        ).fetchall()
        names = [str(row[0]) for row in columns]
        types = [str(row[1]) for row in columns]
    finally:
        connection.close()
    assert names == [column.name for column in SPLIT_SCHEMA]
    assert types == [column.physical_type for column in SPLIT_SCHEMA]
    assert rows == [
        ("b.parquet", 1, "train"),
        ("a.parquet", 0, "validation"),
        ("a.parquet", 2, "test"),
    ]


def test_write_temporal_split_rejects_empty_partition(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    write_identity_parquet(
        canonical,
        (
            (TRAIN_END, FINGERPRINT, "a.parquet", 0),
            (VALIDATION_END, "b" * 64, "a.parquet", 1),
        ),
    )
    with pytest.raises(KaggleV6SplitError, match="empty"):
        write_temporal_split(
            canonical,
            tmp_path / "splits.parquet",
            train_end=TRAIN_END,
            validation_end=VALIDATION_END,
            config=CONFIG,
            temp_directory=tmp_path / "tmp",
        )


def test_write_temporal_split_rejects_cutover_outside_range(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    write_identity_parquet(
        canonical,
        (
            (datetime(2026, 6, 12, tzinfo=UTC), FINGERPRINT, "a.parquet", 0),
            (datetime(2026, 6, 18, tzinfo=UTC), "b" * 64, "a.parquet", 1),
            (datetime(2026, 6, 22, tzinfo=UTC), "c" * 64, "a.parquet", 2),
        ),
    )
    with pytest.raises(KaggleV6SplitError, match="outside"):
        write_temporal_split(
            canonical,
            tmp_path / "splits.parquet",
            train_end=datetime(2026, 6, 1, tzinfo=UTC),
            validation_end=VALIDATION_END,
            config=CONFIG,
            temp_directory=tmp_path / "tmp",
        )
