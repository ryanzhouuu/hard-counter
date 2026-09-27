from pathlib import Path

import duckdb
import polars as pl
import pytest
from test_tower_attention import tower_schema
from tower_dataset_fixture import VERSION, stamp, tower_rows, write_tower_source

from clash_sos.infrastructure.clash_royale.attention_snapshot import (
    export_snapshot,
    write_snapshot_parts,
)


def write_parts(source: Path, workspace: Path) -> int:
    return write_snapshot_parts(
        source,
        workspace / "parts",
        schema=tower_schema(),
        dataset_version=VERSION,
        start=stamp(1),
        end=stamp(27),
        batch_rows=2,
    )


def test_snapshot_streams_validated_parts_and_exports_a_temporal_split(tmp_path: Path) -> None:
    source = tmp_path / "normalized.jsonl"
    write_tower_source(source)
    assert write_parts(source, tmp_path) == 6
    assert len(tuple((tmp_path / "parts").glob("*.parquet"))) == 3
    connection = duckdb.connect()
    try:
        counts = export_snapshot(
            connection, tmp_path, train_end=stamp(10), validation_end=stamp(20)
        )
    finally:
        connection.close()
    assert counts == {"train": 3, "validation": 2, "test": 1}
    canonical = pl.read_parquet(tmp_path / "canonical.parquet")
    assert canonical.height == 6
    assert canonical.get_column("side_a_tower").n_unique() == 4
    assert canonical.get_column("side_a_tower_level").to_list() == [16] * 6


@pytest.mark.parametrize(
    "changes",
    [
        {"balance_era_id": "2026-06"},
        {"dataset_version": "other"},
        {"timestamp": stamp(28)},
        {"side_a_tower": "unknown:tower"},
        {"side_a_tower": None},
        {"side_a_tower_level": 11},
    ],
)
def test_snapshot_rejects_invalid_normalized_inputs(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    source = tmp_path / "normalized.jsonl"
    row = tower_rows()[0].model_copy(update=changes)
    write_tower_source(source, (row,))
    with pytest.raises(ValueError, match="line 1"):
        write_parts(source, tmp_path)


def test_snapshot_rejects_duplicate_battle_identities(tmp_path: Path) -> None:
    source = tmp_path / "normalized.jsonl"
    row = tower_rows()[0]
    write_tower_source(source, (row, row))
    write_parts(source, tmp_path)
    connection = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="duplicate or conflicting"):
            export_snapshot(connection, tmp_path, train_end=stamp(10), validation_end=stamp(20))
    finally:
        connection.close()


def test_snapshot_requires_a_positive_batch_bound(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch size must be positive"):
        write_snapshot_parts(
            tmp_path / "source.jsonl",
            tmp_path / "parts",
            schema=tower_schema(),
            dataset_version=VERSION,
            start=stamp(1),
            end=stamp(27),
            batch_rows=0,
        )
    assert not (tmp_path / "parts").exists()
