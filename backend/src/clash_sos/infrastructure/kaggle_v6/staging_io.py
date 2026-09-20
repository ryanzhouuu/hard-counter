"""DuckDB scans and Polars part writes for bounded Kaggle v6 staging."""

import numbers
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from zipfile import ZipFile

import duckdb
import polars as pl

from clash_sos.domain.manifests import SchemaColumnManifest
from clash_sos.domain.staged_dataset import (
    STAGED_SCHEMA,
    UNADAPTABLE_SCHEMA,
    StagedBattleRow,
    UnadaptableRow,
)
from clash_sos.infrastructure.kaggle_v6.audit_io import validate_archive_members


class KaggleV6StagingError(ValueError):
    pass


PolarsSchemaType = pl.DataType | type[pl.DataType]
_POLARS_PHYSICAL_TYPES: dict[str, PolarsSchemaType] = {
    "BIGINT": pl.Int64,
    "TIMESTAMP WITH TIME ZONE": pl.Datetime(time_zone="UTC"),
    "UTINYINT[]": pl.List(pl.UInt8),
    "VARCHAR": pl.String,
    "VARCHAR[]": pl.List(pl.String),
}


def polars_schema(columns: Sequence[SchemaColumnManifest]) -> dict[str, PolarsSchemaType]:
    """Translate DuckDB physical types from a staged schema into Polars dtypes."""
    return {column.name: _POLARS_PHYSICAL_TYPES[column.physical_type] for column in columns}


def connect_staging_duckdb(
    *, memory_limit: str, threads: int, temp_directory: Path
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB session with UTC timestamps and bounded resource settings.

    Disables preserve_insertion_order so large scans do not keep extra buffers.
    """
    connection = duckdb.connect(
        config={
            "memory_limit": memory_limit,
            "threads": str(threads),
            "temp_directory": str(temp_directory),
        }
    )
    connection.execute("SET TimeZone='UTC'")
    connection.execute("SET preserve_insertion_order=false")
    return connection


def list_parquet_members(archive: ZipFile) -> tuple[str, ...]:
    """Return sorted Parquet member paths from a validated Kaggle v6 archive."""
    members = validate_archive_members(archive)
    return tuple(member.filename for member in members if member.filename.endswith(".parquet"))


def python_cell(value: object) -> object:
    """Coerce DuckDB cell values into types the Kaggle adapter accepts."""
    if value is None or type(value) is bool or isinstance(value, str):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value
        return value.replace(tzinfo=UTC)
    if isinstance(value, numbers.Integral):
        return int(value)
    return value


def row_mapping(columns: Sequence[str], values: Sequence[object]) -> dict[str, object]:
    """Zip DuckDB column names with coerced cell values."""
    return {name: python_cell(value) for name, value in zip(columns, values, strict=True)}


def part_path(
    workspace: Path,
    kind: Literal["staging", "unadaptable"],
    archive_member: str,
    part_index: int,
) -> Path:
    """Build the deterministic part-file path for one staging batch."""
    return workspace / kind / PurePosixPath(archive_member).stem / f"part-{part_index:05d}.parquet"


def staged_row_to_record(row: StagedBattleRow) -> dict[str, object]:
    return {
        "source_id": row.source_id,
        "timestamp": row.timestamp,
        "mode": row.mode,
        "balance_era_id": row.balance_era_id,
        "outcome": row.outcome.value,
        "event_key": row.event_key,
        "fingerprint": row.fingerprint,
        "side_a_player_id": row.side_a_player_id.value,
        "side_b_player_id": row.side_b_player_id.value,
        "side_a_card_ids": list(row.side_a_card_ids),
        "side_a_card_forms": list(row.side_a_card_forms),
        "side_a_card_levels": list(row.side_a_card_levels),
        "side_a_deck_hash": row.side_a_deck_hash,
        "side_b_card_ids": list(row.side_b_card_ids),
        "side_b_card_forms": list(row.side_b_card_forms),
        "side_b_card_levels": list(row.side_b_card_levels),
        "side_b_deck_hash": row.side_b_deck_hash,
        "observation_issues": [issue.value for issue in row.observation_issues],
        "archive_member": row.archive_member,
        "row_number": row.row_number,
    }


def unadaptable_row_to_record(row: UnadaptableRow) -> dict[str, object]:
    return {
        "archive_member": row.archive_member,
        "row_number": row.row_number,
        "state": row.state.value,
        "issues": [issue.value for issue in row.issues],
    }


def write_staged_part(path: Path, rows: Sequence[StagedBattleRow], *, row_group_rows: int) -> None:
    """Write one staging part with the staged schema so nullable eras survive leading nulls."""
    if not rows:
        raise KaggleV6StagingError("refusing to write an empty staging part")
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [staged_row_to_record(row) for row in rows],
        schema=polars_schema(STAGED_SCHEMA),
    ).write_parquet(path, compression="zstd", row_group_size=row_group_rows)


def write_unadaptable_part(
    path: Path, rows: Sequence[UnadaptableRow], *, row_group_rows: int
) -> None:
    """Write one unadaptable part with the unadaptable schema. Raises on an empty batch."""
    if not rows:
        raise KaggleV6StagingError("refusing to write an empty unadaptable part")
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [unadaptable_row_to_record(row) for row in rows],
        schema=polars_schema(UNADAPTABLE_SCHEMA),
    ).write_parquet(path, compression="zstd", row_group_size=row_group_rows)


def directory_byte_size(path: Path) -> int:
    """Sum file sizes under a directory tree, or zero when the path is absent."""
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
