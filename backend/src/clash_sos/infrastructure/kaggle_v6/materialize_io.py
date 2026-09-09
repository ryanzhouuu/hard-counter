"""DuckDB join, sorted COPY, and schema inspection for canonical Parquet."""

from collections.abc import Iterator
from pathlib import Path

import duckdb

from clash_sos.domain.canonical_dataset import CANONICAL_SCHEMA, CanonicalBattleRow
from clash_sos.domain.manifests import SchemaColumnManifest
from clash_sos.infrastructure.kaggle_v6.staging_io import row_mapping

_CANONICAL_SELECT = """
SELECT
    CAST(? AS VARCHAR) AS dataset_version,
    CAST(s.source_id AS VARCHAR) AS source_id,
    CAST(s.timestamp AS TIMESTAMP WITH TIME ZONE) AS timestamp,
    CAST(s.mode AS VARCHAR) AS mode,
    CAST(s.balance_era_id AS VARCHAR) AS balance_era_id,
    CAST(s.outcome AS VARCHAR) AS outcome,
    CAST(s.event_key AS VARCHAR) AS event_key,
    CAST(s.fingerprint AS VARCHAR) AS fingerprint,
    CAST(s.side_a_player_id AS VARCHAR) AS side_a_player_id,
    CAST(s.side_b_player_id AS VARCHAR) AS side_b_player_id,
    CAST(s.side_a_card_ids AS VARCHAR[]) AS side_a_card_ids,
    CAST(s.side_a_card_forms AS VARCHAR[]) AS side_a_card_forms,
    CAST(s.side_a_card_levels AS UTINYINT[]) AS side_a_card_levels,
    CAST(s.side_a_deck_hash AS VARCHAR) AS side_a_deck_hash,
    CAST(s.side_b_card_ids AS VARCHAR[]) AS side_b_card_ids,
    CAST(s.side_b_card_forms AS VARCHAR[]) AS side_b_card_forms,
    CAST(s.side_b_card_levels AS UTINYINT[]) AS side_b_card_levels,
    CAST(s.side_b_deck_hash AS VARCHAR) AS side_b_deck_hash,
    CAST(s.archive_member AS VARCHAR) AS archive_member,
    CAST(s.row_number AS BIGINT) AS row_number
FROM read_parquet(?) AS s
INNER JOIN read_parquet(?) AS d
    ON s.archive_member = d.archive_member AND s.row_number = d.row_number
WHERE d.state = 'valid'
ORDER BY s.timestamp, s.fingerprint, s.archive_member, s.row_number
"""


class KaggleV6MaterializeError(ValueError):
    pass


def read_canonical_schema(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> tuple[SchemaColumnManifest, ...]:
    """Return DuckDB DESCRIBE columns for a canonical Parquet file."""
    rows = connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
    observed = tuple(
        SchemaColumnManifest(
            name=str(row[0]),
            physical_type=str(row[1]),
            nullable=row[2] == "YES",
        )
        for row in rows
    )
    observed_names = tuple(column.name for column in observed)
    expected_names = tuple(column.name for column in CANONICAL_SCHEMA)
    if observed_names != expected_names:
        raise KaggleV6MaterializeError("canonical parquet columns do not match the schema")
    return observed


def write_canonical_parquet(
    connection: duckdb.DuckDBPyConnection,
    *,
    staging_files: tuple[Path, ...],
    disposition_files: tuple[Path, ...],
    output_path: Path,
    dataset_version: str,
    row_group_rows: int,
) -> int:
    """Join valid dispositions to staging rows and COPY sorted zstd canonical Parquet."""
    if output_path.exists():
        raise KaggleV6MaterializeError("canonical parquet already exists")
    if not disposition_files:
        raise KaggleV6MaterializeError("disposition parquet is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute("DROP TABLE IF EXISTS canonical_rows")
    if staging_files:
        connection.execute(
            f"CREATE TEMP TABLE canonical_rows AS {_CANONICAL_SELECT}",
            [
                dataset_version,
                [str(path) for path in staging_files],
                [str(path) for path in disposition_files],
            ],
        )
    else:
        columns = ", ".join(f"{column.name} {column.physical_type}" for column in CANONICAL_SCHEMA)
        connection.execute(f"CREATE TEMP TABLE canonical_rows ({columns})")
    counted = connection.execute("SELECT COUNT(*) FROM canonical_rows").fetchone()
    if counted is None:
        raise KaggleV6MaterializeError("canonical row count is unavailable")
    row_count = int(counted[0])
    target = str(output_path).replace("'", "''")
    connection.execute(
        f"""
        COPY canonical_rows TO '{target}' (
            FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(row_group_rows)}
        )
        """
    )
    connection.execute("DROP TABLE canonical_rows")
    return row_count


def iter_canonical_rows(path: Path, *, batch_rows: int) -> Iterator[CanonicalBattleRow]:
    """Yield canonical rows in sort order without loading the full file."""
    if batch_rows < 1:
        raise KaggleV6MaterializeError("batch_rows must be at least 1")
    connection = duckdb.connect()
    try:
        result = connection.execute(
            """
            SELECT * FROM read_parquet(?)
            ORDER BY timestamp, fingerprint, archive_member, row_number
            """,
            [str(path)],
        )
        columns = [str(column[0]) for column in result.description]
        while True:
            batch = result.fetchmany(batch_rows)
            if not batch:
                break
            for values in batch:
                yield CanonicalBattleRow.model_validate(row_mapping(columns, values))
    finally:
        connection.close()
