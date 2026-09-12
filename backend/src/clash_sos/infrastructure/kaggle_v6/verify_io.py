"""DuckDB checks for processed canonical, disposition, and split artifacts."""

from datetime import datetime
from pathlib import Path

import duckdb

from clash_sos.domain.canonical_dataset import CANONICAL_SCHEMA
from clash_sos.infrastructure.kaggle_v6.staging_io import python_cell


class KaggleV6VerifyError(ValueError):
    pass


def _as_int(value: object) -> int:
    cell = python_cell(value)
    if type(cell) is not int:
        raise KaggleV6VerifyError("count must be an integer")
    return cell


def _count(connection: duckdb.DuckDBPyConnection, sql: str, params: list[object]) -> int:
    row = connection.execute(sql, params).fetchone()
    if row is None:
        raise KaggleV6VerifyError("count is unavailable")
    return _as_int(row[0])


def _require_zero(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[object],
    message: str,
) -> None:
    if _count(connection, sql, params):
        raise KaggleV6VerifyError(message)


def verify_processed_artifacts(
    connection: duckdb.DuckDBPyConnection,
    *,
    canonical_path: Path,
    disposition_files: tuple[Path, ...],
    temporal_split_path: Path,
    player_split_path: Path,
    train_end: datetime,
    validation_end: datetime,
    excluded_bridge_rows: int,
) -> None:
    """Raise unless every frozen processed-dataset check passes."""
    canonical = str(canonical_path)
    _require_zero(
        connection,
        """
        SELECT COUNT(*) FROM (
            SELECT 1 FROM read_parquet(?)
            GROUP BY timestamp, fingerprint, archive_member, row_number
            HAVING COUNT(*) > 1
        )
        """,
        [canonical],
        "canonical identity is not unique",
    )
    rows = connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [canonical]).fetchall()
    names = tuple(str(row[0]) for row in rows)
    types = tuple(str(row[1]) for row in rows)
    if names != tuple(column.name for column in CANONICAL_SCHEMA):
        raise KaggleV6VerifyError("canonical schema names do not match")
    if types != tuple(column.physical_type for column in CANONICAL_SCHEMA):
        raise KaggleV6VerifyError("canonical schema types do not match")
    _require_zero(
        connection,
        "SELECT COALESCE(SUM(stats_null_count), 0) FROM parquet_metadata(?)",
        [canonical],
        "canonical parquet contains nulls",
    )
    if not disposition_files:
        raise KaggleV6VerifyError("disposition parquet is required")
    if not player_split_path.is_file():
        raise KaggleV6VerifyError("player-disjoint split is required")
    if not temporal_split_path.is_file():
        raise KaggleV6VerifyError("temporal split is required")
    canonical_count = _count(connection, "SELECT COUNT(*) FROM read_parquet(?)", [canonical])
    dispositions = [str(path) for path in disposition_files]
    valid = _count(
        connection,
        "SELECT COUNT(*) FROM read_parquet(?) WHERE state = 'valid'",
        [dispositions],
    )
    if valid != canonical_count:
        raise KaggleV6VerifyError("disposition valid count does not match canonical rows")
    _require_zero(
        connection,
        """
        SELECT COUNT(*)
        FROM read_parquet(?) canonical
        ANTI JOIN (
            SELECT archive_member, row_number FROM read_parquet(?) WHERE state = 'valid'
        ) dispositions
          ON canonical.archive_member = dispositions.archive_member
         AND canonical.row_number = dispositions.row_number
        """,
        [canonical, dispositions],
        "canonical rows are missing valid dispositions",
    )
    retained = _count(connection, "SELECT COUNT(*) FROM read_parquet(?)", [str(player_split_path)])
    if retained + excluded_bridge_rows != canonical_count:
        raise KaggleV6VerifyError("player-disjoint bridge count does not reconcile")
    _require_zero(
        connection,
        """
        SELECT COUNT(*) FROM (
            SELECT player_id FROM (
                SELECT canonical.side_a_player_id AS player_id, splits.partition
                FROM read_parquet(?) canonical
                JOIN read_parquet(?) splits
                  ON canonical.timestamp = splits.timestamp
                 AND canonical.fingerprint = splits.fingerprint
                 AND canonical.archive_member = splits.archive_member
                 AND canonical.row_number = splits.row_number
                UNION ALL
                SELECT canonical.side_b_player_id, splits.partition
                FROM read_parquet(?) canonical
                JOIN read_parquet(?) splits
                  ON canonical.timestamp = splits.timestamp
                 AND canonical.fingerprint = splits.fingerprint
                 AND canonical.archive_member = splits.archive_member
                 AND canonical.row_number = splits.row_number
            )
            GROUP BY 1 HAVING COUNT(DISTINCT partition) > 1
        )
        """,
        [canonical, str(player_split_path), canonical, str(player_split_path)],
        "player-disjoint split leaks a player across partitions",
    )
    coverage = _count(
        connection, "SELECT COUNT(*) FROM read_parquet(?)", [str(temporal_split_path)]
    )
    if coverage != canonical_count:
        raise KaggleV6VerifyError("temporal split does not cover every canonical row")
    _require_zero(
        connection,
        """
        SELECT COUNT(*) FROM (
            SELECT timestamp, fingerprint, archive_member, row_number FROM read_parquet(?)
            EXCEPT
            SELECT timestamp, fingerprint, archive_member, row_number FROM read_parquet(?)
        )
        """,
        [canonical, str(temporal_split_path)],
        "temporal split is missing canonical identities",
    )
    _require_zero(
        connection,
        """
        SELECT COUNT(*)
        FROM read_parquet(?) splits
        JOIN read_parquet(?) canonical
          ON splits.timestamp = canonical.timestamp
         AND splits.fingerprint = canonical.fingerprint
         AND splits.archive_member = canonical.archive_member
         AND splits.row_number = canonical.row_number
        WHERE splits.partition != CASE
            WHEN canonical.timestamp < ? THEN 'train'
            WHEN canonical.timestamp < ? THEN 'validation'
            ELSE 'test'
        END
        """,
        [str(temporal_split_path), canonical, train_end, validation_end],
        "temporal split boundaries do not match cutovers",
    )
