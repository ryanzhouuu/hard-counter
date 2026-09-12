"""DuckDB COPY writers for temporal and player-disjoint split Parquet."""

from datetime import datetime
from pathlib import Path

import duckdb

from clash_sos.domain.processed_manifest import (
    TEMPORAL_PARTITIONS,
    TemporalPartitionSummary,
    TemporalSplitManifest,
)
from clash_sos.infrastructure.kaggle_v6.staging_io import python_cell


class KaggleV6SplitError(ValueError):
    pass


_TEMPORAL_SELECT = """
SELECT
    CAST(timestamp AS TIMESTAMP WITH TIME ZONE) AS timestamp,
    CAST(fingerprint AS VARCHAR) AS fingerprint,
    CAST(archive_member AS VARCHAR) AS archive_member,
    CAST(row_number AS BIGINT) AS row_number,
    CASE
        WHEN timestamp < ? THEN 'train'
        WHEN timestamp < ? THEN 'validation'
        ELSE 'test'
    END AS partition
FROM read_parquet(?)
ORDER BY timestamp, fingerprint, archive_member, row_number
"""


def _require_timezone(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise KaggleV6SplitError(f"{label} must be timezone-aware")
    return value


def _as_datetime(value: object) -> datetime:
    cell = python_cell(value)
    if not isinstance(cell, datetime):
        raise KaggleV6SplitError("timestamp bounds must be datetimes")
    return _require_timezone(cell, label="canonical timestamp")


def write_temporal_split_parquet(
    connection: duckdb.DuckDBPyConnection,
    *,
    canonical_path: Path,
    output_path: Path,
    train_end: datetime,
    validation_end: datetime,
    row_group_rows: int,
) -> TemporalSplitManifest:
    """COPY a sorted zstd temporal split and return partition summaries."""
    train_end = _require_timezone(train_end, label="train_end")
    validation_end = _require_timezone(validation_end, label="validation_end")
    if train_end >= validation_end:
        raise KaggleV6SplitError("train_end must be earlier than validation_end")
    if output_path.exists():
        raise KaggleV6SplitError("temporal split already exists")
    if not canonical_path.is_file():
        raise KaggleV6SplitError("canonical parquet is required")
    bounds = connection.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM read_parquet(?)",
        [str(canonical_path)],
    ).fetchone()
    if bounds is None or bounds[0] is None or bounds[1] is None:
        raise KaggleV6SplitError("canonical parquet has no timestamps")
    timestamp_min = _as_datetime(bounds[0])
    timestamp_max = _as_datetime(bounds[1])
    if train_end < timestamp_min or train_end > timestamp_max:
        raise KaggleV6SplitError("train_end is outside the accepted timestamp range")
    if validation_end < timestamp_min or validation_end > timestamp_max:
        raise KaggleV6SplitError("validation_end is outside the accepted timestamp range")
    connection.execute("DROP TABLE IF EXISTS temporal_split")
    connection.execute(
        f"CREATE TEMP TABLE temporal_split AS {_TEMPORAL_SELECT}",
        [train_end, validation_end, str(canonical_path)],
    )
    counts = {
        str(partition): (int(count), _as_datetime(minimum), _as_datetime(maximum))
        for partition, count, minimum, maximum in connection.execute(
            """
            SELECT partition, COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM temporal_split
            GROUP BY 1
            """
        ).fetchall()
    }
    partitions: list[TemporalPartitionSummary] = []
    for label in TEMPORAL_PARTITIONS:
        if label not in counts:
            raise KaggleV6SplitError(f"temporal {label} partition is empty")
        count, minimum, maximum = counts[label]
        partitions.append(
            TemporalPartitionSummary(
                partition=label,
                row_count=count,
                timestamp_min=minimum,
                timestamp_max=maximum,
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = str(output_path).replace("'", "''")
    connection.execute(
        f"""
        COPY temporal_split TO '{target}' (
            FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(row_group_rows)}
        )
        """
    )
    connection.execute("DROP TABLE temporal_split")
    return TemporalSplitManifest(
        train_end=train_end,
        validation_end=validation_end,
        partitions=tuple(partitions),
    )
