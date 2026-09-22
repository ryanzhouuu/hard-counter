"""Resolve attention row populations from joined Parquet before cache creation.

The caller chooses the watch fraction and slice bounds. Outcomes never enter
boundary selection or row identity hashing.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from clash_sos.domain.attention_protocol import AttentionSlice, Partition, RowKey, digest_row_keys
from clash_sos.infrastructure.kaggle_v6.staging_io import python_cell

_JOIN = """
FROM read_parquet(?) AS c
JOIN read_parquet(?) AS s
USING (timestamp, fingerprint, archive_member, row_number)
WHERE s.partition = ? AND c.timestamp >= ? AND c.timestamp < ?
"""


class AttentionProtocolResolveError(ValueError):
    """The requested population cannot form a valid nonempty attention slice."""


@dataclass(frozen=True)
class WatchBoundary:
    """The actual tail fraction may differ because timestamp ties stay together."""

    timestamp: datetime
    fit_rows: int
    watch_rows: int
    actual_fraction: float


def _require_bounds(start: datetime, end: datetime) -> None:
    """Require a timezone-aware, nonempty half-open interval."""
    if start.tzinfo is None or start.utcoffset() is None:
        raise AttentionProtocolResolveError("slice bounds must be timezone-aware")
    if end.tzinfo is None or end.utcoffset() is None:
        raise AttentionProtocolResolveError("slice bounds must be timezone-aware")
    if start >= end:
        raise AttentionProtocolResolveError("slice start must precede end")


def _params(
    canonical_path: Path,
    split_path: Path,
    partition: Partition,
    start: datetime,
    end: datetime,
) -> list[object]:
    """Keep source paths and half-open bounds aligned across resolver queries."""
    if not canonical_path.is_file() or not split_path.is_file():
        raise AttentionProtocolResolveError("canonical and split Parquet files are required")
    _require_bounds(start, end)
    return [str(canonical_path), str(split_path), partition, start, end]


def _timestamp(value: object) -> datetime:
    """Reject untyped timestamps before choosing or recording a boundary."""
    parsed = python_cell(value)
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AttentionProtocolResolveError("joined timestamp must be timezone-aware")
    return parsed


def resolve_attention_slice(
    connection: duckdb.DuckDBPyConnection,
    *,
    canonical_path: Path,
    split_path: Path,
    partition: Partition,
    start: datetime,
    end: datetime,
    batch_rows: int = 10_000,
) -> AttentionSlice:
    """Hash unique sorted joined row keys; an empty or duplicated slice fails."""
    if batch_rows < 1:
        raise AttentionProtocolResolveError("batch_rows must be positive")
    rows = connection.execute(
        f"""
        SELECT c.timestamp, c.fingerprint, c.archive_member, c.row_number
        {_JOIN}
        ORDER BY c.timestamp, c.fingerprint, c.archive_member, c.row_number
        """,
        _params(canonical_path, split_path, partition, start, end),
    )
    count = 0

    def keys() -> Iterator[RowKey]:
        """Bound Python allocation to one DuckDB fetch batch."""
        nonlocal count
        while batch := rows.fetchmany(batch_rows):
            for stamp, fingerprint, member, number in batch:
                if not isinstance(fingerprint, str) or not isinstance(member, str):
                    raise AttentionProtocolResolveError("joined row key is invalid")
                if type(number) is not int:
                    raise AttentionProtocolResolveError("joined row number is invalid")
                count += 1
                yield (_timestamp(stamp), fingerprint, member, number)

    try:
        digest = digest_row_keys(keys())
    except ValueError as error:
        raise AttentionProtocolResolveError(str(error)) from error
    if count == 0:
        raise AttentionProtocolResolveError("attention slice is empty")
    return AttentionSlice(
        partition=partition, start=start, end=end, row_count=count, row_keys_sha256=digest
    )


def resolve_watch_boundary(
    connection: duckdb.DuckDBPyConnection,
    *,
    canonical_path: Path,
    split_path: Path,
    start: datetime,
    end: datetime,
    target_fraction: float,
    batch_rows: int = 10_000,
) -> WatchBoundary:
    """Choose the nearest nonempty watch tail; equal distances retain more fit rows."""
    if not 0 < target_fraction < 1:
        raise AttentionProtocolResolveError("target watch fraction must be between zero and one")
    if batch_rows < 1:
        raise AttentionProtocolResolveError("batch_rows must be positive")
    params = _params(canonical_path, split_path, "train", start, end)
    total_row = connection.execute(f"SELECT COUNT(*) {_JOIN}", params).fetchone()
    total = 0 if total_row is None else int(total_row[0])
    if total < 2:
        raise AttentionProtocolResolveError("watch split requires at least two rows")
    groups = connection.execute(
        f"SELECT c.timestamp, COUNT(*) {_JOIN} GROUP BY c.timestamp ORDER BY c.timestamp",
        params,
    )
    fit_rows = 0
    best: WatchBoundary | None = None
    while batch := groups.fetchmany(batch_rows):
        for stamp, group_count in batch:
            if 0 < fit_rows < total:
                watch_rows = total - fit_rows
                candidate = WatchBoundary(
                    timestamp=_timestamp(stamp),
                    fit_rows=fit_rows,
                    watch_rows=watch_rows,
                    actual_fraction=watch_rows / total,
                )
                if best is None or (
                    abs(candidate.watch_rows - target_fraction * total),
                    -candidate.fit_rows,
                ) < (abs(best.watch_rows - target_fraction * total), -best.fit_rows):
                    best = candidate
            fit_rows += int(group_count)
    if best is None or fit_rows != total:
        raise AttentionProtocolResolveError("watch split needs two distinct timestamps")
    return best
