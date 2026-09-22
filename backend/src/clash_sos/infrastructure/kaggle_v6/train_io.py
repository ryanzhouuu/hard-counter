"""DuckDB join, deterministic mirroring, and count aggregates for matchup training.

Reads canonical battles joined to a split parquet. Winner-first rows are mirrored
with the same SHA-256 rule as `should_mirror_sides` so labels are not constantly 1.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import duckdb

from clash_sos.infrastructure.kaggle_v6.staging_io import python_cell


class KaggleV6TrainError(ValueError):
    pass


@dataclass(frozen=True)
class OrientedExample:
    """One mirrored training or evaluation row with oriented decks and label."""

    label: int
    side_a_keys: tuple[str, ...]
    side_b_keys: tuple[str, ...]
    deck_a_hash: str
    deck_b_hash: str
    side_a_player_id: str = ""
    side_b_player_id: str = ""


@dataclass(frozen=True)
class OrientedCacheRow:
    """Keep canonical identity and validated levels beside the legacy orientation."""

    timestamp: datetime
    fingerprint: str
    archive_member: str
    row_number: int
    example: OrientedExample
    side_a_levels: tuple[int, ...]
    side_b_levels: tuple[int, ...]


def _oriented_sql(*, include_levels: bool, windowed: bool = False) -> str:
    """Keep one mirror rule while legacy fixtures retain their narrow projection."""
    joined_levels = "c.side_a_card_levels, c.side_b_card_levels," if include_levels else ""
    oriented_levels = (
        "CASE WHEN mirrored THEN side_b_card_levels ELSE side_a_card_levels END AS side_a_levels,"
        "CASE WHEN mirrored THEN side_a_card_levels ELSE side_b_card_levels END AS side_b_levels,"
        if include_levels
        else ""
    )
    window_filter = "AND c.timestamp >= ? AND c.timestamp < ?" if windowed else ""
    return f"""
WITH joined AS (
    SELECT
        c.timestamp,
        c.fingerprint,
        c.archive_member,
        c.row_number,
        c.side_a_player_id,
        c.side_b_player_id,
        c.side_a_card_ids,
        c.side_a_card_forms,
        c.side_b_card_ids,
        c.side_b_card_forms,
        {joined_levels}
        c.side_a_deck_hash,
        c.side_b_deck_hash,
        (
            CAST(
                '0x' || substr(sha256(CAST(? AS VARCHAR) || ':' || c.fingerprint), 1, 2)
                AS INTEGER
            ) % 2 = 1
        ) AS mirrored
    FROM read_parquet(?) AS c
    INNER JOIN read_parquet(?) AS s
        ON c.timestamp = s.timestamp
        AND c.fingerprint = s.fingerprint
        AND c.archive_member = s.archive_member
        AND c.row_number = s.row_number
    WHERE s.partition = ? {window_filter}
),
oriented AS (
    SELECT
        timestamp,
        fingerprint,
        archive_member,
        row_number,
        CASE WHEN mirrored THEN side_b_player_id ELSE side_a_player_id END AS side_a_player_id,
        CASE WHEN mirrored THEN side_a_player_id ELSE side_b_player_id END AS side_b_player_id,
        CASE WHEN mirrored THEN 0 ELSE 1 END AS label,
        list_transform(
            list_zip(
                CASE WHEN mirrored THEN side_b_card_ids ELSE side_a_card_ids END,
                CASE WHEN mirrored THEN side_b_card_forms ELSE side_a_card_forms END
            ),
            x -> x[1] || ':' || x[2]
        ) AS side_a_keys,
        list_transform(
            list_zip(
                CASE WHEN mirrored THEN side_a_card_ids ELSE side_b_card_ids END,
                CASE WHEN mirrored THEN side_a_card_forms ELSE side_b_card_forms END
            ),
            x -> x[1] || ':' || x[2]
        ) AS side_b_keys,
        {oriented_levels}
        CASE WHEN mirrored THEN side_b_deck_hash ELSE side_a_deck_hash END AS deck_a_hash,
        CASE WHEN mirrored THEN side_a_deck_hash ELSE side_b_deck_hash END AS deck_b_hash
    FROM joined
)
"""


_ORIENTED_SQL = _oriented_sql(include_levels=False)
_CACHE_ORIENTED_SQL = _oriented_sql(include_levels=True)
_CACHE_WINDOWED_SQL = _oriented_sql(include_levels=True, windowed=True)


def _as_int(value: object) -> int:
    cell = python_cell(value)
    if type(cell) is not int:
        raise KaggleV6TrainError("count must be an integer")
    return cell


def _as_str(value: object) -> str:
    cell = python_cell(value)
    if not isinstance(cell, str):
        raise KaggleV6TrainError("hash and identity values must be strings")
    return cell


def _as_keys(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise KaggleV6TrainError("card identities must be a list")
    return tuple(_as_str(item) for item in cast(list[object], value))


def _as_levels(value: object) -> tuple[int, ...]:
    """Reject malformed level lists before the attention encoder sees them."""
    if not isinstance(value, list):
        raise KaggleV6TrainError("card levels must be a list")
    return tuple(_as_int(item) for item in cast(list[object], value))


def _require_paths(canonical_path: Path, split_path: Path) -> None:
    if not canonical_path.is_file():
        raise KaggleV6TrainError("canonical parquet is required")
    if not split_path.is_file():
        raise KaggleV6TrainError("split parquet is required")


def _oriented_params(
    canonical_path: Path, split_path: Path, *, partition: str, seed: int
) -> list[object]:
    return [seed, str(canonical_path), str(split_path), partition]


def require_partition_rows(
    connection: duckdb.DuckDBPyConnection,
    canonical_path: Path,
    split_path: Path,
    *,
    split: str,
    partition: str,
    seed: int,
    expected_rows: int,
) -> int:
    """Return the joined row count, or raise if the partition is empty or incomplete."""
    _require_paths(canonical_path, split_path)
    row = connection.execute(
        f"{_ORIENTED_SQL} SELECT COUNT(*) FROM oriented",
        _oriented_params(canonical_path, split_path, partition=partition, seed=seed),
    ).fetchone()
    if row is None or _as_int(row[0]) == 0:
        raise KaggleV6TrainError(f"split {partition} partition is empty")
    count = _as_int(row[0])
    if count != expected_rows:
        raise KaggleV6TrainError(f"{split} {partition} join count {count} != {expected_rows}")
    return count


def aggregate_card_counts(
    connection: duckdb.DuckDBPyConnection,
    *,
    canonical_path: Path,
    split_path: Path,
    partition: str,
    seed: int,
) -> dict[str, tuple[int, int]]:
    """Return per-card (wins, trials) after mirroring rows in one split partition."""
    _require_paths(canonical_path, split_path)
    rows = connection.execute(
        f"""
        {_ORIENTED_SQL}
        SELECT identity, SUM(won)::BIGINT, COUNT(*)::BIGINT
        FROM (
            SELECT unnest(side_a_keys) AS identity, label AS won FROM oriented
            UNION ALL
            SELECT unnest(side_b_keys) AS identity, 1 - label AS won FROM oriented
        )
        GROUP BY 1
        ORDER BY 1
        """,
        _oriented_params(canonical_path, split_path, partition=partition, seed=seed),
    ).fetchall()
    return {_as_str(identity): (_as_int(wins), _as_int(trials)) for identity, wins, trials in rows}


def aggregate_matchup_counts(
    connection: duckdb.DuckDBPyConnection,
    *,
    canonical_path: Path,
    split_path: Path,
    partition: str,
    seed: int,
) -> dict[tuple[str, str], tuple[int, int]]:
    """Return per oriented deck-pair (wins, trials) in one split partition."""
    _require_paths(canonical_path, split_path)
    rows = connection.execute(
        f"""
        {_ORIENTED_SQL}
        SELECT deck_a_hash, deck_b_hash, SUM(label)::BIGINT, COUNT(*)::BIGINT
        FROM oriented
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        _oriented_params(canonical_path, split_path, partition=partition, seed=seed),
    ).fetchall()
    return {
        (_as_str(deck_a), _as_str(deck_b)): (_as_int(wins), _as_int(trials))
        for deck_a, deck_b, wins, trials in rows
    }


def iter_oriented_examples(
    connection: duckdb.DuckDBPyConnection,
    *,
    canonical_path: Path,
    split_path: Path,
    partition: str,
    seed: int,
    batch_rows: int = 10_000,
    order_by_time: bool = False,
) -> Iterator[OrientedExample]:
    """Yield mirrored examples for scoring without materializing the full corpus.

    order_by_time sorts by timestamp, fingerprint, archive member, and row number
    so a skill pass can use only earlier battles.
    """
    if batch_rows < 1:
        raise KaggleV6TrainError("batch_rows must be positive")
    _require_paths(canonical_path, split_path)
    order_clause = (
        "ORDER BY timestamp, fingerprint, archive_member, row_number" if order_by_time else ""
    )
    result = connection.execute(
        f"""
        {_ORIENTED_SQL}
        SELECT label, side_a_keys, side_b_keys, deck_a_hash, deck_b_hash,
               side_a_player_id, side_b_player_id
        FROM oriented
        {order_clause}
        """,
        _oriented_params(canonical_path, split_path, partition=partition, seed=seed),
    )
    while True:
        batch = result.fetchmany(batch_rows)
        if not batch:
            return
        for label, side_a, side_b, deck_a, deck_b, player_a, player_b in batch:
            yield OrientedExample(
                label=_as_int(label),
                side_a_keys=_as_keys(side_a),
                side_b_keys=_as_keys(side_b),
                deck_a_hash=_as_str(deck_a),
                deck_b_hash=_as_str(deck_b),
                side_a_player_id=_as_str(player_a),
                side_b_player_id=_as_str(player_b),
            )


def iter_oriented_cache_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    canonical_path: Path,
    split_path: Path,
    partition: str,
    seed: int,
    batch_rows: int = 10_000,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Iterator[OrientedCacheRow]:
    """Stream ordered cache rows; optional half-open windows bound DuckDB memory."""
    if batch_rows < 1:
        raise KaggleV6TrainError("batch_rows must be positive")
    if (start is None) != (end is None):
        raise KaggleV6TrainError("cache window requires both bounds")
    if start is not None and end is not None and start >= end:
        raise KaggleV6TrainError("cache window must be nonempty")
    _require_paths(canonical_path, split_path)
    sql = _CACHE_WINDOWED_SQL if start is not None else _CACHE_ORIENTED_SQL
    params = _oriented_params(canonical_path, split_path, partition=partition, seed=seed)
    if start is not None and end is not None:
        params.extend((start, end))
    result = connection.execute(
        f"""
        {sql}
        SELECT timestamp, fingerprint, archive_member, row_number, label,
               side_a_keys, side_b_keys, side_a_levels, side_b_levels,
               deck_a_hash, deck_b_hash, side_a_player_id, side_b_player_id
        FROM oriented
        ORDER BY timestamp, fingerprint, archive_member, row_number
        """,
        params,
    )
    while batch := result.fetchmany(batch_rows):
        for row in batch:
            timestamp = python_cell(row[0])
            if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
                raise KaggleV6TrainError("cache row timestamp must be timezone-aware")
            yield OrientedCacheRow(
                timestamp=timestamp,
                fingerprint=_as_str(row[1]),
                archive_member=_as_str(row[2]),
                row_number=_as_int(row[3]),
                example=OrientedExample(
                    label=_as_int(row[4]),
                    side_a_keys=_as_keys(row[5]),
                    side_b_keys=_as_keys(row[6]),
                    deck_a_hash=_as_str(row[9]),
                    deck_b_hash=_as_str(row[10]),
                    side_a_player_id=_as_str(row[11]),
                    side_b_player_id=_as_str(row[12]),
                ),
                side_a_levels=_as_levels(row[7]),
                side_b_levels=_as_levels(row[8]),
            )
