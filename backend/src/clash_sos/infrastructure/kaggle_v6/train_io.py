"""DuckDB join, deterministic mirroring, and count aggregates for matchup training.

Reads canonical battles joined to a split parquet. Winner-first rows are mirrored
with the same SHA-256 rule as `should_mirror_sides` so labels are not constantly 1.
"""

from collections.abc import Iterator
from dataclasses import dataclass
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


_ORIENTED_SQL = """
WITH joined AS (
    SELECT
        c.side_a_card_ids,
        c.side_a_card_forms,
        c.side_b_card_ids,
        c.side_b_card_forms,
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
    WHERE s.partition = ?
),
oriented AS (
    SELECT
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
        CASE WHEN mirrored THEN side_b_deck_hash ELSE side_a_deck_hash END AS deck_a_hash,
        CASE WHEN mirrored THEN side_a_deck_hash ELSE side_b_deck_hash END AS deck_b_hash
    FROM joined
)
"""


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
) -> Iterator[OrientedExample]:
    """Yield mirrored examples for scoring without materializing the full corpus."""
    if batch_rows < 1:
        raise KaggleV6TrainError("batch_rows must be positive")
    _require_paths(canonical_path, split_path)
    result = connection.execute(
        f"""
        {_ORIENTED_SQL}
        SELECT label, side_a_keys, side_b_keys, deck_a_hash, deck_b_hash
        FROM oriented
        """,
        _oriented_params(canonical_path, split_path, partition=partition, seed=seed),
    )
    while True:
        batch = result.fetchmany(batch_rows)
        if not batch:
            return
        for label, side_a, side_b, deck_a, deck_b in batch:
            yield OrientedExample(
                label=_as_int(label),
                side_a_keys=_as_keys(side_a),
                side_b_keys=_as_keys(side_b),
                deck_a_hash=_as_str(deck_a),
                deck_b_hash=_as_str(deck_b),
            )
