"""Build train-only deck support for attention evaluation on local disk.

The index reads sidecar deck hashes from the refit slice and never reads labels.
SQLite bounds memory even when nearly every observed deck pair is unique.
"""

import sqlite3
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from clash_sos.domain.attention_cache import SliceRole
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import AttentionCache


def _unordered_pair(first: str, second: str) -> tuple[str, str]:
    """Use one key for both orientations of a deck matchup."""
    return (first, second) if first <= second else (second, first)


def iter_attention_sidecars(cache: AttentionCache, role: SliceRole) -> Iterator[dict[str, object]]:
    """Yield one verified slice's sidecars in cache-array ordinal order."""
    offset = next((item for item in cache.manifest.slices if item.role == role), None)
    if offset is None:
        raise ValueError(f"cache has no {role} slice")
    partition = next(
        item for item in cache.manifest.partitions if item.partition == offset.partition
    )
    expected = offset.start
    for relative in partition.sidecar_paths:
        if expected == offset.stop:
            return
        for raw in pl.read_parquet(cache.directory / relative).iter_rows(named=True):
            row: dict[str, object] = raw
            ordinal = row.get("row_ordinal")
            if not isinstance(ordinal, int):
                raise ValueError("attention sidecar ordinal is invalid")
            if ordinal < offset.start:
                continue
            if ordinal >= offset.stop:
                break
            if ordinal != expected:
                raise ValueError("attention sidecar is not aligned with cache arrays")
            yield row
            expected += 1
    if expected != offset.stop:
        raise ValueError("attention sidecar slice is incomplete")


class AttentionSupportIndex:
    """Count training deck and unordered-pair appearances without outcome data."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self.connection = connection

    @classmethod
    def build(cls, cache: AttentionCache, path: Path) -> "AttentionSupportIndex":
        """Create a fresh local index; remove it if the build fails."""
        if path.exists():
            raise FileExistsError(f"attention support index already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                "CREATE TABLE decks (hash TEXT PRIMARY KEY, appearances INTEGER NOT NULL);"
                "CREATE TABLE pairs (first_hash TEXT NOT NULL, second_hash TEXT NOT NULL,"
                " appearances INTEGER NOT NULL, PRIMARY KEY(first_hash, second_hash));"
            )
            decks: Counter[str] = Counter()
            pairs: Counter[tuple[str, str]] = Counter()
            pending = 0
            for row in iter_attention_sidecars(cache, "refit"):
                first, second = row.get("deck_a_hash"), row.get("deck_b_hash")
                if not isinstance(first, str) or not isinstance(second, str):
                    raise ValueError("attention sidecar deck hashes are invalid")
                decks[first] += 1
                decks[second] += 1
                pairs[_unordered_pair(first, second)] += 1
                pending += 1
                if pending == 50_000:
                    cls._flush(connection, decks, pairs)
                    pending = 0
            cls._flush(connection, decks, pairs)
            return cls(path, connection)
        except BaseException:
            connection.close()
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _flush(
        connection: sqlite3.Connection,
        decks: Counter[str],
        pairs: Counter[tuple[str, str]],
    ) -> None:
        """Combine each bounded chunk before applying indexed upserts."""
        with connection:
            connection.executemany(
                "INSERT INTO decks VALUES (?, ?) ON CONFLICT(hash) DO UPDATE "
                "SET appearances = appearances + excluded.appearances",
                decks.items(),
            )
            connection.executemany(
                "INSERT INTO pairs VALUES (?, ?, ?) ON CONFLICT(first_hash, second_hash) "
                "DO UPDATE SET appearances = appearances + excluded.appearances",
                ((first, second, count) for (first, second), count in pairs.items()),
            )
        decks.clear()
        pairs.clear()

    def lookup(self, first: str, second: str) -> tuple[int, int, int]:
        """Return refit deck supports and unordered-pair support for one matchup."""
        pair = _unordered_pair(first, second)
        a = self.connection.execute("SELECT appearances FROM decks WHERE hash = ?", (first,))
        b = self.connection.execute("SELECT appearances FROM decks WHERE hash = ?", (second,))
        both = self.connection.execute(
            "SELECT appearances FROM pairs WHERE first_hash = ? AND second_hash = ?", pair
        )
        a_row, b_row, pair_row = a.fetchone(), b.fetchone(), both.fetchone()
        return (
            int(a_row[0]) if a_row else 0,
            int(b_row[0]) if b_row else 0,
            int(pair_row[0]) if pair_row else 0,
        )

    def close(self) -> None:
        """Release the local database before its owning workspace is removed."""
        self.connection.close()
