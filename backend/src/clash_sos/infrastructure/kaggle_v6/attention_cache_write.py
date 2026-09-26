"""Stream one oriented split partition into disk-backed attention inputs.

The caller owns the unpublished workspace and removes it after any failure.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import chain
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from clash_sos.domain.attention_cache import CachePartition, CacheSliceRange, SliceRole
from clash_sos.domain.attention_protocol import AttentionSlice, Partition, RowKey, RowKeyDigest
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.infrastructure.kaggle_v6.staging_io import python_cell
from clash_sos.infrastructure.kaggle_v6.train_io import iter_oriented_cache_rows


class AttentionCacheWriteError(ValueError):
    """Joined rows cannot satisfy the declared partition or slice identities."""


@dataclass
class _SliceState:
    """Track one protocol slice while a partition is written only once."""

    role: SliceRole
    source: AttentionSlice
    digest: RowKeyDigest = field(default_factory=RowKeyDigest)
    start: int | None = None
    stop: int | None = None
    count: int = 0

    def update(self, row: RowKey, ordinal: int) -> None:
        """Add an in-bounds row and preserve its contiguous offset range."""
        if self.source.start <= row[0] < self.source.end:
            if self.start is None:
                self.start = ordinal
            self.stop = ordinal + 1
            self.count += 1
            self.digest.update(row)

    def finish(self, partition: Partition) -> CacheSliceRange:
        """Refuse counts or row identities that differ from the frozen protocol."""
        if (
            self.start is None
            or self.stop is None
            or self.count != self.source.row_count
            or self.stop - self.start != self.count
            or self.digest.hexdigest() != self.source.row_keys_sha256
        ):
            raise AttentionCacheWriteError(f"{self.role} slice count or row digest mismatch")
        return CacheSliceRange(
            role=self.role, partition=partition, start=self.start, stop=self.stop
        )


def _write_sidecar(rows: list[dict[str, object]], path: Path) -> None:
    """Flush a bounded metadata chunk with ordinal and canonical row identity."""
    pl.DataFrame(rows).write_parquet(path, compression="zstd")


def _partition_windows(
    connection: duckdb.DuckDBPyConnection, split_path: Path, partition: Partition
) -> tuple[tuple[datetime, datetime], ...]:
    """Keep the ordered join below DuckDB's 1 GB working-memory limit."""
    row = connection.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM read_parquet(?) WHERE partition = ?",
        [str(split_path), partition],
    ).fetchone()
    if row is None:
        raise AttentionCacheWriteError("split partition is empty")
    minimum, maximum = python_cell(row[0]), python_cell(row[1])
    if not isinstance(minimum, datetime) or not isinstance(maximum, datetime):
        raise AttentionCacheWriteError("split partition timestamps are invalid")
    minimum = minimum.astimezone(UTC)
    maximum = maximum.astimezone(UTC)
    current = minimum.replace(hour=minimum.hour // 6 * 6, minute=0, second=0, microsecond=0)
    step = timedelta(hours=6)
    windows: list[tuple[datetime, datetime]] = []
    while current <= maximum:
        windows.append((current, current + step))
        current += step
    return tuple(windows)


def write_attention_partition(
    connection: duckdb.DuckDBPyConnection,
    *,
    canonical_path: Path,
    split_path: Path,
    partition: Partition,
    expected_rows: int,
    mirror_seed: int,
    schema: AttentionCardSchema,
    slices: dict[SliceRole, AttentionSlice],
    workspace: Path,
    batch_rows: int = 10_000,
    sidecar_rows: int = 50_000,
) -> tuple[CachePartition, tuple[CacheSliceRange, ...]]:
    """Validate and write a single ordered partition without retaining examples."""
    if expected_rows < 1 or batch_rows < 1 or sidecar_rows < 1:
        raise AttentionCacheWriteError("partition and batch sizes must be positive")
    if any(item.partition != partition for item in slices.values()):
        raise AttentionCacheWriteError("slice partition does not match output partition")
    directory = workspace / partition
    directory.mkdir(parents=True, exist_ok=False)
    tokens_path = directory / "tokens.npy"
    labels_path = directory / "labels.npy"
    tokens = np.lib.format.open_memmap(
        tokens_path, mode="w+", dtype=np.uint16, shape=(expected_rows, 2, 8)
    )
    labels = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.uint8, shape=(expected_rows,)
    )
    states = tuple(_SliceState(role, item) for role, item in slices.items())
    sidecar: list[dict[str, object]] = []
    sidecar_paths: list[str] = []
    previous: RowKey | None = None
    count = 0
    rows = chain.from_iterable(
        iter_oriented_cache_rows(
            connection,
            canonical_path=canonical_path,
            split_path=split_path,
            partition=partition,
            seed=mirror_seed,
            batch_rows=batch_rows,
            start=start,
            end=end,
        )
        for start, end in _partition_windows(connection, split_path, partition)
    )
    for row in rows:
        key: RowKey = (row.timestamp, row.fingerprint, row.archive_member, row.row_number)
        if previous is not None and key <= previous:
            raise AttentionCacheWriteError("joined row keys must be unique and sorted")
        if count >= expected_rows:
            raise AttentionCacheWriteError("joined partition exceeds expected row count")
        if row.example.label not in (0, 1):
            raise AttentionCacheWriteError("oriented label must be binary")
        try:
            tokens[count, 0, :] = schema.encode_deck(
                row.example.side_a_keys, levels=row.side_a_levels
            )
            tokens[count, 1, :] = schema.encode_deck(
                row.example.side_b_keys, levels=row.side_b_levels
            )
        except ValueError as error:
            raise AttentionCacheWriteError(f"invalid deck at row {count}: {error}") from error
        labels[count] = row.example.label
        sidecar.append(
            {
                "row_ordinal": count,
                "timestamp": row.timestamp,
                "fingerprint": row.fingerprint,
                "archive_member": row.archive_member,
                "row_number": row.row_number,
                "deck_a_hash": row.example.deck_a_hash,
                "deck_b_hash": row.example.deck_b_hash,
                "side_a_player_id": row.example.side_a_player_id,
                "side_b_player_id": row.example.side_b_player_id,
            }
        )
        for state in states:
            state.update(key, count)
        if len(sidecar) == sidecar_rows:
            relative = f"{partition}/sidecar-{len(sidecar_paths):05d}.parquet"
            _write_sidecar(sidecar, workspace / relative)
            sidecar_paths.append(relative)
            sidecar.clear()
        previous = key
        count += 1
    if sidecar:
        relative = f"{partition}/sidecar-{len(sidecar_paths):05d}.parquet"
        _write_sidecar(sidecar, workspace / relative)
        sidecar_paths.append(relative)
    tokens.flush()
    labels.flush()
    del tokens, labels
    if count != expected_rows:
        raise AttentionCacheWriteError(f"joined partition count {count} != {expected_rows}")
    offsets = tuple(state.finish(partition) for state in states)
    return (
        CachePartition(
            partition=partition,
            row_count=count,
            tokens_path=f"{partition}/tokens.npy",
            labels_path=f"{partition}/labels.npy",
            sidecar_paths=tuple(sidecar_paths),
        ),
        offsets,
    )
