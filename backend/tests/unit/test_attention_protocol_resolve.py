"""Resolve timestamp tails and canonical slice identities from tiny Parquet inputs."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from clash_sos.application.attention_protocol_resolve import (
    AttentionProtocolResolveError,
    resolve_attention_slice,
    resolve_watch_boundary,
)
from clash_sos.domain.attention_protocol import digest_row_keys

START = datetime(2026, 6, 1, tzinfo=UTC)
END = START + timedelta(days=4)


def sources(
    tmp_path: Path, stamps: tuple[datetime, ...], *, duplicate_split: bool = False
) -> tuple[Path, Path]:
    """Write joined keys with two rows at each supplied timestamp."""
    canonical = tmp_path / "canonical.parquet"
    split = tmp_path / "splits-temporal.parquet"
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    connection.execute(
        "CREATE TABLE rows (timestamp TIMESTAMPTZ, fingerprint VARCHAR, "
        "archive_member VARCHAR, row_number BIGINT, partition VARCHAR)"
    )
    for index, stamp in enumerate(stamps):
        connection.execute(
            "INSERT INTO rows VALUES (?, ?, 'part.parquet', ?, 'train')",
            [stamp, f"fp-{index}", index],
        )
    connection.execute(f"COPY (SELECT * EXCLUDE partition FROM rows) TO '{canonical}'")
    connection.execute(f"COPY (SELECT * FROM rows) TO '{split}'")
    if duplicate_split:
        connection.execute("INSERT INTO rows SELECT * FROM rows LIMIT 1")
        split.unlink()
        connection.execute(f"COPY (SELECT * FROM rows) TO '{split}'")
    connection.close()
    return canonical, split


def test_watch_boundary_keeps_ties_and_favors_more_fit_rows(tmp_path: Path) -> None:
    stamps = tuple(START + timedelta(days=day) for day in (0, 0, 1, 1, 2, 2))
    canonical, split = sources(tmp_path, stamps)
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    try:
        boundary = resolve_watch_boundary(
            connection,
            canonical_path=canonical,
            split_path=split,
            start=START,
            end=END,
            target_fraction=0.5,
            batch_rows=1,
        )
    finally:
        connection.close()
    assert boundary.timestamp == START + timedelta(days=2)
    assert (boundary.fit_rows, boundary.watch_rows) == (4, 2)
    assert boundary.actual_fraction == pytest.approx(1 / 3)


def test_slice_resolves_count_and_sorted_row_digest(tmp_path: Path) -> None:
    stamps = (START, START, START + timedelta(days=1))
    canonical, split = sources(tmp_path, stamps)
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    try:
        result = resolve_attention_slice(
            connection,
            canonical_path=canonical,
            split_path=split,
            partition="train",
            start=START,
            end=START + timedelta(days=1),
            batch_rows=1,
        )
    finally:
        connection.close()
    assert result.row_count == 2
    assert result.row_keys_sha256 == digest_row_keys(
        ((START, "fp-0", "part.parquet", 0), (START, "fp-1", "part.parquet", 1))
    )


def test_duplicate_join_and_empty_slice_fail(tmp_path: Path) -> None:
    canonical, split = sources(tmp_path, (START,), duplicate_split=True)
    connection = duckdb.connect()
    try:
        with pytest.raises(AttentionProtocolResolveError, match="unique and sorted"):
            resolve_attention_slice(
                connection,
                canonical_path=canonical,
                split_path=split,
                partition="train",
                start=START,
                end=END,
            )
        with pytest.raises(AttentionProtocolResolveError, match="empty"):
            resolve_attention_slice(
                connection,
                canonical_path=canonical,
                split_path=split,
                partition="validation",
                start=START,
                end=END,
            )
    finally:
        connection.close()


def test_watch_rejects_invalid_fraction_and_unsplittable_ties(tmp_path: Path) -> None:
    canonical, split = sources(tmp_path, (START, START))
    connection = duckdb.connect()
    try:
        for fraction in (0.0, 1.0):
            with pytest.raises(AttentionProtocolResolveError, match="fraction"):
                resolve_watch_boundary(
                    connection,
                    canonical_path=canonical,
                    split_path=split,
                    start=START,
                    end=END,
                    target_fraction=fraction,
                )
        with pytest.raises(AttentionProtocolResolveError, match="distinct timestamps"):
            resolve_watch_boundary(
                connection,
                canonical_path=canonical,
                split_path=split,
                start=START,
                end=END,
                target_fraction=0.5,
            )
    finally:
        connection.close()
