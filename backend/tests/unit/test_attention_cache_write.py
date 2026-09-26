"""Check streamed tokens, mirror semantics, and failures on tiny Parquet joins."""

from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest
from attention_cache_fixture import LOSE, WIN, write_cache_dataset
from catalog_fixture import expanded_catalog

from clash_sos.domain.attention_schema import AttentionModelConfig, build_attention_schema
from clash_sos.domain.canonical_dataset import deck_content_hash
from clash_sos.infrastructure.kaggle_v6.attention_cache_write import (
    AttentionCacheWriteError,
    write_attention_partition,
)


def test_partition_writer_streams_aligned_mirrored_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    try:
        partition, slices = write_attention_partition(
            connection,
            canonical_path=dataset / "canonical.parquet",
            split_path=dataset / "splits-temporal.parquet",
            partition="train",
            expected_rows=2,
            mirror_seed=0,
            schema=schema,
            slices={
                "selection_fit": protocol.selection_fit,
                "watch": protocol.watch,
                "refit": protocol.refit,
            },
            workspace=tmp_path / "cache",
            batch_rows=1,
            sidecar_rows=1,
        )
    finally:
        connection.close()
    tokens = np.load(tmp_path / "cache" / partition.tokens_path, mmap_mode="r")
    labels = np.load(tmp_path / "cache" / partition.labels_path, mmap_mode="r")
    assert tokens.shape == (2, 2, 8)
    assert tokens.dtype == np.uint16
    assert labels.dtype == np.uint8
    assert labels.tolist() == [1, 0]
    assert tokens[0, 0].tolist() == list(schema.encode_deck(tuple(f"{card}:base" for card in WIN)))
    assert tokens[1, 0].tolist() == list(schema.encode_deck(tuple(f"{card}:base" for card in LOSE)))
    assert len(partition.sidecar_paths) == 2
    sidecars = [pl.read_parquet(tmp_path / "cache" / path) for path in partition.sidecar_paths]
    assert [part["row_ordinal"][0] for part in sidecars] == [0, 1]
    assert sidecars[1]["side_a_player_id"][0] == "#LOSE1"
    assert [(item.role, item.start, item.stop) for item in slices] == [
        ("selection_fit", 0, 1),
        ("watch", 1, 2),
        ("refit", 0, 2),
    ]


def test_partition_writer_rejects_count_and_duplicate_join(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    connection = duckdb.connect()
    try:
        with pytest.raises(AttentionCacheWriteError, match="count 2 != 3"):
            write_attention_partition(
                connection,
                canonical_path=dataset / "canonical.parquet",
                split_path=dataset / "splits-temporal.parquet",
                partition="train",
                expected_rows=3,
                mirror_seed=0,
                schema=schema,
                slices={"refit": protocol.refit},
                workspace=tmp_path / "wrong-count",
            )
        duplicate = tmp_path / "duplicate.parquet"
        connection.execute(
            f"COPY (SELECT * FROM read_parquet('{dataset / 'splits-temporal.parquet'}') "
            "UNION ALL SELECT * FROM read_parquet(?) WHERE row_number = 0) "
            f"TO '{duplicate}' (FORMAT PARQUET)",
            [str(dataset / "splits-temporal.parquet")],
        )
        with pytest.raises(AttentionCacheWriteError, match="unique and sorted"):
            write_attention_partition(
                connection,
                canonical_path=dataset / "canonical.parquet",
                split_path=duplicate,
                partition="train",
                expected_rows=3,
                mirror_seed=0,
                schema=schema,
                slices={"refit": protocol.refit},
                workspace=tmp_path / "duplicate-cache",
            )
    finally:
        connection.close()


def test_partition_writer_rejects_invalid_card_level(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    canonical = tmp_path / "bad-level.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(
            "COPY (SELECT * REPLACE (CASE WHEN row_number = 0 THEN "
            "[15,16,16,16,16,16,16,16] ELSE side_a_card_levels END "
            f"AS side_a_card_levels) FROM read_parquet('{dataset / 'canonical.parquet'}')) "
            f"TO '{canonical}' (FORMAT PARQUET)",
        )
        with pytest.raises(AttentionCacheWriteError, match="level 16"):
            write_attention_partition(
                connection,
                canonical_path=canonical,
                split_path=dataset / "splits-temporal.parquet",
                partition="train",
                expected_rows=2,
                mirror_seed=0,
                schema=schema,
                slices={"refit": protocol.refit},
                workspace=tmp_path / "invalid-cache",
            )
    finally:
        connection.close()


def test_partition_writer_preserves_tokens_above_255(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _, protocol = write_cache_dataset(dataset)
    catalog, attributes = expanded_catalog()
    schema = build_attention_schema(
        catalog.serialize(), attributes=attributes, network=AttentionModelConfig()
    )
    cards = tuple(f"z-future-{index:03}" for index in range(82, 90))
    expected = schema.encode_deck(tuple(f"{card}:base" for card in cards))
    assert max(expected) > 255
    canonical = tmp_path / "expanded.parquet"
    pl.read_parquet(dataset / "canonical.parquet").with_columns(
        pl.lit(list(cards)).alias("side_a_card_ids"),
        pl.lit(deck_content_hash(cards, ("base",) * 8, (16,) * 8)).alias("side_a_deck_hash"),
    ).write_parquet(canonical)
    connection = duckdb.connect()
    try:
        partition, _ = write_attention_partition(
            connection,
            canonical_path=canonical,
            split_path=dataset / "splits-temporal.parquet",
            partition="train",
            expected_rows=2,
            mirror_seed=0,
            schema=schema,
            slices={"refit": protocol.refit},
            workspace=tmp_path / "cache",
        )
    finally:
        connection.close()
    tokens = np.load(tmp_path / "cache" / partition.tokens_path)
    assert tokens.dtype == np.uint16
    assert tokens[0, 0].tolist() == list(expected)
    assert tokens[1, 1].tolist() == list(expected)
