from hashlib import sha256
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest
from attention_cache_fixture import write_cache_dataset
from test_tower_attention import tower_schema

from clash_sos.infrastructure.kaggle_v6.attention_cache_read import _check_arrays
from clash_sos.infrastructure.kaggle_v6.attention_cache_write import write_attention_partition


def test_cache_mirrors_towers_with_their_decks_and_checks_slot_types(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _, protocol = write_cache_dataset(dataset)
    schema = tower_schema()
    canonical = dataset / "canonical.parquet"
    pl.read_parquet(canonical).with_columns(
        pl.lit("cannoneer:tower").alias("side_a_tower"),
        pl.lit("dagger-duchess:tower").alias("side_b_tower"),
        pl.lit(16).alias("side_a_tower_level"),
        pl.lit(16).alias("side_b_tower_level"),
    ).write_parquet(canonical)
    output = tmp_path / "cache"
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
            workspace=output,
        )
    finally:
        connection.close()
    path = output / partition.tokens_path
    tokens = np.load(path)
    assert tokens.shape == (2, 2, 9)
    assert tokens[:, :, 8].tolist() == [
        [schema.identity_index["cannoneer:tower"], schema.identity_index["dagger-duchess:tower"]],
        [schema.identity_index["dagger-duchess:tower"], schema.identity_index["cannoneer:tower"]],
    ]
    source = pl.read_parquet(canonical).sort("timestamp").row(0, named=True)
    sidecar = pl.read_parquet(output / partition.sidecar_paths[0])
    expected = sha256(f"{source['side_a_deck_hash']}|cannoneer:tower|16".encode()).hexdigest()
    assert sidecar.row(0, named=True)["deck_a_hash"] == expected
    _check_arrays(output, partition, schema)
    tokens[0, 0, 0], tokens[0, 0, 8] = tokens[0, 0, 8], tokens[0, 0, 0]
    np.save(path, tokens)
    with pytest.raises(ValueError, match="ninth slot"):
        _check_arrays(output, partition, schema)
    tokens[0, 0, 0] = len(schema.identity_vocab)
    np.save(path, tokens)
    with pytest.raises(ValueError, match=r"outside.*vocabulary"):
        _check_arrays(output, partition, schema)
