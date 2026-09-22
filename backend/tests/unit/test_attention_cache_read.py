"""Reopen disk caches only when their arrays and metadata remain aligned."""

from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest
from attention_cache_fixture import write_cache_dataset

from clash_sos.domain.attention_cache import (
    AttentionCacheManifest,
    CacheFile,
    CachePartition,
    CacheSliceRange,
    SliceRole,
    dump_cache_manifest,
)
from clash_sos.domain.attention_protocol import AttentionProtocol, AttentionSlice, Partition
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import (
    AttentionCacheReadError,
    load_attention_cache,
)
from clash_sos.infrastructure.kaggle_v6.attention_cache_write import write_attention_partition
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file


def write_cache(tmp_path: Path) -> tuple[Path, Path, AttentionCardSchema, AttentionProtocol]:
    """Publish all fixture partitions using the real streaming writer."""
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    directory = tmp_path / "cache"
    directory.mkdir()
    roles: tuple[SliceRole, ...] = ("selection_fit", "watch", "refit", "development", "reporting")
    partitions: list[CachePartition] = []
    offsets: list[CacheSliceRange] = []
    declared: dict[SliceRole, AttentionSlice | None] = {
        "selection_fit": protocol.selection_fit,
        "watch": protocol.watch,
        "refit": protocol.refit,
        "development": protocol.development,
        "reporting": protocol.reporting,
    }
    partition_rows: tuple[tuple[Partition, int], ...] = (
        ("train", 2),
        ("validation", 2),
        ("test", 1),
    )
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    try:
        for partition_name, expected in partition_rows:
            selected: dict[SliceRole, AttentionSlice] = {
                role: source_slice
                for role in roles
                if (source_slice := declared[role]) is not None
                and source_slice.partition == partition_name
            }
            partition, ranges = write_attention_partition(
                connection,
                canonical_path=dataset / "canonical.parquet",
                split_path=dataset / "splits-temporal.parquet",
                partition=partition_name,
                expected_rows=expected,
                mirror_seed=0,
                schema=schema,
                slices=selected,
                workspace=directory,
                batch_rows=1,
                sidecar_rows=1,
            )
            partitions.append(partition)
            offsets.extend(ranges)
    finally:
        connection.close()
    paths = tuple(
        path
        for partition in partitions
        for path in (partition.tokens_path, partition.labels_path, *partition.sidecar_paths)
    )
    files = tuple(
        CacheFile(path=path, size_bytes=size, sha256=digest)
        for path in paths
        for size, digest in (hash_file(directory / path, 1024),)
    )
    manifest = AttentionCacheManifest(
        protocol=protocol,
        encoding_sha256=schema.fingerprint(),
        partitions=tuple(partitions),
        slices=tuple(offsets),
        files=files,
        watch_actual_fraction=protocol.watch.row_count / protocol.refit.row_count,
    )
    (directory / "manifest.json").write_bytes(dump_cache_manifest(manifest))
    return dataset, directory, schema, protocol


def test_reopened_cache_yields_repeatable_fit_batches_and_ordered_evaluation(
    tmp_path: Path,
) -> None:
    dataset, directory, schema, protocol = write_cache(tmp_path)
    cache = load_attention_cache(dataset, directory, protocol, schema, chunk_size=1024)
    first = list(cache.iter_batches("refit", batch_size=1, seed=4, epoch=2))
    second = list(cache.iter_batches("refit", batch_size=1, seed=4, epoch=2))
    assert len(first) == 2
    assert [labels.tolist() for _, labels in first] == [labels.tolist() for _, labels in second]
    assert all(tokens.shape == (1, 2, 8) and tokens.dtype == np.uint8 for tokens, _ in first)
    evaluation = list(cache.iter_batches("development", batch_size=1, seed=99, epoch=4))
    assert len(evaluation) == 2
    assert [labels.tolist() for _, labels in evaluation] == [
        labels.tolist() for _, labels in cache.iter_batches("development", batch_size=1)
    ]


def test_loader_rejects_stale_source_and_partial_cache(tmp_path: Path) -> None:
    dataset, directory, schema, protocol = write_cache(tmp_path)
    with (dataset / "canonical.parquet").open("ab") as output:
        output.write(b"stale")
    with pytest.raises(AttentionCacheReadError, match="input hash mismatch"):
        load_attention_cache(dataset, directory, protocol, schema)
    with pytest.raises(AttentionCacheReadError, match="manifest is required"):
        load_attention_cache(dataset, tmp_path / "partial", protocol, schema)


def test_loader_rejects_corrupt_file_and_sidecar_alignment(tmp_path: Path) -> None:
    dataset, directory, schema, protocol = write_cache(tmp_path)
    token_path = directory / "train/tokens.npy"
    with token_path.open("ab") as output:
        output.write(b"changed")
    with pytest.raises(AttentionCacheReadError, match="member hash mismatch"):
        load_attention_cache(dataset, directory, protocol, schema)
    with token_path.open("rb+") as output:
        output.truncate(output.seek(0, 2) - len(b"changed"))
    manifest_path = directory / "manifest.json"
    manifest = AttentionCacheManifest.model_validate_json(manifest_path.read_bytes())
    relative = manifest.partitions[0].sidecar_paths[0]
    sidecar_path = directory / relative
    pl.read_parquet(sidecar_path).with_columns(pl.lit(99).alias("row_ordinal")).write_parquet(
        sidecar_path
    )
    files = tuple(
        item.model_copy(update={"size_bytes": size, "sha256": digest})
        if item.path == relative
        else item
        for item in manifest.files
        for size, digest in (hash_file(sidecar_path, 1024),)
    )
    manifest_path.write_bytes(dump_cache_manifest(manifest.model_copy(update={"files": files})))
    with pytest.raises(AttentionCacheReadError, match="ordinals are not contiguous"):
        load_attention_cache(dataset, directory, protocol, schema)
