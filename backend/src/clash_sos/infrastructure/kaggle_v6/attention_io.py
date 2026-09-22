"""Build and publish reusable attention inputs from a frozen experiment protocol.

The selected published dataset is read-only; incomplete output stays in an
owned sibling workspace and is never mistaken for a finished cache.
"""

from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from clash_sos.application.dataset_staging import StagingConfig
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
    AttentionCache,
    load_attention_cache,
)
from clash_sos.infrastructure.kaggle_v6.attention_cache_write import write_attention_partition
from clash_sos.infrastructure.kaggle_v6.attention_sources import validate_attention_sources
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.publish_io import publish_processed_version
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb


def _declared_slices(protocol: AttentionProtocol) -> dict[SliceRole, AttentionSlice]:
    """Collect each present role once, preserving protocol order."""
    possible: tuple[tuple[SliceRole, AttentionSlice | None], ...] = (
        ("selection_fit", protocol.selection_fit),
        ("watch", protocol.watch),
        ("refit", protocol.refit),
        ("development", protocol.development),
        ("calibration", protocol.calibration),
        ("reporting", protocol.reporting),
    )
    return {role: item for role, item in possible if item is not None}


def _inventory(
    workspace: Path, partitions: tuple[CachePartition, ...], chunk_size: int
) -> tuple[CacheFile, ...]:
    """Hash only declared cache members, never temporary DuckDB spill files."""
    paths = sorted(
        path
        for item in partitions
        for path in (item.tokens_path, item.labels_path, *item.sidecar_paths)
    )
    return tuple(
        CacheFile(path=path, size_bytes=size, sha256=digest)
        for path in paths
        for size, digest in (hash_file(workspace / path, chunk_size),)
    )


def build_attention_cache(
    dataset: Path,
    destination: Path,
    protocol: AttentionProtocol,
    schema: AttentionCardSchema,
    *,
    config: StagingConfig | None = None,
    sidecar_rows: int = 50_000,
) -> AttentionCache:
    """Reuse an exact cache or atomically publish a fully verified new one."""
    settings = config or StagingConfig()
    if sidecar_rows < 1:
        raise ValueError("sidecar_rows must be positive")
    if destination.exists():
        return load_attention_cache(
            dataset, destination, protocol, schema, chunk_size=settings.chunk_size
        )
    sources = validate_attention_sources(dataset, protocol, schema, chunk_size=settings.chunk_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    workspace = destination.with_name(f".{destination.name}.building-{uuid4().hex}")
    workspace.mkdir()
    temp_directory = workspace / "duckdb-temp"
    temp_directory.mkdir()
    connection = None
    try:
        connection = connect_staging_duckdb(
            memory_limit=settings.memory_limit,
            threads=settings.threads,
            temp_directory=temp_directory,
        )
        declared = _declared_slices(protocol)
        partitions: list[CachePartition] = []
        offsets: list[CacheSliceRange] = []
        partition_order: tuple[Partition, ...] = ("train", "validation", "test")
        for partition in partition_order:
            selected: dict[SliceRole, AttentionSlice] = {
                role: item for role, item in declared.items() if item.partition == partition
            }
            if not selected:
                continue
            stored, ranges = write_attention_partition(
                connection,
                canonical_path=sources.canonical_path,
                split_path=sources.split_path,
                partition=partition,
                expected_rows=sources.partition_counts[partition],
                mirror_seed=protocol.mirror_seed,
                schema=schema,
                slices=selected,
                workspace=workspace,
                batch_rows=settings.batch_rows,
                sidecar_rows=sidecar_rows,
            )
            partitions.append(stored)
            offsets.extend(ranges)
        connection.close()
        connection = None
        rmtree(temp_directory)
        manifest = AttentionCacheManifest(
            protocol=protocol,
            encoding_sha256=schema.fingerprint(),
            partitions=tuple(partitions),
            slices=tuple(offsets),
            files=_inventory(workspace, tuple(partitions), settings.chunk_size),
            watch_actual_fraction=protocol.watch.row_count / protocol.refit.row_count,
        )
        (workspace / "manifest.json").write_bytes(dump_cache_manifest(manifest))
        load_attention_cache(dataset, workspace, protocol, schema, chunk_size=settings.chunk_size)
        publish_processed_version(workspace, destination)
        return AttentionCache(destination, manifest)
    finally:
        if connection is not None:
            connection.close()
        if workspace.exists():
            rmtree(workspace, ignore_errors=True)
