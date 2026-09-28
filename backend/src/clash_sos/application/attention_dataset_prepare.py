"""Publish normalized official battles with frozen temporal attention populations."""

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from shutil import rmtree
from typing import Literal, cast
from uuid import uuid4

from clash_sos.application.attention_protocol_resolve import (
    resolve_attention_slice,
    resolve_watch_boundary,
)
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.attention_cache import CacheFile
from clash_sos.domain.attention_dataset import AttentionDatasetManifest
from clash_sos.domain.attention_protocol import AttentionProtocol, AttentionSlice, Partition
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.infrastructure.clash_royale.attention_snapshot import (
    export_snapshot,
    write_snapshot_parts,
)
from clash_sos.infrastructure.kaggle_v6.attention_sources import validate_attention_sources
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.publish_io import publish_processed_version
from clash_sos.infrastructure.kaggle_v6.staging_io import connect_staging_duckdb


def prepare_official_attention_dataset(
    source: Path,
    destination: Path,
    *,
    schema: AttentionCardSchema,
    dataset_version: str,
    start: datetime,
    train_end: datetime,
    validation_end: datetime,
    end: datetime,
    watch_fraction: float = 0.1,
    mirror_seed: int = 0,
    config: StagingConfig | None = None,
) -> Path:
    """Freeze one era of official rows under their exact input schema and split."""
    if schema.tower_catalog is None:
        raise ValueError("official snapshot preparation requires a tower-aware schema")
    bounds = (start, train_end, validation_end, end)
    if any(value.tzinfo is None or value.utcoffset() is None for value in bounds):
        raise ValueError("snapshot bounds must be timezone-aware")
    if not start < train_end < validation_end < end:
        raise ValueError("snapshot bounds must increase strictly")
    if not 0 < watch_fraction < 1 or mirror_seed < 0:
        raise ValueError("watch fraction must be between zero and one; seed must be nonnegative")
    if not source.is_file():
        raise ValueError("normalized official JSONL source is required")
    if destination.exists():
        raise ValueError("published official snapshot already exists")
    settings = config or StagingConfig()
    destination.parent.mkdir(parents=True, exist_ok=True)
    workspace = destination.with_name(f".{destination.name}.building-{uuid4().hex}")
    workspace.mkdir()
    temp = workspace / "duckdb-temp"
    temp.mkdir()
    connection = None
    try:
        count = write_snapshot_parts(
            source,
            workspace / "parts",
            schema=schema,
            dataset_version=dataset_version,
            start=start,
            end=end,
            batch_rows=settings.batch_rows,
        )
        connection = connect_staging_duckdb(
            memory_limit=settings.memory_limit,
            threads=settings.threads,
            temp_directory=temp,
        )
        counts = export_snapshot(
            connection,
            workspace,
            train_end=train_end,
            validation_end=validation_end,
        )
        files = tuple(
            CacheFile(path=name, size_bytes=size, sha256=digest)
            for name in ("canonical.parquet", "splits-temporal.parquet")
            for size, digest in (hash_file(workspace / name, settings.chunk_size),)
        )
        manifest = AttentionDatasetManifest(
            canonical_schema_version=cast(
                Literal["official-ranked16-schema:v1", "official-ranked16-schema:v2"],
                schema.canonical_schema_version,
            ),
            dataset_version=dataset_version,
            balance_era_id=schema.balance_era_id,
            catalog_version=schema.catalog_version,
            tower_catalog_version=schema.tower_catalog.catalog_version,
            row_count=count,
            partition_counts={
                "train": counts.get("train", 0),
                "validation": counts.get("validation", 0),
                "test": counts.get("test", 0),
            },
            start=start,
            train_end=train_end,
            validation_end=validation_end,
            end=end,
            files=files,
        )
        contents = canonical_json_bytes(manifest.model_dump(mode="python")) + b"\n"
        (workspace / "manifest.json").write_bytes(contents)
        canonical, split = workspace / files[0].path, workspace / files[1].path
        boundary = resolve_watch_boundary(
            connection,
            canonical_path=canonical,
            split_path=split,
            start=start,
            end=train_end,
            target_fraction=watch_fraction,
            batch_rows=settings.batch_rows,
        )
        active_connection = connection

        def resolved(partition: Partition, first: datetime, last: datetime) -> AttentionSlice:
            return resolve_attention_slice(
                active_connection,
                canonical_path=canonical,
                split_path=split,
                partition=partition,
                start=first,
                end=last,
                batch_rows=settings.batch_rows,
            )

        protocol = AttentionProtocol(
            family="temporal",
            dataset_version=dataset_version,
            balance_era_id=schema.balance_era_id,
            processed_manifest_sha256=sha256(contents).hexdigest(),
            canonical_sha256=files[0].sha256,
            split_sha256=files[1].sha256,
            split_file="splits-temporal.parquet",
            encoding_sha256=schema.fingerprint(),
            mirror_seed=mirror_seed,
            selection_fit=resolved("train", start, boundary.timestamp),
            watch=resolved("train", boundary.timestamp, train_end),
            refit=resolved("train", start, train_end),
            development=resolved("validation", train_end, validation_end),
            reporting=resolved("test", validation_end, end),
        )
        (workspace / "protocol.json").write_bytes(
            canonical_json_bytes(protocol.model_dump(mode="python")) + b"\n"
        )
        (workspace / "feature-schema.json").write_bytes(
            canonical_json_bytes(schema.model_dump(mode="python")) + b"\n"
        )
        validate_attention_sources(workspace, protocol, schema, chunk_size=settings.chunk_size)
        connection.close()
        connection = None
        rmtree(workspace / "parts")
        rmtree(temp)
        return publish_processed_version(workspace, destination)
    finally:
        if connection is not None:
            connection.close()
        if workspace.exists():
            rmtree(workspace, ignore_errors=True)
