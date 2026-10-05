"""Load bounded immutable official search populations through verified attention caches."""

from pathlib import Path

import duckdb
import numpy as np

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.attention_cache import CachePartition
from clash_sos.domain.attention_dataset import (
    AttentionDatasetManifest,
    official_row_type,
)
from clash_sos.domain.attention_protocol import AttentionProtocol, AttentionSlice
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import AttentionCache
from clash_sos.infrastructure.kaggle_v6.attention_io import build_attention_cache
from experiments.common.artifacts import file_record
from experiments.common.contracts import FileRecord, PopulationIdentity
from experiments.common.data_access import ResearchRow, RoleAccess
from experiments.common.protocol import require_search
from experiments.common.source_rows import encode_official_row, population_identity


def _partition_rows(
    dataset: Path,
    cache: AttentionCache,
    partition: CachePartition,
    slices: tuple[AttentionSlice, ...],
    schema: AttentionCardSchema,
    row_cap: int,
) -> tuple[ResearchRow, ...]:
    paths = [str(cache.directory / path) for path in partition.sidecar_paths]
    condition = " OR ".join("(c.timestamp >= ? AND c.timestamp < ?)" for _ in slices)
    query = f"""
        SELECT c.*, m.row_ordinal AS cache_ordinal,
               m.side_a_player_id AS cache_player_a, m.side_b_player_id AS cache_player_b
        FROM read_parquet(?) c JOIN read_parquet(?) m
        ON c.timestamp = m.timestamp AND c.fingerprint = m.fingerprint
        AND c.archive_member = m.archive_member AND c.row_number = m.row_number
        WHERE {condition}
        ORDER BY c.timestamp, c.fingerprint, c.archive_member, c.row_number
        LIMIT ?
    """
    parameters: list[object] = [str(dataset / "canonical.parquet"), paths]
    parameters.extend(bound for item in slices for bound in (item.start, item.end))
    parameters.append(row_cap + 1)
    tokens = np.load(cache.directory / partition.tokens_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(cache.directory / partition.labels_path, mmap_mode="r", allow_pickle=False)
    rows: list[ResearchRow] = []
    with duckdb.connect(config={"memory_limit": "256MB", "threads": "1"}) as connection:
        cursor = connection.execute(query, parameters)
        names = tuple(item[0] for item in cursor.description)
        records = cursor.fetchmany(row_cap + 1)
        if len(records) > row_cap:
            raise ValueError("source join exceeds explicit row cap")
        for record in records:
            payload: dict[str, object] = dict(zip(names, record, strict=True))
            ordinal = payload.pop("cache_ordinal")
            player_a, player_b = payload.pop("cache_player_a"), payload.pop("cache_player_b")
            source = official_row_type(schema.canonical_schema_version).model_validate(payload)
            if source.dataset_version != cache.manifest.protocol.dataset_version:
                raise ValueError("canonical row dataset version disagrees with frozen protocol")
            row = encode_official_row(source, schema, cache.manifest.protocol.mirror_seed)
            if type(ordinal) is not int or not 0 <= ordinal < partition.row_count:
                raise ValueError("cache ordinal must identify an existing array row")
            encoded = (
                tuple(int(value) for value in tokens[ordinal, 0]),
                tuple(int(value) for value in tokens[ordinal, 1]),
            )
            if (encoded, int(labels[ordinal]), player_a, player_b) != (
                row.tokens,
                row.label,
                row.player_a,
                row.player_b,
            ):
                raise ValueError(
                    "canonical event join disagrees with cache tokens, labels, or players"
                )
            rows.append(row)
    return tuple(rows)


def load_snapshot(
    dataset: Path,
    protocol_path: Path,
    schema_path: Path,
    cache: Path,
    row_cap: int | None = None,
) -> tuple[RoleAccess, PopulationIdentity, AttentionCardSchema]:
    """Fail on cap overflow; tuple-based access never silently truncates frozen roles."""
    if row_cap is None or row_cap <= 0:
        raise ValueError("tuple-based source loading requires an explicit positive row cap")
    manifest = AttentionDatasetManifest.model_validate_json(
        (dataset / "manifest.json").read_bytes()
    )
    protocol = AttentionProtocol.model_validate_json(protocol_path.read_bytes())
    schema = AttentionCardSchema.model_validate_json(schema_path.read_bytes())
    require_search(protocol)
    calibration = protocol.calibration
    assert calibration is not None
    declared = (protocol.refit, calibration, protocol.development)
    if sum(item.row_count for item in declared) > row_cap:
        raise ValueError("snapshot exceeds row cap; prepare a fresh bounded exploratory snapshot")
    verified = build_attention_cache(
        dataset,
        cache,
        protocol,
        schema,
        config=StagingConfig(batch_rows=min(row_cap, 10_000)),
    )
    if any(part.partition == "test" for part in verified.manifest.partitions):
        raise ValueError("search cache cannot contain physical test rows")
    rows = tuple(
        sorted(
            (
                row
                for partition in verified.manifest.partitions
                for row in _partition_rows(
                    dataset,
                    verified,
                    partition,
                    tuple(item for item in declared if item.partition == partition.partition),
                    schema,
                    row_cap,
                )
            ),
            key=lambda row: row.key,
        )
    )
    access = RoleAccess(protocol, rows)
    files = tuple(
        FileRecord(
            path=file.path,
            sha256=file.sha256,
            size_bytes=file.size_bytes,
        )
        for file in manifest.files
    )
    files = (*files, file_record(dataset / "manifest.json", dataset))
    identity = population_identity(
        rows,
        files,
        schema,
        protocol.mirror_seed,
        start=protocol.refit.start,
        end=protocol.development.end,
    )
    return access, identity, schema
