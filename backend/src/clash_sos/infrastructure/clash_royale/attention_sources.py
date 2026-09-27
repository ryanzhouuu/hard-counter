"""Verify tower-aware snapshots without attributing them to the Kaggle corpus."""

from pathlib import Path

from clash_sos.domain.attention_dataset import AttentionDatasetManifest
from clash_sos.domain.attention_protocol import AttentionProtocol, Partition
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file


def validate_official_sources(
    dataset: Path,
    manifest_bytes: bytes,
    protocol: AttentionProtocol,
    schema: AttentionCardSchema,
    chunk_size: int,
) -> tuple[Path, Path, dict[Partition, int]]:
    manifest = AttentionDatasetManifest.model_validate_json(manifest_bytes)
    if (
        protocol.family != "temporal"
        or schema.tower_catalog is None
        or schema.canonical_schema_version != manifest.canonical_schema_version
        or manifest.dataset_version != protocol.dataset_version
        or manifest.balance_era_id != protocol.balance_era_id
        or schema.balance_era_id != protocol.balance_era_id
        or manifest.catalog_version != schema.catalog_version
        or manifest.tower_catalog_version != schema.tower_catalog.catalog_version
        or schema.fingerprint() != protocol.encoding_sha256
    ):
        raise ValueError("official snapshot, protocol, and encoding disagree")
    expected_hashes = {
        "canonical.parquet": protocol.canonical_sha256,
        "splits-temporal.parquet": protocol.split_sha256,
    }
    for file in manifest.files:
        path = dataset / file.path
        if not path.is_file():
            raise ValueError(f"official snapshot input is missing: {file.path}")
        size, digest = hash_file(path, chunk_size)
        if (size, digest) != (file.size_bytes, file.sha256) or digest != expected_hashes[file.path]:
            raise ValueError(f"official snapshot input hash mismatch: {file.path}")
    return (
        dataset / "canonical.parquet",
        dataset / "splits-temporal.parquet",
        manifest.partition_counts,
    )
