"""Bind an attention cache to the published processed dataset and its hashes.

Only the selected canonical and split inputs are read; source files remain immutable.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from clash_sos.domain.attention_protocol import AttentionProtocol, Partition
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.domain.processed_manifest import ProcessedDatasetManifest, ProcessedOutputFile
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file
from clash_sos.infrastructure.kaggle_v6.catalog import KAGGLE_V6_CARDS


class AttentionCacheSourceError(ValueError):
    """Published input identity differs from the declared experiment protocol."""


@dataclass(frozen=True)
class AttentionSources:
    """Verified paths and expected split counts for one protocol family."""

    manifest_path: Path
    canonical_path: Path
    split_path: Path
    partition_counts: dict[Partition, int]


def _inventoried_file(
    manifest: ProcessedDatasetManifest, kind: str, expected_name: str
) -> ProcessedOutputFile:
    """Require the selected split name rather than trusting a matching file kind."""
    matches = tuple(item for item in manifest.files if item.kind == kind)
    if len(matches) != 1 or matches[0].path != expected_name:
        raise AttentionCacheSourceError(f"processed {kind} file is missing or renamed")
    return matches[0]


def _verify_file(path: Path, file: ProcessedOutputFile, expected_hash: str, chunk: int) -> None:
    """Compare physical bytes to both published inventory and protocol identity."""
    if not path.is_file():
        raise AttentionCacheSourceError(f"processed input is missing: {file.path}")
    size, digest = hash_file(path, chunk)
    if size != file.size_bytes or digest != file.sha256 or digest != expected_hash:
        raise AttentionCacheSourceError(f"processed input hash mismatch: {file.path}")


def validate_attention_sources(
    dataset: Path,
    protocol: AttentionProtocol,
    schema: AttentionCardSchema,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> AttentionSources:
    """Reject stale manifests, split files, and schema mismatches before joining."""
    if chunk_size < 1:
        raise AttentionCacheSourceError("hash chunk size must be positive")
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        raise AttentionCacheSourceError("processed manifest is required")
    manifest_bytes = manifest_path.read_bytes()
    if sha256(manifest_bytes).hexdigest() != protocol.processed_manifest_sha256:
        raise AttentionCacheSourceError("processed manifest hash mismatch")
    try:
        manifest = ProcessedDatasetManifest.model_validate_json(manifest_bytes)
    except ValidationError as error:
        raise AttentionCacheSourceError("processed manifest is invalid") from error
    if (
        manifest.dataset_version != protocol.dataset_version
        or manifest.catalog_version != KAGGLE_V6_CARDS.version
        or len(manifest.accepted.eras) != 1
        or manifest.accepted.eras[0].era_id != protocol.balance_era_id
        or schema.balance_era_id != protocol.balance_era_id
        or schema.fingerprint() != protocol.encoding_sha256
    ):
        raise AttentionCacheSourceError("processed dataset, protocol, and encoding disagree")
    required_identities = {entry.card.identity_key for entry in KAGGLE_V6_CARDS.entries}
    if not required_identities.issubset(schema.identity_vocab):
        raise AttentionCacheSourceError("selected schema does not cover the Kaggle source catalog")
    canonical = _inventoried_file(manifest, "canonical", "canonical.parquet")
    split_kind = "temporal_split" if protocol.family == "temporal" else "player_disjoint_split"
    split = _inventoried_file(manifest, split_kind, protocol.split_file)
    canonical_path = dataset / canonical.path
    split_path = dataset / split.path
    _verify_file(canonical_path, canonical, protocol.canonical_sha256, chunk_size)
    _verify_file(split_path, split, protocol.split_sha256, chunk_size)
    summaries = (
        manifest.temporal_split.partitions
        if protocol.family == "temporal"
        else manifest.player_disjoint_split.partitions
    )
    counts: dict[Partition, int] = {item.partition: item.row_count for item in summaries}
    return AttentionSources(manifest_path, canonical_path, split_path, counts)
