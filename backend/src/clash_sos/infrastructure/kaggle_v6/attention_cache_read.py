"""Verify a published attention cache and expose bounded NumPy batches.

Metadata is checked against its ordinal, source protocol, and file hashes before
any trainer receives token arrays.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import ValidationError

from clash_sos.domain.attention_cache import AttentionCacheManifest, CachePartition, SliceRole
from clash_sos.domain.attention_protocol import AttentionProtocol, RowKey, RowKeyDigest
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.infrastructure.kaggle_v6.attention_sources import validate_attention_sources
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file


class AttentionCacheReadError(ValueError):
    """The requested cache is incomplete, stale, corrupt, or incompatible."""


@dataclass(frozen=True)
class AttentionCache:
    """Validated file-backed inputs; arrays are opened only while iterating."""

    directory: Path
    manifest: AttentionCacheManifest

    def iter_batches(
        self, role: SliceRole, *, batch_size: int, seed: int = 0, epoch: int = 0
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Shuffle fit rows by bounded blocks; emit evaluation rows in source order."""
        if batch_size < 1 or seed < 0 or epoch < 0:
            raise AttentionCacheReadError("batch size must be positive; seed and epoch nonnegative")
        offset = next((item for item in self.manifest.slices if item.role == role), None)
        if offset is None:
            raise AttentionCacheReadError(f"cache has no {role} slice")
        partition = next(
            item for item in self.manifest.partitions if item.partition == offset.partition
        )
        tokens = np.load(self.directory / partition.tokens_path, mmap_mode="r")
        labels = np.load(self.directory / partition.labels_path, mmap_mode="r")
        if role in ("selection_fit", "refit"):
            block_rows = max(batch_size, 8192)
            blocks = (offset.stop - offset.start + block_rows - 1) // block_rows
            rng = np.random.default_rng(seed + epoch)
            for block in rng.permutation(blocks):
                first = offset.start + int(block) * block_rows
                indices = np.arange(first, min(first + block_rows, offset.stop))
                rng.shuffle(indices)
                for start in range(0, len(indices), batch_size):
                    selected = indices[start : start + batch_size]
                    yield np.asarray(tokens[selected]), np.asarray(labels[selected])
        else:
            for first in range(offset.start, offset.stop, batch_size):
                stop = min(first + batch_size, offset.stop)
                yield np.asarray(tokens[first:stop]), np.asarray(labels[first:stop])


def _check_arrays(directory: Path, partition: CachePartition) -> None:
    """Reject a validly hashed file whose NumPy header has the wrong layout."""
    try:
        tokens = np.load(directory / partition.tokens_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(directory / partition.labels_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise AttentionCacheReadError("cache arrays cannot be opened") from error
    if (
        tokens.shape != (partition.row_count, 2, 8)
        or labels.shape != (partition.row_count,)
        or tokens.dtype != np.dtype("uint8")
        or labels.dtype != np.dtype("uint8")
    ):
        raise AttentionCacheReadError("cache array shape or dtype mismatch")
    for first in range(0, partition.row_count, 100_000):
        if np.any(labels[first : first + 100_000] > 1):
            raise AttentionCacheReadError("cache labels must be binary")


def _row_key(row: dict[str, object]) -> RowKey:
    """Read one canonical identity from a sidecar without coercing bad fields."""
    stamp = row.get("timestamp")
    fingerprint = row.get("fingerprint")
    member = row.get("archive_member")
    number = row.get("row_number")
    if (
        not isinstance(stamp, datetime)
        or stamp.tzinfo is None
        or stamp.utcoffset() is None
        or not isinstance(fingerprint, str)
        or not isinstance(member, str)
        or type(number) is not int
    ):
        raise AttentionCacheReadError("sidecar row key is invalid")
    return stamp, fingerprint, member, number


def _check_sidecars(directory: Path, manifest: AttentionCacheManifest) -> None:
    """Verify ordinal continuity and the declared digest of every protocol slice."""
    for partition in manifest.partitions:
        offsets = tuple(item for item in manifest.slices if item.partition == partition.partition)
        digests = {item.role: RowKeyDigest() for item in offsets}
        ordinal = 0
        previous: RowKey | None = None
        for relative in partition.sidecar_paths:
            try:
                rows = pl.read_parquet(directory / relative).iter_rows(named=True)
                for raw in rows:
                    row: dict[str, object] = raw
                    if row.get("row_ordinal") != ordinal:
                        raise AttentionCacheReadError("sidecar ordinals are not contiguous")
                    key = _row_key(row)
                    if previous is not None and key <= previous:
                        raise AttentionCacheReadError("sidecar row keys are not unique and sorted")
                    for offset in offsets:
                        if offset.start <= ordinal < offset.stop:
                            digests[offset.role].update(key)
                    previous = key
                    ordinal += 1
            except (OSError, pl.exceptions.PolarsError) as error:
                raise AttentionCacheReadError(f"sidecar cannot be read: {relative}") from error
        if ordinal != partition.row_count:
            raise AttentionCacheReadError("sidecar row count does not match arrays")
        for offset in offsets:
            source_slice = getattr(manifest.protocol, offset.role)
            if (
                source_slice is None
                or digests[offset.role].hexdigest() != source_slice.row_keys_sha256
            ):
                raise AttentionCacheReadError(f"{offset.role} sidecar row digest mismatch")


def load_attention_cache(
    dataset: Path,
    directory: Path,
    protocol: AttentionProtocol,
    schema: AttentionCardSchema,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> AttentionCache:
    """Reject stale source bytes, missing members, and broken sidecar alignment."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise AttentionCacheReadError("attention cache manifest is required")
    try:
        manifest = AttentionCacheManifest.model_validate_json(manifest_path.read_bytes())
    except ValidationError as error:
        raise AttentionCacheReadError("attention cache manifest is invalid") from error
    if manifest.protocol != protocol or manifest.encoding_sha256 != schema.fingerprint():
        raise AttentionCacheReadError("attention cache protocol or schema mismatch")
    try:
        validate_attention_sources(dataset, protocol, schema, chunk_size=chunk_size)
    except ValueError as error:
        raise AttentionCacheReadError(str(error)) from error
    for file in manifest.files:
        path = directory / file.path
        if not path.is_file():
            raise AttentionCacheReadError(f"cache member is missing: {file.path}")
        size, digest = hash_file(path, chunk_size)
        if size != file.size_bytes or digest != file.sha256:
            raise AttentionCacheReadError(f"cache member hash mismatch: {file.path}")
    for partition in manifest.partitions:
        _check_arrays(directory, partition)
    _check_sidecars(directory, manifest)
    return AttentionCache(directory, manifest)
