"""Versioned inventory and row ranges for a reusable attention input cache.

Cache readers validate this contract before exposing arrays to a trainer.
"""

from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, model_validator

from clash_sos.domain.attention_protocol import AttentionProtocol, Partition
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.domain.manifests import ManifestModel, Sha256

CACHE_VERSION = "attention-input-cache:v1"
SliceRole = Literal["selection_fit", "watch", "refit", "development", "calibration", "reporting"]


class CacheFile(ManifestModel):
    """One immutable cache member, inventoried by size and SHA-256."""

    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: Sha256

    @model_validator(mode="after")
    def check_path(self) -> Self:
        """Forbid paths that escape the cache directory."""
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.path:
            raise ValueError("cache file path must be relative and normalized")
        return self


class CachePartition(ManifestModel):
    """One split partition stored once, even when refit overlaps fit and watch."""

    partition: Partition
    row_count: int = Field(gt=0)
    tokens_path: str
    labels_path: str
    sidecar_paths: tuple[str, ...] = Field(min_length=1)


class CacheSliceRange(ManifestModel):
    """Half-open offsets into one partition's ordered token and label arrays."""

    role: SliceRole
    partition: Partition
    start: int = Field(ge=0)
    stop: int = Field(gt=0)

    @model_validator(mode="after")
    def check_range(self) -> Self:
        """Reject empty offsets before mapping them to protocol slices."""
        if self.start >= self.stop:
            raise ValueError("cache slice range must be nonempty")
        return self


class AttentionCacheManifest(ManifestModel):
    """A complete cache can be reused only with its exact source and protocol."""

    cache_version: Literal["attention-input-cache:v1"] = CACHE_VERSION
    protocol: AttentionProtocol
    encoding_sha256: Sha256
    token_dtype: Literal["uint8"] = "uint8"
    label_dtype: Literal["uint8"] = "uint8"
    partitions: tuple[CachePartition, ...]
    slices: tuple[CacheSliceRange, ...]
    files: tuple[CacheFile, ...]
    watch_actual_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def check_alignment(self) -> Self:
        """Tie partition files and offsets to every declared protocol slice."""
        if self.encoding_sha256 != self.protocol.encoding_sha256:
            raise ValueError("cache encoding does not match protocol")
        partitions = {item.partition: item for item in self.partitions}
        if len(partitions) != len(self.partitions):
            raise ValueError("cache partitions must be unique")
        declared = {
            role: item
            for role in (
                "selection_fit",
                "watch",
                "refit",
                "development",
                "calibration",
                "reporting",
            )
            if (item := getattr(self.protocol, role)) is not None
        }
        ranges = {item.role: item for item in self.slices}
        if len(ranges) != len(self.slices) or set(ranges) != set(declared):
            raise ValueError("cache ranges must match protocol slices exactly")
        for role, source_slice in declared.items():
            offset = ranges[role]
            partition = partitions.get(offset.partition)
            if (
                partition is None
                or offset.partition != source_slice.partition
                or offset.stop > partition.row_count
                or offset.stop - offset.start != source_slice.row_count
            ):
                raise ValueError(f"cache range does not match {role}")
        fit, watch, refit = (ranges[role] for role in ("selection_fit", "watch", "refit"))
        if not (
            fit.partition == watch.partition == refit.partition
            and fit.stop == watch.start
            and (refit.start, refit.stop) == (fit.start, watch.stop)
        ):
            raise ValueError("refit offsets must cover fit and watch")
        if self.watch_actual_fraction != (watch.stop - watch.start) / (refit.stop - refit.start):
            raise ValueError("watch fraction does not match cached ranges")
        expected_paths: set[str] = set()
        for partition in self.partitions:
            expected_paths.update(
                (partition.tokens_path, partition.labels_path, *partition.sidecar_paths)
            )
        files = {item.path for item in self.files}
        if len(files) != len(self.files) or files != expected_paths:
            raise ValueError("cache file inventory must match partitions")
        return self


def dump_cache_manifest(manifest: AttentionCacheManifest) -> bytes:
    """Use the repository's canonical JSON representation for cache metadata."""
    return canonical_json_bytes(manifest.model_dump(mode="python")) + b"\n"
