"""Validate cache inventory and slice offsets independently of file I/O."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from clash_sos.domain.attention_cache import (
    AttentionCacheManifest,
    CacheFile,
    CachePartition,
    CacheSliceRange,
    dump_cache_manifest,
)
from clash_sos.domain.attention_protocol import AttentionProtocol, AttentionSlice

STAMP = datetime(2026, 6, 1, tzinfo=UTC)
HASH = "a" * 64


def slice_at(partition: str, first: int, last: int, count: int) -> AttentionSlice:
    """Use distinct half-open bounds for a compact protocol fixture."""
    return AttentionSlice.model_validate(
        {
            "partition": partition,
            "start": STAMP + timedelta(days=first),
            "end": STAMP + timedelta(days=last),
            "row_count": count,
            "row_keys_sha256": HASH,
        }
    )


def cache_manifest() -> AttentionCacheManifest:
    """Build a valid refit-overlap cache with one evaluation partition."""
    protocol = AttentionProtocol(
        family="temporal",
        dataset_version="kaggle-v6-ranked16-v2",
        balance_era_id="2026-06",
        processed_manifest_sha256=HASH,
        canonical_sha256=HASH,
        split_file="splits-temporal.parquet",
        split_sha256=HASH,
        encoding_sha256=HASH,
        mirror_seed=0,
        selection_fit=slice_at("train", 0, 1, 4),
        watch=slice_at("train", 1, 2, 2),
        refit=slice_at("train", 0, 2, 6),
        development=slice_at("validation", 2, 3, 2),
    )
    partitions = (
        CachePartition(
            partition="train",
            row_count=6,
            tokens_path="train/tokens.npy",
            labels_path="train/labels.npy",
            sidecar_paths=("train/sidecar-00000.parquet",),
        ),
        CachePartition(
            partition="validation",
            row_count=2,
            tokens_path="validation/tokens.npy",
            labels_path="validation/labels.npy",
            sidecar_paths=("validation/sidecar-00000.parquet",),
        ),
    )
    ranges = (
        CacheSliceRange(role="selection_fit", partition="train", start=0, stop=4),
        CacheSliceRange(role="watch", partition="train", start=4, stop=6),
        CacheSliceRange(role="refit", partition="train", start=0, stop=6),
        CacheSliceRange(role="development", partition="validation", start=0, stop=2),
    )
    paths = tuple(
        path
        for item in partitions
        for path in (item.tokens_path, item.labels_path, *item.sidecar_paths)
    )
    return AttentionCacheManifest(
        protocol=protocol,
        encoding_sha256=HASH,
        partitions=partitions,
        slices=ranges,
        files=tuple(CacheFile(path=path, size_bytes=1, sha256=HASH) for path in paths),
        watch_actual_fraction=2 / 6,
    )


def test_manifest_round_trips_and_tracks_exact_ranges() -> None:
    manifest = cache_manifest()
    assert manifest.partitions[0].row_count == 6
    assert manifest.slices[2].start == 0
    assert AttentionCacheManifest.model_validate_json(dump_cache_manifest(manifest)) == manifest


def test_manifest_rejects_missing_range_and_wrong_fraction() -> None:
    payload = cache_manifest().model_dump(mode="python")
    payload["slices"] = payload["slices"][:-1]
    with pytest.raises(ValidationError, match="ranges must match"):
        AttentionCacheManifest.model_validate(payload)
    payload = cache_manifest().model_dump(mode="python")
    payload["watch_actual_fraction"] = 0.5
    with pytest.raises(ValidationError, match="watch fraction"):
        AttentionCacheManifest.model_validate(payload)


def test_manifest_rejects_missing_inventory_and_escaping_path() -> None:
    payload = cache_manifest().model_dump(mode="python")
    payload["files"] = payload["files"][:-1]
    with pytest.raises(ValidationError, match="inventory"):
        AttentionCacheManifest.model_validate(payload)
    with pytest.raises(ValidationError, match="relative and normalized"):
        CacheFile(path="../escape", size_bytes=1, sha256=HASH)
