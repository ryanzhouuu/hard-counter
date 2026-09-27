from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from attention_cache_fixture import write_cache_dataset
from pydantic import ValidationError
from test_canonical_dataset import valid_row_payload
from test_tower_attention import tower_schema

from clash_sos.domain.attention_cache import CacheFile
from clash_sos.domain.attention_dataset import AttentionDatasetManifest, TowerBattleRow
from clash_sos.domain.canonical_dataset import canonical_json_bytes
from clash_sos.infrastructure.kaggle_v6.attention_sources import validate_attention_sources
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file


def test_official_row_requires_recorded_towers_and_normalized_levels() -> None:
    payload = valid_row_payload(
        source_id="official-api",
        side_a_tower="cannoneer:tower",
        side_b_tower="dagger-duchess:tower",
        side_a_tower_level=16,
        side_b_tower_level=16,
    )
    row = TowerBattleRow.model_validate(payload)
    assert row.side_a_tower == "cannoneer:tower"
    for changes in (
        {"source_id": "kaggle-v6"},
        {"side_a_tower": None},
        {"side_b_tower": "knight:base"},
        {"side_a_tower_level": 11},
    ):
        with pytest.raises(ValidationError):
            TowerBattleRow.model_validate({**payload, **changes})


def test_official_snapshot_inventory_and_source_hashes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _, protocol = write_cache_dataset(dataset)
    schema = tower_schema()
    start = datetime(2026, 9, 1, tzinfo=UTC)
    manifest = AttentionDatasetManifest(
        dataset_version=protocol.dataset_version,
        balance_era_id=schema.balance_era_id,
        catalog_version=schema.catalog_version,
        tower_catalog_version="official-towers:2026-09-26",
        row_count=5,
        partition_counts={"train": 2, "validation": 2, "test": 1},
        start=start,
        train_end=start + timedelta(days=1),
        validation_end=start + timedelta(days=2),
        end=start + timedelta(days=3),
        files=tuple(
            CacheFile(path=name, size_bytes=size, sha256=digest)
            for name in ("canonical.parquet", "splits-temporal.parquet")
            for size, digest in (hash_file(dataset / name, 1024),)
        ),
    )
    contents = canonical_json_bytes(manifest.model_dump(mode="python"))
    (dataset / "manifest.json").write_bytes(contents)
    protocol = protocol.model_copy(
        update={
            "balance_era_id": schema.balance_era_id,
            "encoding_sha256": schema.fingerprint(),
            "processed_manifest_sha256": sha256(contents).hexdigest(),
        }
    )
    sources = validate_attention_sources(dataset, protocol, schema)
    assert sources.partition_counts == manifest.partition_counts
    with (dataset / "canonical.parquet").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="input hash mismatch"):
        validate_attention_sources(dataset, protocol, schema)
    for changes in (
        {"row_count": 6},
        {"partition_counts": {"train": 5}},
        {"files": ()},
        {"train_end": start},
        {"start": start.replace(tzinfo=None)},
    ):
        with pytest.raises(ValidationError):
            AttentionDatasetManifest.model_validate({**manifest.model_dump(), **changes})


def test_tower_schema_rejects_untowered_kaggle_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _, protocol = write_cache_dataset(dataset)
    with pytest.raises(ValueError, match="Kaggle rows have no towers"):
        validate_attention_sources(dataset, protocol, tower_schema())
