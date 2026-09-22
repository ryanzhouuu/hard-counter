"""Reject source changes before an attention cache is built or reused."""

from pathlib import Path

import pytest
from attention_cache_fixture import write_cache_dataset

from clash_sos.infrastructure.kaggle_v6.attention_sources import (
    AttentionCacheSourceError,
    validate_attention_sources,
)


def test_sources_match_published_manifest_and_protocol(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    sources = validate_attention_sources(dataset, protocol, schema, chunk_size=17)
    assert sources.partition_counts == {"train": 2, "validation": 2, "test": 1}
    assert sources.canonical_path == dataset / "canonical.parquet"


def test_sources_reject_changed_data_and_protocol(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    with (dataset / "canonical.parquet").open("ab") as output:
        output.write(b"stale")
    with pytest.raises(AttentionCacheSourceError, match="input hash mismatch"):
        validate_attention_sources(dataset, protocol, schema)
    (dataset / "canonical.parquet").unlink()
    with pytest.raises(AttentionCacheSourceError, match="input is missing"):
        validate_attention_sources(dataset, protocol, schema)


def test_sources_reject_changed_manifest_and_schema(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    wrong = protocol.model_copy(update={"encoding_sha256": "b" * 64})
    with pytest.raises(AttentionCacheSourceError, match="encoding disagree"):
        validate_attention_sources(dataset, wrong, schema)
    with (dataset / "manifest.json").open("ab") as output:
        output.write(b" ")
    with pytest.raises(AttentionCacheSourceError, match="manifest hash mismatch"):
        validate_attention_sources(dataset, protocol, schema)
