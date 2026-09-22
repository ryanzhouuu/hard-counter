"""Publish complete attention caches and reuse only exact verified matches."""

from pathlib import Path
from typing import NoReturn

import pytest
from attention_cache_fixture import write_cache_dataset

from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import AttentionCacheReadError
from clash_sos.infrastructure.kaggle_v6.attention_io import build_attention_cache

CONFIG = StagingConfig(memory_limit="256MB", threads=1, batch_rows=1, chunk_size=1024)


def test_builder_publishes_and_reuses_exact_cache(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    destination = tmp_path / "data" / "cache"
    built = build_attention_cache(
        dataset, destination, protocol, schema, config=CONFIG, sidecar_rows=1
    )
    assert built.directory == destination
    assert (destination / "manifest.json").is_file()
    assert len(built.manifest.partitions) == 3
    assert built.manifest.watch_actual_fraction == 0.5
    first_bytes = (destination / "manifest.json").read_bytes()
    reused = build_attention_cache(
        dataset, destination, protocol, schema, config=CONFIG, sidecar_rows=2
    )
    assert reused.manifest == built.manifest
    assert (destination / "manifest.json").read_bytes() == first_bytes
    assert not tuple(destination.parent.glob(".*.building-*"))


def test_builder_rejects_existing_mismatch_without_overwrite(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    destination = tmp_path / "cache"
    build_attention_cache(dataset, destination, protocol, schema, config=CONFIG)
    before = (destination / "manifest.json").read_bytes()
    changed = protocol.model_copy(update={"mirror_seed": 1})
    with pytest.raises(AttentionCacheReadError, match="protocol or schema mismatch"):
        build_attention_cache(dataset, destination, changed, schema, config=CONFIG)
    assert (destination / "manifest.json").read_bytes() == before


def test_builder_cleans_owned_workspace_after_stream_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    schema, protocol = write_cache_dataset(dataset)
    destination = tmp_path / "cache"

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        """Simulate a failed partition stream after the workspace was created."""
        raise RuntimeError("stream interrupted")

    monkeypatch.setattr(
        "clash_sos.infrastructure.kaggle_v6.attention_io.write_attention_partition", fail
    )
    with pytest.raises(RuntimeError, match="stream interrupted"):
        build_attention_cache(dataset, destination, protocol, schema, config=CONFIG)
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".cache.building-*"))
