from pathlib import Path

import numpy as np
import pytest
from test_tower_attention import tower_schema
from tower_dataset_fixture import VERSION, stamp, tower_rows, write_tower_source

from clash_sos.application.attention_dataset_prepare import prepare_official_attention_dataset
from clash_sos.application.dataset_staging import StagingConfig
from clash_sos.domain.attention_dataset import AttentionDatasetManifest
from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.attention_schema import AttentionCardSchema
from clash_sos.infrastructure.kaggle_v6.attention_io import build_attention_cache


def prepare(source: Path, destination: Path) -> Path:
    return prepare_official_attention_dataset(
        source,
        destination,
        schema=tower_schema(),
        dataset_version=VERSION,
        start=stamp(1),
        train_end=stamp(10),
        validation_end=stamp(20),
        end=stamp(27),
        watch_fraction=0.3,
        config=StagingConfig(batch_rows=2, memory_limit="256MB"),
    )


def test_published_snapshot_builds_and_reuses_a_verified_tower_cache(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_tower_source(source)
    original = source.read_bytes()
    destination = prepare(source, tmp_path / "dataset")
    manifest = AttentionDatasetManifest.model_validate_json(
        (destination / "manifest.json").read_bytes()
    )
    protocol = AttentionProtocol.model_validate_json((destination / "protocol.json").read_bytes())
    schema = AttentionCardSchema.model_validate_json(
        (destination / "feature-schema.json").read_bytes()
    )
    assert manifest.partition_counts == {"train": 3, "validation": 2, "test": 1}
    assert protocol.selection_fit.row_count == 2 and protocol.watch.row_count == 1
    assert schema == tower_schema()
    cache = build_attention_cache(destination, tmp_path / "cache", protocol, schema)
    assert cache.manifest.cache_version == "attention-input-cache:v2"
    batches = list(cache.iter_batches("refit", batch_size=2))
    assert sum(len(tokens) for tokens, _ in batches) == 3
    assert all(tokens.shape[1:] == (2, 9) and tokens.dtype == np.uint16 for tokens, _ in batches)
    assert build_attention_cache(destination, tmp_path / "cache", protocol, schema) == cache
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="already exists"):
        prepare(source, destination)
    assert not tuple(tmp_path.glob(".*.building-*"))


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "missing_partition", "tied_train"])
def test_preparation_removes_only_owned_output_after_failure(tmp_path: Path, invalid: str) -> None:
    source = tmp_path / "source.jsonl"
    rows = tower_rows()
    if invalid == "empty":
        source.write_text("")
    elif invalid == "duplicate":
        write_tower_source(source, (*rows, rows[0]))
    elif invalid == "missing_partition":
        write_tower_source(source, rows[:3])
    else:
        write_tower_source(
            source,
            tuple(
                row.model_copy(update={"timestamp": stamp(2)}) if index < 3 else row
                for index, row in enumerate(rows)
            ),
        )
    before = source.read_bytes()
    with pytest.raises(ValueError):
        prepare(source, tmp_path / "dataset")
    assert source.read_bytes() == before
    assert not (tmp_path / "dataset").exists()
    assert not tuple(tmp_path.glob(".*.building-*"))
