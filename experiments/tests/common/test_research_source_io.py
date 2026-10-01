from pathlib import Path

import numpy as np
import polars as pl
import pytest
from experiments.common.protocol import search_protocol
from experiments.common.source_io import load_snapshot
from experiments.common.source_rows import event_mapping_digest
from test_attention_dataset_prepare import prepare
from tower_dataset_fixture import stamp, tower_rows, write_tower_source

from clash_sos.domain.attention_cache import AttentionCacheManifest
from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.matchup_baseline import should_mirror_sides
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file


def bounded_snapshot(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source.jsonl"
    write_tower_source(source)
    dataset = prepare(source, tmp_path / "dataset")
    original = AttentionProtocol.model_validate_json((dataset / "protocol.json").read_bytes())
    resolved = search_protocol(original, dataset, calibration_end=stamp(11))
    protocol = tmp_path / "search.json"
    protocol.write_text(resolved.model_dump_json())
    return dataset, protocol, dataset / "feature-schema.json", tmp_path / "cache"


def test_snapshot_join_preserves_events_players_labels_tokens_and_search_roles(
    tmp_path: Path,
) -> None:
    paths = bounded_snapshot(tmp_path)
    access, population, schema = load_snapshot(*paths, row_cap=5)
    rows = (
        *access.read("refit", "fit"),
        *access.read("calibration", "calibrate"),
        *access.read("development", "compare"),
    )
    assert len(rows) == 5
    assert population.event_mapping_sha256 == event_mapping_digest(rows)
    assert population.catalog_sha256 == schema.catalog_sha256
    assert len(population.snapshot_files) == 3
    originals = {row.event_key: row for row in tower_rows()}
    for row in rows:
        original = originals[row.event_key]
        mirrored = should_mirror_sides(original.fingerprint, seed=access.protocol.mirror_seed)
        assert row.mirrored == mirrored
        assert row.label == int(not mirrored)
        assert row.player_a == (
            original.side_b_player_id.value if mirrored else original.side_a_player_id.value
        )
        assert len(row.tokens[0]) == 9
    assert not (paths[-1] / "test").exists()
    assert load_snapshot(*paths, row_cap=5)[1] == population
    with pytest.raises(PermissionError):
        access.read("reporting", "compare")


def test_cap_and_search_role_checks_fail_before_cache_creation(tmp_path: Path) -> None:
    paths = bounded_snapshot(tmp_path)
    for cap in (None, 0, 4):
        with pytest.raises(ValueError, match="row cap"):
            load_snapshot(*paths, row_cap=cap)
        assert not paths[-1].exists()
    with pytest.raises(ValueError, match="exclude reporting"):
        load_snapshot(paths[0], paths[0] / "protocol.json", paths[2], paths[3], row_cap=5)


def test_snapshot_loader_rejects_cache_corruption(tmp_path: Path) -> None:
    paths = bounded_snapshot(tmp_path)
    load_snapshot(*paths, row_cap=5)
    tokens_path = paths[-1] / "train/tokens.npy"
    tokens = np.load(tokens_path)
    tokens[0, 0, 0] += 1
    with tokens_path.open("wb") as output:
        np.save(output, tokens, allow_pickle=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(*paths, row_cap=5)


def test_resealed_cache_player_changes_fail_canonical_orientation_join(tmp_path: Path) -> None:
    paths = bounded_snapshot(tmp_path)
    load_snapshot(*paths, row_cap=5)
    manifest_path = paths[-1] / "manifest.json"
    manifest = AttentionCacheManifest.model_validate_json(manifest_path.read_bytes())
    sidecar = manifest.partitions[0].sidecar_paths[0]
    sidecar_path = paths[-1] / sidecar
    pl.read_parquet(sidecar_path).with_columns(
        pl.lit("unrelated-player").alias("side_a_player_id")
    ).write_parquet(sidecar_path)
    size, digest = hash_file(sidecar_path, 1024)
    modified = manifest.model_copy(
        update={
            "files": tuple(
                item.model_copy(update={"size_bytes": size, "sha256": digest})
                if item.path == sidecar
                else item
                for item in manifest.files
            )
        }
    )
    manifest_path.write_text(modified.model_dump_json())
    with pytest.raises(ValueError, match="canonical event join"):
        load_snapshot(*paths, row_cap=5)
