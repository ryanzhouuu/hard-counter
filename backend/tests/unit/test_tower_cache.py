from hashlib import sha256
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_attention_dataset_prepare import prepare
from test_tower_attention import tower_schema
from tower_dataset_fixture import write_tower_source

from clash_sos.domain.attention_cache import AttentionCacheManifest, dump_cache_manifest
from clash_sos.domain.attention_protocol import AttentionProtocol
from clash_sos.domain.matchup_baseline import should_mirror_sides
from clash_sos.infrastructure.kaggle_v6.attention_cache_read import load_attention_cache
from clash_sos.infrastructure.kaggle_v6.attention_io import build_attention_cache
from clash_sos.infrastructure.kaggle_v6.audit_io import hash_file


def test_cache_mirrors_towers_with_their_decks_and_checks_slot_types(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_tower_source(source)
    dataset = prepare(source, tmp_path / "dataset")
    protocol = AttentionProtocol.model_validate_json((dataset / "protocol.json").read_bytes())
    schema = tower_schema()
    cache = build_attention_cache(dataset, tmp_path / "cache", protocol, schema)
    partition = cache.manifest.partitions[0]
    path = cache.directory / partition.tokens_path
    tokens = np.load(path)
    assert tokens.shape == (3, 2, 9)
    canonical = pl.read_parquet(dataset / "canonical.parquet").sort("timestamp").head(3)
    for index, row in enumerate(canonical.iter_rows(named=True)):
        towers = [row["side_a_tower"], row["side_b_tower"]]
        if should_mirror_sides(row["fingerprint"], seed=protocol.mirror_seed):
            towers.reverse()
        assert tokens[index, :, 8].tolist() == [schema.identity_index[tower] for tower in towers]
    first = canonical.row(0, named=True)
    side = "b" if should_mirror_sides(first["fingerprint"], seed=protocol.mirror_seed) else "a"
    expected = sha256(
        f"{first[f'side_{side}_deck_hash']}|{first[f'side_{side}_tower']}|16".encode()
    ).hexdigest()
    sidecar = pl.read_parquet(cache.directory / partition.sidecar_paths[0])
    assert sidecar.row(0, named=True)["deck_a_hash"] == expected
    manifest_path = cache.directory / "manifest.json"
    for corruption in ("wrong_slot", "outside_vocabulary"):
        changed = tokens.copy()
        if corruption == "wrong_slot":
            changed[0, 0, 0], changed[0, 0, 8] = changed[0, 0, 8], changed[0, 0, 0]
        else:
            changed[0, 0, 0] = len(schema.identity_vocab)
        np.save(path, changed)
        size, digest = hash_file(path, 1024)
        manifest = AttentionCacheManifest.model_validate_json(manifest_path.read_bytes())
        files = tuple(
            file.model_copy(update={"size_bytes": size, "sha256": digest})
            if file.path == partition.tokens_path
            else file
            for file in manifest.files
        )
        manifest_path.write_bytes(dump_cache_manifest(manifest.model_copy(update={"files": files})))
        message = "ninth slot" if corruption == "wrong_slot" else r"outside.*vocabulary"
        with pytest.raises(ValueError, match=message):
            load_attention_cache(dataset, cache.directory, protocol, schema)
