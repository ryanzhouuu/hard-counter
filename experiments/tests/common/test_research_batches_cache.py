from pathlib import Path

import numpy as np
import pytest
from experiments.common.batches import aligned_batches, rms_scales
from experiments.common.cache import (
    FeatureCacheIdentity,
    load_feature_cache,
    oriented_digest,
    write_feature_cache,
)
from experiments.tests.common.research_fixture import HASH, rows

from clash_sos.domain.attention_protocol import digest_row_keys


def test_one_shuffle_keeps_tokens_labels_features_and_players_aligned() -> None:
    population = rows()
    features = np.arange(len(population) * 2).reshape(-1, 2)
    visited: list[int] = []
    for batch in aligned_batches(population, features, batch_size=3, shuffle=True, seed=9):
        visited.extend(int(i) for i in batch.indices)
        for index, token, label, feature, player in zip(
            batch.indices, batch.tokens, batch.labels, batch.features, batch.players, strict=True
        ):
            source = population[int(index)]
            assert np.array_equal(token, source.tokens)
            assert label == source.label
            assert np.array_equal(feature, features[index])
            assert player == (source.player_a, source.player_b)
    assert sorted(visited) == list(range(len(population)))
    assert visited != sorted(visited)


def test_rms_preserves_swap_and_zero_columns() -> None:
    values = np.array([[0.0, 2.0], [0.0, 4.0]])
    scales = rms_scales(values)
    assert np.array_equal(scales, [1.0, np.sqrt(10.0)])
    assert np.array_equal(-values / scales, -(values / scales))


def test_feature_cache_rejects_changed_rows_mechanics_and_bytes(tmp_path: Path) -> None:
    population = rows()
    values = np.zeros((len(population), 2))
    identity = FeatureCacheIdentity(
        oriented_sha256=oriented_digest(population),
        encoding_sha256=HASH,
        mechanics_sha256=HASH,
        feature_names=("a", "b"),
        formulas=("difference", "context"),
        training_rows_sha256=HASH,
        scales=(1.0, 1.0),
        row_keys_sha256=digest_row_keys(r.key for r in population),
        row_count=len(population),
    )
    destination = tmp_path / "cache"
    write_feature_cache(destination, population, values, identity)
    assert np.array_equal(load_feature_cache(destination, identity), values)
    with pytest.raises(ValueError, match="configuration"):
        load_feature_cache(destination, identity.model_copy(update={"mechanics_sha256": "b" * 64}))
    with pytest.raises(ValueError, match="orientation"):
        write_feature_cache(
            tmp_path / "bad", tuple(r.swapped() for r in population), values, identity
        )
    with (destination / "features.npy").open("ab") as output:
        output.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_feature_cache(destination, identity)
