from collections.abc import Iterator
from math import log

import pytest

from clash_sos.domain.matchup_clusters import (
    CLUSTER_COUNT,
    ClusterBattle,
    ClusterMatchupTable,
    fit_prefix_matchups,
)

IDENTITIES = ("archers:base", "giant:base", "knight:base", "musketeer:base")


def _table() -> ClusterMatchupTable:
    return ClusterMatchupTable.from_payload(
        {
            "alpha": 1.0,
            "batch_size": 4096,
            "centroids": [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            "cluster_count": 2,
            "max_iter": 100,
            "n_init": 1,
            "seed": 0,
            "wins": [[0, 3], [1, 4]],
        },
        IDENTITIES,
    )


def test_known_wins_are_antisymmetric_log_odds() -> None:
    table = _table()
    assert table.cluster_of(("knight:base",)) == 0
    assert table.cluster_of(("archers:base",)) == 1
    assert table.log_odds(("knight:base",), ("archers:base",)) == pytest.approx(log(2))
    assert table.log_odds(("archers:base",), ("knight:base",)) == pytest.approx(-log(2))


def test_same_cluster_unseen_pair_and_empty_side_are_zero() -> None:
    table = _table()
    assert table.log_odds(("knight:base",), ("knight:base",)) == 0.0
    fresh = ClusterMatchupTable.from_payload(
        {
            "alpha": 1.0,
            "batch_size": 8,
            "centroids": [[1.0, 0.0], [0.0, 1.0]],
            "cluster_count": 2,
            "max_iter": 10,
            "n_init": 1,
            "seed": 0,
            "wins": [[0, 0], [0, 0]],
        },
        ("archers:base", "knight:base"),
    )
    assert fresh.log_odds(("archers:base",), ("knight:base",)) == 0.0
    assert table.log_odds((), ("knight:base",)) == 0.0
    assert table.cluster_of(()) is None


def test_equidistant_centroids_take_the_lower_index() -> None:
    table = ClusterMatchupTable.from_payload(
        {
            "alpha": 1.0,
            "batch_size": 8,
            "centroids": [[0.0, 0.0], [0.0, 0.0]],
            "cluster_count": 2,
            "max_iter": 10,
            "n_init": 1,
            "seed": 0,
            "wins": [[0, 0], [0, 0]],
        },
        ("archers:base", "knight:base"),
    )
    assert table.cluster_of(("knight:base",)) == 0


def test_payload_round_trip_keeps_log_odds() -> None:
    table = _table()
    restored = ClusterMatchupTable.from_payload(table.to_payload(), IDENTITIES)
    assert restored.log_odds(("knight:base",), ("archers:base",)) == pytest.approx(log(2))


def _battles() -> Iterator[ClusterBattle]:
    prefix = (
        ClusterBattle("a", "b", ("knight:base",), ("archers:base",), 1),
        ClusterBattle("a", "b", ("knight:base",), ("archers:base",), 1),
        ClusterBattle("a", "b", ("knight:base",), ("archers:base",), 1),
    )
    tail = ClusterBattle("a", "b", ("knight:base",), ("archers:base",), 0)
    yield from prefix
    yield tail


def test_watch_tail_outcome_stays_out_of_the_win_table() -> None:
    identities = ("archers:base", "knight:base")
    prefix = fit_prefix_matchups(
        _battles,
        prefix_rows=3,
        identities=identities,
        cluster_count=2,
        batch_size=8,
        max_iter=20,
    )
    including_tail = fit_prefix_matchups(
        _battles,
        prefix_rows=4,
        identities=identities,
        cluster_count=2,
        batch_size=8,
        max_iter=20,
    )
    assert prefix.wins_total() == 3
    assert including_tail.wins_total() == 4
    assert prefix.log_odds(("knight:base",), ("archers:base",)) > including_tail.log_odds(
        ("knight:base",), ("archers:base",)
    )


def test_refitting_the_same_prefix_keeps_centroids() -> None:
    identities = ("archers:base", "knight:base")
    first = fit_prefix_matchups(
        _battles,
        prefix_rows=3,
        identities=identities,
        cluster_count=2,
        batch_size=8,
        max_iter=20,
        seed=0,
    )
    second = fit_prefix_matchups(
        _battles,
        prefix_rows=3,
        identities=identities,
        cluster_count=2,
        batch_size=8,
        max_iter=20,
        seed=0,
    )
    assert first.to_payload()["centroids"] == second.to_payload()["centroids"]


def test_repeated_hash_with_different_cards_fails() -> None:
    def battles() -> Iterator[ClusterBattle]:
        yield ClusterBattle("h", "g", ("knight:base",), ("archers:base",), 1)
        yield ClusterBattle("h", "g", ("musketeer:base",), ("archers:base",), 1)

    with pytest.raises(ValueError, match="deck hash has two card lists"):
        fit_prefix_matchups(
            battles,
            prefix_rows=2,
            identities=IDENTITIES,
            cluster_count=1,
            batch_size=8,
        )


def test_cluster_count_above_the_prefix_fails() -> None:
    with pytest.raises(ValueError, match="cluster count exceeds distinct prefix decks"):
        fit_prefix_matchups(
            _battles,
            prefix_rows=3,
            identities=("archers:base", "knight:base"),
            cluster_count=3,
            batch_size=8,
        )


def test_default_cluster_count_is_64() -> None:
    assert CLUSTER_COUNT == 64
